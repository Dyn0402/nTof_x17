#!/usr/bin/env python3
"""
37_threading_displays.py — event displays that answer, per event, whether the
M3 reference track threads the *reconstructed cluster* better under the
waveform-first reconstruction than under the production (aggregate-time) one.

The comparison has to avoid the obvious circularity: the forward fit returns a
line, so drawing its own charge profile along that line proves nothing.  The
cluster shown here is therefore obtained from a **line-free 2-D deconvolution**:

    data(i,t) = sum_{j,k} Q[j,k] . [ h0(t-t0-u_k)            i = j
                                   + c1 h1(t-t0-u_k-tau)     |i-j| = 1
                                   + c2 h2(t-t0-u_k-2tau)    |i-j| = 2 ]

Q[j,k] >= 0 is the charge on strip j arriving in depth bin k.  Only the
calibrated impulse template and the resistive-sharing kernel enter (script 03 /
13 products); no track, no slope, no position model.  Solved by NNLS with a
second-difference Tikhonov penalty along depth.  The M3 line is then overlaid
on that density -- an honest "does the track go through the cluster" test.

The production cluster on the same axes is the hit tree the analysis actually
uses (significance floor applied, as in cosmic_micro_tpc_analysis), each strip
placed at the depth implied by its aggregate hit time.  Both clusters use the
same z origin (the forward fit's t0 = charge arrival from the mesh) and the
same drift velocity, so the panels differ only in how the charge is timed.

Outputs (in <Analysis>/mx17_3/waveform_first/threading_displays/):
    event_<eid>_planes.png   raw waveforms + both clusters, X and Y
    event_<eid>_3d.png       3-D display, production cloud vs deconvolved cloud
    threading_census.png     population deviation-from-reference, both methods
    threading_census.json    the numbers behind it

Usage:
    ../../.venv/bin/python 37_threading_displays.py            # 6 displays + census
    ../../.venv/bin/python 37_threading_displays.py --n 12 --census 400
    ../../.venv/bin/python 37_threading_displays.py --eids 729,9852
"""
import os
import sys
import json
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import nnls

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model2 as fm2
from wft_reco import WFTReco

BASE = fm2.BASE
OUT = os.path.join(BASE, 'threading_displays')
HYPER_JSON = os.path.join(BASE, 'hyper_v2.json')

KDEEP = 20                 # depth bins in the deconvolution (20 x 60 ns)
LAMBDA = 1.0               # Tikhonov weight along depth (see --lam)
SIG_REL_FLOOR = 0.10       # production per-plane relative significance floor
PROD_GAP_MM = 12.0         # production spatial clustering gap (GAP_THRESHOLD_MM)
# production's own calibrated drift velocity for this run, per plane
# (alignment_tpc_veto50/angular_resolution.json) — used only for the
# "production in its own frame" robustness check
V_PROD = {'x': 31.5, 'y': 33.5}
GAP_MM = 29.0
Z_DEEP = 15.0              # "upper half of the gap" for the depth-resolved metric
QFRAC = 0.05               # depth bins below this fraction of peak charge are noise
DEPTH_SLICES = [(0.0, 6.0, 'z0'), (6.0, 12.0, 'z1'), (12.0, 18.0, 'z2'),
                (18.0, 24.0, 'z3'), (24.0, 29.0, 'z4')]


# ----------------------------------------------------------------- deconvolution
def deconv2d(P, plane, t0, hyper, lam=LAMBDA, K=KDEEP):
    """Line-free 2-D charge deconvolution.  Returns (Q, model, chi2, dof, uk, pos)."""
    W, noise, pos, sat = fm2.prep_plane(P, plane)
    n, ns = W.shape
    kY = hyper['kY'] if plane == 'y' else 1.0
    c1, c2, tau = hyper['c1'] * kY, hyper['c2'] * kY, hyper['tau_s']
    tmpl, sm = fm2._templates(plane, hyper['sigma_s'])
    uk = (np.arange(K) + 0.5) * fm2.DT
    TS = fm2.TS
    H0 = np.stack([np.interp(TS - (t0 + u), fm2.TGRID, tmpl, left=0, right=0) for u in uk])
    H1 = np.stack([np.interp(TS - (t0 + u + tau), fm2.TGRID, sm, left=0, right=0) for u in uk])
    H2 = np.stack([np.interp(TS - (t0 + u + 2 * tau), fm2.TGRID, sm, left=0, right=0) for u in uk])

    A = np.zeros((n, ns, n, K))
    for j in range(n):
        A[j, :, j, :] += H0.T
        for off, c, H in ((1, c1, H1), (2, c2, H2)):
            if j + off < n:
                A[j + off, :, j, :] += c * H.T
            if j - off >= 0:
                A[j - off, :, j, :] += c * H.T
    A = A.reshape(n * ns, n * K)

    ok = ~sat.reshape(-1)
    wgt = np.repeat(1.0 / noise, ns)
    Aw = (A * wgt[:, None])[ok]
    yw = (W / noise[:, None]).reshape(-1)[ok]

    # second difference along depth, per strip
    rows = []
    for j in range(n):
        for k in range(K - 2):
            r = np.zeros(n * K)
            r[j * K + k:j * K + k + 3] = (1.0, -2.0, 1.0)
            rows.append(r)
    L = np.asarray(rows)
    q, _ = nnls(np.vstack([Aw, lam * L]),
                np.concatenate([yw, np.zeros(len(L))]), maxiter=60 * n * K)
    Q = q.reshape(n, K)
    model = (A @ q).reshape(n, ns)
    chi2 = float((((W - model) / noise[:, None]) ** 2)[~sat].sum())
    return Q, model, chi2, int((~sat).sum()), uk, pos


def free_ladder(P, plane, t0, hyper, mu_init, K=KDEEP, sweeps=3,
                span=2.0, step=0.04):
    """Line-free *sub-pitch* cluster: one free transverse position mu_k and one
    charge q_k per 60 ns depth bin, no relation imposed between bins.

    The strip-quantised centroid of the 2-D deconvolution collapses onto a
    single strip when the cloud is narrow (< 1 pitch); here each depth bin's
    charge is instead placed at a continuous position, folded through the same
    strip integration + resistive-sharing kernel + impulse template.  Solved by
    exact coordinate descent: for a trial mu_k the optimal q_k is a one-line
    non-negative least-squares update, so a fine position grid is cheap.

    Returns (mu, q) with q = 0 for bins the data do not support.
    """
    W, noise, pos, sat = fm2.prep_plane(P, plane)
    n, ns = W.shape
    kY = hyper['kY'] if plane == 'y' else 1.0
    c1, c2, tau = hyper['c1'] * kY, hyper['c2'] * kY, hyper['tau_s']
    tmpl, sm = fm2._templates(plane, hyper['sigma_s'])
    uk = (np.arange(K) + 0.5) * fm2.DT
    ok = (~sat).reshape(-1)
    y = (W / noise[:, None]).reshape(-1)[ok]
    inv = (1.0 / noise)[:, None]

    H0 = np.stack([np.interp(fm2.TS - (t0 + u), fm2.TGRID, tmpl, left=0, right=0) for u in uk])
    H1 = np.stack([np.interp(fm2.TS - (t0 + u + tau), fm2.TGRID, sm, left=0, right=0) for u in uk])
    H2 = np.stack([np.interp(fm2.TS - (t0 + u + 2 * tau), fm2.TGRID, sm, left=0, right=0) for u in uk])
    sig_k = np.sqrt(hyper['sigma_p0'] ** 2 + hyper['Dp'] ** 2 * uk)

    from scipy.special import erf

    def column(k, mu):
        s = np.sqrt(2) * max(sig_k[k], 1e-3)
        F = 0.5 * (erf((pos + fm2.PITCH / 2 - mu) / s)
                   - erf((pos - fm2.PITCH / 2 - mu) / s))
        col = np.outer(F, H0[k])
        sh1 = np.zeros_like(F); sh1[1:] += c1 * F[:-1]; sh1[:-1] += c1 * F[1:]
        col += np.outer(sh1, H1[k])
        sh2 = np.zeros_like(F); sh2[2:] += c2 * F[:-2]; sh2[:-2] += c2 * F[2:]
        col += np.outer(sh2, H2[k])
        return (col * inv).reshape(-1)[ok]

    mu = np.array(mu_init, float)
    fill = np.nanmedian(mu) if np.isfinite(mu).any() else float(np.mean(pos))
    mu[~np.isfinite(mu)] = fill
    A = np.stack([column(k, mu[k]) for k in range(K)], axis=1)
    q = np.zeros(K)
    # A depth bin whose response has almost left the readout window is
    # unconstrained: (r.a)^2/(a.a) blows up as a.a -> 0.  Freeze those bins.
    dens0 = (A * A).sum(axis=0)
    den_floor = 1e-3 * float(dens0.max()) if dens0.max() > 0 else 0.0
    usable = dens0 >= den_floor
    for _ in range(sweeps):
        for k in range(K):
            if not usable[k]:
                q[k] = 0.0
                continue
            r = y - A @ q + A[:, k] * q[k]          # residual without bin k
            grid = mu[k] + np.arange(-span, span + 1e-9, step)
            best = (None, -np.inf, 0.0)
            for m in grid:
                a = column(k, m)
                den = float(a @ a)
                if den < den_floor:
                    continue
                qk = max(0.0, float(r @ a) / den)
                gain = qk * qk * den                # chi2 reduction from bin k
                if gain > best[1]:
                    best = (m, gain, qk)
            if best[0] is None:
                q[k] = 0.0
                continue
            mu[k], q[k] = best[0], best[2]
            A[:, k] = column(k, mu[k])
    return mu, q


def _wq(d, w, q):
    o = np.argsort(d)
    cw = np.cumsum(w[o]) / w.sum()
    return float(d[o][min(np.searchsorted(cw, q), len(d) - 1)])


def cluster_deviation(zc, pc, wc, p0_ref, tan_ref, zlo=0.0, zhi=GAP_MM):
    """Charge-weighted median / 90th pct |cluster - reference line| in a depth slice."""
    m = (np.isfinite(zc) & np.isfinite(pc) & (wc > 0)
         & (zc >= zlo) & (zc <= zhi))
    if m.sum() == 0:
        return np.nan, np.nan
    d = np.abs(pc[m] - (p0_ref + tan_ref * zc[m]))
    return _wq(d, wc[m], 0.5), _wq(d, wc[m], 0.9)


# ----------------------------------------------------------------- per event
def prod_cluster(ev, plane, pos_map, feu):
    """The cluster the production reconstruction actually fits: hits past the
    per-plane relative significance floor, spatially clustered with the
    production gap threshold, largest cluster kept (cosmic_micro_tpc_analysis
    _fit_single_axis).  Returns (hit time, position, amplitude)."""
    empty = (np.array([]), np.array([]), np.array([]))
    h = ev[plane].get('hits')
    if not h:
        return empty
    ch, amp, t, sig = h['ch'], h['amp'], h['time'], h['sig']
    keep = np.isfinite(sig) & (sig > 0)
    if keep.any():                                   # per-plane relative floor
        keep &= sig >= SIG_REL_FLOOR * np.nanmax(sig[keep])
    else:
        keep = np.ones(len(ch), bool)
    p = pos_map[feu][ch.astype(int)]
    keep &= np.isfinite(p)
    if keep.sum() == 0:
        return empty
    p, t, amp = p[keep], t[keep], amp[keep]
    o = np.argsort(p)
    p, t, amp = p[o], t[o], amp[o]
    lab = np.concatenate([[0], np.cumsum(np.diff(p) > PROD_GAP_MM)])
    big = np.bincount(lab).argmax()
    m = lab == big
    return t[m], p[m], amp[m]


def analyse_event(ev, reco, meta, feus, lam=LAMBDA):
    """Fit + deconvolve both planes.  Returns a dict per plane plus event info."""
    out = {'eid': ev['eid'], 'radial_residual': ev['radial_residual']}
    # z origin of the production event display: earliest hit time over both planes
    t0_own = min(ev['prod']['t0_x'], ev['prod']['t0_y'])
    for plane in ('x', 'y'):
        P = ev[plane]
        tan_ref = ev[f'tan_{plane}']
        p0_ref = ev[f'ref_mesh_{plane}']
        f = reco.fit_plane(P['W'], P['pos'], P['noise'], P['ch'], plane,
                           tan_seed=tan_ref, p0_seed=p0_ref)
        Q, model, chi2, dof, uk, pos = deconv2d(P, plane, f['t0'], reco.hyper, lam=lam)
        z = uk * reco.v_drift * 1e-3
        wbin = Q.sum(axis=0)
        cen = np.full(len(z), np.nan)
        nz = wbin > 0
        cen[nz] = (Q * pos[:, None]).sum(axis=0)[nz] / wbin[nz]
        live = wbin > QFRAC * wbin.max() if wbin.max() > 0 else np.zeros(len(z), bool)

        # sub-pitch line-free ladder, seeded from the deconvolution centroids
        mu, qk = free_ladder(P, plane, f['t0'], reco.hyper, cen)
        live_mu = qk > QFRAC * qk.max() if qk.max() > 0 else np.zeros(len(z), bool)

        tp, pp, ap = prod_cluster(ev, plane, meta['pos_map'], feus[plane])
        # common frame (this study): forward-fit t0 and v for both methods
        zp = (tp - f['t0']) * reco.v_drift * 1e-3
        # production's own frame (what the existing 3-D displays draw): earliest
        # hit as z=0 and the production per-plane calibrated velocity
        zp_own = (tp - t0_own) * V_PROD[plane] * 1e-3
        dev_ff = cluster_deviation(z[live_mu], mu[live_mu], qk[live_mu],
                                   p0_ref, tan_ref)
        dev_cen = cluster_deviation(z[live], cen[live], wbin[live], p0_ref, tan_ref)
        dev_pr = cluster_deviation(zp, pp, ap, p0_ref, tan_ref)
        dev_pr_own = cluster_deviation(zp_own, pp, ap, p0_ref, tan_ref)
        deep_ff = cluster_deviation(z[live_mu], mu[live_mu], qk[live_mu],
                                    p0_ref, tan_ref, zlo=Z_DEEP)
        deep_pr = cluster_deviation(zp, pp, ap, p0_ref, tan_ref, zlo=Z_DEEP)
        deep_pr_own = cluster_deviation(zp_own, pp, ap, p0_ref, tan_ref, zlo=Z_DEEP)

        # Production ladder line, re-expressed in the common (position, depth)
        # frame: production gives position(t) = det_p + (t - t0_prod)/slope with
        # slope in ns/mm, and here t = t0_ff + z*1e3/v.
        pr = ev['prod']
        slope = pr[f'slope_{plane}']                       # ns/mm
        if np.isfinite(slope) and abs(slope) > 1e-9:
            tan_prod = 1e3 / (slope * reco.v_drift)
            p0_prod = pr[f'det_{plane}'] + (f['t0'] - pr[f't0_{plane}']) / slope
        else:
            tan_prod = p0_prod = np.nan

        out[plane] = dict(
            fit=f, Q=Q, model=model, chi2_dec=chi2, dof_dec=dof, pos=pos, z=z,
            wbin=wbin, cen=cen, live=live, mu=mu, qk=qk, live_mu=live_mu,
            dev_cen=dev_cen, tan_ref=tan_ref, p0_ref=p0_ref,
            zp=zp, zp_own=zp_own, pp=pp, ap=ap,
            tan_prod=tan_prod, p0_prod=p0_prod,
            dev_ff=dev_ff, dev_pr=dev_pr, dev_pr_own=dev_pr_own,
            deep_ff=deep_ff, deep_pr=deep_pr, deep_pr_own=deep_pr_own,
            tan_ff=f['tan_theta'], t0=f['t0'])
    return out


# ----------------------------------------------------------------- figures
def fig_planes(R, v, path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for row, plane in enumerate(('x', 'y')):
        d = R[plane]
        pos, z = d['pos'], d['z']
        zline = np.linspace(0, GAP_MM, 2)

        # --- (a) raw waveforms, all three lines, in time space
        ax = axes[row, 0]
        Wdata = d['Wdata']
        pm = ax.pcolormesh(np.append(pos - 0.39, pos[-1] + 0.39),
                           np.append(fm2.TS - 30, fm2.TS[-1] + 30), Wdata.T,
                           cmap='viridis', vmin=-0.05 * Wdata.max(), vmax=Wdata.max())
        plt.colorbar(pm, ax=ax, label='ADC')
        t_of_z = d['t0'] + zline * 1e3 / v
        ax.plot(d['p0_ref'] + d['tan_ref'] * zline, t_of_z, 'w--', lw=2,
                label='M3 reference')
        ax.plot(d['fit']['p0'] + d['tan_ff'] * zline, t_of_z, '-', color='red', lw=1.8,
                label='waveform-first fit')
        if np.isfinite(d['tan_prod']):
            ax.plot(d['p0_prod'] + d['tan_prod'] * zline, t_of_z, '-', color='orange',
                    lw=1.8, label='production ladder')
        ax.set_xlabel(f'{plane} [mm]'); ax.set_ylabel('sample time [ns]')
        ax.set_title(f'{plane.upper()} plane — waveforms\n'
                     f'tan: ref {d["tan_ref"]:+.3f} | wf-first {d["tan_ff"]:+.3f} | '
                     f'prod {d["tan_prod"]:+.3f}')
        ax.legend(fontsize=7, loc='upper right')

        # --- (b) production cluster in depth space
        ax = axes[row, 1]
        if len(d['zp']):
            sc = ax.scatter(d['pp'], d['zp'], c=d['ap'], cmap='Oranges', s=70,
                            edgecolor='k', lw=0.3, vmin=0)
            plt.colorbar(sc, ax=ax, label='amplitude')
        ax.plot(d['p0_ref'] + d['tan_ref'] * zline, zline, 'g-', lw=2.5,
                label='M3 reference')
        if np.isfinite(d['tan_prod']):
            ax.plot(d['p0_prod'] + d['tan_prod'] * zline, zline, '--', color='orange',
                    lw=1.6, label='production ladder')
        ax.set_ylim(-2, GAP_MM + 2); ax.set_xlim(pos[0] - 0.5, pos[-1] + 0.5)
        ax.set_xlabel(f'{plane} [mm]'); ax.set_ylabel('drift depth [mm]')
        ax.set_title(f'production cluster (aggregate strip times)\n'
                     f'|dev| to M3: {d["dev_pr"][0]:.2f} mm all depths, '
                     f'{d["deep_pr"][0]:.2f} mm above {Z_DEEP:.0f} mm')
        ax.legend(fontsize=7); ax.grid(alpha=0.25)

        # --- (c) deconvolved cluster in depth space
        ax = axes[row, 2]
        Q = d['Q']
        pm = ax.pcolormesh(np.append(pos - 0.39, pos[-1] + 0.39),
                           np.append(z - 0.5 * (z[1] - z[0]), z[-1] + 0.5 * (z[1] - z[0])),
                           Q.T, cmap='Blues', vmin=0)
        plt.colorbar(pm, ax=ax, label='deconvolved charge')
        ax.plot(d['cen'][d['live']], z[d['live']], 'k.', ms=4, alpha=0.45,
                label=f'strip centroid ({d["dev_cen"][0]:.2f} mm)')
        ax.plot(d['mu'][d['live_mu']], z[d['live_mu']], 'ko', ms=6, mfc='none',
                mew=1.6, label='free ladder (sub-pitch)')
        ax.plot(d['p0_ref'] + d['tan_ref'] * zline, zline, 'g-', lw=2.5,
                label='M3 reference')
        ax.set_ylim(-2, GAP_MM + 2); ax.set_xlim(pos[0] - 0.5, pos[-1] + 0.5)
        ax.set_xlabel(f'{plane} [mm]'); ax.set_ylabel('drift depth [mm]')
        ax.set_title(f'waveform-first cluster (2-D deconvolution, line-free)\n'
                     f'|dev| to M3: {d["dev_ff"][0]:.2f} mm all depths, '
                     f'{d["deep_ff"][0]:.2f} mm above {Z_DEEP:.0f} mm')
        ax.legend(fontsize=7); ax.grid(alpha=0.25)

    fig.suptitle(f'det3 (mx17_3) sat_det3 — event {R["eid"]}   '
                 f'v = {v:.1f} um/ns, common z origin = waveform-first t0   '
                 f'(M3 radial residual {R["radial_residual"]:.2f} mm)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=105)
    plt.close(fig)


def fig_3d(R, v, path):
    """Production point cloud vs deconvolved cloud, both with the M3 track."""
    fig = plt.figure(figsize=(15, 7))
    zline = np.linspace(0, GAP_MM, 60)
    xr = R['x']['p0_ref'] + R['x']['tan_ref'] * zline
    yr = R['y']['p0_ref'] + R['y']['tan_ref'] * zline
    # common limits so the two panels are visually comparable
    lim = {}
    for p, ref in (('x', xr), ('y', yr)):
        d = R[p]
        vals = [ref, d['pp'], d['mu'][d['live_mu']]]
        vals = np.concatenate([np.asarray(a, float).ravel() for a in vals if len(a)])
        vals = vals[np.isfinite(vals)]
        c, half = 0.5 * (vals.min() + vals.max()), 0.5 * (vals.max() - vals.min())
        lim[p] = (c - max(half, 2.0) - 0.5, c + max(half, 2.0) + 0.5)

    for col, mode in enumerate(('production', 'waveform-first')):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        if mode == 'production':
            # x strips: x measured, y from the production Y ladder at that depth
            zx, px, ax_amp = R['x']['zp'], R['x']['pp'], R['x']['ap']
            zy, py, ay_amp = R['y']['zp'], R['y']['pp'], R['y']['ap']
            ypred = R['y']['p0_prod'] + R['y']['tan_prod'] * zx
            xpred = R['x']['p0_prod'] + R['x']['tan_prod'] * zy
            if len(zx):
                ax.scatter(px, ypred, zx, c=ax_amp, cmap='Reds', s=45, alpha=0.9,
                           label='X strips (x meas., y pred.)')
            if len(zy):
                ax.scatter(xpred, py, zy, c=ay_amp, cmap='Blues', s=45, alpha=0.9,
                           label='Y strips (x pred., y meas.)')
            dm = (R['x']['dev_pr'][0], R['y']['dev_pr'][0])
            dd = (R['x']['deep_pr'][0], R['y']['deep_pr'][0])
        else:
            # deconvolved bin centroids; complementary coordinate from the other
            # plane's deconvolved centroid at the same depth bin (no line used)
            lx, ly = R['x']['live_mu'], R['y']['live_mu']
            zc = R['x']['z']
            cx, cy = R['x']['mu'], R['y']['mu']
            wx = R['x']['qk']
            both = lx & ly & (zc <= GAP_MM)
            if both.any():
                ax.scatter(cx[both], cy[both], zc[both], c=wx[both], cmap='Purples',
                           s=60, alpha=0.95,
                           label='free-ladder charge position (both planes)')
            dm = (R['x']['dev_ff'][0], R['y']['dev_ff'][0])
            dd = (R['x']['deep_ff'][0], R['y']['deep_ff'][0])

        ax.plot(xr, yr, zline, color='green', lw=2.5, label='M3 reference track')
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_zlabel('drift depth [mm]')
        ax.set_xlim(*lim['x']); ax.set_ylim(*lim['y']); ax.set_zlim(-1, GAP_MM + 1)
        ax.set_title(f'{mode} cluster\n|dev| to reference (all depths | above '
                     f'{Z_DEEP:.0f} mm):  x {dm[0]:.2f} | {dd[0]:.2f} mm,   '
                     f'y {dm[1]:.2f} | {dd[1]:.2f} mm')
        ax.legend(fontsize=7, loc='upper left')
        ax.view_init(elev=16, azim=-58)
    fig.suptitle(f'det3 event {R["eid"]} — 3-D display, same events, same v = {v:.1f} um/ns',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=105)
    plt.close(fig)


# ----------------------------------------------------------------- census
def _census_one(args):
    eid, lam = args
    ev = _EV[eid]
    R = analyse_event(ev, _RECO, _META, _FEUS, lam=lam)
    rec = dict(eid=eid)
    for p in ('x', 'y'):
        d = R[p]
        rec[f'{p}_ff'] = d['dev_ff'][0]
        rec[f'{p}_pr'] = d['dev_pr'][0]
        rec[f'{p}_ff90'] = d['dev_ff'][1]
        rec[f'{p}_pr90'] = d['dev_pr'][1]
        rec[f'{p}_own'] = d['dev_pr_own'][0]
        rec[f'{p}_ff_deep'] = d['deep_ff'][0]
        rec[f'{p}_pr_deep'] = d['deep_pr'][0]
        rec[f'{p}_own_deep'] = d['deep_pr_own'][0]
        rec[f'{p}_tan_ref'] = d['tan_ref']
        rec[f'{p}_chi2dof'] = d['fit']['chi2'] / max(d['fit']['dof'], 1)
        # deviation profile vs depth, all three variants (divergence plot)
        for lo, hi, tag in DEPTH_SLICES:
            rec[f'{p}_ff_{tag}'] = cluster_deviation(
                d['z'][d['live_mu']], d['mu'][d['live_mu']], d['qk'][d['live_mu']],
                d['p0_ref'], d['tan_ref'], zlo=lo, zhi=hi)[0]
            rec[f'{p}_pr_{tag}'] = cluster_deviation(
                d['zp'], d['pp'], d['ap'], d['p0_ref'], d['tan_ref'],
                zlo=lo, zhi=hi)[0]
            rec[f'{p}_own_{tag}'] = cluster_deviation(
                d['zp_own'], d['pp'], d['ap'], d['p0_ref'], d['tan_ref'],
                zlo=lo, zhi=hi)[0]
    return rec


def _init(evfile, lam):
    global _EV, _RECO, _META, _FEUS
    d = pickle.load(open(evfile, 'rb'))
    _EV, _META = d['events'], d['meta']
    _RECO = WFTReco()
    _FEUS = _feu_map(_META)


def _feu_map(meta):
    feus = sorted(meta['pos_map'])
    # the cache stores {feu: position array}; X is the lower FEU number (7), Y = 8
    return {'x': feus[0], 'y': feus[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6, help='number of event displays')
    ap.add_argument('--eids', type=str, default='', help='explicit comma-separated eids')
    ap.add_argument('--census', type=int, default=300, help='events in the census (0=skip)')
    ap.add_argument('--lam', type=float, default=LAMBDA)
    ap.add_argument('--jobs', type=int, default=12)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(BASE, 'wfcache.pkl')
    d = pickle.load(open(cache, 'rb'))
    events, meta = d['events'], d['meta']
    reco = WFTReco()
    feus = _feu_map(meta)
    v = reco.v_drift
    print(f'{len(events):,} cached events; v = {v:.2f} um/ns; FEU map {feus}')

    eids = list(events)
    tan3 = np.array([np.hypot(events[e]['tan_x'], events[e]['tan_y']) for e in eids])
    if args.eids:
        picks = [int(s) for s in args.eids.split(',')]
    else:
        qs = np.linspace(35, 96, args.n)
        picks, seen = [], set()
        for q in qs:
            i = int(np.argmin(np.abs(tan3 - np.percentile(tan3, q))))
            while eids[i] in seen:
                i += 1
            seen.add(eids[i]); picks.append(eids[i])

    for eid in picks:
        ev = events[eid]
        R = analyse_event(ev, reco, meta, feus, lam=args.lam)
        for p in ('x', 'y'):
            R[p]['Wdata'] = fm2.prep_plane(ev[p], p)[0]
        fig_planes(R, v, os.path.join(OUT, f'event_{eid}_planes.png'))
        fig_3d(R, v, os.path.join(OUT, f'event_{eid}_3d.png'))
        print(f'event {eid}: tan3={np.hypot(ev["tan_x"], ev["tan_y"]):.3f}  '
              f'dev(prod) x {R["x"]["dev_pr"][0]:.2f} y {R["y"]["dev_pr"][0]:.2f} | '
              f'dev(wf-first) x {R["x"]["dev_ff"][0]:.2f} y {R["y"]["dev_ff"][0]:.2f} mm')

    if args.census:
        rng = np.random.default_rng(7)
        pool = [e for e, t in zip(eids, tan3) if 0.08 <= t <= 0.45]
        sel = rng.choice(pool, size=min(args.census, len(pool)), replace=False)
        print(f'census on {len(sel)} events, {args.jobs} jobs ...')
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                                 initargs=(cache, args.lam)) as ex:
            recs = list(ex.map(_census_one, [(int(e), args.lam) for e in sel],
                               chunksize=4))
        census_figure(recs, v)


def census_figure(recs, v):
    import pandas as pd
    df = pd.DataFrame(recs)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    summ = {}
    for i, p in enumerate(('x', 'y')):
        ax = axes[i]
        b = np.linspace(0, 4, 90)
        for key, lab, c in (('pr', 'production cluster', 'orange'),
                            ('own', 'production, own t0 and v', 'darkred'),
                            ('ff', 'waveform-first cluster', 'C0')):
            a = df[f'{p}_{key}'].to_numpy()
            a = a[np.isfinite(a)]
            ax.hist(a, bins=b, cumulative=True, density=True, histtype='step',
                    lw=2, color=c, ls='--' if key == 'own' else '-',
                    label=f'{lab}: median {np.median(a):.2f} mm, '
                          f'<1 mm {100*np.mean(a<1):.0f}%')
            summ[f'{p}_{key}'] = dict(
                median=float(np.median(a)), frac_lt_0p5=float(np.mean(a < 0.5)),
                frac_lt_1=float(np.mean(a < 1.0)), n=int(len(a)),
                median_deep=float(np.nanmedian(
                    df[f'{p}_{key}_deep'].to_numpy())))
        ax.set_xlabel(f'{p}: charge-weighted |cluster - M3 line| [mm]')
        ax.set_ylabel('cumulative fraction'); ax.grid(alpha=0.3)
        ax.legend(fontsize=8); ax.set_xlim(0, 3)

    # --- the headline: deviation vs depth (the "diverges as it ascends" claim)
    ax = axes[2]
    zc = [0.5 * (lo + hi) for lo, hi, _ in DEPTH_SLICES]
    prof = {}
    for p, ls in (('x', '-'), ('y', '--')):
        for meth, lab, c in (('pr', 'production', 'orange'),
                             ('own', 'production own frame', 'darkred'),
                             ('ff', 'waveform-first', 'C0')):
            med = [np.nanmedian(df[f'{p}_{meth}_{tag}']) for _, _, tag in DEPTH_SLICES]
            prof[f'{p}_{meth}'] = [float(m) for m in med]
            ax.plot(zc, med, ls, marker='o' if p == 'x' else 's', color=c,
                    label=f'{p} {lab}')
    ax.set_xlabel('drift depth [mm]')
    ax.set_ylabel('median |cluster - M3 line| [mm]')
    ax.set_title('divergence with depth'); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[3]
    for p, mk in (('x', 'o'), ('y', 's')):
        at = df[f'{p}_tan_ref'].abs()
        bins = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]
        ctr = [0.5 * (a + b) for a, b in bins]
        for meth, lab, c in (('pr', 'production', 'orange'),
                             ('ff', 'waveform-first', 'C0')):
            med = [np.nanmedian(df[f'{p}_{meth}'][(at >= a) & (at < b)])
                   for a, b in bins]
            ax.plot(ctr, med, mk + '-', color=c, label=f'{p} {lab}')
    ax.set_xlabel('|tan(theta)| reference'); ax.set_ylabel('median |dev| [mm]')
    ax.set_title('threading vs track angle'); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle(f'det3 sat_det3 — does the M3 track thread the reconstructed cluster?  '
                 f'({len(df)} events, v = {v:.1f} um/ns, common z origin)')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, 'threading_census.png'), dpi=110)
    json.dump(dict(n_events=int(len(df)), v=float(v), summary=summ,
                   depth_profile=dict(z=zc, **prof)),
              open(os.path.join(OUT, 'threading_census.json'), 'w'), indent=2)
    df.to_csv(os.path.join(OUT, 'threading_census.csv'), index=False)
    print(json.dumps(summ, indent=2))
    print('depth profile', json.dumps(prof, indent=2))


if __name__ == '__main__':
    main()
