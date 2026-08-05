#!/usr/bin/env python3
"""
Part IV figures — how one plane is actually fitted: the chi2 landscape, the
reference-free global start, Nelder-Mead, the errors, and the two things that
make the linear sub-problem behave (NNLS non-negativity and saturation
censoring).
"""
from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import wftdoc as K
from wftdoc import C, save

from wft import model as wm
from wft import reco as wr

EID = 1663
CAL = None
EVS = None


def setup():
    global CAL, EVS
    CAL = K.install()
    EVS = K.calib_events()
    return CAL, EVS


def prep(eid=EID, plane='x'):
    P = K.trim_window(EVS[eid][plane])
    if np.asarray(P['W']).shape[1] != wm.NSAMP:
        wm.set_nsamp(np.asarray(P['W']).shape[1])
    W, noise, pos, sat = wm.prep_plane(P, plane)
    return P, W, noise, pos, sat


# ------------------------------------------------------------ chi2 landscape
def fig_chi2_surface():
    plane = 'x'
    P, W, noise, pos, sat = prep(plane=plane)
    h = dict(CAL.hyper)
    v = CAL.v_drift
    ev = EVS[EID]
    p0_ref, tan_ref = ev['ref_mesh_x'], ev['tan_x']
    w_ref = tan_ref * v * 1e-3

    r = wm.fit_plane_raw(P, plane, p0_ref, w_ref, 400.0, hyper=h)
    print(f'[fit] free fit: p0 {r["p0"]:.3f} mm (ref {p0_ref:.3f}), '
          f'tan {r["w"]*1e3/v:+.4f} (ref {tan_ref:+.4f}), t0 {r["t0"]:.0f} ns, '
          f'chi2/dof {r["chi2"]/r["dof"]:.1f}')

    def chi(p0, w, t0):
        return wm.chi2_plane(plane, W, noise, pos, sat, p0, w, t0, h,
                             snap_t0=False)[0]

    n = 55
    p0s = np.linspace(r['p0'] - 2.0, r['p0'] + 2.0, n)
    ws = np.linspace(r['w'] - 0.010, r['w'] + 0.010, n)
    t0s = np.linspace(r['t0'] - 260, r['t0'] + 260, n)

    t_start = time.time()
    Cw = np.array([[chi(p, w, r['t0']) for w in ws] for p in p0s])
    Ct = np.array([[chi(p, r['w'], t) for t in t0s] for p in p0s])
    print(f'[fit] two {n}x{n} chi2 maps in {time.time()-t_start:.1f} s '
          f'({2*n*n} NNLS solves)')

    fig = plt.figure(figsize=(13, 3.9))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.95], wspace=0.32)
    s = max(r['chi2'] / r['dof'], 1.0)          # chi2/dof, the error scale
    lev = np.array([1, 4, 9, 25]) * s

    ax = fig.add_subplot(gs[0])
    im = ax.pcolormesh(ws * 1e3 / v, p0s, np.log10(Cw), cmap='viridis_r',
                       shading='auto')
    ax.contour(ws * 1e3 / v, p0s, Cw - np.nanmin(Cw), levels=lev,
               colors=['w'], linewidths=[1.4, 1.0, 0.8, 0.6])
    ax.plot(r['w'] * 1e3 / v, r['p0'], '*', ms=14, color=C['orange'],
            label='fit minimum')
    ax.plot(tan_ref, p0_ref, 'o', ms=8, color=C['ref'], mfc='none', mew=2,
            label='M3 reference')
    ax.set_xlabel(r'tan$\theta$  =  $w / v$')
    ax.set_ylabel('$p_0$  [mm]')
    ax.set_title(r'$\chi^2(p_0, w)$ at the fitted $t_0$'
                 '\n(white: 1, 2, 3, 5$\\sigma$)', loc='left')
    ax.legend(fontsize=7.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'$\log_{10}\chi^2$', color=K.CHROME)
    cb.ax.tick_params(colors=K.CHROME)
    cb.outline.set_edgecolor(K.CHROME)

    ax = fig.add_subplot(gs[1])
    ax.pcolormesh(t0s, p0s, np.log10(Ct), cmap='viridis_r', shading='auto')
    ax.contour(t0s, p0s, Ct - np.nanmin(Ct), levels=lev, colors=['w'],
               linewidths=[1.4, 1.0, 0.8, 0.6])
    ax.plot(r['t0'], r['p0'], '*', ms=14, color=C['orange'])
    ax.set_xlabel('$t_0$  [ns]')
    ax.set_ylabel('$p_0$  [mm]')
    ax.set_title(r'$\chi^2(p_0, t_0)$ at the fitted $w$ —'
                 '\nthe valley is a real degeneracy', loc='left')
    ax.grid(False)

    ax = fig.add_subplot(gs[2])
    prof = np.array([chi(r['p0'], w, r['t0']) for w in ws])
    ax.plot(ws * 1e3 / v, prof / r['dof'], color=C['blue'])
    ax.axvline(r['w'] * 1e3 / v, color=C['orange'], ls='--', label='fit')
    ax.axvline(tan_ref, color=C['ref'], ls=':', label='reference')
    ax.set_xlabel(r'tan$\theta$')
    ax.set_ylabel(r'$\chi^2/\mathrm{dof}$')
    ax.set_title('the slope is well determined', loc='left')
    ax.legend(fontsize=7.5)
    save(fig, 'chi2_surface')
    return r


# ------------------------------------------------------------- global start
def fig_global_start(r):
    plane = 'x'
    P, W, noise, pos, sat = prep(plane=plane)
    h = dict(CAL.hyper)
    v = CAL.v_drift

    def chi(p0, w, t0):
        return wm.chi2_plane(plane, W, noise, pos, sat, p0, w, t0, h)[0]

    p0_seed, _w0, t0_seed = wm.init_guess(P, plane)
    amp = np.maximum(W.max(axis=1), 0.0)
    p_c = float((pos * amp).sum() / amp.sum())
    p0s = p_c + np.arange(-wr.P0_SCAN_HALF, wr.P0_SCAN_HALF + 1e-9,
                          wr.P0_SCAN_STEP)
    t0s = np.arange(t0_seed - wr.T0_SCAN_HALF, t0_seed + wr.T0_SCAN_HALF + 1e-9,
                    wr.T0_SCAN_STEP)
    ws = np.arange(-wr.W_SCAN_HALF, wr.W_SCAN_HALF + 1e-9, wr.W_SCAN_STEP)

    S1 = np.array([[chi(p, 0.0, t) for t in t0s] for p in p0s])
    j = np.unravel_index(np.argmin(S1), S1.shape)
    t0b = t0s[j[1]]
    S2 = np.array([[chi(p, w, t0b) for w in ws] for p in p0s])
    j2 = np.unravel_index(np.argmin(S2), S2.shape)
    print(f'[fit] global start: brightest-strip seed p0 {p0_seed:.2f} mm, '
          f'half-max t0 {t0_seed:.0f} ns -> scan start '
          f'p0 {p0s[j2[0]]:.2f}, w {ws[j2[1]]:.4f} '
          f'(tan {ws[j2[1]]*1e3/v:+.3f}), t0 {t0b:.0f} ns; '
          f'{S1.size + S2.size} evaluations')

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
    ax = axs[0]
    im = ax.pcolormesh(t0s, p0s, np.log10(S1), cmap='viridis_r', shading='auto')
    ax.plot(t0b, p0s[j[0]], '*', ms=14, color=C['orange'],
            label='stage-1 best')
    ax.plot(t0_seed, p0_seed, 'o', ms=6, color=C['red'], mfc='none', mew=1.8,
            label='crude seed (brightest strip, half-max)')
    ax.set_xlabel('$t_0$ [ns]'); ax.set_ylabel('$p_0$ [mm]')
    ax.set_title(r'stage 1: scan $(p_0, t_0)$ at zero slope', loc='left')
    ax.legend(fontsize=7); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'$\log_{10}\chi^2$', color=K.CHROME)
    cb.ax.tick_params(colors=K.CHROME)
    cb.outline.set_edgecolor(K.CHROME)

    ax = axs[1]
    ax.pcolormesh(ws * 1e3 / v, p0s, np.log10(S2), cmap='viridis_r',
                  shading='auto')
    ax.plot(ws[j2[1]] * 1e3 / v, p0s[j2[0]], '*', ms=14, color=C['orange'],
            label='stage-2 best = fit start')
    ax.plot(r['w'] * 1e3 / v, r['p0'], 'P', ms=10, color=C['pink'],
            label='final Nelder-Mead result')
    ax.axhspan(p0s[0], p0s[-1], color=C['grey'], alpha=0.0)
    ax.axhline(p0s[0], color=C['red'], lw=1.0, ls='--')
    ax.axhline(p0s[-1], color=C['red'], lw=1.0, ls='--',
               label='edge of the $p_0$ scan')
    ax.set_ylim(min(p0s[0], r['p0']) - 0.6, max(p0s[-1], r['p0']) + 0.4)
    ax.set_xlabel(r'tan$\theta$'); ax.set_ylabel('$p_0$ [mm]')
    ax.set_title(r'stage 2: scan $(p_0, w)$ at the best $t_0$', loc='left')
    ax.legend(fontsize=7); ax.grid(False)

    ax = axs[2]
    ax.plot(ws * 1e3 / v, S2.min(axis=0) / r['dof'], color=C['blue'],
            marker='o', ms=3)
    ax.axvline(ws[j2[1]] * 1e3 / v, color=C['orange'], ls='--',
               label='scan best')
    ax.axvline(r['w'] * 1e3 / v, color=C['pink'], ls=':', label='final fit')
    ax.set_xlabel(r'tan$\theta$')
    ax.set_ylabel(r'best $\chi^2/\mathrm{dof}$ over $p_0$')
    ax.set_title('why the scan has to be wide:\nthe slope basin is narrow',
                 loc='left')
    ax.legend(fontsize=7.5)
    save(fig, 'global_start')


def fig_scan_coverage(n_scan=160):
    """How well does the p0 scan window actually bracket the answer? It is
    centred on the window's charge centroid, but p0 is the position at the
    MESH — for an inclined track those differ by half the transverse span."""
    cal = CAL
    rows = []
    for eid in sorted(EVS)[:n_scan]:
        e = EVS[eid]
        if 'x' not in e:
            continue
        P = K.trim_window(e['x'])
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        W, noise, pos, sat = wm.prep_plane(P, 'x')
        amp = np.maximum(W.max(axis=1), 0.0)
        if amp.sum() <= 0:
            continue
        pc = float((pos * amp).sum() / amp.sum())
        f = wr.fit_plane(P, 'x', cal)
        if f is None:
            continue
        rows.append((pc, f.p0, e['ref_mesh_x'], e['tan_x'], f.tan_theta))
    a = np.array(rows)
    d_ref = a[:, 2] - a[:, 0]
    out = 100 * np.mean(np.abs(d_ref) > wr.P0_SCAN_HALF)
    dp = a[:, 1] - a[:, 2]
    print(f'[fit] scan coverage: |p0_ref - centroid| > {wr.P0_SCAN_HALF} mm '
          f'for {out:.0f}% of planes; |p0_fit - p0_ref| median '
          f'{np.median(np.abs(dp)):.2f} mm, within 1 mm '
          f'{100*np.mean(np.abs(dp) < 1):.0f}%')

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    ax = axs[0]
    ax.scatter(a[:, 3], d_ref, s=12, alpha=0.7, color=C['blue'])
    ax.axhline(wr.P0_SCAN_HALF, color=C['red'], ls='--',
               label=f'scan half-width, ±{wr.P0_SCAN_HALF} mm')
    ax.axhline(-wr.P0_SCAN_HALF, color=C['red'], ls='--')
    ax.set_xlabel(r'tan$\theta$ (reference)')
    ax.set_ylabel('true $p_0$ − scan centre [mm]')
    ax.set_title(f'the scan centre is the charge centroid,\nnot the mesh — '
                 f'{out:.0f}% of planes fall outside', loc='left')
    ax.legend(fontsize=7.5)

    ax = axs[1]
    ax.hist(dp, bins=np.linspace(-4, 4, 70), color=C['blue'], alpha=0.85)
    ax.set_xlabel('fitted $p_0$ − reference $p_0$ [mm]')
    ax.set_ylabel('planes')
    ax.set_title('Nelder-Mead walks out of the scan box\nand mostly gets there '
                 f'anyway (median |Δ| {np.median(np.abs(dp)):.2f} mm)',
                 loc='left')

    ax = axs[2]
    inside = np.abs(d_ref) <= wr.P0_SCAN_HALF
    tails = {}
    for m, lab, col in ((inside, 'inside the scan box', C['blue']),
                        (~inside, 'outside', C['orange'])):
        if m.sum() < 5:
            continue
        tails[lab] = 100 * np.mean(np.abs(dp[m]) > 2.0)
        ax.hist(np.abs(dp[m]), bins=np.linspace(0, 4, 40), histtype='step',
                lw=1.8, color=col, density=True, cumulative=True,
                label=f'{lab} (n={m.sum()}, med '
                      f'{np.median(np.abs(dp[m])):.2f} mm, '
                      f'{tails[lab]:.0f}% beyond 2 mm)')
    ax.set_xlabel('|fitted − reference| $p_0$ [mm]')
    ax.set_ylabel('cumulative fraction')
    ax.set_title('the core is unaffected; the failure tail is not\n'
                 '(same median, 5× the fraction beyond 2 mm)', loc='left')
    ax.legend(fontsize=7)
    print('[fit] fraction beyond 2 mm:', {k: round(v, 1) for k, v in tails.items()})
    save(fig, 'scan_coverage')


# ------------------------------------------------- NNLS profile degeneracy
def fig_nnls_profile():
    """Two facts about the charge profile, side by side: per event it is sparse
    (adjacent 60 ns bins are degenerate under a 350 ns response), and in the
    ensemble it is flat with a sharp edge."""
    h = dict(CAL.hyper)
    v = CAL.v_drift
    eids = K.rank_events(EVS, 'x', tan_lo=0.10, tan_hi=0.45, n_scan=200,
                         n_keep=40, single_cluster=False)
    profs = []
    for eid in eids:
        ev = EVS[eid]
        P = K.trim_window(ev['x'])
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        p0, w = ev['ref_mesh_x'], ev['tan_x'] * v * 1e-3
        r = wm.fit_plane_raw(P, 'x', p0, w, 400.0, hyper=h, fix_p0w=(p0, w))
        q = np.asarray(r['q'], float)
        if q.sum() > 0:
            profs.append(q / q.sum())
    profs = np.array(profs)
    occ = float((profs > 0.005).mean())
    print(f'[fit] {len(profs)} ref-pinned profiles, per-bin occupancy {occ:.0%}')

    z = wm.UK * v * 1e-3
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axs[0]
    for p in profs[:12]:
        ax.step(z, p, where='mid', lw=1.0, alpha=0.65)
    ax.set_xlabel('drift depth [mm]'); ax.set_ylabel('$q_k$ / total')
    ax.set_title(f'12 individual events: sparse and spiky\n'
                 f'(only {occ:.0%} of bins non-zero)', loc='left')

    ax = axs[1]
    med = np.median(profs, axis=0)
    mean = profs.mean(axis=0)
    ax.step(z, med, where='mid', color=C['grey'], label='per-bin median')
    ax.step(z, mean, where='mid', color=C['blue'], lw=2, label='mean')
    ax.fill_between(z, 0, mean, step='mid', color=C['blue'], alpha=0.22)
    ax.axvline(27.9, color=C['red'], ls='--', lw=1,
               label="det3's measured column, 27.9 mm")
    ax.set_xlabel('drift depth [mm]'); ax.set_ylabel('mean $q_k$ / total')
    ax.set_title('the ensemble: flat, with an edge\n'
                 '(the median under-reads — sparsity truncates it)', loc='left')
    ax.legend(fontsize=7)

    # WHY it is sparse: non-negativity acting on a noisy, correlated system.
    ax = axs[2]
    ev = EVS[EID]
    P = K.trim_window(ev['x'])
    W, noise, pos, sat = wm.prep_plane(P, 'x')
    p0, w = ev['ref_mesh_x'], ev['tan_x'] * v * 1e-3
    r = wm.fit_plane_raw(P, 'x', p0, w, 400.0, hyper=h, fix_p0w=(p0, w))
    q = np.asarray(r['q'], float)
    M = wm.build_matrix('x', pos, p0, w, r['t0'], h)
    A = M * np.repeat(1.0 / noise, wm.NSAMP)[:, None]
    y = (W / noise[:, None]).reshape(-1)
    ok = ~sat.reshape(-1)
    ls, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    cond = np.linalg.cond(A)
    ax.step(z, ls / q.sum(), where='mid', color=C['grey'],
            label=f'unconstrained least squares\n({(ls<0).sum()} of 18 bins '
                  'negative)')
    ax.step(z, q / q.sum(), where='mid', color=C['blue'], lw=2,
            label=f'NNLS ({(q<=0).sum()} bins clipped to 0)')
    ax.axhline(0, color=K.CHROME, lw=0.8)
    ax.set_xlabel('drift depth [mm]'); ax.set_ylabel('$q_k$ / total')
    ax.set_title('sparsity is the non-negativity constraint,\n'
                 f'not ill-conditioning (cond(A) = {cond:.0f})', loc='left')
    ax.legend(fontsize=7)
    print(f'[fit] design-matrix condition number {cond:.1f}; '
          f'unconstrained LS puts {(ls<0).sum()}/18 bins negative')
    save(fig, 'nnls_profile')


# ------------------------------------------------------------------ errors
def fig_errors():
    plane = 'x'
    P, W, noise, pos, sat = prep(plane=plane)
    h = dict(CAL.hyper)
    v = CAL.v_drift
    r = wm.fit_plane_raw(P, plane, EVS[EID]['ref_mesh_x'],
                         EVS[EID]['tan_x'] * v * 1e-3, 400.0, hyper=h)
    ep, ew = wr._errors(P, plane, r, h)
    scale = max(r['chi2'] / max(r['dof'], 1), 1.0)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    ax = axs[0]
    dps = np.linspace(-0.6, 0.6, 41)
    cs = [wm.chi2_plane(plane, W, noise, pos, sat, r['p0'] + d, r['w'],
                        r['t0'], h, snap_t0=False)[0] for d in dps]
    ax.plot(dps, (np.array(cs) - r['chi2']) / scale, color=C['blue'])
    ax.axhline(1, color=C['red'], ls='--', lw=1, label=r'$\Delta\chi^2 = 1$')
    ax.axvline(ep, color=C['orange'], ls=':', lw=1.4)
    ax.axvline(-ep, color=C['orange'], ls=':', lw=1.4,
               label=fr'curvature $\sigma$ = {ep*1000:.0f} µm')
    ax.set_ylim(-0.5, 12)
    ax.set_xlabel('$p_0$ offset [mm]')
    ax.set_ylabel(r'$\Delta\chi^2 / (\chi^2/\mathrm{dof})$')
    ax.set_title(r'the $p_0$ error from curvature', loc='left')
    ax.legend(fontsize=7.5)

    ax = axs[1]
    dws = np.linspace(-0.004, 0.004, 41)
    cs = [wm.chi2_plane(plane, W, noise, pos, sat, r['p0'], r['w'] + d,
                        r['t0'], h, snap_t0=False)[0] for d in dws]
    ax.plot(dws * 1e3 / v, (np.array(cs) - r['chi2']) / scale, color=C['blue'])
    ax.axhline(1, color=C['red'], ls='--', lw=1)
    for s in (+1, -1):
        ax.axvline(s * ew * 1e3 / v, color=C['orange'], ls=':', lw=1.4)
    ax.set_ylim(-0.5, 12)
    ax.set_xlabel(r'tan$\theta$ offset')
    ax.set_ylabel(r'$\Delta\chi^2 / (\chi^2/\mathrm{dof})$')
    ax.set_title(fr'the slope error: statistical $\sigma$(tan) = '
                 fr'{ew*1e3/v:.4f} = {np.degrees(np.arctan(ew*1e3/v)):.2f}°',
                 loc='left')

    ax = axs[2]
    df = K.events_table()
    for p, col in (('x', C['x']), ('y', C['y'])):
        e = df.loc[df[f'{p}_ok'], f'{p}_tan_err'].to_numpy()
        e = e[np.isfinite(e)]
        ax.hist(np.degrees(np.arctan(e)), bins=np.linspace(0, 4, 80),
                histtype='step', lw=1.6, color=col,
                label=f'{p}: median {np.degrees(np.arctan(np.median(e))):.2f}°')
    ax.axvline(np.degrees(np.arctan(wr.FLOOR_TAN)), color=C['red'], ls='--',
               label=f'physics floor, {np.degrees(np.arctan(wr.FLOOR_TAN)):.1f}°')
    ax.set_xlabel('reported angle error [deg]')
    ax.set_ylabel('events')
    ax.set_title('reported errors over the whole run\n'
                 '(curvature ⊕ physics floor)', loc='left')
    ax.legend(fontsize=7.5)
    save(fig, 'errors')


# ------------------------------------------------------------- saturation
def fig_saturation():
    """Find a real saturated event and show what censoring does."""
    h = dict(CAL.hyper)
    v = CAL.v_drift
    hit = None
    for eid in sorted(EVS):
        e = EVS[eid]
        if 'x' not in e:
            continue
        P = K.trim_window(e['x'])
        if np.asarray(P['W']).max() >= wm.SAT:
            hit = (eid, P)
            break
    if hit is None:
        print('[fit] no saturated event in the cache — skipping')
        return
    eid, P = hit
    if np.asarray(P['W']).shape[1] != wm.NSAMP:
        wm.set_nsamp(np.asarray(P['W']).shape[1])
    W, noise, pos, sat = wm.prep_plane(P, 'x')
    e = EVS[eid]
    p0, w = e['ref_mesh_x'], e['tan_x'] * v * 1e-3
    r = wm.fit_plane_raw(P, 'x', p0, w, 400.0, hyper=h)
    model = wm.model_waveforms('x', pos, r['p0'], r['w'], r['t0'], r['q'], h)
    t = np.arange(wm.NSAMP) * 0.06
    i = int(np.argmax(sat.sum(axis=1)))
    print(f'[fit] saturated event {eid}: {sat.sum()} clipped samples on '
          f'{int((sat.any(axis=1)).sum())} strips')

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4))
    ax = axs[0]
    ax.plot(t, W[i], color=K.CHROME, marker='o', ms=3, label='data (clipped)')
    ax.plot(t, model[i], color=C['orange'], label='model')
    ax.axhline(wm.SAT, color=C['red'], ls='--', lw=1,
               label=f'saturation, {wm.SAT:.0f} ADC')
    m = sat[i]
    ax.plot(t[m], W[i][m], 'x', ms=9, color=C['red'], mew=2,
            label='censored samples')
    ax.set_xlabel('time [µs]'); ax.set_ylabel('ADC')
    ax.set_title(f'event {eid}, strip {pos[i]:.1f} mm', loc='left')
    ax.legend(fontsize=7)

    ax = axs[1]
    im = ax.imshow(sat, aspect='auto', origin='lower', cmap='Reds',
                   extent=[0, wm.NSAMP * .06, pos[0] - .39, pos[-1] + .39],
                   interpolation='nearest')
    ax.set_xlabel('time [µs]'); ax.set_ylabel('strip position [mm]')
    ax.set_title('the censoring mask: excluded from the fit, penalised only\n'
                 'if the model dips below the clipped value', loc='left')
    ax.grid(False)
    save(fig, 'saturation')


# ---------------------------------------------------------------- timing
def fig_timing():
    P, W, noise, pos, sat = prep()
    h = dict(CAL.hyper)
    v = CAL.v_drift

    def timeit(fn, n=12):
        fn()
        t0 = time.time()
        for _ in range(n):
            fn()
        return (time.time() - t0) / n

    t_mat = timeit(lambda: wm.build_matrix('x', pos, 150.0, 0.008, 400.0, h))
    t_chi = timeit(lambda: wm.chi2_plane('x', W, noise, pos, sat, 150.0, 0.008,
                                         400.0, h))
    wm._tt_cache.clear()
    t_cold = timeit(lambda: (wm._tt_cache.clear(),
                             wm.build_matrix('x', pos, 150.0, 0.008, 400.0, h)),
                    n=6)
    t0 = time.time()
    wr.fit_plane(P, 'x', CAL)
    t_plane = time.time() - t0

    labels = ['build_matrix\n(cached tensors)', 'build_matrix\n(cold cache)',
              'chi2_plane\n(matrix + NNLS)', 'fit_plane\n(scan + NM, total)']
    vals = [t_mat * 1e3, t_cold * 1e3, t_chi * 1e3, t_plane * 1e3]
    print('[fit] timing ms:', dict(zip(labels, np.round(vals, 2))))

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    b = ax.barh(labels, vals, color=[C['blue'], C['grey'], C['teal'],
                                     C['orange']], height=0.6)
    for rect, val in zip(b, vals):
        ax.text(val * 1.05, rect.get_y() + rect.get_height() / 2,
                f'{val:.2f} ms' if val < 100 else f'{val/1000:.2f} s',
                va='center', fontsize=8.5, color=K.CHROME)
    ax.set_xscale('log')
    ax.set_xlabel('time per call [ms]')
    ax.set_title('where the time goes, one plane of one event '
                 '(single core, this machine)', loc='left')
    ax.set_xlim(vals[0] * 0.4, max(vals) * 4)
    save(fig, 'timing')


def main():
    setup()
    r = fig_chi2_surface()
    fig_global_start(r)
    fig_scan_coverage()
    fig_nnls_profile()
    fig_errors()
    fig_saturation()
    fig_timing()


if __name__ == '__main__':
    main()
