#!/usr/bin/env python3
"""
Part III figures — the forward model, taken apart piece by piece.

Everything here uses the live `sat_det3` lp calibration bundle and real
waveform windows from the ref-pinned calibration cache, so each panel shows
the model that production actually runs.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import wftdoc as K
from wftdoc import C, save

from wft import model as wm


CAL = None
EVS = None


def setup():
    global CAL, EVS
    CAL = K.install()
    EVS = K.calib_events()
    print('[model]', CAL.summary())
    print(f'[model] {len(EVS)} calibration-cache events')
    return CAL, EVS


# --------------------------------------------------------------- 1. template
def fig_template():
    g = np.asarray(CAL.grid, float)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))

    ax = axs[0]
    for p, col in (('x', C['x']), ('y', C['y'])):
        t = np.asarray(CAL.tmpl[p], float)
        ax.plot(g, t, color=col, label=f'{p} plane')
        i10 = np.argmax(t >= 0.1)
        i90 = np.argmax(t >= 0.9)
        print(f'[model] template {p}: rise10-90 {g[i90]-g[i10]:.0f} ns, '
              f'peak {g[int(np.argmax(t))]:.0f} ns, undershoot {t.min():+.3f}')
    ax.axhline(0, color=K.CHROME, lw=0.7)
    ax.set_xlabel('time since the charge arrives at the mesh [ns]')
    ax.set_ylabel('normalised response  h(t)')
    ax.set_title('the measured impulse response — one electron cluster in, '
                 'this shape out', loc='left')
    ax.legend()
    ax.set_xlim(-300, 1400)

    ax = axs[1]
    for p, col in (('x', C['x']), ('y', C['y'])):
        t = np.asarray(CAL.tmpl[p], float)
        ax.plot(g, t, color=col, label=f'{p} plane')
    ax.axhline(0, color=K.CHROME, lw=0.7)
    ax.set_xlim(200, 1400)
    ax.set_ylim(-0.14, 0.22)
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('h(t)')
    ax.set_title('the tail, magnified: Y undershoots several times deeper '
                 'than X', loc='left')
    ax.legend()
    save(fig, 'template')


# ------------------------------------------------------------ 2. the kernels
def fig_kernels():
    """The two sharing-kernel forms, drawn from the same template with the
    bundle's own tau_s and sigma_s."""
    g = np.asarray(CAL.grid, float)
    h = CAL.hyper
    tau, sig = h['tau_s'], h['sigma_s']
    base = np.linspace(-300, 2400, 900)

    wm.set_share_mode('lp')
    H1_lp, H2_lp = wm._copy_responses('y', base, h)
    wm.set_share_mode('delay')
    H1_d, H2_d = wm._copy_responses('y', base, h)
    wm.set_share_mode(CAL.share_mode)
    H0 = np.interp(base, g, np.asarray(CAL.tmpl['y'], float), left=0, right=0)

    c1, c2 = h['c1'], h['c2']
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))

    ax = axs[0]
    ax.plot(base, H0, color=K.CHROME, lw=2, label='own charge, $h(t)$')
    ax.plot(base, c1 * H1_d, color=C['orange'], ls='--',
            label=fr'delay: $c_1 h_s(t-\tau_s)$, $\tau_s$={tau:.0f} ns')
    ax.plot(base, c2 * H2_d, color=C['red'], ls='--',
            label=r'delay: $c_2 h_s(t-2\tau_s)$')
    ax.axhline(0, color=K.CHROME, lw=0.7)
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('response')
    ax.set_title('"delay" kernel — a shifted, smeared copy', loc='left')
    ax.legend(fontsize=7.5)

    ax = axs[1]
    ax.plot(base, H0, color=K.CHROME, lw=2, label='own charge, $h(t)$')
    ax.plot(base, c1 * H1_lp, color=C['orange'],
            label=fr'lp: $c_1 (h * K_\tau)$, $\tau$={tau:.0f} ns')
    ax.plot(base, c2 * H2_lp, color=C['red'],
            label=r'lp: $c_2 (h * K_\tau * K_\tau)$')
    ax.axhline(0, color=K.CHROME, lw=0.7)
    ax.set_xlabel('time [ns]')
    ax.set_title('"lp" kernel — an RC-dispersed copy (the one in use)',
                 loc='left')
    ax.legend(fontsize=7.5)

    ax = axs[2]
    n0 = H0 / H0.max()
    ax.plot(base, n0, color=K.CHROME, lw=2, label='own charge')
    ax.plot(base, H1_lp / H1_lp.max(), color=C['orange'],
            label='lp ±1 copy, peak-normalised')
    ax.plot(base, H1_d / H1_d.max(), color=C['orange'], ls='--',
            label='delay ±1 copy, peak-normalised')
    pk_lp = base[int(np.argmax(H1_lp))] - base[int(np.argmax(H0))]
    pk_d = base[int(np.argmax(H1_d))] - base[int(np.argmax(H0))]
    ax.axhline(0, color=K.CHROME, lw=0.7)
    ax.set_xlim(-200, 1600)
    ax.set_xlabel('time [ns]')
    ax.set_title(f'peak shift of the ±1 copy: lp {pk_lp:+.0f} ns, '
                 f'delay {pk_d:+.0f} ns', loc='left')
    ax.legend(fontsize=7.5)
    print(f'[model] +-1 copy peak shift: lp {pk_lp:+.0f} ns, delay {pk_d:+.0f} ns')
    save(fig, 'kernels')


# --------------------------------------------------- 3. the strip fractions
def fig_strip_fractions():
    h = dict(CAL.hyper)
    pos = np.arange(-6, 7) * wm.PITCH
    v = CAL.v_drift
    fig = plt.figure(figsize=(13, 3.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

    # (a) sigma_p(u)
    ax = fig.add_subplot(gs[0])
    u = wm.UK
    for sp0, dp, lab, col in ((h['sigma_p0'], h['Dp'],
                               f"bundle: $\\sigma_{{p0}}$={h['sigma_p0']:.2f} mm, "
                               f"$D_p$={h['Dp']:.4f}", C['blue']),
                              (0.098, 0.0114, 'det3 delay-mode calibration',
                               C['grey'])):
        ax.plot(u * v * 1e-3, np.sqrt(sp0 ** 2 + dp ** 2 * u), color=col,
                label=lab)
    ax.axhline(wm.PITCH, color=C['red'], ls=':', label='one strip pitch')
    ax.set_xlabel('drift depth [mm]')
    ax.set_ylabel(r'transverse spread $\sigma_p(u)$ [mm]')
    ax.set_title('the cloud grows as it drifts', loc='left')
    ax.legend(fontsize=7.5)

    # (b) F_ik for an inclined track
    ax = fig.add_subplot(gs[1])
    F = wm.strip_fractions(pos, 0.0, 0.28 * v * 1e-3, h['sigma_p0'], h['Dp'])
    im = ax.imshow(F, aspect='auto', origin='lower', cmap='viridis',
                   extent=[-0.5, wm.K - 0.5, pos[0] - 0.39, pos[-1] + 0.39])
    ax.set_xlabel('depth bin $k$  (60 ns each)')
    ax.set_ylabel('strip position [mm]')
    ax.set_title(r'$F_{ik}$ for tan$\theta$ = 0.28', loc='left')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('charge fraction', color=K.CHROME)
    cb.ax.tick_params(colors=K.CHROME)
    cb.outline.set_edgecolor(K.CHROME)

    # (c) profiles for a few bins
    ax = fig.add_subplot(gs[2])
    for k, col in ((0, C['blue']), (6, C['green']), (12, C['orange']),
                   (17, C['red'])):
        ax.plot(pos, F[:, k], 'o-', ms=3, color=col,
                label=f'k={k}  (u={wm.UK[k]:.0f} ns, z={wm.UK[k]*v*1e-3:.1f} mm)')
    ax.set_xlabel('strip position [mm]')
    ax.set_ylabel('fraction of that bin on the strip')
    ax.set_title('a depth bin spreads over more strips as it drifts', loc='left')
    ax.legend(fontsize=7)
    save(fig, 'strip_fractions')


# ------------------------------------------------------- 4. design matrix
def fig_design_matrix(eid):
    ev = EVS[eid]
    plane = 'x'
    P = K.trim_window(ev[plane])
    h = dict(CAL.hyper)
    v = CAL.v_drift
    pos = np.asarray(P['pos'], float)
    p0, w, t0 = ev['ref_mesh_x'], ev['tan_x'] * v * 1e-3, 420.0
    wm.set_nsamp(np.asarray(P['W']).shape[1])
    M = wm.build_matrix(plane, pos, p0, w, t0, h)
    Mr = M.reshape(len(pos), wm.NSAMP, wm.K)

    fig = plt.figure(figsize=(13, 6.6))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.15], hspace=0.45,
                  wspace=0.35)

    ks = [0, 5, 10, 15]
    vmax = Mr.max()
    for j, k in enumerate(ks):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(Mr[:, :, k], aspect='auto', origin='lower', cmap='magma',
                  vmin=0, vmax=vmax * 0.9,
                  extent=[0, wm.NSAMP * 60 / 1000, pos[0] - .39, pos[-1] + .39])
        ax.set_title(f'column k={k}\nz = {wm.UK[k]*v*1e-3:.1f} mm', loc='left',
                     fontsize=9)
        ax.set_xlabel('time [µs]')
        if j == 0:
            ax.set_ylabel('strip position [mm]')
        ax.grid(False)

    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(M, aspect='auto', origin='lower', cmap='magma',
                   interpolation='nearest')
    for i in range(1, len(pos)):
        ax.axhline(i * wm.NSAMP - 0.5, color='w', lw=0.35, alpha=0.35)
    ax.set_xlabel('depth bin $k$   (the 18 unknowns solved by NNLS)')
    ax.set_ylabel('row = (strip, sample), stacked')
    ax.set_title(f'the whole design matrix M: '
                 f'{M.shape[0]} rows × {M.shape[1]} columns', loc='left')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=K.CHROME)
    cb.outline.set_edgecolor(K.CHROME)

    ax = fig.add_subplot(gs[1, 2:])
    ic = int(np.argmax(Mr.sum(axis=(1, 2))))
    for k in range(0, wm.K, 2):
        ax.plot(np.arange(wm.NSAMP) * 0.06, Mr[ic, :, k],
                color=plt.cm.viridis(k / (wm.K - 1)), lw=1.3)
    ax.set_xlabel('time [µs]')
    ax.set_ylabel('model amplitude per unit charge')
    ax.set_title(f'one strip (position {pos[ic]:.1f} mm) — what each depth bin '
                 'contributes to it\n(dark = shallow, light = deep)', loc='left')
    save(fig, 'design_matrix')
    return eid


# ------------------------------------------------- 5. sharing decomposition
def fig_sharing_decomposition(eid):
    """Split the fitted model of a real event into own-charge, ±1 and ±2
    contributions. This is the picture that explains why a per-strip hit time
    cannot be a drift time."""
    ev = EVS[eid]
    h = dict(CAL.hyper)
    v = CAL.v_drift
    plane = 'x'
    P = K.trim_window(ev[plane])
    wm.set_nsamp(np.asarray(P['W']).shape[1])
    W, noise, pos, sat = wm.prep_plane(P, plane)
    p0l, wl = ev['ref_mesh_x'], ev['tan_x'] * v * 1e-3
    r = wm.fit_plane_raw(P, plane, p0l, wl, 400.0, hyper=h, fix_p0w=(p0l, wl))
    q, t0 = r['q'], r['t0']

    # rebuild the three pieces by zeroing c1/c2
    def piece(c1, c2):
        hh = dict(h, c1=c1, c2=c2)
        return (wm.build_matrix(plane, pos, p0l, wl, t0, hh) @ q).reshape(
            len(pos), wm.NSAMP)
    own = piece(0.0, 0.0)
    with1 = piece(h['c1'], 0.0)
    full = piece(h['c1'], h['c2'])
    sh1, sh2 = with1 - own, full - with1

    t = np.arange(wm.NSAMP) * 0.06
    ic = int(np.argmax(W.max(axis=1)))
    sel = [max(ic - 3, 0), max(ic - 1, 0), ic, min(ic + 1, len(pos) - 1),
           min(ic + 3, len(pos) - 1)]
    sel = sorted(set(sel))
    fig, axs = plt.subplots(1, len(sel), figsize=(2.8 * len(sel), 3.4),
                            sharex=True)
    for ax, i in zip(np.atleast_1d(axs), sel):
        ax.fill_between(t, 0, own[i], color=C['blue'], alpha=0.55,
                        label='own charge')
        ax.fill_between(t, own[i], own[i] + sh1[i], color=C['orange'],
                        alpha=0.6, label='±1 neighbours')
        ax.fill_between(t, own[i] + sh1[i], full[i], color=C['red'],
                        alpha=0.6, label='±2 neighbours')
        ax.plot(t, W[i], color=K.CHROME, lw=1.2, marker='o', ms=2.5,
                label='data')
        ax.set_title(f'{pos[i]:.1f} mm', loc='left', fontsize=9)
        ax.set_xlabel('time [µs]')
    np.atleast_1d(axs)[0].set_ylabel('ADC')
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.suptitle(f'event {eid}, X plane — what each strip is actually made of',
                 color=K.CHROME, fontsize=10.5, x=0.01, ha='left')
    save(fig, 'sharing_decomposition')
    return dict(q=q, t0=t0, p0=p0l, w=wl, W=W, noise=noise, pos=pos, sat=sat,
                full=full, own=own)


# ------------------------------------------------------ 6. model vs data
def fig_model_vs_data(eid, st):
    W, pos, noise = st['W'], st['pos'], st['noise']
    full, q, t0 = st['full'], st['q'], st['t0']
    t = np.arange(wm.NSAMP) * 0.06
    v = CAL.v_drift

    fig = plt.figure(figsize=(13, 6.2))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.05],
                  hspace=0.4, wspace=0.3)

    vmax = max(W.max(), full.max())
    for j, (A, lab) in enumerate(((W, 'data'), (full, 'model'),
                                  (W - full, 'residual'))):
        ax = fig.add_subplot(gs[0, j])
        cmap = 'magma' if j < 2 else 'coolwarm'
        kw = dict(vmin=0, vmax=vmax) if j < 2 else dict(vmin=-vmax*.3,
                                                        vmax=vmax*.3)
        im = ax.imshow(A, aspect='auto', origin='lower', cmap=cmap,
                       extent=[0, wm.NSAMP * .06, pos[0] - .39, pos[-1] + .39],
                       interpolation='nearest', **kw)
        ax.set_title(lab, loc='left')
        ax.set_xlabel('time [µs]')
        if j == 0:
            ax.set_ylabel('strip position [mm]')
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.ax.tick_params(colors=K.CHROME)
        cb.outline.set_edgecolor(K.CHROME)

    ax = fig.add_subplot(gs[1, :2])
    off = 0.32 * np.nanmax(W)
    for i in range(len(pos)):
        ax.plot(t, W[i] + i * off, color=K.CHROME, lw=1.0, marker='o', ms=2,
                alpha=0.85)
        ax.plot(t, full[i] + i * off, color=C['orange'], lw=1.5)
        ax.text(t[-1] + 0.02, i * off, f'{pos[i]:.1f}', fontsize=6.5,
                color=K.CHROME, va='center')
    ax.plot([], [], color=K.CHROME, marker='o', ms=3, label='data')
    ax.plot([], [], color=C['orange'], label='model')
    ax.set_yticks([])
    ax.set_xlabel('time [µs]')
    ax.set_ylabel('strip (offset), labelled by position [mm]')
    ax.set_title('every strip in the window, simultaneously', loc='left')
    ax.legend(loc='upper left')
    ax.set_xlim(0, t[-1] + 0.25)

    ax = fig.add_subplot(gs[1, 2])
    z = wm.UK * v * 1e-3
    ax.step(z, q, where='mid', color=C['green'])
    ax.fill_between(z, 0, q, step='mid', color=C['green'], alpha=0.3)
    tot = q.sum()
    cum = np.cumsum(q) / tot
    for frac, col, lab in ((0.5, C['blue'], 'u50'), (0.9, C['purple'], 'u90')):
        ax.axvline(np.interp(frac, cum, z), color=col, ls='--', lw=1,
                   label=f'{lab} = {np.interp(frac, cum, wm.UK):.0f} ns')
    ax.set_xlabel('drift depth  $z = v\\,u$  [mm]')
    ax.set_ylabel('fitted charge  $q_k$')
    ax.set_title('the NNLS charge profile', loc='left')
    ax.legend(fontsize=7.5)
    save(fig, 'model_vs_data')


# ------------------------------------------------------ 7. depth ladder
def fig_depth_ladder(eid, st):
    """The model as a sum over depth bins: each 60 ns slice of drift lands at
    its own transverse position and its own time."""
    pos, q, t0 = st['pos'], st['q'], st['t0']
    p0, w = st['p0'], st['w']
    h = dict(CAL.hyper)
    v = CAL.v_drift
    M = wm.build_matrix('x', pos, p0, w, t0, h).reshape(len(pos), wm.NSAMP, wm.K)
    t = np.arange(wm.NSAMP) * 0.06

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9),
                            gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    ic = int(np.argmax((M * q).sum(axis=(1, 2))))
    run = np.zeros(wm.NSAMP)
    for k in range(wm.K):
        add = M[ic, :, k] * q[k]
        ax.fill_between(t, run, run + add, color=plt.cm.viridis(k / (wm.K - 1)),
                        lw=0, alpha=0.95)
        run = run + add
    ax.plot(t, st['W'][ic], color=K.CHROME, lw=1.4, marker='o', ms=3,
            label='data')
    ax.set_xlabel('time [µs]')
    ax.set_ylabel('ADC')
    ax.set_title(f'one strip ({pos[ic]:.1f} mm), decomposed by depth bin\n'
                 '(dark = charge from the mesh, light = from the cathode)',
                 loc='left')
    ax.legend()

    ax = axs[1]
    zs = wm.UK * v * 1e-3
    line = p0 + w * wm.UK
    ax.scatter(line, zs, s=40 * q / max(q.max(), 1e-9) + 4, color=C['green'],
               zorder=3, label='fitted charge, sized by $q_k$')
    ax.plot(p0 + w * np.array([0, wm.UK[-1]]), [0, zs[-1]], color=C['ref'],
            lw=1.2, ls='--', label=f'the fitted line, tan = {w*1e3/v:+.3f}')
    ax.set_xlabel('transverse position [mm]')
    ax.set_ylabel('drift depth [mm]')
    ax.set_title('the same event in the (position, depth) plane', loc='left')
    ax.legend(fontsize=7.5)
    ax.invert_yaxis()
    save(fig, 'depth_ladder')


WORKED_EID = 1663      # clean on both planes; see wftdoc.rank_events


def main():
    setup()
    fig_template()
    fig_kernels()
    fig_strip_fractions()
    eid = WORKED_EID
    print(f'[model] worked example: event {eid}')
    fig_design_matrix(eid)
    st = fig_sharing_decomposition(eid)
    fig_model_vs_data(eid, st)
    fig_depth_ladder(eid, st)


if __name__ == '__main__':
    main()
