#!/usr/bin/env python3
"""
Part VI figures — the calibration: what is measured, how, and the one failure
mode that has cost the most time (v trading against the sharing kernel).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import wftdoc as K
from wftdoc import C, save

from wft import model as wm
from wft.calibrate import TEMPLATE_GRID, TEMPLATE_TAN_MIN, TEMPLATE_MIN_AMP

CAL = None
EVS = None


def setup():
    global CAL, EVS
    CAL = K.install()
    EVS = K.calib_events()
    return CAL, EVS


# ----------------------------------------------------------- 1. corridor
def fig_corridor():
    """How the calibration cache cuts its windows: along the reference
    corridor, not around a seed cluster — the point is to see the whole track
    including the strips the hit finder missed."""
    e = EVS[1663]
    P = e['x']
    W = np.asarray(P['W'], float)
    pos = np.asarray(P['pos'], float)
    v = CAL.v_drift
    p0, tan = e['ref_mesh_x'], e['tan_x']

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    im = ax.imshow(W, aspect='auto', origin='lower', cmap='magma',
                   extent=[0, W.shape[1] * 0.06, pos[0] - .39, pos[-1] + .39],
                   interpolation='nearest')
    for z, ls, lab in ((-3.0, ':', None), (33.0, ':', None)):
        ax.axhline(p0 + z * tan, color=C['ref'], ls=ls, lw=1.4, label=lab)
    ax.axhline(p0 + -3.0 * tan - 5.0, color=C['blue'], lw=1.2,
               label='corridor ± 5 mm pad')
    ax.axhline(p0 + 33.0 * tan + 5.0, color=C['blue'], lw=1.2)
    ax.plot([], [], color=C['ref'], ls=':', lw=1.4,
            label='reference line, z = −3 … 33 mm')
    ax.set_xlabel('time [µs]'); ax.set_ylabel('strip position [mm]')
    ax.set_title(f'calibration window, event 1663 X: every strip the reference\n'
                 f'track could have touched (tan = {tan:+.3f})', loc='left')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.ax.tick_params(colors=K.CHROME); cb.outline.set_edgecolor(K.CHROME)
    save(fig, 'calib_corridor')


# ----------------------------------------------------------- 2. template
def fig_template_build():
    """Rebuild the impulse template from the cache exactly as
    wft.calibrate.measure_templates does, and show the ingredients."""
    def t50(w):
        ipk = int(np.argmax(w))
        a = w[ipk]
        for k in range(1, ipk + 1):
            if w[k] >= 0.5 * a > w[k - 1]:
                return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
        return np.nan

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for ax, plane, col in ((axs[0], 'x', C['x']), (axs[1], 'y', C['y'])):
        acc = []
        for ev in EVS.values():
            if plane not in ev or abs(ev[f'tan_{plane}']) < TEMPLATE_TAN_MIN:
                continue
            W = np.asarray(ev[plane]['W'], np.float32)
            ns = W.shape[1]
            amax = W.max(axis=1)
            for i in np.argsort(amax)[::-1][:2]:
                w = W[i]
                a = w.max(); ipk = int(np.argmax(w))
                if a < TEMPLATE_MIN_AMP or a > 3550 or ipk < 6 or ipk > ns - 12:
                    continue
                c = t50(w)
                if np.isfinite(c):
                    tt = (np.arange(ns) - c) * 60.0
                    acc.append(np.interp(TEMPLATE_GRID, tt, w / a,
                                         left=np.nan, right=np.nan))
        A = np.array(acc)
        for r in A[:120]:
            ax.plot(TEMPLATE_GRID, r, color=col, alpha=0.06, lw=0.9)
        t = np.nanmedian(A, axis=0)
        t -= np.nanmedian(t[TEMPLATE_GRID < -250])
        ax.plot(TEMPLATE_GRID, t, color=K.CHROME, lw=2.2,
                label=f'median of {len(A)} pulses')
        ax.plot(TEMPLATE_GRID, np.asarray(CAL.tmpl[plane], float), color='k',
                lw=1.0, ls='--', label='the bundle\'s template')
        ax.axhline(0, color=K.CHROME, lw=0.7)
        ax.set_xlim(-350, 1400); ax.set_ylim(-0.2, 1.15)
        ax.set_xlabel('time relative to the 50 % crossing [ns]')
        ax.set_ylabel('amplitude / peak')
        ax.set_title(f'{plane} plane: bright strips of inclined tracks '
                     f'(|tan| > {TEMPLATE_TAN_MIN})', loc='left')
        ax.legend(fontsize=8)
        print(f'[calib] {plane}: {len(A)} template candidates')
    save(fig, 'template_build')


# ------------------------------------------------- 3. the v <-> c1 degeneracy
def _event_chi2_at(eid, hyper, v, planes=('x', 'y')):
    ev = EVS[eid]
    tot = 0.0
    for plane in planes:
        if plane not in ev:
            continue
        P = K.trim_window(ev[plane])
        W = np.asarray(P['W'], float)
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        Wp, noise, pos, sat = wm.prep_plane(P, plane)
        wline = ev[f'tan_{plane}'] * v * 1e-3
        p0l = ev[f'ref_mesh_{plane}']
        grid = np.arange(180.0, 720.0, 30.0)
        cs = [wm.chi2_plane(plane, Wp, noise, pos, sat, p0l, wline, float(t),
                            hyper)[0] for t in grid]
        c = float(np.nanmin(cs))
        if np.isfinite(c):
            tot += c
    return tot


def fig_degeneracy(n_ev=32, n_grid=11):
    """The failure mode that produced det7's c1 = 0.004 and det4's kY = 3.2:
    the calibration chi2 has a long flat valley in (sharing, v)."""
    h0 = dict(CAL.hyper)
    v0 = CAL.v_drift
    eids = [e for e in sorted(EVS)[:200]
            if 'x' in EVS[e] and 'y' in EVS[e]
            and 0.10 < abs(EVS[e]['tan_x']) < 0.45][:n_ev]
    c1s = np.linspace(0.05, 0.60, n_grid)
    vs = np.linspace(28.0, 46.0, n_grid)
    Z = np.zeros((len(c1s), len(vs)))
    for i, c1 in enumerate(c1s):
        for j, v in enumerate(vs):
            hh = dict(h0, c1=float(c1))
            Z[i, j] = sum(_event_chi2_at(e, hh, float(v)) for e in eids)
        print(f'[calib]   c1={c1:.2f} done', flush=True)
    Z /= 1e6
    i0, j0 = np.unravel_index(np.argmin(Z), Z.shape)
    print(f'[calib] degeneracy map minimum at c1={c1s[i0]:.3f}, '
          f'v={vs[j0]:.1f}; bundle sits at c1={h0["c1"]:.3f}, v={v0:.1f}')

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9),
                            gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    im = ax.pcolormesh(vs, c1s, Z, cmap='viridis_r', shading='auto')
    lv = Z.min() * (1 + np.array([0.002, 0.005, 0.01, 0.02, 0.05]))
    ax.contour(vs, c1s, Z, levels=lv, colors='w', linewidths=0.8)
    ax.plot(vs[j0], c1s[i0], '*', ms=15, color=C['orange'],
            label='minimum of this slice')
    ax.plot(v0, h0['c1'], 'o', ms=9, color=C['red'], mfc='none', mew=2,
            label='the bundle (c1 pinned to the beam value)')
    ax.set_xlabel(r'drift velocity $v$ [µm/ns]')
    ax.set_ylabel(r'sharing amplitude $c_1$')
    ax.set_title(r'ref-pinned $\chi^2$ over the training set —'
                 '\nwhite contours at +0.2, 0.5, 1, 2, 5 %', loc='left')
    ax.legend(fontsize=7.5); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'total $\chi^2$  [$10^6$]', color=K.CHROME)
    cb.ax.tick_params(colors=K.CHROME); cb.outline.set_edgecolor(K.CHROME)

    ax = axs[1]
    ax.plot(vs, Z.min(axis=0) / Z.min(), color=C['blue'], marker='o', ms=4,
            label=r'best over $c_1$ at each $v$')
    ax.plot(vs, Z[i0] / Z.min(), color=C['grey'], ls='--',
            label=fr'at the fitted $c_1$ = {c1s[i0]:.2f}')
    ax.axvline(v0, color=C['red'], ls=':', label=f'bundle v = {v0:.1f}')
    ax.set_xlabel(r'$v$ [µm/ns]')
    ax.set_ylabel(r'$\chi^2$ / best')
    ax.set_title('profiled over the kernel, $v$ is almost free:\n'
                 'a 10 % move costs well under 1 % of $\\chi^2$', loc='left')
    ax.legend(fontsize=7.5)
    save(fig, 'degeneracy')


# --------------------------------------------------- 4. lp vs delay kernel
def fig_share_modes(n_ev=60):
    h = dict(CAL.hyper)
    v = CAL.v_drift
    eids = [e for e in sorted(EVS)[:200]
            if 'x' in EVS[e] and abs(EVS[e]['tan_x']) > 0.10][:n_ev]
    out = {}
    for mode in ('lp', 'delay'):
        wm.set_share_mode(mode)
        out[mode] = np.array([_event_chi2_at(e, h, v, planes=('x',))
                              for e in eids])
    wm.set_share_mode(CAL.share_mode)
    r = out['delay'] / out['lp']
    print(f'[calib] delay/lp chi2 ratio: median {np.median(r):.3f}, '
          f'lp better in {100*np.mean(r>1):.0f}% of planes')

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axs[0]
    ax.scatter(out['lp'] / 1e5, out['delay'] / 1e5, s=16, color=C['blue'],
               alpha=0.75)
    lim = [0, max(out['lp'].max(), out['delay'].max()) / 1e5 * 1.05]
    ax.plot(lim, lim, color=K.CHROME, lw=1, ls='--')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r'$\chi^2$, lp kernel  [$10^5$]')
    ax.set_ylabel(r'$\chi^2$, delay kernel  [$10^5$]')
    ax.set_title('same events, same hypers, only the kernel form changes',
                 loc='left')

    ax = axs[1]
    ax.hist(r, bins=np.linspace(0.8, 1.6, 50), color=C['blue'], alpha=0.85)
    ax.axvline(1, color=K.CHROME, lw=1)
    ax.axvline(np.median(r), color=C['orange'], ls='--',
               label=f'median {np.median(r):.3f}')
    ax.set_xlabel(r'$\chi^2$(delay) / $\chi^2$(lp)')
    ax.set_ylabel('planes')
    ax.set_title(f'the RC-dispersed copy fits better on\n'
                 f'{100*np.mean(r>1):.0f} % of planes '
                 '(at these hypers, which were fitted for lp)', loc='left')
    ax.legend(fontsize=8)
    save(fig, 'share_modes')


# ------------------------------------------------------ 5. fleet kernels
FLEET = {
    # detector: (c1, kY, tau_s [ns], v [um/ns], source)
    'det3 (A)': (0.306, 1.222, 127.0, 36.6, 'live lp bundle, this document'),
    'det2 (B)': (0.290, 1.420, 49.0, 39.9, 'WAVEFORM_FIRST_THREADING §22'),
    'det4 (E)': (0.250, 2.100, 141.0, 34.0, 'EXTRACTION 2026-08-05 arm C'),
    'det6': (0.300, 1.500, 60.0, 26.4, 'FLEET_2026-07-29 (indicative)'),
    'det7 (bad)': (0.004, 6.600, 47.0, 36.7, 'degenerate fit — do not use'),
}


def fig_fleet():
    names = list(FLEET)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.2))
    for ax, idx, lab, ref in ((axs[0], 0, r'sharing amplitude $c_1$', 0.3),
                              (axs[1], 1, r'plane asymmetry $k_Y$', 1.0),
                              (axs[2], 3, r'drift velocity $v$ [µm/ns]', None)):
        vals = [FLEET[n][idx] for n in names]
        cols = [C['red'] if 'bad' in n else C['blue'] for n in names]
        ax.barh(names, vals, color=cols, height=0.55)
        if ref is not None:
            ax.axvline(ref, color=K.CHROME, ls='--', lw=1)
        for i, v_ in enumerate(vals):
            ax.text(v_ * 1.02, i, f'{v_:g}', va='center', fontsize=8,
                    color=K.CHROME)
        ax.set_xlabel(lab)
        ax.invert_yaxis()
    axs[0].set_title('kernels are per detector — and a physically impossible\n'
                     'one is how you recognise a degenerate fit', loc='left')
    save(fig, 'fleet_kernels')


def main():
    setup()
    fig_corridor()
    fig_template_build()
    fig_share_modes()
    fig_fleet()
    fig_degeneracy()


if __name__ == '__main__':
    main()
