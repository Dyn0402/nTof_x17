#!/usr/bin/env python3
"""
An event gallery: real det3 muons through the production fit, spanning the
range of behaviour from textbook to broken. Selected by *outcome* (angle
residual against the reference), not by eye, so the failures are honest.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import wftdoc as K
from wftdoc import C, save

from wft import model as wm, reco as wr

CAL = None
EVS = None


def setup():
    global CAL, EVS
    CAL = K.install()
    EVS = K.calib_events()
    return CAL, EVS


def fit_all(plane='x', n=300):
    v = CAL.v_drift
    out = []
    for eid in sorted(EVS)[:n]:
        e = EVS[eid]
        if plane not in e:
            continue
        P = K.trim_window(e[plane])
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        try:
            f = wr.fit_plane(P, plane, CAL)
        except Exception:
            continue
        if f is None:
            continue
        d = (np.degrees(np.arctan(f.tan_theta))
             - np.degrees(np.arctan(e[f'tan_{plane}'])))
        out.append(dict(eid=eid, P=P, fit=f, dtheta=d,
                        tan_ref=e[f'tan_{plane}'], p0_ref=e[f'ref_mesh_{plane}'],
                        dp0=f.p0 - e[f'ref_mesh_{plane}']))
    return out


def panel(ax, rec, plane, show_ref=True):
    P, f = rec['P'], rec['fit']
    W = np.asarray(P['W'], float)
    pos = np.asarray(P['pos'], float)
    v = CAL.v_drift
    h = dict(CAL.hyper)
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    Wp, noise, _p, sat = wm.prep_plane(P, plane)
    q = None
    c, q = wm.chi2_plane(plane, Wp, noise, pos, sat, f.p0, f.w, f.t0, h,
                         snap_t0=False)
    ax.imshow(W, aspect='auto', origin='lower', cmap='magma',
              extent=[0, W.shape[1] * 0.06, pos[0] - .39, pos[-1] + .39],
              interpolation='nearest')
    # overlay the fitted and reference lines in (time, position)
    u = np.linspace(0, 900, 40)
    ax.plot((f.t0 + u) * 1e-3, f.p0 + f.w * u, color=C['orange'], lw=1.8,
            label=f'fit  {np.degrees(np.arctan(f.tan_theta)):+.1f}°')
    if show_ref:
        w_ref = rec['tan_ref'] * v * 1e-3
        ax.plot((f.t0 + u) * 1e-3, rec['p0_ref'] + w_ref * u, color=C['ref'],
                lw=1.6, ls='--',
                label=f'M3  {np.degrees(np.arctan(rec["tan_ref"])):+.1f}°')
    ax.set_xlim(0, W.shape[1] * 0.06)
    ax.set_ylim(pos[0] - .39, pos[-1] + .39)
    ax.grid(False)
    ax.legend(fontsize=6.5, loc='upper left', framealpha=0.35,
              labelcolor='w', facecolor='k')
    ax.set_title(f'ev {rec["eid"]}  Δθ {rec["dtheta"]:+.1f}°  '
                 f'Δp₀ {rec["dp0"]:+.2f} mm\n'
                 f'χ²/dof {f.chi2/max(f.dof,1):.0f}, {f.n_strips} strips',
                 loc='left', fontsize=8)


def fig_gallery(plane='x'):
    recs = fit_all(plane)
    recs = [r for r in recs if abs(r['tan_ref']) > 0.10]
    recs.sort(key=lambda r: abs(r['dtheta']))
    n = len(recs)
    print(f'[gallery] {n} fitted planes; |Δθ| median {np.median([abs(r["dtheta"]) for r in recs]):.2f}°')
    picks = ([('best', recs[0]), ('median', recs[n // 2]),
              ('75th percentile', recs[int(0.75 * n)]),
              ('90th percentile', recs[int(0.90 * n)]),
              ('97th percentile', recs[int(0.97 * n)]),
              ('worst', recs[-1])])
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.0))
    for ax, (lab, r) in zip(axs.ravel(), picks):
        panel(ax, r, plane)
        ax.set_xlabel('time [µs]', fontsize=8)
        ax.set_ylabel('position [mm]', fontsize=8)
        ax.text(0.98, 0.03, lab, transform=ax.transAxes, ha='right',
                fontsize=8.5, color='w')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.suptitle('det3 X-plane fits ordered by angle error — the tail is real '
                 'and is mostly δ-rays, second tracks and saturation',
                 color=K.CHROME, fontsize=11, x=0.01, y=0.985, ha='left')
    save(fig, 'gallery', tight=False)
    for lab, r in picks:
        print(f'[gallery] {lab:18s} ev {r["eid"]:5d} dtheta {r["dtheta"]:+6.2f} '
              f'dp0 {r["dp0"]:+6.2f} chi2/dof '
              f'{r["fit"].chi2/max(r["fit"].dof,1):7.0f} '
              f'nstrips {r["fit"].n_strips}')


def main():
    setup()
    fig_gallery('x')


if __name__ == '__main__':
    main()
