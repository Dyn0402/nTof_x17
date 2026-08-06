#!/usr/bin/env python3
"""
edge_fringe_test.py — discriminate mechanical gap topography from edge fringe
fields in the charge-visible gap maps.

Physics: boundary-condition distortions of the drift field in a parallel-plate
gap of depth d obey Laplace decay into the volume: the lowest transverse mode
falls as exp(-pi*s/d) with distance s from the edge (e-folding d/pi ~ 9.5 mm
for d = 30 mm). A fringe artefact in the apparent gap must therefore be an
edge-universal function of s that is dead (<1 %) by s ~ 45 mm — and, since
det2 and det3 share the chamber construction, the SAME function on both.
A mechanically dished/tilted cathode has no reason to organise by s and
survives at arbitrary distance from the edges.

Kernel-free version of the gap fit (gap_map_hires.py smooths with r = 45 mm,
which bleeds edge features inward): events are binned directly by their
distance to the nearest active-area edge; the stacked-profile erfc endpoint
(T) AND its softness (sig) are fitted per bin. Fringe predicts, in the edge
bins only: T pulled + sig inflated; the interior bins are fringe-free by
construction.

    ../../.venv/bin/python mx_june_wft/bench/edge_fringe_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO]

U = (np.arange(18) + 0.5) * 60.0
GAP_MECH = 30.0
# drift-volume (nominal strip-frame) boundary, detector-local mm
XLO, XHI = 0.0, 398.58
YLO, YHI = 0.0, 398.58
EDGES = np.array([0, 12, 24, 36, 48, 62, 78, 96, 116, 140, 170, 200])
D_FRINGE_DEAD = 45.0   # exp(-pi*45/30) ~ 0.9 %

DETS = [
    ('det3', '/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_'
     '6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/gap_study/'
     'event_profiles.parquet', 36.83, '#d01c8b'),
    ('det2', '/home/dylan/x17/cosmic_bench/Analysis/mx17_det2_det3_overnight_'
     '6-22-26/longer_run/mx17_2/wft/gap_study/event_profiles.parquet',
     39.04, '#5e3c99'),
]


def sharp(u, A, T, sig):
    return A * 0.5 * erfc((u - T) / (np.sqrt(2) * sig))


def fit_T_sig(P):
    m = P.mean(axis=0)
    e = np.maximum(P.std(axis=0) / np.sqrt(len(P)), 1e-5)
    sel = U < 1050
    try:
        p, c = curve_fit(sharp, U[sel], m[sel], p0=[m[:5].mean(), 700, 60],
                         sigma=e[sel], absolute_sigma=True, maxfev=20000)
        return (float(p[1]), float(np.sqrt(c[1, 1])),
                float(abs(p[2])), float(np.sqrt(c[2, 2])))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def profiles_by_edge_dist(path, v_geom):
    df = pd.read_parquet(path)
    g = df[(df.plane == 'x') & df.contained & (df.chi2dof < 250)].copy()
    Q = g[[f'q{i}' for i in range(18)]].to_numpy()
    Q = Q / Q.sum(axis=1, keepdims=True)
    x, y = g.ref_x.to_numpy(), g.ref_y.to_numpy()
    s = np.minimum.reduce([x - XLO, XHI - x, y - YLO, YHI - y])
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum() < 80:
            continue
        T, Te, sg, sge = fit_T_sig(Q[m])
        rows.append(dict(s=0.5 * (lo + hi), n=int(m.sum()),
                         gap=T * v_geom / 1e3, gap_err=Te * v_geom / 1e3,
                         sig=sg * v_geom / 1e3, sig_err=sge * v_geom / 1e3))
    interior = s > 100.0
    Ti, Tie, sgi, _ = fit_T_sig(Q[interior])
    return pd.DataFrame(rows), (Ti * v_geom / 1e3, Tie * v_geom / 1e3,
                                sgi * v_geom / 1e3, int(interior.sum()))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), layout='constrained')
    fig.suptitle('Fringe-field test: gap and endpoint softness vs distance to '
                 'nearest edge (kernel-free, X plane, contained tracks)',
                 fontsize=11)
    results = {}
    for det, path, vg, col in DETS:
        prof, (gi, gie, sgi, ni) = profiles_by_edge_dist(path, vg)
        results[det] = (prof, gi, gie, sgi, ni)
        for ax, key in zip(axs, ('gap', 'sig')):
            ax.errorbar(prof.s, prof[key], yerr=prof[f'{key}_err'],
                        color=col, lw=1.8, marker='o', ms=4, capsize=2,
                        label=f'{det} (interior: {gi:.1f} mm)' if key == 'gap'
                        else det)
        print(f'{det}: interior (s>100 mm, n={ni}) gap = {gi:.2f} '
              f'+- {gie:.2f} mm, endpoint sigma = {sgi:.2f} mm')
        print(prof.to_string(index=False,
                             float_format=lambda v: f'{v:.2f}'))

    for ax in axs:
        ax.axvspan(0, D_FRINGE_DEAD, color='0.85', alpha=0.6, zorder=0)
        ax.set_xlabel('distance to nearest edge s [mm]')
        ax.grid(alpha=0.25)
    axs[0].axhline(GAP_MECH, color='k', lw=1, ls=':')
    axs[0].text(100, GAP_MECH + 0.12, '30 mm mechanical', fontsize=8)
    axs[0].text(2, 25.6, 'fringe-permitted zone\n(exp(-πs/d): <1% '
                'beyond 45 mm)', fontsize=7.5, color='0.35')
    axs[0].set_ylabel('charge-visible gap [mm]')
    axs[0].set_ylim(25.3, 31.8)
    axs[0].legend(fontsize=8, loc='center right')
    axs[0].set_title('endpoint position', fontsize=10)
    axs[1].set_ylabel('fitted erfc endpoint width [mm]')
    axs[1].set_title('endpoint softness (fringe would inflate edge bins)',
                     fontsize=10)

    out = os.path.join(REPO, 'mx_june_wft', 'edge_fringe_test.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
