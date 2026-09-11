#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plastic_acceptance.py -- the trigger's own acceptance, from the Geant geometry.

WHY.  The measured hit maps show a dip down the middle of each chamber in the
in-plane coordinate.  Dylan's explanation: the production trigger needs a
coincidence with the TWO plastic bars behind each wall, and there is a gap
between them, so a charged particle threading that gap makes no plastic signal
and no trigger.  The chamber is not inefficient there; the trigger is blind.

That is a prediction, not a story, because the geometry is known.  This module
computes it and nothing is invented: every number is read from the Geant
config (``~/CLionProjects/MX17_Full_Geant/include/SimConfig.hh``) and the two
independent sources agree where they overlap --

    bscTape_hu + bsc_gap/2 = 100.22 + 1.5 = 101.72 mm  ==  PLASTIC_U_OFFSET
    mm_pinwheel_shift_cm    = {1.55, 1.575, 1.635, 1.73}  ==  PINWHEEL for D,B,A,C

WHAT IS COMPUTED.  For each point on a chamber's active surface, the fraction
of the He-3 capsule that can see it *through active plastic*: sample source
points in the capsule, draw the straight line to the surface point, extend it
to the plastic plane, and ask whether it lands on scintillator.  The answer is
an acceptance map, and it is **not** a hard edge, because the source is 60 mm
long and 20 mm across -- the penumbra is the physics.

WHAT IT IS NOT.  Only the plastics are imposed.  The SiPM wall, the chamber's
own efficiency, dead readout channels (chamber D loses ~130 of 512, STATUS.md)
and any material budget are all absent, so this is the *trigger geometry's*
ceiling and not a prediction of the measured map.  Comparing the two is the
point: what the geometry does not explain is what the detector is doing.

    python -m sept26_prelim_analysis.plastic_acceptance --scan
    python -m sept26_prelim_analysis.plastic_acceptance --dead-edge 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import figstyle as fs  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')

# --------------------------------------------------------------- the geometry
#: He-3 capsule: cylinder r = 10 mm, half-length 20 mm, hemispherical end caps
#: (SimConfig he3_radius_cm = 1.0, he3_half_length_cm = 2.0).
HE3_R_MM, HE3_HALF_L_MM = 10.0, 20.0

#: Plastic bars, per SimConfig: each 200 mm (u) x 300 mm (v) x 20 mm, wrapped in
#: 20 um Al + 200 um black mylar, with a 3 mm gap between the WRAPPED bars.
BSC_U_MM, BSC_V_MM, BSC_GAP_MM = 200.0, 300.0, 3.0
BSC_TAPE_UM, BSC_AL_UM = 200.0, 20.0
#: Half-width of the wrapped bar, and hence the bar centre offset.
_WRAP = (2 * BSC_AL_UM + 2 * BSC_TAPE_UM) / 1000.0
BSC_TAPE_HU = (BSC_U_MM + _WRAP) / 2.0
U_OFFSET = BSC_TAPE_HU + BSC_GAP_MM / 2.0          # 101.72 mm

#: Distances in the reconstruction's frame, which is what the measured maps use.
D_PERP_MM = 234.6
STRIPS_TO_PLASTIC = {'A': 188.1, 'B': 186.1, 'C': 186.1, 'D': 190.1}
PINWHEEL = {'D': 15.5, 'B': 15.75, 'A': 16.35, 'C': 17.3}
#: MM active area (SimConfig mm_size_u_cm / mm_size_v_cm).
MM_U_MM, MM_V_MM = 399.0, 360.0


def sample_source(n: int, rng) -> np.ndarray:
    """Points uniform in the capsule: cylinder plus hemispherical caps.

    Rejection inside the bounding cylinder of half-length (L + r), which is the
    capsule's actual extent; the caps make it a stadium of revolution.
    """
    out = []
    need = n
    while need > 0:
        m = int(need * 1.6) + 64
        y = rng.uniform(-(HE3_HALF_L_MM + HE3_R_MM), HE3_HALF_L_MM + HE3_R_MM, m)
        x = rng.uniform(-HE3_R_MM, HE3_R_MM, m)
        z = rng.uniform(-HE3_R_MM, HE3_R_MM, m)
        r2 = x * x + z * z
        inside = r2 <= HE3_R_MM ** 2
        cap = np.abs(y) > HE3_HALF_L_MM
        dy = np.abs(y) - HE3_HALF_L_MM
        inside &= np.where(cap, r2 + dy ** 2 <= HE3_R_MM ** 2, True)
        p = np.c_[x[inside], y[inside], z[inside]]
        out.append(p)
        need -= len(p)
    return np.vstack(out)[:n]


def acceptance_map(arm: str, dead_edge_mm: float = 0.0,
                   gap_mm: float | None = None, n_src: int = 400,
                   n_bins: int = 40, seed: int = 1) -> tuple:
    """Fraction of the capsule that sees each surface point through plastic.

    Works entirely in the chamber's local frame: u across the strips, v along
    the beam, and the plastic centred on the MM (the wall is centred on the
    structure instead, which is why only the wall carries the pinwheel term).
    """
    rng = np.random.default_rng(seed)
    S = sample_source(n_src, rng)
    off = U_OFFSET if gap_mm is None else (BSC_TAPE_HU + gap_mm / 2.0)
    inner = off - BSC_U_MM / 2.0 + dead_edge_mm      # inner active edge
    outer = off + BSC_U_MM / 2.0 - dead_edge_mm      # outer active edge
    v_half = BSC_V_MM / 2.0 - dead_edge_mm

    L = STRIPS_TO_PLASTIC[arm]
    # The source's own offset from the chamber axis, in the chamber frame: the
    # capsule's y is the beam axis = the chamber's v, and its transverse
    # coordinates project onto u with the pinwheel foot.
    su = S[:, 0] - (-PINWHEEL[arm])       # transverse, about the perpendicular foot
    sv = S[:, 1]
    sw = -D_PERP_MM + S[:, 2] * 0.0       # depth: the capsule is thin in w vs 234.6

    e = np.linspace(-MM_U_MM / 2, MM_U_MM / 2, n_bins + 1)
    ev = np.linspace(-MM_V_MM / 2, MM_V_MM / 2, n_bins + 1)
    cu, cv = 0.5 * (e[:-1] + e[1:]), 0.5 * (ev[:-1] + ev[1:])
    U, V = np.meshgrid(cu, cv, indexing='ij')

    # A straight line from (su, sv, sw) through (U, V, 0) reaches the plastic
    # plane at w = +L, i.e. a lever of (L - sw)/(0 - sw) beyond the MM.
    A = np.zeros_like(U)
    for i in range(len(su)):
        t = (L - sw[i]) / (0.0 - sw[i])
        up = su[i] + (U - su[i]) * t
        vp = sv[i] + (V - sv[i]) * t
        ok = ((np.abs(up) >= inner) & (np.abs(up) <= outer)
              & (np.abs(vp) <= v_half))
        A += ok
    A /= len(su)
    return cu, cv, A


def measured_profile(arm: str, tier: str = 'scint') -> tuple:
    """The measured x-projection of that tier, normalised to its own median."""
    from sept26_prelim_analysis.hit_maps import selection_tiers
    subs = ['stat090_0000', 'stat090_0001', 'stat090_0002']
    fp = str(paths.out('fullpass') / 'run_145')
    S = selection_tiers('run_145', subs, arm, fp)
    m = S[tier].to_numpy()
    e = np.arange(-200, 201, 10.0)
    h, _ = np.histogram(S.x.to_numpy()[m], bins=e)
    c = 0.5 * (e[:-1] + e[1:])
    core = h[(c > -140) & (c < 140)]
    return c, h / (np.median(core[core > 0]) if (core > 0).any() else 1.0)


def fig_expected(dead_edge_mm: float, n_src: int, out):
    """Two rows: the trigger geometry's ceiling, and it drawn over the data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sept26_prelim_analysis.hit_maps import selection_tiers, EDGES
    fs.use()

    subs = ['stat090_0000', 'stat090_0001', 'stat090_0002']
    fp = str(paths.out('fullpass') / 'run_145')
    rows = []
    with plt.rc_context({'font.size': fs.BASE_PT * 0.77,
                         'axes.titlesize': fs.BASE_PT * 0.86,
                         'axes.labelsize': fs.BASE_PT * 0.77,
                         'xtick.labelsize': fs.BASE_PT * 0.64,
                         'ytick.labelsize': fs.BASE_PT * 0.64}):
        fig, axes = plt.subplots(2, 4, figsize=(fs.BANNER[0], 5.33),
                                 constrained_layout=True)
        for c, arm in enumerate(ARMS):
            cu, cv, A = acceptance_map(arm, dead_edge_mm=dead_edge_mm,
                                       n_src=n_src, n_bins=60)
            ax = axes[0, c]
            im = ax.imshow(A.T, origin='lower', aspect='equal', cmap='magma',
                           vmin=0, vmax=max(A.max(), 1e-9),
                           extent=[cu[0], cu[-1], cv[0], cv[-1]])
            ax.set_title(f'chamber {arm}', color=fs.DET_COLOR[arm], pad=5)
            ax.set_xlabel('u local [mm]')
            ax.set_xlim(-200, 200); ax.set_ylim(-200, 200)
            if c == 0:
                ax.set_ylabel('v local (along beam) [mm]')
            for i, u in enumerate(cu):
                for j, v in enumerate(cv):
                    rows.append(dict(arm=arm, u=u, v=v, acceptance=A[i, j],
                                     dead_edge_mm=dead_edge_mm))

            # the data, with the model's penumbra drawn on it
            S = selection_tiers('run_145', subs, arm, fp)
            tier = 'target' if S.target.any() else 'pointing'
            m = S[tier].to_numpy()
            H, _, _ = np.histogram2d(S.x.to_numpy()[m], S.y.to_numpy()[m],
                                     bins=[EDGES, EDGES])
            ax2 = axes[1, c]
            ax2.imshow(H.T, origin='lower', aspect='equal', cmap='viridis',
                       extent=[EDGES[0], EDGES[-1], EDGES[0], EDGES[-1]])
            # The penumbra IS the message: three levels, because an extended
            # 60 mm source cannot cast a hard edge.
            X, Y = np.meshgrid(cu, cv, indexing='ij')
            mx = A.max() if A.max() > 0 else 1.0
            ax2.contour(X, Y, A / mx, levels=[0.2, 0.5, 0.8],
                        colors=['#ff4f36', '#ff4f36', '#ff4f36'],
                        linewidths=[0.8, 1.8, 0.8],
                        linestyles=[':', '-', ':'])
            ax2.set_title(f'{arm} — {tier}, n={int(m.sum()):,}',
                          color=fs.DET_COLOR[arm], pad=5)
            ax2.set_xlabel('u local [mm]')
            ax2.set_xlim(-200, 200); ax2.set_ylim(-200, 200)
            if c == 0:
                ax2.set_ylabel('v local (along beam) [mm]')
        fig.colorbar(im, ax=axes[0, :], fraction=0.02, pad=0.01,
                     label='fraction of the He-3 capsule seen through plastic')
        fig.suptitle(f'Trigger acceptance from the plastic geometry '
                     f'(dead edge {dead_edge_mm:.0f} mm) — top: predicted; '
                     f'bottom: measured, with the 20/50/80 % contours over it',
                     fontsize=fs.BASE_PT * 0.96)
        fs.preliminary(axes[0, 0], loc='lower left')
        fs.save(fig, out / 'plastic_acceptance', data=pd.DataFrame(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--dead-edge', type=float, default=0.0,
                    help='mm of inactive scintillator at every bar edge')
    ap.add_argument('--gap', type=float, default=None,
                    help='override the 3 mm wrapped-bar gap')
    ap.add_argument('--n-src', type=int, default=400)
    ap.add_argument('--scan', action='store_true',
                    help='compare a range of dead-edge values against the data')
    a = ap.parse_args()

    print(f'plastic bars {BSC_U_MM:.0f} x {BSC_V_MM:.0f} mm, wrapped half-width '
          f'{BSC_TAPE_HU:.2f} mm, gap {BSC_GAP_MM:.1f} mm')
    print(f'bar centre offset {U_OFFSET:.2f} mm  (reco PLASTIC_U_OFFSET = 101.72)')
    print(f'capsule r {HE3_R_MM:.0f} mm, half-length {HE3_HALF_L_MM:.0f} mm '
          f'+ hemispherical caps -> {2 * (HE3_HALF_L_MM + HE3_R_MM):.0f} mm long\n')

    if a.scan:
        cm_, meas = measured_profile('A', 'scint')
        print('dip depth at the chamber centre, model vs measurement (arm A):')
        print(f'{"dead edge":>10} {"gap":>6} {"model min":>10} {"width@50%":>10}')
        for de in (0.0, 2.0, 5.0, 8.0, 12.0, 16.0):
            cu, cv, A = acceptance_map('A', dead_edge_mm=de, n_src=a.n_src)
            prof = A.mean(axis=1)
            prof = prof / np.median(prof[np.abs(cu) < 140])
            lo = prof.min()
            below = cu[prof < 0.5]
            w = (below.max() - below.min()) if len(below) else 0.0
            print(f'{de:>10.1f} {BSC_GAP_MM:>6.1f} {lo:>10.2f} {w:>10.0f}')
        mm_ = meas[(cm_ > -60) & (cm_ < 80)]
        print(f'\nmeasured (arm A, scint tier): min {mm_.min():.2f} of median')
        below = cm_[(meas < 0.5) & (np.abs(cm_) < 100)]
        if len(below):
            print(f'  below half over x in [{below.min():+.0f}, {below.max():+.0f}] '
                  f'-> width {below.max() - below.min() + 10:.0f} mm')
        return 0

    od = paths.figures('acceptance')
    fig_expected(a.dead_edge, a.n_src, od)
    print(f'\nwrote {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
