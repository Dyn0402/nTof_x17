#!/usr/bin/env python3
"""
make_pose2_report.py -- the capsule pose refit on the real EAR2 beam
(capsule_pose pose2), as <out>/capsule_pose_ear2/report.html with figures/.

    python -m sept26_prelim_analysis.make_pose2_report
"""
from __future__ import annotations

import datetime as dt
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402

OUT = paths.out('capsule_pose_ear2')
FIG = OUT / 'figures'


def load() -> pd.DataFrame:
    fs_ = [f for f in glob.glob(str(OUT / 'pose2_*.csv')) if not f.endswith('_all.csv')]
    R = pd.concat([pd.read_csv(f) for f in fs_], ignore_index=True)
    if 'x_b' not in R:
        R['x_b'] = 0.0
    R['x_b'] = R.x_b.fillna(0.0)
    R = R.drop_duplicates(['x_b', 'x_c', 'theta', 'phi'])
    R['rel'] = R.x_c - R.x_b
    return R


def fig_tilt(R):
    import matplotlib.pyplot as plt
    fs.use()
    g = R[R.x_b == 0]
    T = g.groupby(['theta', 'phi']).chi2_shape.min().unstack()
    fig, ax = fs.figure(fs.HALF)
    th = T.index.to_numpy()
    for ph, col in ((0, fs.TRACK), (90, fs.MUTED), (135, '#0072B2'), (180, fs.INK), (225, '#009E73'), (315, fs.COPPER)):
        if ph in T.columns:
            y = T[ph].copy()
            y.loc[0.0] = T.iloc[0].dropna().iloc[0]
            ax.plot(th, y.reindex(th), '-o', ms=4, lw=1.5, color=col, label=f'φ = {ph}°')
    ax.set_xlabel('capsule tilt θ (deg)')
    ax.set_ylabel('χ² of the x-vs-height shape (10 bins)')
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fs.title(ax, 'The shape wants the top leaning toward −x', 'beam at the frame centre; best capsule x per tilt')
    plt.tight_layout()
    fs.save(fig, FIG / 'tilt_shape', data=T.reset_index())


def fig_degeneracy(R):
    fs.use()
    fig, ax = fs.figure(fs.HALF)
    best = R.loc[R.groupby(['x_b', 'x_c']).chi2.idxmin()]
    for xb, col in ((0.0, fs.INK), (-5.0, '#0072B2'), (-10.0, fs.COPPER)):
        g = best[best.x_b == xb].sort_values('x_c')
        ax.plot(g.x_c, g.chi2, '-o', color=col, ms=5, lw=1.5, label=f'beam at x = {xb:g} mm')
    ax.set_xlabel('capsule x (mm, frame)')
    ax.set_ylabel('χ² (x crossings + shape), best tilt')
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=8.5)
    fs.title(ax, 'Beam and capsule trade off sideways', 'each point: the best tilt at that pose')
    fs.save(fig, FIG / 'degeneracy', data=best[['x_b', 'x_c', 'theta', 'phi', 'chi2', 'chi2_x', 'chi2_shape', 'gas', 'dy_c']])


def build():
    R = load()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_tilt(R)
    fig_degeneracy(R)
    best = R.loc[R.groupby(['x_b', 'x_c']).chi2.idxmin()].sort_values(['x_b', 'x_c'])
    cols = ['x_b', 'x_c', 'theta', 'phi', 'gas', 'al_nose', 'al_barrel', 'x_A', 'x_C', 'chi2_x', 'chi2_shape', 'chi2', 'dy_c']
    tbl = best[cols].round(2).to_html(index=False, border=0)
    unt = R[(R.theta == 0) & (R.x_b == 0)].chi2_shape
    til = R[(R.theta >= 7) & R.phi.isin([135, 180, 225])].chi2_shape
    b0 = best[best.x_b == 0].chi2.min()
    bb = best.chi2.min()
    gas0 = best[best.x_b == 0].sort_values('chi2').gas.iloc[0]
    css = ('body{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1b2430;'
           'line-height:1.5}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #d5dae1;padding:2px 6px;'
           'text-align:right}img{max-width:100%}.verdict{background:#f6f0f7;border-left:4px solid #8a3f8f;padding:10px 14px}'
           '.cap{color:#6a7583;font-size:14px}')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Capsule pose on the real beam</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{css}</style></head><body>
<h1>Capsule pose on the real EAR2 beam</h1>
<p class="cap">Generated {dt.date.today().isoformat()} by <code>make_pose2_report.py</code> from <code>capsule_pose pose2</code>
(beam: <code>ear2_beam</code>, capsule at 19.5 m; data: is2_v2 cf, late window 20–75 ms, A and C).</p>
<div class="verdict"><p><b>Verdict.</b> With the real beam the old two "readings" are gone. Three things come out:
<b>the capsule top leans toward −x by ~7–15°</b> (untilted shape χ² {unt.min():.0f}–{unt.max():.0f} for 10 points, tilted
toward −x {til.min():.0f}–{np.percentile(til, 25):.0f}); <b>it sits ~8–12 mm lower than drawn</b> in every good pose; and
<b>sideways, beam and capsule trade off</b>: with the beam at the frame centre the capsule is ~15 mm off it (best χ² {b0:.1f},
³He sees {gas0:.2f} of the beam), and with the beam ~10 mm off the frame the capsule sits on or near it (best χ² {bb:.1f},
³He ~0.55). The literature puts the beam on the pipe axis to 1–2 mm.</p></div>
<h2>Best pose per (beam x, capsule x)</h2>{tbl}
<p class="cap">gas: fraction of the beam that reaches the ³He; dy_c: how far the capsule must move in height (mm) to put its
y crossing on the data's. χ²_x uses 1 mm per chamber crossing (chain systematic); χ²_shape uses the data errors and the toy's
Geant4-library spread per height bin.</p>
<img src="figures/tilt_shape.png" alt="shape chi2 vs tilt"><img src="figures/degeneracy.png" alt="chi2 vs capsule x for three beam positions">
<h2>What this does not rule out</h2><ul>
<li>The shape χ² is limited by the Geant4 track library (~10 k tracks per arm); Δχ² of ~15 between beam positions is not decisive.</li>
<li>The tilt's z component: A and C see x only. B/D z-vs-height would pin φ (135° vs 225°); D needs its hot columns masked first.</li>
<li>The late (20–75 ms) capsule tracks are ~100× more than the direct beam gives at those times (n_TOF phase space); the fit
assumes they are lit like the beam. The 4–12 ms window alone is lower purity.</li>
<li>The capsule at 19.95 m (broader beam) is not refitted; nor is the big collimator.</li>
</ul></body></html>'''
    (OUT / 'report.html').write_text(page)
    print('wrote', OUT / 'report.html')


if __name__ == '__main__':
    build()
