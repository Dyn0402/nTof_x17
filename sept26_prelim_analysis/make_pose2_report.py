#!/usr/bin/env python3
"""
make_pose2_report.py -- the capsule pose refit on the real EAR2 beam
(capsule_pose pose2), as <out>/capsule_pose_ear2/report.html with figures/.

    python -m sept26_prelim_analysis.make_pose2_report
    POSE2_L=19.95 POSE2_OUT=capsule_pose_ear2_L19p95 python -m sept26_prelim_analysis.make_pose2_report
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402

OUT = paths.out(os.environ.get('POSE2_OUT', 'capsule_pose_ear2'))
L_M = float(os.environ.get('POSE2_L', 19.5))
FIG = OUT / 'figures'


def load() -> pd.DataFrame:
    fs_ = [f for f in glob.glob(str(OUT / 'pose2_*.csv')) if not f.endswith(('_all.csv', '_xz.csv'))]
    R = pd.concat([pd.read_csv(f) for f in fs_], ignore_index=True)
    if 'x_b' not in R:
        R['x_b'] = 0.0
    R['x_b'] = R.x_b.fillna(0.0)
    R = R.drop_duplicates(['x_b', 'x_c', 'theta', 'phi'])
    R['rel'] = R.x_c - R.x_b
    return R


def load_z():
    """pose2z (D's z vs height) at alpha = 0, and the D-rotation scan."""
    Z = pd.concat([pd.read_csv(f) for f in glob.glob(str(OUT / 'pose2z_*.csv'))], ignore_index=True)
    if 'alpha' not in Z:
        Z['alpha'] = 0.0
    Z['alpha'] = Z.alpha.fillna(0.0)
    return Z


def merge_xz(R, Z):
    M = R.merge(Z[Z.alpha == 0][['x_b', 'x_c', 'theta', 'phi', 'chi2_z', 'n_z', 'shear_z']],
                on=['x_b', 'x_c', 'theta', 'phi'])
    M['chi2_xz'] = M.chi2 + M.chi2_z
    M.to_csv(OUT / 'pose2_xz.csv', index=False)
    return M


def fig_z(M):
    """D's z vs height against the best pose at each tilt azimuth."""
    import matplotlib.pyplot as plt
    from sept26_prelim_analysis.capsule_pose import H_EDGES
    fs.use()
    DZ = pd.read_csv(OUT / 'data_shear_z.csv')
    fig, ax = fs.figure(fs.HALF)
    hc = 0.5 * (H_EDGES[:-1] + H_EDGES[1:])
    for arm, col, mk in (('D', fs.INK, 'o'), ('B', fs.MUTED, 's')):
        d = DZ[(DZ.arm == arm) & DZ.h.between(-100, 40)]
        ax.errorbar(d.h, d.z_cross, d.z_err, fmt=mk, color=col, ms=4, lw=1, capsize=0,
                    label=f'{arm} data (25–75 ms)' + (', no drift field' if arm == 'B' else ''))
    best = M[M.theta > 0].sort_values('chi2_xz').groupby('phi').head(1)
    for ph, col in ((135, '#0072B2'), (180, fs.COPPER), (225, '#009E73')):
        b = best[best.phi == ph]
        if len(b):
            b = b.iloc[0]
            # pose2z's toy heights are already moved onto D's y crossing
            z = np.array([float(v) for v in b.shear_z.split()])
            y0 = DZ[DZ.arm == 'D'].y_cross.iloc[0]
            m = (hc >= -90) & (hc <= 30)
            ax.plot(hc[m], z[m], '-', lw=1.6, color=col,
                    label=f'φ {ph}°: θ {b.theta:g}°, χ²_z {b.chi2_z:.0f}')
    ax.set_xlabel('reconstructed height at the axis plane (mm, +y up)')
    ax.set_ylabel('z crossing (mm)')
    ax.set_ylim(-12, 10)
    ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
    fs.title(ax, 'D: the source moves toward +z with height', 'best x+z pose at each azimuth')
    plt.tight_layout()
    fs.save(fig, FIG / 'z_shear', data=DZ)


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
    Z = load_z()
    M = merge_xz(R, Z)
    FIG.mkdir(parents=True, exist_ok=True)
    fig_tilt(R)
    fig_degeneracy(R)
    fig_z(M)
    zc = ['x_b', 'x_c', 'theta', 'phi', 'gas', 'chi2_x', 'chi2_shape', 'chi2_z', 'chi2_xz', 'dy_c']
    perphi = M[M.theta > 0].sort_values('chi2_xz').groupby('phi').head(1).sort_values('phi')
    unt_xz = M[M.theta == 0].sort_values('chi2_xz').head(1)
    tbl_phi = pd.concat([perphi, unt_xz])[zc].round(1).to_html(index=False, border=0)
    A = Z[Z.alpha != 0]
    if len(A):
        # the same poses at alpha = 0, so the two columns compare like with like
        key = ['x_b', 'x_c', 'theta', 'phi']
        A0 = Z[Z.alpha == 0].merge(A[key].drop_duplicates(), on=key)
        both = pd.concat([A, A0])
        both, A0 = both[both.theta > 0], A0[A0.theta > 0]
        rows = []
        for ph, g in both.groupby('phi'):
            b = g.loc[g.chi2_z.idxmin()]
            rows.append(dict(phi=ph, chi2_z_alpha0=A0[A0.phi == ph].chi2_z.min(), chi2_z_best=b.chi2_z,
                             alpha_at_best=b.alpha))
        tbl_alpha = pd.DataFrame(rows).round(1).to_html(index=False, border=0)
        a_rng = f'{A.alpha.min():g} to {A.alpha.max():+g}°'
    else:
        tbl_alpha, a_rng = '<p>(not run)</p>', ''
    rot_txt = (f'An in-plane rotation of D ({a_rng}) does not change that ordering.' if a_rng
               else 'The D-rotation scan was run at 19.5 m only (capsule_pose_ear2/report.html).')
    b135 = perphi[perphi.phi == 135].chi2_xz.iloc[0]
    b180 = perphi[perphi.phi == 180].chi2_xz.iloc[0]
    b225 = perphi[perphi.phi == 225].chi2_xz.iloc[0]
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
(beam: <code>ear2_beam</code>, capsule at {L_M:g} m; data: is2_v2 cf, late window 20–75 ms, A and C).</p>
<div class="verdict"><p><b>Verdict.</b> With the real beam the old two "readings" are gone. Three things come out:
<b>the capsule top leans toward −x by ~7–15°</b> (untilted shape χ² {unt.min():.0f}–{unt.max():.0f} for 10 points, tilted
toward −x {til.min():.0f}–{np.percentile(til, 25):.0f}); <b>it sits ~8–12 mm lower than drawn</b> in every good pose; and
<b>sideways, beam and capsule trade off</b>: with the beam at the frame centre the capsule is ~15 mm off it (best χ² {b0:.1f},
³He sees {gas0:.2f} of the beam), and with the beam ~10 mm off the frame the capsule sits on or near it (best χ² {bb:.1f},
³He ~0.55). The literature puts the beam on the pipe axis to 1–2 mm.</p>
<p><b>Tilt azimuth (chamber D, z vs height): φ ≈ 135°.</b> In D the capture source moves toward +z by ~5 mm from
80 mm below the nose to its height, in every 5–15 ms slice from 25 to 75 ms and in every quarter of the runs. Adding that curve
to the fit, the best pose leans toward −x <i>and</i> +z (toward B and A; south-east in the hall): best x+z χ² {b135:.0f} at
φ 135°, against {b180:.0f} at 180° and {b225:.0f} at 225°. {rot_txt} The +z sign is pinned by the surveyed arm layout (A at +z) and the x-view in-plane sign measured on the target image
(COORDINATES.md), not by a self-consistency test.</p></div>
<h2>Best pose per (beam x, capsule x)</h2>{tbl}
<p class="cap">gas: fraction of the beam that reaches the ³He; dy_c: how far the capsule must move in height (mm) to put its
y crossing on the data's. χ²_x uses 1 mm per chamber crossing (chain systematic); χ²_shape uses the data errors and the toy's
Geant4-library spread per height bin.</p>
<img src="figures/tilt_shape.png" alt="shape chi2 vs tilt"><img src="figures/degeneracy.png" alt="chi2 vs capsule x for three beam positions">
<h2>Tilt azimuth from D</h2>
<p>D measures global z. Its late (25–75 ms) wall-coincident tracks from the production chain give the z crossing per bin of
reconstructed source height (<code>capsule_pose data_z</code>). Below 25 ms D's crossing is junk (noise in its x view), so the
window starts there. The toy has Geant4 track libraries only for A and C. The capsule and beam are close to axisymmetric, so D's
z curve for a pose is read off the A/C x curve of the same pose turned −90° about the beam (<code>capsule_pose pose2z</code>).
χ²_z: the curve's shape over the 5 populated bins (−60 to +20 mm), as for x.</p>
<img src="figures/z_shear.png" alt="D z crossing vs height with the best pose at phi 135, 180, 225">
<p class="cap">B (opposite D) is shown but not fitted: it has no drift field, so its y angles, and hence the heights it bins by, are
unreliable; its flat curve (best-fit scale on D's shape −0.25 ± 0.41) cannot test the shear.</p>
<h3>Best pose at each azimuth, x + z</h3>{tbl_phi}
<h3>Could an in-plane rotation of D fake it?</h3>
<p>The stand-in chamber is turned in its plane by α (beam −10, capsule −10/−15, θ 7–15°). Best χ²_z over α against α = 0:</p>
{tbl_alpha}
<h2>What this does not rule out</h2><ul>
<li>The shape χ² is limited by the Geant4 track library (~10 k tracks per arm); Δχ² of ~15 between beam positions is not decisive.</li>
<li>The z shear rests on one chamber (D) and on the A/C track response standing in for D's. B cannot confirm it. D's crossing also
depends on the hit height on the plane (+30 mm/m at fixed source height, against +6 to +10 for A and C): a D-specific
effect (y angle scale, alignment) is present, though the rotation scan says it does not flip the azimuth.</li>
<li>No pose reproduces D's whole curve. D rises steadily, ~4 mm from 60 mm below the nose to 20 mm above it. The best
φ 135° pose is flat below the nose and rises ~2 mm above it. The fit picks the azimuth from the sign of the rise (135° rises,
225° falls), not from a matched shape; χ²_z ≈ 13 for 5 points.</li>
<li>The φ grid is 45° wide; φ is ~135° ± 25°, not measured to better than that.</li>
<li>The late (20–75 ms) capsule tracks are ~100× more than the direct beam gives at those times (n_TOF phase space); the fit
assumes they are lit like the beam. The 4–12 ms window alone is lower purity.</li>
<li>Capsule distance: 19.5 m and 19.95 m are fitted separately (capsule_pose_ear2 and capsule_pose_ear2_L19p95); neither the big collimator nor a real-gun Geant4 library is in yet.</li>
</ul></body></html>'''
    (OUT / 'report.html').write_text(page)
    print('wrote', OUT / 'report.html')


if __name__ == '__main__':
    build()
