#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py -- report.html for the n_TOF pass-1 tracking-improvement study (rerun/ntof/improve/).

Reads the products of line_residuals.py, bow_check.py and angle_map.py in <out> (track_quality.out_dir()) and writes
<out>/report.html with figures in <out>/figures/ (each with its CSV).  Re-run after any of those steps.

    python -m rerun.ntof.improve.make_report
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sept26_prelim_analysis import figstyle as fs  # noqa: E402
from rerun.ntof.improve.track_quality import out_dir, load_tracks, source_x  # noqa: E402
from rerun.ntof.improve.angle_map import correct  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def fig_maps(out: Path, maps: dict) -> str:
    fs.use()
    fig, axs = plt.subplots(2, 4, figsize=(9.6, 5.0), constrained_layout=True)
    rows = []
    c = np.asarray(next(iter(maps.values()))['centres'])
    for col, key in enumerate(('A_x', 'A_y', 'C_x', 'C_y')):
        m = maps[key]
        for r, (q, lim, lab) in enumerate((('a', 0.05, 'tilt a'), ('s', 0.15, 'scale s'))):
            Z = np.asarray(m[q])
            N = np.asarray(m['n'])
            Zp = np.where(N >= 80, Z, np.nan)
            ax = axs[r, col]
            im = ax.imshow(Zp.T, origin='lower', extent=[-200, 200, -200, 200], cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.set_title(f'{key.replace("_", " view ")}: {lab}', fontsize=9.5)
            ax.set_xlabel('x_local [mm]')
            if col == 0:
                ax.set_ylabel('y_local [mm]')
            if col == 3:
                fig.colorbar(im, ax=axs[r, :], shrink=0.85, label=lab)
            for i, xc in enumerate(c):
                for k, yc in enumerate(c):
                    rows.append(dict(map=key, term=q, x=xc, y=yc, value=Z[i, k], n=N[i, k]))
    fs.save(fig, out / 'figures' / 'angle_maps.png', data=pd.DataFrame(rows))
    plt.close(fig)
    return 'figures/angle_maps.png'


def fig_heldout(out: Path, H: pd.DataFrame) -> str:
    fs.use()
    g = H.groupby(['arm', 'view'])[['sig_clean_before', 'sig_clean_after', 'sig_edge_before', 'sig_edge_after']].mean()
    fig, ax = plt.subplots(figsize=fs.FIG, constrained_layout=True)
    lab = [f'{a} {v}' for a, v in g.index]
    x = np.arange(len(lab))
    for k, (col, name, off, alpha) in enumerate((('sig_clean_before', 'all cells, pass 1', -0.3, 0.45),
                                                 ('sig_clean_after', 'all cells, with map', -0.1, 1.0),
                                                 ('sig_edge_before', 'edge cells, pass 1', 0.1, 0.45),
                                                 ('sig_edge_after', 'edge cells, with map', 0.3, 1.0))):
        color = fs.ACCENT if 'clean' in col else fs.COPPER
        ax.bar(x + off, g[col] * 1000, width=0.2, color=color, alpha=alpha, label=name)
    ax.set_xticks(x, lab)
    ax.set_ylabel('single-view resolution σ (MAD) [× 10⁻³ tan]')
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    fs.title(ax, 'Held-out cosmic lines: resolution with and without the in-situ map',
             'trained on one run parity, tested on the other (mean of both splits); edge = |x| or |y| > 125 mm')
    fs.save(fig, out / 'figures' / 'heldout_sigma.png', data=g.reset_index())
    plt.close(fig)
    return 'figures/heldout_sigma.png'


def fig_beam(out: Path, maps: dict) -> str:
    d = load_tracks()
    xs = source_x(d)
    d = d[(d.ms >= 20) & (d.tanx.abs() >= 0.1)]
    rng = np.random.default_rng(7)
    edges = np.linspace(-60, 60, 121)
    mid = 0.5 * (edges[1:] + edges[:-1])
    fs.use()
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE, constrained_layout=True, sharey=True)
    rows = []
    for ax, (arm, g) in zip(axs, d.groupby('arm')):
        sx = (g.d_x / g.d_z).to_numpy()
        pz, pos = g.p0_z.to_numpy(), g.pos.to_numpy()
        for lab, sl, ls in (('pass 1', sx, '--'),
                            ('with cosmic map', correct(sx, g.x_local.to_numpy(), g.y_local.to_numpy(),
                                                        maps[f'{arm}_x']), '-')):
            disp = -pz * sl
            h = np.histogram(pos + disp - xs[arm], edges)[0]
            h0 = np.histogram(pos + disp[rng.permutation(len(disp))] - xs[arm], edges)[0]
            ex = (h - h0) / len(disp)
            ax.plot(mid, ex * 100, ls, color=fs.DET_COLOR[arm], label=lab, lw=1.4 if ls == '-' else 1.0)
            rows += [dict(arm=arm, version=lab, miss_mm=m, excess_pct=e * 100) for m, e in zip(mid, ex)]
        ax.set_xlabel('signed miss at the source plane [mm]')
        ax.set_title(f'chamber {arm}, ≥ 20 ms, |tan x| ≥ 0.1', fontsize=9.5)
        ax.legend(frameon=False, fontsize=8.5)
    axs[0].set_ylabel('tracks minus shuffled null [% per mm]')
    fs.save(fig, out / 'figures' / 'beam_pointing.png', data=pd.DataFrame(rows))
    plt.close(fig)
    return 'figures/beam_pointing.png'


def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    h = ['<table><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns) + '</tr>']
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            cells.append(f'<td>{(f.format(v) if f and isinstance(v, (int, float, np.floating)) else html.escape(str(v)))}</td>')
        h.append('<tr>' + ''.join(cells) + '</tr>')
    return '\n'.join(h) + '</table>'


def main() -> int:
    out = out_dir()
    (out / 'figures').mkdir(exist_ok=True)
    maps = json.load(open(out / 'angle_map.json'))['maps']
    H = pd.read_csv(out / 'angle_map_heldout.csv')
    Bm = pd.read_csv(out / 'angle_map_beam.csv')
    Bw = pd.read_csv(out / 'bow_check.csv')
    L = pd.read_parquet(out / 'lines.parquet', columns=['run'])
    f_maps, f_held, f_beam = fig_maps(out, maps), fig_heldout(out, H), fig_beam(out, maps)

    hm = H.groupby(['arm', 'view'])[['sig_clean_before', 'sig_clean_after', 'sig_edge_before', 'sig_edge_after']] \
        .mean().reset_index()
    hm['gain'] = 1 - hm.sig_clean_after / hm.sig_clean_before
    bw = Bw[Bw.variant.isin(['on', 'off'])].pivot_table(index=['arm', 'view'], columns='variant',
                                                         values=['sigma', 'cell_rms']).reset_index()
    bw.columns = ['arm', 'view', 'cell rms off', 'cell rms on', 'σ off', 'σ on']
    bw = bw[['arm', 'view', 'σ on', 'σ off', 'cell rms on', 'cell rms off']]
    bm = Bm[['arm', 'map', 'exc15', 'w68', 'tail', 'focus_s', 'edge_exc15', 'edge_w68']].rename(columns={
        'map': 'angles', 'exc15': 'pointing excess < 15 mm', 'w68': 'core half-width 68 % [mm]',
        'tail': 'tail 15–45 mm share', 'focus_s': 'sharpest focus s', 'edge_exc15': 'edge |x|>125: excess',
        'edge_w68': 'edge: core [mm]'})
    bm['angles'] = bm['angles'].map({'before': 'pass 1', 'after': 'with cosmic map'})
    sA = Bm.set_index(['arm', 'map']).focus_s
    asym_b = sA[('A', 'before')] - sA[('C', 'before')]
    asym_a = sA[('A', 'after')] - sA[('C', 'after')]
    g_lo, g_hi = hm.gain.min() * 100, hm.gain.max() * 100

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>n_TOF pass-1 tracking: what to improve</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;color:#1b2430;line-height:1.45}}
table{{border-collapse:collapse;margin:0.6em 0 1.2em;font-size:0.9em}}td,th{{border:1px solid #c9ced6;padding:3px 8px;text-align:right}}
th{{background:#f1f3f6}}img{{max-width:100%}}.verdict{{background:#f6f0f7;border-left:4px solid #8a3f8f;padding:0.6em 1em}}
figcaption{{font-size:0.9em;color:#4a5563;margin-bottom:1.5em}}code{{background:#f1f3f6;padding:0 3px}}</style></head><body>
<h1>n_TOF pass 1: where the tracking can still improve</h1>
<p>Study code <code>rerun/ntof/improve/</code>; inputs: the pass-1 analysis view (v(t) angles, gated = true |tan| &lt; 0.5).
Cosmic truth: {len(L):,} A–C lines from {L.run.nunique()} beam-off runs (one gated track per chamber, lines within 60 mm).
Beam check: the late (≥ 20 ms) capsule tracks of the production runs.</p>

<div class="verdict"><b>Verdict.</b> The largest fixable error in the A and C angles is a <b>position-dependent angle
distortion</b> of 0.008–0.016 rms in tan (up to 0.04–0.05 at the edges), stable in time. An in-situ correction map fitted
on the cosmic lines improves the held-out single-view resolution by {g_lo:.0f}–{g_hi:.0f} %. On the beam capsule tracks,
which are independent of the map, the map also raises the capsule-pointing excess, narrows the core and removes the
A−C focus asymmetry (A − C: {asym_b:+.3f} → {asym_a:+.3f}). The bench bow carried into pass 1 helps C and is neutral on A, so it stays.
The 13 % angle tail on the cosmic line is mostly impure truth (showers, two-particle events), not failed fits.</div>

<h2>1. The angle tails are mostly in the truth sample</h2>
<p>13–14 % of views miss the joined A–C line by more than 0.15 in tan. The misses are shared: A and C are both wrong
3× more often than chance (x: 5.5 % against 1.9 %). When the other chamber and this chamber's other view agree with the
line within 0.05, a view's own tail is 1.0–2.1 %. The flagged views carry about half the charge, a fifth of the Δχ²,
and 5–9 competing-cluster strips on A: extra particles in the chamber. So this is not a fit failure to chase.</p>

<h2>2. The bench bow block: keep it</h2>
<p>The plan (NTOF_PASS_PLAN §5) asked for bow-on against bow-off on the A–C line. Bow-off is the constant mapping
w/(kw v) from the stored fit speed; "cell rms" is the rms of the median residual over 50 mm cells.</p>
{table(bw, {c: '{:.4f}' for c in bw.columns[2:]})}
<p>On C the bow lowers σ by 8–11 % and the local bias by a third. On A it is neutral (x slightly worse in the cell rms,
y the same).</p>

<h2>3. A position-dependent angle distortion remains</h2>
<figure><img src="{f_maps}"><figcaption><b>Figure 1.</b> The in-situ map: tilt a (top) and scale s (bottom) of the
residual r = a + s·j per 50 mm cell, for A and C and each view. Cells with fewer than 80 clean lines are blank (they
take the chamber-wide fit). C has a scale gradient along y in both views (about +0.1 at y ≈ −150 mm to −0.1…−0.2 at
+175 mm). A has edge tilts in x of −0.03 and +0.04 at x ≈ ∓175 mm.</figcaption></figure>
<p>The pattern is real. Split by run parity, the two halves agree cell by cell with correlation 0.94–0.97. Split
early/late in the campaign, the tilt terms agree at 0.70–0.97 and C's scale at 0.84–0.92. The cell statistical error is
~0.003.</p>

<h2>4. The correction, held out and on the beam</h2>
{table(hm.rename(columns={'sig_clean_before': 'σ pass 1', 'sig_clean_after': 'σ map', 'sig_edge_before': 'σ edge pass 1',
                          'sig_edge_after': 'σ edge map', 'gain': 'gain'}),
       {'σ pass 1': '{:.4f}', 'σ map': '{:.4f}', 'σ edge pass 1': '{:.4f}', 'σ edge map': '{:.4f}', 'gain': '{:.1%}'})}
<figure><img src="{f_held}"><figcaption><b>Figure 2.</b> Single-view resolution on held-out clean cosmic lines, before and
after the map, all cells and edge cells. Each split trains on one run parity and tests on the other.</figcaption></figure>
{table(bm, {c: '{:.3f}' for c in bm.columns[2:]})}
<figure><img src="{f_beam}"><figcaption><b>Figure 3.</b> Late capsule tracks, x view: the signed miss at the source minus
a shuffled-angle null, with and without the cosmic map. The map is trained only on beam-off cosmics.</figcaption></figure>
<p>The A−C antisymmetric focus residual of pass 1 (4.6 % in focus_split) had been read as geometry (a distance or an
axis offset). The cosmic map removes it without knowing where the source is, so most of it is the angle distortion above.
D's source z crossing (−5.55 mm) is therefore not confirmed by the A/C focus. It may carry D's own distortion, which
this study cannot measure: the B–D cosmic lines do not close (D's line misses B's point by a median +117 mm in both
coordinates), so B cannot yet serve as D's truth.</p>

<h2>What this does not rule out</h2>
<ul><li>The cosmic truth uses A and C positions in the global frame. A position error of 1 mm moves j by 0.002, well below
the map, but an error in a chamber's global placement that varies over the plate would enter the map as an angle term.</li>
<li>The map is fitted on cosmic angles (|j| &lt; 0.45, mostly steep in y). Beam electrons at the edges of the angle
range rely on the per-cell linear term.</li>
<li>Edge cells with few lines fall back to the chamber-wide fit; the map clamps at the outer cell centres (±175 mm).</li>
<li>D and B have no map; D's angles keep whatever distortion they have.</li>
<li>Nothing here changes the same-chamber data/Geant4 factor ~10, which is the source model, not the angles.</li></ul>

<h2>Next steps (existing data only)</h2>
<ol><li>Decide whether the map becomes a table-level correction in the analysis view (like the v(t) κ step), with
<code>&lt;col&gt;_map</code> twins, and rerun the angle-sensitive downstream (capsule focus and pose, same-chamber and two-arm
opening angles, vertex imaging).</li>
<li>Fold the map into a bundle-level angle correction for the bench-bow terms on A (whose bench bow is neutral).</li>
<li>Find why the B–D cosmic lines do not close; with them, D gets its own map and its source z can be re-measured.</li></ol>
</body></html>"""
    (out / 'report.html').write_text(page)
    print('wrote', out / 'report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
