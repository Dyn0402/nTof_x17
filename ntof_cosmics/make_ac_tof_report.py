#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_ac_tof_report.py -- report.html for the arm-to-arm time resolution (ac_tof_beamoff.py, ac_tof.py), built from
their CSVs.   .venv/bin/python ntof_cosmics/make_ac_tof_report.py   -> $WORK_DATA/nTof_x17/ac_tof/report.html"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import pandas as pd

OUT = Path(os.environ.get('WORK_DATA', '/media/dylan/data/x17')) / 'nTof_x17' / 'ac_tof'


def table(df, cols, fmt):
    h = '<table><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cols) + '</tr>'
    for _, r in df.iterrows():
        h += '<tr>' + ''.join(f'<td>{fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else ""}</td>' for c in cols) + '</tr>'
    return h + '</table>'


def main() -> int:
    S = pd.read_csv(OUT / 'beamoff_summary.csv')
    meta = json.loads((OUT / 'beamoff_meta.json').read_text())
    g = lambda k: S[S['sample'] == k].iloc[0]
    al = g('walls, all')
    tag = S[S['sample'].str.startswith('direction-tagged')]
    pl = S[S['sample'].str.startswith('plastics')]
    cols = ['sample', 'n', 'sigma_AC_ns', 'err_ns', 'per_arm_ns', 'note']
    fmt = {'n': '{:.0f}', 'sigma_AC_ns': '{:.3f}', 'err_ns': '{:.3f}', 'per_arm_ns': '{:.3f}'}
    beam = OUT / 'ac_tof_summary.csv'
    btab = ''
    if beam.exists():
        B = pd.read_csv(beam)
        btab = table(B, ['sample', 'n', 's68_ns', 'core_ns', 'per_arm_core'],
                     {'n': '{:.0f}', 's68_ns': '{:.2f}', 'core_ns': '{:.2f}', 'per_arm_core': '{:.2f}'})
    tcore = ', '.join(f'{r.per_arm_ns:.2f}' for r in tag.itertuples())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>A–C time resolution</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 16px;line-height:1.45}}
table{{border-collapse:collapse;margin:1em 0;font-size:0.9em}}td,th{{border:1px solid #ccc;padding:3px 8px;text-align:left}}
.v{{background:#eef6ee;padding:0.8em 1em;border-left:4px solid #393}}img{{max-width:100%}}</style></head><body>
<h1>Arm-to-arm time resolution of the SiPM walls, from cosmic muons (A–C)</h1>
<p class="v"><b>Verdict.</b> With no beam, the SiPM walls of arms A and C give <b>σ_t ≈ 0.52–0.64 ns per arm</b>
(σ(t_A − t_C) = {al.sigma_AC_ns:.2f} ± {al.err_ns:.2f} ns from the two-peak fit of {int(al.n)} muons; direction-tagged
peak cores {tcore} ns per arm), as read out at n_TOF (official pulse fit, 1 GS/s). The plastics give
{pl.iloc[0].per_arm_ns:.2f} ns per arm (one PMT per bar, {int(pl.iloc[0].n)} muons). The muon's own time of flight
between the walls (±2.3–2.5 ns) is resolved: the distribution has two clean peaks. During beam the same muons give
~6 ns (the pulse fit's floor under beam), so only the beam-off data measure the detector.</p>
<h2>1. What was measured</h2>
<p>Muons crossing the SiPM walls of A and C, with both ends of a wall group firing in each arm, unsaturated, within
±{meta['win_ns']:.0f} ns of each other, while n_TOF recorded with no protons (run_149, n_TOF {meta['runs'][0]}–{meta['runs'][-1]}).
Wall time = mean of the two ends (cancels the light path along the vertical bars).</p>
<p>Only 17 of these muons also have an A–C Micromegas line, so each muon's own TOF cannot be subtracted. Instead
t_A − t_C is fitted as a mixture of ±T (T = L/c from the geometry of the beam-run A–C muons that do have a line:
median {meta['T_median']:.2f} ns), a Gaussian of width σ, per-group offsets, a time walk in log(amplitude) per arm,
the direction probability from the bars' top−bottom differences, and a flat accidental fraction.</p>
<h2>2. Results</h2>{table(S, cols, fmt)}
<p>Per arm = σ(A−C)/√2 (arms assumed equal). The fit's σ includes the spread of T and any template mismatch; the
direction-tagged cores need no template. The plain s68 with the TOF left in ({g('s68 of t_A - t_C after the fitted offsets + walk (muon TOF NOT removed)').sigma_AC_ns:.2f} ns)
is what the first look quoted as an upper bound.</p>
<p><img src="figures/beamoff_walls.png" alt="t_A - t_C after offsets and walk, and against the bar height difference"></p>
<p>Left: t_A − t_C after the fitted offsets and walk; the two peaks are the two directions of the muon. Right: the
bars' top−bottom difference sorts the muons by direction.</p>
<h2>3. Checks</h2><ul>
<li>Toy closure: the fit returns its input ({g('toy closure (5 toys of the "walls, all" fit)').note}, refit
{g('toy closure (5 toys of the "walls, all" fit)').sigma_AC_ns:.3f} ns).</li>
<li>Stable in time (two halves of the n_TOF runs), with amplitude (both above median), and with the T template scaled
±10 % (0.92–0.96 ns).</li>
<li>Peak separation 5.0 ns, i.e. T ≈ 2.5 ns, against the template's 2.3: the beam-off muons are somewhat steeper.</li></ul>
<h2>4. Beam runs (ac_tof.py): why they do not measure σ_t</h2>
<p>A–C muons at 20–80 ms after the flash with a Micromegas line (the TOF subtracted per muon) give t_A − t_C cores of
~6 ns whatever is corrected. Injected test pulses show the n_TOF pulse fit's wall timing degrades to 2.5–4.5 ns per hit
under beam (0.4 ns without), which accounts for it.</p>{btab}
<h2>5. What this does not rule out</h2><ul>
<li>It is the n_TOF readout. The LNL plan (V1742 at 5 GS/s for the walls, a CFD digitiser for the PMTs) could do
better or worse; this is the as-built detector with this electronics, not a prediction for LNL.</li>
<li>Arms assumed equal; A and C only. The per-arm split cannot be separated with two arms.</li>
<li>The plastic number rests on 477 muons and a crude position correction (wall group, bar height).</li>
<li>Offsets were fitted on the same sample (8 of 2,579 degrees of freedom).</li></ul>
</body></html>"""
    (OUT / 'report.html').write_text(doc)
    print('->', OUT / 'report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
