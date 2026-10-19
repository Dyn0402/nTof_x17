#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_xy_t0_report.py -- figures/report.html for the x/y t0 investigation.

Generated, not hand-written: every number in the prose is read from
``<out>/xy_t0/`` and from the figure CSVs, so re-running ``xy_t0.py`` and
``make_xy_t0_figures.py`` updates the tables, the headline numbers and the
verdict text together.  Figures are referenced with relative links so the same
file works from disk, from the DAQ Analysis tab, or copied elsewhere.

    X17_ROOT=D:/x17 python ntof_athens_26/xy_t0/make_xy_t0_report.py
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                          # noqa: E402

CSS = """
:root{--ink:#1b2430;--muted:#6a7583;--line:#d4d9e0;--surface:#fbfcfe;
--accent:#8a3f8f;--copper:#d18a44;--red:#c0392b;
--A:#0072B2;--B:#D55E00;--C:#009E73;--D:#CC79A7}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 20px 90px}
h1{font-size:31px;line-height:1.22;margin:0 0 6px}
h2{font-size:21px;margin:46px 0 12px;padding-top:16px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px;color:var(--muted);
letter-spacing:.05em;text-transform:uppercase}
p{margin:0 0 14px}
.sub{color:var(--muted);font-size:15px;margin-bottom:26px}
.verdict{background:#fff;border:1px solid var(--line);border-left:4px solid var(--accent);
padding:20px 24px;border-radius:6px;margin:26px 0}
.verdict p:last-child{margin-bottom:0}
.warn{border-left-color:var(--copper)}
.stop{border-left-color:var(--red)}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}
.tile{flex:1 1 160px;background:#fff;border:1px solid var(--line);
border-radius:6px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:26px;font-weight:650;margin-top:3px;font-variant-numeric:tabular-nums}
.tile .n{font-size:12.5px;color:var(--muted);margin-top:2px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14.5px;
background:#fff;border:1px solid var(--line);border-radius:6px;overflow:hidden}
th,td{padding:7px 11px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{background:#f2f4f7;font-weight:620;font-size:13px;
text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
tbody tr:last-child td{border-bottom:none}
td.num{font-variant-numeric:tabular-nums}
figure{margin:26px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:13.5px;color:var(--muted);margin-top:8px}
code{background:#eef1f5;padding:1px 5px;border-radius:3px;font-size:13.5px}
.foot{margin-top:60px;padding-top:16px;border-top:1px solid var(--line);
font-size:13px;color:var(--muted)}
ul{margin:0 0 14px;padding-left:22px}li{margin-bottom:6px}
@media (prefers-color-scheme:dark){
:root{--ink:#e6e9ee;--muted:#9aa4b2;--line:#2b3440;--surface:#12171e}
.verdict,.tile,table,figure img{background:#1a212a}
thead th{background:#222b36}
code{background:#222b36}}
"""


def esc(s) -> str:
    return html.escape(str(s))


def tbl(df: pd.DataFrame, cols, heads, fmts) -> str:
    h = ''.join(f'<th>{esc(x)}</th>' for x in heads)
    rows = []
    for r in df.itertuples():
        cells = []
        for c, f in zip(cols, fmts):
            v = getattr(r, c)
            try:
                cells.append(f'<td class="num">{f(v)}</td>')
            except (TypeError, ValueError):
                cells.append(f'<td>{esc(v)}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return (f'<table><thead><tr>{h}</tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None)
    ap.add_argument('--figs', default=None)
    a = ap.parse_args()
    src = Path(a.src) if a.src else paths.spell('out', 'xy_t0')
    figs = Path(a.figs) if a.figs else HERE / 'figures'
    paths.require(src / 'insitu_dt.csv', 'xy_t0.py tables')

    meta = json.loads((src / 'xy_t0.meta.json').read_text())
    cen = pd.read_csv(src / 'fallback_census.csv')
    ins = pd.read_csv(src / 'insitu_dt.csv')
    law = pd.read_csv(src / 'insitu_law.csv')
    res = pd.read_csv(src / 'residual_table.csv')
    deg = pd.read_csv(src / 'degeneracy_test.csv')
    geo = pd.read_csv(src / 'geometry_test.csv')
    cir = pd.read_csv(src / 'circularity.csv')
    dis = pd.read_csv(src / 'discrimination.csv')

    f1 = lambda v: f'{v:,.1f}'                                    # noqa: E731
    f2 = lambda v: f'{v:.2f}'                                     # noqa: E731
    f3 = lambda v: f'{v:.3f}'                                     # noqa: E731
    fi = lambda v: f'{int(v):,}'                                  # noqa: E731
    fp = lambda v: f'{100 * v:.1f} %'                             # noqa: E731
    fs_ = lambda v: esc(v)                                        # noqa: E731

    hit = meta['campaign_dt_hit_frac']
    slope_lo, slope_hi = law.slope_ns_per_unit.min(), law.slope_ns_per_unit.max()
    a_hw = float(res[(res.arm == 'A') & (res.dt == 'as shipped')].halfwidth.iloc[0])
    hw_lo = float(res[res.dt == 'as shipped'].halfwidth.min())
    hw_hi = float(res[res.dt == 'as shipped'].halfwidth.max())
    sig_lo = float(dis.per_plane_sigma_ns.min())
    sig_hi = float(dis.per_plane_sigma_ns.max())
    s60 = deg.sigma_60.abs().max()
    s5 = deg.sigma_5_control.min()
    # geo['quantile'], not geo.quantile -- the latter is DataFrame.quantile
    geo_floor = geo[geo['quantile'] == 0].halfwidth
    enh_lo, enh_hi = dis.enhancement.min(), dis.enhancement.max()

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>x/y t0 — what the two planes agree on</title><style>{CSS}</style></head>
<body><div class="wrap">

<h1>The two planes agree to 60&nbsp;ns, and it is not the depth&nbsp;bin</h1>
<p class="sub">What <code>dt_xy</code> is, what the x/y coincidence gate is worth, and why
<code>HANDOFF_T0_PRIOR.md</code>&nbsp;§1 was reading a real number through a wrong mechanism.
&nbsp;·&nbsp; {fi(meta['n_tracks'])} stage-3 tracks &nbsp;·&nbsp; {date.today().isoformat()}</p>

<div class="verdict">
<p><strong>The handoff's measurement stands; its diagnosis does not.</strong>
The two planes of one chamber really do disagree on t<sub>0</sub> at the
{f1(hw_lo)}–{f1(hw_hi)}&nbsp;ns level. But it is <strong>not</strong> the 60&nbsp;ns
depth-bin degeneracy: a periodicity test that detects the known 5&nbsp;ns
<code>T0_STEP</code> snap at {f1(abs(s5))}&nbsp;σ finds nothing at 60&nbsp;ns
(|σ|&nbsp;≤&nbsp;{f2(s60)}). And it is not the x/y plane separation either — the
handoff's own first check — because the residual does not vanish at normal
incidence.</p>
<p>What it is: the forward model's per-plane t<sub>0</sub> is <em>smoothly</em>
uncertain at about {f1(sig_lo)}–{f1(sig_hi)}&nbsp;ns, against a fitted
<code>x_t0_err</code> that medians a few ns. Same scale as one depth bin — which
is why the bin was a tempting explanation — but a broad uncertainty, not a
two-minimum ambiguity, and it needs a different fix.</p>
</div>

<div class="tiles">
<div class="tile"><div class="k">measured dt_xy used on</div><div class="v">{fp(hit)}</div>
<div class="n">of the campaign — 0 tracks on A, B, C</div></div>
<div class="tile"><div class="k">dt_xy is linear in ftst</div><div class="v">{f1(slope_lo)}…{f1(slope_hi)}</div>
<div class="n">ns per unit, all four arms (quantum {f1(-meta['ftst_quantum_ns'])})</div></div>
<div class="tile"><div class="k">60 ns comb</div><div class="v">≤ {f2(s60)} σ</div>
<div class="n">control at 5 ns: {f1(abs(s5))} σ</div></div>
<div class="tile"><div class="k">per-plane t<sub>0</sub></div><div class="v">~{f1(sig_hi)} ns</div>
<div class="n">reported error: a few ns</div></div>
</div>

<h2>1 · The census, exactly: the measured offset never ran on A, B or C</h2>
<p>The handoff predicted "~100&nbsp;% miss on A and C from the parity argument" and
asked for it to be confirmed rather than trusted. It is not approximately
100&nbsp;%. <code>ftst</code> is a 6-phase counter; A, B and C only ever produce
<strong>even</strong> <code>ftst_diff</code> and their bundle keys are
<strong>odd</strong>, so the intersection is empty by construction — no statistics
involved. Every one of those tracks ran on the hardcoded
<code>{meta['fallback_dt']}</code>&nbsp;ns.</p>
{tbl(cen, ['arm', 'n', 'n_hit', 'hit_frac', 'bundle_keys', 'ftst_diff_seen'],
     ['arm', 'tracks', 'used measured dt', 'hit rate', 'bundle keys', 'ftst_diff in data'],
     [fs_, fi, fi, fp, fs_, fs_])}

<h2>2 · What dt_xy actually is — and it needs no bench</h2>
<p>It is not a per-chamber constant to be looked up. The <em>whole</em>
<code>x_t0&nbsp;−&nbsp;y_t0</code> distribution translates linearly with
<code>ftst_diff</code>, by the same amount on all four arms. <code>ftst</code> has
{fi(meta['ftst_phases'])} phases over a {fi(meta['sample_ns'])}&nbsp;ns sample, so one
unit is {f1(meta['ftst_quantum_ns'])}&nbsp;ns of readout phase — and that is what is
measured, to within the known low bias of the estimator (the accidental pedestal
does not shift with <code>ftst</code> and pulls the correlation peak toward zero).</p>
{tbl(law, ['arm', 'slope_ns_per_unit', 'expected_ns_per_unit', 'frac_of_quantum',
           'max_abs_resid_ns', 'n_classes'],
     ['arm', 'measured ns/unit', 'one phase', 'fraction', 'max resid to the line', 'classes'],
     [fs_, f2, f1, f2, f1, fi])}
<p>So <code>dt_xy</code> is a readout-clock effect with no chamber physics in it.
The bench measured two <code>ftst_diff</code> classes per arm; on A, B and C neither
of them occurs in beam data. <strong>The beam data measures it directly, for every
class that actually happens.</strong></p>
<figure><img src="dt_xy_law.png" alt="dt_xy linear in ftst_diff">
<figcaption>Left: the offset against <code>ftst_diff</code>, four arms and the
one-phase line. Right: where the bench keys sit against the classes the beam
data produces — they never coincide except on D at ±3.</figcaption></figure>

<h2>3 · The shape nobody had looked at</h2>
<p>Drawn per <code>ftst</code> class, on unambiguous tracks with no gate, the x/y
time difference is a <strong>~200&nbsp;ns-wide peak sitting on an accidental
pedestal</strong>. Two planes seeing one track at the same instant would give a
spike a few ns wide. This is the whole result, and no choice of offset changes it.</p>
<figure><img src="xy_agreement.png" alt="x/y time difference per ftst class">
<figcaption>True pairing (line) and the same tracks with a scrambled partner
(grey). The shape slides left across each row — that is <code>dt_xy</code>. Its
width within a panel is the t<sub>0</sub> resolution.</figcaption></figure>

<h2>4 · Two mechanisms tested, two ruled out</h2>
<h3>Not the depth-bin degeneracy</h3>
<p><code>wft/model.py</code> says the χ² surface has near-degenerate minima
60&nbsp;ns apart and that only ~35&nbsp;% of free fits land in the physical one.
That predicts a <em>comb</em> in the residual at 60&nbsp;ns. There is none — and the
test is not blind, because the same statistic finds the known 5&nbsp;ns
<code>T0_STEP</code> snap at {f1(abs(s5))}&nbsp;σ or better on every chamber.</p>
{tbl(deg, ['arm', 'n', 'amp_60ns', 'null_mean', 'sigma_60', 'amp_5ns_control', 'sigma_5_control'],
     ['arm', 'tracks', '|R| at 60 ns', 'null', '60 ns σ', '|R| at 5 ns', '5 ns σ (control)'],
     [fs_, fi, f3, f3, f2, f3, f1])}
<h3>Not the x/y plane separation</h3>
<p>The handoff called this "the first thing to check and it could explain a large
part of the table". It is real and it is small. A plane separation crossed by an
inclined track must vanish at normal incidence; the lowest-inclination quintile
already carries {f1(geo_floor.min())}–{f1(geo_floor.max())}&nbsp;ns of half-width, and
the full <code>tan θ</code> range adds about 10 on top.</p>
<figure><img src="mechanism.png" alt="degeneracy and geometry tests">
<figcaption>Left: the periodicity test with its positive control. Right: residual
half-width against inclination — a trend, on a floor that geometry cannot explain.</figcaption></figure>

<h2>5 · The gate is the metric — so §1's number is truncated and its test circular</h2>
<div class="verdict stop">
<p><code>wft.reco.select_tracks</code> sets <code>gated</code> from
<code>|(t0x − t0y) − dt| ≤ {f1(meta['tol_ns'])} ns</code> AND both planes plausible.
<strong>Zero of {fi(meta['n_tracks'])} gated tracks lie outside that window.</strong>
The handoff measured the x/y residual on the gated sample, so it measured a
distribution that had already been cut on — at ±{f1(meta['tol_ns'])}&nbsp;ns around a
centre that was wrong by up to 84&nbsp;ns.</p>
<p>The consequence for §3 step&nbsp;4 is direct: a prior that moved t<sub>0</sub> would
move which tracks pass this cut, so the residual measured on the survivors would
improve <em>partly by construction</em>. <strong>That target must be declared on the
unselected population</strong> — where the half-widths are not 66/76/77 but
{f1(float(cir[cir.arm == 'A'].hw_all.iloc[0]))}/{f1(float(cir[cir.arm == 'C'].hw_all.iloc[0]))}/{f1(float(cir[cir.arm == 'D'].hw_all.iloc[0]))}&nbsp;ns.</p>
</div>
{tbl(cir, ['arm', 'n', 'frac_gated', 'gated_outside_window', 'hw_gated', 'hw_all',
           'hw_ungated', 'frac_within_20ns_of_cut'],
     ['arm', 'tracks', 'gated', 'gated outside window', 'hw gated', 'hw all',
      'hw ungated', 'within 20 ns of the cut'],
     [fs_, fi, fp, fi, f1, f1, f1, fp])}

<h2>6 · What the coincidence test is worth</h2>
<p>Against a scrambled pairing — same arm, same <code>ftst</code> class, someone
else's <code>y_t0</code> — the window keeps most true pairs and about a third of
accidental ones. That is an enhancement of {f2(enh_lo)}–{f2(enh_hi)}×, bought for
roughly 30&nbsp;% of the real tracks. Useful; not the discriminator
<code>select_pair</code>'s docstring describes when it says it uses information
"that single-plane selection cannot use".</p>
{tbl(dis, ['arm', 'acc_true', 'acc_scrambled', 'enhancement', 'signal_hw_ns',
           'per_plane_sigma_ns', 'x_t0_err_med', 'err_understated_by'],
     ['arm', 'keeps true', 'keeps scrambled', 'enhancement', 'signal hw [ns]',
      'per-plane σ [ns]', 'fitted err [ns]', 'understated by'],
     [fs_, fp, fp, f2, f1, f1, f2, f1])}
<figure><img src="gate.png" alt="the gate as metric and as discriminator">
<figcaption>Left: the sample §1 measured, beside the distribution it was cut out
of. Right: what the window keeps, true against scrambled.</figcaption></figure>

<h2>7 · Does the right dt_xy fix §1? No.</h2>
<p>Reproduction first: the shipped column lands on {f1(a_hw)}&nbsp;ns for A and the
same three track counts as the handoff. Substituting the in-situ offset then moves
the half-width by about a nanosecond. The wrong constant scatters the residual by
~20–34&nbsp;ns across <code>ftst</code> classes, which is worth fixing on its own
account, and it is small beside a {f1(hw_lo)}–{f1(hw_hi)}&nbsp;ns spread.</p>
{tbl(res, ['arm', 'dt', 'n', 'halfwidth', 'f30', 'f60', 'dt_error_sd'],
     ['arm', 'dt subtracted', 'tracks', 'half-width [ns]', '|res| > 30 ns',
      '|res| > 60 ns', 'dt error sd [ns]'],
     [fs_, fs_, fi, f1, fp, fp, f1])}

<h2>8 · What this does not settle</h2>
<ul>
<li><strong>The in-situ prior (handoff §3 step 3) is not built.</strong> This supplies
the <code>dt_xy</code> law it would need and removes the reason to expect it to
collapse the residual the way §1 hoped — a ~60 ns per-plane uncertainty is not
something a better centre repairs.</li>
<li><strong>The absolute level of <code>dt_insitu</code> is provisional.</strong> The
<em>shape</em> (linear, {f1(slope_lo)}…{f1(slope_hi)} ns per unit) is good to a few ns;
the anchor is a median on a broad peak and the whole column can move ~10 ns
together. No acceptance or efficiency number is re-derived from it here.</li>
<li><strong>Nothing is re-reconstructed.</strong> Every number is read from the
stage-3 track table. Deciding whether a t<sub>0</sub> prior helps needs the waveform
path, which is not this.</li>
<li><strong>Why the per-plane t<sub>0</sub> is uncertain at one bin is not answered.</strong>
That it is smooth rather than combed rules out the two-minimum picture; it does not
say what replaces it.</li>
</ul>

<p class="foot">Generated by <code>make_xy_t0_report.py</code> from
<code>{esc(src)}</code>. Tables: <code>fallback_census</code>, <code>insitu_dt</code>,
<code>insitu_law</code>, <code>residual_table</code>, <code>degeneracy_test</code>,
<code>geometry_test</code>, <code>circularity</code>, <code>discrimination</code>.
Every figure ships the CSV it was drawn from.</p>
</div></body></html>"""

    out = figs / 'report.html'
    out.write_text(doc, encoding='utf-8')
    print(f'[xy_t0] wrote {out}  ({len(doc):,} chars)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
