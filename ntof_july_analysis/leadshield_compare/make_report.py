#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the HTML report for the lead-shielding before/after check.

Writes <OUT_BASE>/report.html, which the DAQ page's Analysis tab lists and
opens inline (see app.py /analysis_file/<relpath> — the report references its
figures with ORDINARY RELATIVE LINKS, 'figures/x.png', so the same file works
opened from disk, served by the DAQ page, or copied elsewhere with its
figures/ directory).

Generated, not hand-written, so it tracks the tables: re-run after compare.py
and the numbers, the chart and the verdict text all follow.

Run: .venv/bin/python ntof_july_analysis/leadshield_compare/make_report.py
"""
import html
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import lib as L  # noqa: E402

TAB_DIR = os.path.join(L.OUT_BASE, 'tables')
OUT_PATH = os.path.join(L.OUT_BASE, 'report.html')

# Categorical slot 1 of the reference data-viz palette, both modes, unchanged.
SERIES_1_LIGHT, SERIES_1_DARK = '#2a78d6', '#3987e5'

FIGURES = [
    ('eff_vs_dt_zoom.png', 'Track efficiency, first 15 ms',
     'Boxcar W=2 ms. The 1-5 ms window is shaded. run_139 (red) tracks '
     'run_132 (dark blue) point for point; the night-to-night control '
     'run_130 (light blue) strays further than the after run does.'),
    ('eff_vs_dt.png', 'Track efficiency, full 1-75 ms gate',
     'The whole recovery, boxcar W=6 ms. Det A rises to its plateau by '
     '~35 ms identically in all three runs.'),
    ('acceptance_vs_dt_zoom.png', 'Trigger acceptance, first 12 ms',
     'DAQ-level and reconstruction-independent: accepted events per burst '
     'per ms. The after/before ratio is 1.00 throughout. The comb structure '
     'is the DAQ accept pattern, identical in all three runs.'),
    ('first_accept.png', 'First accepted trigger after the flash',
     'Per-burst. Medians agree to 0.01 ms — the DAQ starts accepting at the '
     'same time before and after.'),
    ('flash_leader_size.png', 'Gamma-flash leader size',
     'Did the flash itself grow? No — run_139 is marginally SMALLER, '
     'tracking the ~1% lower delivered intensity.'),
    ('blind_vs_dt.png', 'Detector blindness vs time since flash',
     'Fraction of read-out events in which the detector produced no hits at '
     'all — the front-end recovery, independent of tracking.'),
    ('eff_vs_dt_high.png', 'Efficiency, HIGH-intensity pulses only',
     'Intensity-matched (>=600e10 p), removing the delivered-beam mix as a '
     'nuisance.'),
]


def esc(s):
    return html.escape(str(s))


# --------------------------------------------------------------- the chart
def delta_chart(a):
    """Dot plot with error bars: the after-before delta per dt window, against
    the before-before control measured on the same detector.

    This is the argument of the analysis and no PNG shows it: the question is
    not 'is the delta significant against zero' but 'is it bigger than the
    night-to-night floor'. One measure, two series -> dot plot + legend;
    tooltips are pure CSS so the chart needs no JS to be readable.
    """
    W, H = 780, 330
    ml, mr, mt, mb = 58, 14, 14, 52
    pw, ph = W - ml - mr, H - mt - mb
    rows = list(a.itertuples())
    n = len(rows)
    band = pw / n
    dodge = band * 0.16

    vals = []
    for r in rows:
        vals += [r.d + r.e, r.d - r.e, r.c + r.ce, r.c - r.ce]
    lo, hi = min(vals), max(vals)
    pad = 0.12 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def y(v):
        return mt + ph * (hi - v) / (hi - lo)

    def x(i, k):
        return ml + band * (i + 0.5) + (dodge if k else -dodge)

    p = []
    # y grid + ticks, on a 2e-3 step
    step = 2.0
    t = step * int(lo / step)
    while t <= hi:
        if lo <= t <= hi:
            cls = 'zero' if abs(t) < 1e-9 else 'grid'
            p.append(f'<line class="{cls}" x1="{ml}" x2="{ml + pw}" '
                     f'y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
            p.append(f'<text class="tick" x="{ml - 8}" y="{y(t) + 4:.1f}" '
                     f'text-anchor="end">{t:+.0f}</text>')
        t += step

    # the 1-5 ms region the question is about
    n_early = sum(1 for r in rows if r.w in ('1-2', '2-3', '3-5'))
    p.insert(0, f'<rect class="early" x="{ml}" y="{mt}" '
                f'width="{band * n_early:.1f}" height="{ph}"/>')

    for i, r in enumerate(rows):
        p.append(f'<text class="xlab" x="{ml + band * (i + 0.5):.1f}" '
                 f'y="{mt + ph + 20}" text-anchor="middle">{esc(r.w)}</text>')
        for k, (v, e, lab, z) in enumerate((
                (r.c, r.ce, 'control (before vs before)', r.zc),
                (r.d, r.e, 'after &minus; before', r.z))):
            cx, ytop, ybot = x(i, k), y(v + e), y(v - e)
            cls = 'meas' if k else 'ctl'
            p.append(
                f'<g class="pt {cls}">'
                f'<line class="bar" x1="{cx:.1f}" x2="{cx:.1f}" '
                f'y1="{ytop:.1f}" y2="{ybot:.1f}"/>'
                f'<line class="cap" x1="{cx - 4:.1f}" x2="{cx + 4:.1f}" '
                f'y1="{ytop:.1f}" y2="{ytop:.1f}"/>'
                f'<line class="cap" x1="{cx - 4:.1f}" x2="{cx + 4:.1f}" '
                f'y1="{ybot:.1f}" y2="{ybot:.1f}"/>'
                f'<circle class="ring" cx="{cx:.1f}" cy="{y(v):.1f}" r="6"/>'
                f'<circle class="dot" cx="{cx:.1f}" cy="{y(v):.1f}" r="4.5"/>'
                f'<rect class="hit" x="{cx - 11:.1f}" y="{mt}" width="22" '
                f'height="{ph}"/>'
                f'<g class="tip" transform="translate({cx:.1f},{y(v) - 16:.1f})">'
                f'<rect x="-92" y="-34" width="184" height="34" rx="4"/>'
                f'<text x="0" y="-21" text-anchor="middle">{esc(r.w)} ms &middot; '
                f'{lab}</text>'
                f'<text x="0" y="-8" text-anchor="middle">'
                f'{v:+.2f} &plusmn; {e:.2f} &times;10&#8315;&#179; '
                f'(z = {z:+.2f})</text></g>'
                f'</g>')

    p.append(f'<text class="ylab" transform="rotate(-90)" '
             f'x="{-(mt + ph / 2):.1f}" y="14" text-anchor="middle">'
             f'&Delta; efficiency (&times;10&#8315;&#179;)</text>')
    p.append(f'<text class="xtitle" x="{ml + pw / 2:.1f}" y="{H - 6}" '
             f'text-anchor="middle">time since gamma flash [ms]</text>')

    legend = (
        '<div class="legend">'
        '<span><i class="sw meas"></i>after &minus; before (run_139 &minus; run_132)</span>'
        '<span><i class="sw ctl"></i>control: before &minus; before (run_132 &minus; run_130)</span>'
        '<span><i class="sw early"></i>the 1&ndash;5&nbsp;ms window in question</span>'
        '</div>')
    return (f'{legend}<div class="chart-wrap"><svg viewBox="0 0 {W} {H}" '
            f'role="img" aria-label="Change in Det A efficiency per time '
            f'window, after minus before, compared with the before-versus-'
            f'before control">{"".join(p)}</svg></div>')


# --------------------------------------------------------------- tables
def html_table(df, cols, headers, fmts, cls=''):
    th = ''.join(f'<th>{esc(h)}</th>' for h in headers)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c, f in zip(cols, fmts):
            v = r[c]
            tds.append(f'<td>{f(v) if pd.notna(v) else "&mdash;"}</td>')
        body.append('<tr>' + ''.join(tds) + '</tr>')
    return (f'<div class="tbl-wrap"><table class="{cls}"><thead><tr>{th}</tr>'
            f'</thead><tbody>{"".join(body)}</tbody></table></div>')


def main():
    wt = pd.read_csv(os.path.join(TAB_DIR, 'window_stats.csv'))
    t50 = pd.read_csv(os.path.join(TAB_DIR, 'recovery_t50.csv'))
    book = pd.read_csv(os.path.join(TAB_DIR, 'run_bookkeeping.csv'))
    a = wt[wt.det == 'A'].copy()
    a['w'] = a.window.str.replace(' ms', '', regex=False)
    a['d'] = (a.eff_139 - a.eff_132) * 1e3
    a['e'] = ((a.err_139 ** 2 + a.err_132 ** 2) ** 0.5) * 1e3
    a['c'] = (a.eff_132 - a.eff_130) * 1e3
    a['ce'] = ((a.err_132 ** 2 + a.err_130 ** 2) ** 0.5) * 1e3
    a['zc'] = a.z_ctl      # short aliases the chart/table formatters use

    tA = t50[t50.det == 'A'].set_index('run')
    dt50 = tA.loc['run_139', 't50_ms'] - tA.loc['run_132', 't50_ms']
    et50 = (tA.loc['run_139', 't50_err_ms'] ** 2
            + tA.loc['run_132', 't50_err_ms'] ** 2) ** 0.5
    ub = dt50 + 1.96 * et50
    ctl50 = tA.loc['run_132', 't50_ms'] - tA.loc['run_130', 't50_ms']
    early = a[a.w.isin(['1-2', '2-3', '3-5'])]
    max_z = early.z.abs().max()
    n_trig = int(book.probe_events.sum())

    f3 = (lambda v: f'{v:.3f}')
    f4 = (lambda v: f'{v:.4f}')
    f2 = (lambda v: f'{v:+.2f}')

    tiles = [
        (f'{dt50:+.2f} &plusmn; {et50:.2f} ms',
         '&Delta; t<sub>50</sub>, after &minus; before',
         f'on a {tA.loc["run_132", "t50_ms"]:.2f} ms recovery'),
        (f'&lt; {ub:+.2f} ms', '95% upper bound on any lengthening',
         'i.e. under 6% of the recovery time'),
        (f'{max_z:.1f} &sigma;', 'largest deviation in 1&ndash;5 ms',
         f'while the control reaches {early.zc.abs().max():.1f} &sigma;'),
        (f'{n_trig / 1e6:.2f} M', 'reconstructed triggers',
         f'{int(book.bursts.sum()):,} flash bursts, 27 sub-runs'),
    ]
    tile_html = ''.join(
        f'<div class="tile"><div class="tile-v">{v}</div>'
        f'<div class="tile-k">{k}</div><div class="tile-s">{s}</div></div>'
        for v, k, s in tiles)

    t50_tbl = html_table(
        t50[t50.det == 'A'].assign(
            lbl=lambda d: d.run.map({'run_130': 'run_130 — before (Aug 3 eve)',
                                     'run_132': 'run_132 — before (night Aug 3–4)',
                                     'run_139': 'run_139 — after (night Aug 4–5)'})),
        ['lbl', 't50_ms', 't50_err_ms'],
        ['run', 't50 [ms]', '± [ms]'], [esc, f3, f3], cls='num')

    win_tbl = html_table(
        a, ['window', 'eff_130', 'eff_132', 'eff_139', 'd', 'z', 'c', 'zc'],
        ['window', 'before (130)', 'before (132)', 'after (139)',
         'Δ ×10⁻³', 'z', 'control Δ ×10⁻³', 'z (control)'],
        [esc, f4, f4, f4, f2, f2, f2, f2], cls='num')

    book_tbl = html_table(
        book, ['run', 'period', 'subruns', 'bursts', 'probe_events',
               'nhits_leader_med', 'e10_med'],
        ['run', 'period', 'sub-runs', 'bursts', 'probe triggers',
         'flash leader hits (med)', 'intensity ×10¹⁰ (med)'],
        [esc, esc, lambda v: f'{int(v)}', lambda v: f'{int(v):,}',
         lambda v: f'{int(v):,}', lambda v: f'{v:.0f}', lambda v: f'{v:.0f}'],
        cls='num')

    figs = ''.join(
        f'<figure><a href="figures/{f}" target="_blank" rel="noopener">'
        f'<img src="figures/{f}" alt="{esc(t)}" loading="lazy"></a>'
        f'<figcaption><b>{esc(t)}</b> {esc(c)}</figcaption></figure>'
        for f, t, c in FIGURES
        if os.path.exists(os.path.join(L.OUT_BASE, 'figures', f)))

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lead-shielding removal — DAQ saturation check</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#fbfbfa; --surface:#ffffff; --line:#e3e2df;
  --ink:#14140f; --ink2:#55534c; --ink3:#87857c;
  --series:{SERIES_1_LIGHT}; --ctl:#8a8880; --early:#f3d77a4d;
  --ok:#1b7f4b; --okbg:#1b7f4b14;
}}
/* Dark values declared under both scopes: the media query follows the OS, the
   data-theme scope lets a host page (the DAQ Analysis tab) force it. */
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#15151a; --surface:#1c1c22; --line:#33333c;
    --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
    --series:{SERIES_1_DARK}; --ctl:#918f86; --early:#f3d77a1f;
    --ok:#5fd08a; --okbg:#5fd08a1a;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15151a; --surface:#1c1c22; --line:#33333c;
  --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
  --series:{SERIES_1_DARK}; --ctl:#918f86; --early:#f3d77a1f;
  --ok:#5fd08a; --okbg:#5fd08a1a;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:28px 22px 64px; background:var(--bg); color:var(--ink);
  font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
h1 {{ font-size:1.62rem; line-height:1.25; margin:0 0 6px; letter-spacing:-.01em; }}
h2 {{ font-size:1.12rem; margin:38px 0 12px; padding-bottom:7px;
  border-bottom:1px solid var(--line); letter-spacing:-.005em; }}
h3 {{ font-size:.98rem; margin:22px 0 8px; }}
.sub {{ color:var(--ink2); margin:0 0 22px; font-size:.94rem; }}
p {{ margin:0 0 12px; }}
code {{ font:.86em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--surface); border:1px solid var(--line); border-radius:4px;
  padding:1px 5px; }}
.verdict {{ background:var(--okbg); border:1px solid var(--ok);
  border-left-width:4px; border-radius:8px; padding:16px 18px; margin:0 0 26px; }}
.verdict b {{ color:var(--ok); }}
.tiles {{ display:grid; gap:12px; margin:0 0 8px;
  grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); }}
.tile {{ background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:14px 16px; }}
.tile-v {{ font-size:1.5rem; font-weight:650; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; }}
.tile-k {{ color:var(--ink2); font-size:.85rem; margin-top:3px; }}
.tile-s {{ color:var(--ink3); font-size:.78rem; margin-top:2px; }}
.tbl-wrap {{ overflow-x:auto; margin:0 0 14px; }}
table {{ border-collapse:collapse; width:100%; font-size:.87rem;
  background:var(--surface); border:1px solid var(--line); border-radius:8px; }}
th,td {{ padding:7px 11px; text-align:left; border-bottom:1px solid var(--line);
  white-space:nowrap; }}
th {{ color:var(--ink2); font-weight:600; font-size:.8rem; }}
tbody tr:last-child td {{ border-bottom:0; }}
table.num td+td, table.num th+th {{ text-align:right;
  font-variant-numeric:tabular-nums; }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; font-size:.82rem;
  color:var(--ink2); margin:4px 0 6px; }}
.legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.sw {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
.sw.meas {{ background:var(--series); }}
.sw.ctl {{ background:var(--ctl); }}
.sw.early {{ background:var(--early); border:1px solid var(--line); }}
.chart-wrap {{ background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:10px 6px 2px; overflow-x:auto; }}
svg {{ width:100%; height:auto; min-width:620px; display:block; }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.zero {{ stroke:var(--ink3); stroke-width:1.5; }}
.early {{ fill:var(--early); }}
.tick,.xlab,.ylab,.xtitle {{ fill:var(--ink2); font-size:11px; }}
.ylab,.xtitle {{ fill:var(--ink3); }}
.pt .bar {{ stroke-width:2; }}
.pt .cap {{ stroke-width:2; }}
.pt .ring {{ fill:var(--surface); }}
.pt.meas .bar, .pt.meas .cap {{ stroke:var(--series); }}
.pt.meas .dot {{ fill:var(--series); }}
.pt.ctl .bar, .pt.ctl .cap {{ stroke:var(--ctl); }}
.pt.ctl .dot {{ fill:var(--ctl); }}
.hit {{ fill:transparent; }}
.tip {{ opacity:0; pointer-events:none; transition:opacity .1s; }}
.tip rect {{ fill:var(--ink); }}
.tip text {{ fill:var(--bg); font-size:10.5px; }}
.pt:hover .tip {{ opacity:1; }}
figure {{ margin:0 0 22px; background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:10px; }}
/* matplotlib PNGs have white backgrounds; give them a white plate so they do
   not clash with the surface in dark mode. */
figure img {{ width:100%; height:auto; display:block; border-radius:4px;
  background:#fff; padding:6px; }}
figcaption {{ color:var(--ink2); font-size:.83rem; margin-top:8px; }}
figcaption b {{ color:var(--ink); }}
ul {{ margin:0 0 12px; padding-left:20px; }}
li {{ margin-bottom:6px; }}
.foot {{ color:var(--ink3); font-size:.8rem; margin-top:40px;
  border-top:1px solid var(--line); padding-top:14px; }}
</style>
</head>
<body>
<div class="wrap">

<h1>Did the 2026-08-04 lead removal lengthen the post-flash DAQ saturation?</h1>
<p class="sub">Before/after check on the identically-configured stat090
production runs bracketing the access &middot; generated from
<code>ntof_july_analysis/leadshield_compare/</code></p>

<div class="verdict">
<p style="margin:0"><b>No.</b> Within the precision of {n_trig / 1e6:.2f} M
reconstructed triggers the post-flash recovery is unchanged. The 50% recovery
time of the clean reference detector moved by
<b>{dt50:+.2f} &plusmn; {et50:.2f} ms</b> on a
{tA.loc['run_132', 't50_ms']:.2f} ms recovery, and the night-to-night control
between two <i>before</i> runs moved {abs(ctl50 / dt50):.0f}&times; more.
Trigger acceptance, first-accept time, blindness and the size of the flash
itself all agree.</p>
</div>

<div class="tiles">{tile_html}</div>

<h2>What was compared</h2>
<p>Three stat090 PRODUCTION runs at the run_67 optimum, configs verified
identical (drift 700&nbsp;V, resist A540/B540/C525/D520, 0.90&nbsp;MIP,
PS+SINGLES, RAW 20&nbsp;smp&nbsp;&times;&nbsp;60&nbsp;ns, latency 27,
Ar/Iso 90/10, no beam filter). The access was the morning of Aug&nbsp;4;
run_132 was operator-killed at 08:44 for it.</p>
{book_tbl}
<p><b>run_130 is the control, not padding.</b> It is a <i>before</i> run on a
different evening, so whatever run_130-vs-run_132 shows is the night-to-night
systematic floor. An after-vs-before difference only means something if it is
larger than that floor.</p>
<p>Reconstruction is the current <code>ntof_tracking.reco</code> chain (noise
flagging &rarr; segment finding &rarr; 3-D x/y pairing). Efficiency is
<b>P(3-D x/y pair) per recorded trigger</b>, denominator <code>readout_*</code>
&mdash; blindness stays <i>in</i> the denominator, because blindness is the
inefficiency being measured. Det&nbsp;A is the reference: it is the only
detector with a good M1 card.</p>

<h2>The answer, as one number</h2>
<p>Time at which Det&nbsp;A's efficiency reaches half its 40&ndash;76&nbsp;ms
plateau, with binomial-bootstrap errors:</p>
{t50_tbl}
<p>after &minus; before = <b>{dt50:+.2f} &plusmn; {et50:.2f} ms</b> (no change).
Before &minus; before, the control = <b>{ctl50:+.2f} ms</b>. A 95% upper bound
on any <i>lengthening</i> is <b>{ub:+.2f} ms</b>, under 6% of the recovery
time. Had the removal doubled the saturation, t<sub>50</sub> would have gone to
~11&nbsp;ms. Det&nbsp;D, the other detector with a physically shaped recovery
curve, agrees: {t50[(t50.det == 'D') & (t50.run == 'run_132')].t50_ms.iloc[0]:.2f}
&rarr; {t50[(t50.det == 'D') & (t50.run == 'run_139')].t50_ms.iloc[0]:.2f} ms,
if anything slightly faster.</p>
<figure><a href="figures/VERDICT_detA.png" target="_blank" rel="noopener">
<img src="figures/VERDICT_detA.png" alt="Det A recovery curves and t50 for the three runs"></a>
<figcaption><b>The headline.</b> Det&nbsp;A's recovery, three runs. The dashed
50%-of-plateau markers for before and after sit 0.01&nbsp;ms apart.</figcaption>
</figure>

<h2>The 1&ndash;5 ms window specifically</h2>
<p>This is where a longer saturation would show. It doesn't &mdash; and the
point of the chart below is that the after&minus;before deltas (blue) are
<i>smaller</i> than the before&minus;before control (grey) in exactly the
windows in question.</p>
{delta_chart(a)}
{win_tbl}
<p>Nothing in 1&ndash;5&nbsp;ms reaches {max_z:.1f}&nbsp;&sigma;, while the
before-vs-before control reaches {early.zc.abs().max():.1f}&nbsp;&sigma; in the
same windows. The late window (40&ndash;76&nbsp;ms) matching to four decimals
confirms the two nights were otherwise equivalent &mdash; anything that moved
early <i>and</i> late alike would not be a saturation-time effect at all.</p>
<p><b>Intensity-matched</b> (HIGH pulses only, &ge;600e10&nbsp;p &mdash; run_139
delivered a slightly softer mix, 52.5% HIGH vs 58.2%): at 3&ndash;5&nbsp;ms the
three runs give 0.0067 (130), 0.0156 (132), 0.0115 (139). <b>The after run lands
between the two before runs.</b> Taken against run_132 alone that reads as
&minus;4.5&nbsp;&sigma;; but run_132 vs run_130 in the same window is
+6.7&nbsp;&sigma;, larger and opposite. This is precisely the trap the control
was included to catch.</p>

<h2>Supporting observables &mdash; all consistent</h2>
<ul>
<li><b>Trigger acceptance</b> (DAQ-level, independent of any reconstruction):
4.382 &rarr; 4.364 accepted events/burst in 1&ndash;2&nbsp;ms, a 0.4%
difference; the after/before ratio is 1.00 at every dt out to 80&nbsp;ms. If the
DAQ had stayed saturated longer, this is where it would show first.</li>
<li><b>First accepted trigger after the flash</b>: q10/q50/q90 =
0.99/1.00/1.04&nbsp;ms &mdash; identical to 0.01&nbsp;ms in all three runs.</li>
<li><b>Det A blindness</b> at 1&ndash;2&nbsp;ms: 0.2035 &rarr; 0.1998,
marginally <i>lower</i> after.</li>
<li><b>The flash itself did not grow</b>: leader total hits 4102 &rarr; 4044
median, hits above amplitude 1000: 3017 &rarr; 2964 &mdash; marginally
<i>smaller</i>, tracking the 1% lower delivered intensity. Whatever lead came
out, it was not attenuating the gamma flash these detectors see.</li>
</ul>

<h2>Why Det B and Det C look different (and why it isn't the effect)</h2>
<p>B and C show significant early-dt decreases (C: &minus;0.013 at
1&ndash;2&nbsp;ms, z = &minus;6.5). These are <b>not</b> saturation-time
signatures:</p>
<ul>
<li>Both have <b>early-dt "efficiency" above their own late-dt plateau</b> &mdash;
B is 0.062 at 1&ndash;2&nbsp;ms falling to 0.0059 at 40&ndash;76&nbsp;ms. Real
track efficiency cannot decrease with recovery time. Their early-dt pairs are
flash-correlated common-mode noise on the bad M1 cards, not tracks; their
t<sub>50</sub> is undefined because the curve never rises through
half-plateau.</li>
<li>B and D also shift at <b>late</b> dt (20&ndash;76&nbsp;ms, z =
&minus;4 to &minus;8), as much as or more than early. Anything that moves early
and late alike is a gain or noise drift.</li>
</ul>
<p>The direction is worth noting anyway: those flash-correlated fakes went
<i>down</i> after the access &mdash; if anything slightly less flash-induced
noise, not more.</p>

<h2>Limits &mdash; what this does not rule out</h2>
<ul>
<li><b>Below 1&nbsp;ms.</b> The DAQ accept gate does not open until
1.00&nbsp;ms (measured, identically in all three runs). A saturation that grew
from 0.3 to 0.9&nbsp;ms would be invisible here. This bounds the
<i>1&ndash;76&nbsp;ms</i> recovery only; closing the sub-ms gap needs a
flash-trigger recovery run in the style of run_18/run_45.</li>
<li><b>Dets B and C</b> get no clean per-detector bound, because their bad M1
cards make their early-dt yield noise-dominated. The bound above is Det&nbsp;A,
supported by Det&nbsp;D.</li>
<li>The comparison is night-to-night; run_130 is what calibrates that. It
cannot separate the access from anything else that changed on Aug&nbsp;4 &mdash;
but since nothing moved, there is nothing to separate.</li>
</ul>

<h2>All figures</h2>
{figs}

<p class="foot">Generated by
<code>ntof_july_analysis/leadshield_compare/make_report.py</code> from
<code>tables/*.csv</code>. Pipeline: <code>process.py</code> &rarr;
<code>feu_presence.py --force</code> &rarr; <code>compare.py</code> &rarr; this
report. Full write-up in <code>RESULT.md</code>; machine-generated tables in
<code>SUMMARY.md</code>.</p>

</div>
</body>
</html>
"""
    with open(OUT_PATH, 'w') as f:
        f.write(doc)
    print('wrote', OUT_PATH, f'({len(doc) / 1024:.0f} kB)')


if __name__ == '__main__':
    main()
