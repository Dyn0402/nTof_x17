#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the HTML report for the flash-reference sweep.

Writes <out>/report.html beside flash_reference.json, which the DAQ page's
Analysis tab lists and opens inline. Figures are referenced with ordinary
relative links (figures/x.png) so the same file works from disk, from the DAQ
page's /analysis_file/<relpath> route, or copied elsewhere.

Generated, not hand-written: re-run after flash_reference_report.py and the
numbers, the charts and the verdict text all follow.

    .venv/bin/python ntof_processing/slim_pipeline/make_flash_report.py \
        --out ntof_processing/slim_pipeline/flash_reference
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

SERIES_1_LIGHT, SERIES_1_DARK = '#2a78d6', '#3987e5'


def esc(s):
    return html.escape(str(s))


def hist_chart(counts, lo, hi, flag_at, title, xlab, log=True):
    """Log-count histogram of a ratio, with the flag threshold marked.

    The whole argument of this analysis is that the two signatures are
    SEPARATED, not merely different in the mean -- so the chart has to show
    the empty region between the population and the threshold, which needs a
    log count axis (the population is 10^5, the outliers are single counts).
    """
    import math
    W, H = 780, 300
    ml, mr, mt, mb = 58, 14, 16, 50
    pw, ph = W - ml - mr, H - mt - mb
    n = len(counts)
    mx = max(counts) if counts else 1

    def y(v):
        if not log:
            return mt + ph * (1 - v / mx)
        top = math.log10(mx + 1)
        return mt + ph * (1 - (math.log10(v + 1) / top if top else 0))

    def x(i):
        return ml + pw * (i / n)

    p = []
    dec = 0
    while 10 ** dec <= mx:
        yy = y(10 ** dec)
        p.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" '
                 f'y1="{yy:.1f}" y2="{yy:.1f}"/>')
        p.append(f'<text class="tick" x="{ml - 8}" y="{yy + 4:.1f}" '
                 f'text-anchor="end">10<tspan dy="-4" font-size="8">{dec}'
                 f'</tspan></text>')
        dec += 1
    for i, c in enumerate(counts):
        if not c:
            continue
        yy = y(c)
        p.append(f'<rect class="bar" x="{x(i):.1f}" y="{yy:.1f}" '
                 f'width="{max(pw / n - 0.4, 0.8):.2f}" '
                 f'height="{mt + ph - yy:.1f}"><title>{lo + (hi - lo) * i / n:.2f}'
                 f'&ndash;{lo + (hi - lo) * (i + 1) / n:.2f}: {c:,}</title></rect>')
    fx = ml + pw * (flag_at - lo) / (hi - lo)
    p.append(f'<line class="thr" x1="{fx:.1f}" x2="{fx:.1f}" y1="{mt}" '
             f'y2="{mt + ph}"/>')
    p.append(f'<text class="thrlab" x="{fx + 5:.1f}" y="{mt + 12}">flag below '
             f'{flag_at:g}&times;</text>')
    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        if lo <= t <= hi:
            tx = ml + pw * (t - lo) / (hi - lo)
            p.append(f'<text class="xlab" x="{tx:.1f}" y="{mt + ph + 18}" '
                     f'text-anchor="middle">{t:g}</text>')
    p.append(f'<text class="xtitle" x="{ml + pw / 2:.1f}" y="{H - 8}" '
             f'text-anchor="middle">{esc(xlab)}</text>')
    p.append(f'<text class="ylab" transform="rotate(-90)" '
             f'x="{-(mt + ph / 2):.1f}" y="14" text-anchor="middle">bursts'
             f'</text>')
    return (f'<div class="chart-wrap"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{esc(title)}">{"".join(p)}</svg></div>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).resolve().parent / 'flash_reference')
    a = ap.parse_args()
    d = json.loads((a.out / 'flash_reference.json').read_text())

    th = d['thresholds']
    flagged = d['flagged']
    since = [f for f in flagged if f['since_run']]
    matched = [f for f in flagged if f['state'] == 'MATCHED']
    both = [f for f in flagged if f['both']]
    ok = not matched

    tiles = [
        (f'{d["n_judged"]:,}', 'bursts examined',
         f'of {d["n_bursts"]:,} clusters, over {d["n_runs"]} DREAM runs'),
        (f'{len(flagged)}', 'with a mis-tagged flash',
         f'{len(both)} flagged by BOTH signatures, {len(since)} since run_79'),
        (f'{len(matched)}', 'of those counted as matched',
         'the silent class: a wrong time base inside a healthy-looking product'
         if matched else 'none — every one was refused by the chain'),
        ('{:.3f}&times;'.format(d['pooled']['gap1_frac_pct'].get('0.01',
                                                                  float('nan'))),
         'lowest gap1 in the bulk',
         f'0.01st percentile; the flag sits at {th["gap_frac"]}&times;'),
    ]
    tile_html = ''.join(
        f'<div class="tile"><div class="tile-v">{v}</div>'
        f'<div class="tile-k">{k}</div><div class="tile-s">{s}</div></div>'
        for v, k, s in tiles)

    DASH = '&mdash;'

    def num(v, fmt='{:.2f}'):
        return DASH if v is None else fmt.format(v)

    rows = []
    for f in sorted(flagged, key=lambda f: (f['run'], f['subrun'],
                                            f['burst_id'])):
        fit = f['fit'] or {}
        st = f['state'] or DASH
        cls = 'bad' if f['state'] == 'MATCHED' else 'ok'
        rows.append(
            '<tr><td>{run}/{sub}</td><td class="n">{bid}</td>'
            '<td class="n">{ntrig}</td><td class="n">{g:.1f}</td>'
            '<td class="n">{gf:.3f}</td><td class="n">{nh:,}</td>'
            '<td class="n">{hf:.3f}</td><td class="{cls}">{st}</td>'
            '<td class="n">{co}</td><td class="n">{fitted}</td>'
            '<td class="n">{da}</td></tr>'.format(
                run=esc(f['run']), sub=esc(f['subrun']), bid=f['burst_id'],
                ntrig=f['n_trig'], g=f['gap1_ns'] / 1e3, gf=f['gap1_frac'],
                nh=f['flash_nhits'], hf=f['nhits_frac'], cls=cls, st=esc(st),
                co=num(f['ledger_frac']),
                fitted=num(fit.get('fitted'), '{:d}'),
                da=num(fit.get('da_ns'), '{:.1f}')))
    table = ('<div class="tbl-wrap"><table class="num"><thead><tr>'
             '<th>sub-run</th><th>burst</th><th>triggers</th>'
             '<th>gap1 [µs]</th><th>gap1 / median</th><th>flash hits</th>'
             '<th>hits / median</th><th>ledger state</th><th>coincidence</th>'
             '<th>fitted</th><th>da [ns]</th></tr></thead><tbody>'
             + ''.join(rows) + '</tbody></table></div>') if rows else \
        '<p class="muted">No burst was flagged.</p>'

    g = d['pooled']['gap1_frac_hist']
    h = d['pooled']['nhits_frac_hist']
    chart_g = hist_chart(g, 0, 2, th['gap_frac'],
                         'gap1 relative to the sub-run median',
                         'flash → first physics trigger, ÷ the sub-run median')
    chart_h = hist_chart(h, 0, 2, th['nhits_frac'],
                         'flash hit count relative to the sub-run median',
                         'hits in the tagged flash event, ÷ the sub-run median')

    verdict = (
        '<p><b>No product carries a mis-tagged gamma flash.</b> Every burst '
        'whose tagged flash fails either signature was already refused by the '
        'matching chain — it appears in the ledger as an unmatched pulse, not '
        'as a matched one. The concern that a dropped flash could be absorbed '
        'into a per-bunch correction and pass silently is not realised '
        'anywhere in the campaign.</p>'
        if ok else
        f'<p><b>{len(matched)} burst(s) with a mis-tagged flash are counted as '
        f'matched.</b> Their DREAM time base is referenced to the wrong '
        f'trigger, so their per-event <code>t_dream_ns</code> and the '
        f'per-bunch correction fitted from them are wrong by the offset in '
        f'the table below, while the product looks healthy. Each needs a '
        f'<code>burst_bruteforce.py</code> scan and a '
        f'<code>burst_fixes.json</code> entry.</p>')

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Is every DREAM burst timed from its gamma flash?</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#fbfbfa; --surface:#ffffff; --line:#e3e2df;
  --ink:#14140f; --ink2:#55534c; --ink3:#87857c;
  --series:{SERIES_1_LIGHT}; --thr:#b4462f;
  --ok:#1b7f4b; --okbg:#1b7f4b14; --bad:#a4243b; --badbg:#a4243b14;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#15151a; --surface:#1c1c22; --line:#33333c;
    --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
    --series:{SERIES_1_DARK}; --thr:#e2765c;
    --ok:#5fd08a; --okbg:#5fd08a1a; --bad:#e8697f; --badbg:#e8697f1a;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15151a; --surface:#1c1c22; --line:#33333c;
  --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
  --series:{SERIES_1_DARK}; --thr:#e2765c;
  --ok:#5fd08a; --okbg:#5fd08a1a; --bad:#e8697f; --badbg:#e8697f1a;
}}
body {{ margin:0; padding:28px 22px 64px; background:var(--bg); color:var(--ink);
  font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .2rem; letter-spacing:-.01em; }}
h2 {{ font-size:1.05rem; margin:2.2rem 0 .6rem; padding-bottom:.25rem;
  border-bottom:1px solid var(--line); }}
.sub {{ color:var(--ink2); margin:0 0 1.4rem; font-size:.92rem; }}
.verdict {{ background:var(--{'okbg' if ok else 'badbg'});
  border-left:4px solid var(--{'ok' if ok else 'bad'});
  padding:.85rem 1.05rem; border-radius:0 7px 7px 0; margin:1.2rem 0; }}
.verdict p {{ margin:.3rem 0; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:12px; margin:1.2rem 0 1.6rem; }}
.tile {{ background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:12px 14px; }}
.tile-v {{ font-size:1.45rem; font-weight:650; letter-spacing:-.02em; }}
.tile-k {{ font-size:.86rem; color:var(--ink2); margin-top:2px; }}
.tile-s {{ font-size:.78rem; color:var(--ink3); margin-top:4px; }}
.chart-wrap {{ background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:8px; margin:.6rem 0 1.2rem; overflow-x:auto; }}
svg {{ display:block; width:100%; height:auto; min-width:560px; }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.bar {{ fill:var(--series); }}
.thr {{ stroke:var(--thr); stroke-width:1.5; stroke-dasharray:4 3; }}
.thrlab, .tick, .xlab {{ fill:var(--ink3); font-size:10.5px; }}
.xtitle, .ylab {{ fill:var(--ink2); font-size:11.5px; }}
.tbl-wrap {{ overflow-x:auto; background:var(--surface);
  border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; }}
th, td {{ padding:.35rem .6rem; border-bottom:1px solid var(--line);
  text-align:left; white-space:nowrap; }}
th {{ font-size:.75rem; letter-spacing:.02em; color:var(--ink2);
  background:var(--bg); position:sticky; top:0; }}
td.n, table.num td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.ok {{ color:var(--ok); }} td.bad {{ color:var(--bad); font-weight:600; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.85em; background:var(--surface); border:1px solid var(--line);
  border-radius:3px; padding:.05em .3em; }}
.muted {{ color:var(--ink3); }}
li {{ margin:.3rem 0; }}
</style>
</head>
<body>
<div class="wrap">
<h1>Is every DREAM burst timed from its gamma flash?</h1>
<p class="sub">Campaign-wide sweep of the burst time reference ·
{d["n_runs"]} DREAM runs, {d["n_bursts"]:,} clusters · generated by
<code>make_flash_report.py</code> from <code>flash_reference.json</code></p>

<div class="verdict">{verdict}</div>

<div class="tiles">{tile_html}</div>

<h2>What is measured, and why it is decisive</h2>
<p><code>bunch_join</code> defines a burst's time base by its <em>first</em>
trigger: that trigger is called the gamma flash, and every other trigger in the
burst is timed from it. If the flash itself was not recorded, the first
scintillator single takes its place and the whole burst sits ~1&nbsp;ms off the
n_TOF clock. Two independent properties say whether the tagged flash is really
the flash, and both come from the DREAM files alone — no n_TOF, no lock, no
fit:</p>
<ul>
<li><b>gap1</b>, the delay from the tagged flash to the first physics trigger.
The N93B gate admits singles only from ~1&nbsp;ms after the flash, so this is a
hard edge; the next gap (single to single) is ~15&nbsp;µs, two orders of
magnitude smaller. A burst whose &ldquo;flash&rdquo; is really a single shows
gap1 in the tens of microseconds.</li>
<li><b>flash hits</b>, the number of hits in the tagged flash event. The flash
saturates every chamber: ~4,000 hits against 16&ndash;830 for a physics
trigger.</li>
</ul>
<p>Both are compared against the <em>sub-run's own</em> medians, never against a
campaign constant: the gate width is a DAQ setting and the flash hit count
depends on which chambers were live. A burst is flagged when gap1 falls below
{th['gap_frac']}&times; its sub-run's median <em>or</em> the flash hit count
below {th['nhits_frac']}&times;. The charts below are why those thresholds
cannot matter — on either axis the bulk and the threshold are separated by an
empty region.</p>

<h2>gap1, relative to each sub-run's median</h2>
{chart_g}
<h2>Flash hit count, relative to each sub-run's median</h2>
{chart_h}

<h2>Every flagged burst, and what the chain did with it</h2>
<p>The join that matters is the last four columns: the pulse ledger's terminal
state for the burst, its measured wall+plastic coincidence, and whether the
product's per-bunch clock fit accepted its bunch. A flagged burst that is
<span class="muted">not</span> MATCHED was refused by the chain and cost
nothing.</p>
{table}

<h2>What this does not settle</h2>
<ul>
<li>Bursts of fewer than {th['min_burst_trig']} triggers have no meaningful
gap1 and are not judged here; they are noise clusters rather than beam bursts
and the ledger already counts them as such.</li>
<li>A flash recorded but <em>mis-ordered</em> — arriving in the file after a
single of the same burst — would not be caught by gap1 (it would look normal)
but would be caught by the hit count, since the tagged event would then be the
single. A flash both recorded and correctly first, yet displaced in time by the
DAQ, would be caught by neither; nothing in the campaign suggests that mode.</li>
<li>This sweep says the time <em>reference</em> is right. It says nothing about
the per-trigger timing resolution within a burst, which the clock fit's
residuals measure.</li>
</ul>

<p class="muted" style="font-size:.82rem">Provenance:
<code>flash_reference_sweep.py</code> (one condor job per DREAM run, reading
two branches of each sub-run's combined hits and the <code>bunches</code> tree
of each published product), <code>flash_reference_report.py</code> (the join
against the pulse ledger), this file. {d['n_subrun_errors']} sub-run(s) could
not be read.</p>
</div>
</body>
</html>
"""
    (a.out / 'report.html').write_text(doc)
    print(f'-> {a.out / "report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
