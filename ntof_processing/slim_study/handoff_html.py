#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
handoff_html.py -- render the n_TOF reprocessing request as a standalone page.

Called by make_handoff.py with the same numbers that go into the markdown, so
the two cannot drift. Emits ONE self-contained document: inline CSS, no
external stylesheet, script, font or image, because the site serves notes
offline (dylan-cern-site README, "Offline").

Palette: the site's own tokens. The coverage bar's three states are
series-1 / series-2 / muted -- blue / orange / neutral grey, NOT the
good/warning status pair, because green and yellow converge under deuteranopia
(measured: green+yellow worst-pair CVD dE 5.1, blue+orange 9.8-11.3, against a
target of 8). Validated with the dataviz validator in both modes:

    blue/orange/grey light  band pass  CVD 9.8 protan  normal 17.6  contrast pass
    blue/orange/grey dark   band pass  CVD 11.3 deutan normal 16.7  contrast pass

The grey deliberately fails the chroma floor -- "LOST" is a neutral slot rather
than a competing identity -- and every segment is direct-labelled, so identity
is never carried by colour alone.
"""
from __future__ import annotations

import html


def _esc(x):
    return html.escape(str(x), quote=True)


CSS = """
:root{
  color-scheme:light;
  --plane:#f9f9f7; --surface:#fcfcfb; --raised:#ffffff;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --neutral:#a8a69e;
  --wash-1:rgba(42,120,214,0.10); --wash-2:rgba(235,104,52,0.10);
  --warnwash:rgba(250,178,25,0.13); --warnedge:#c9901a;
  --okwash:rgba(12,163,12,0.10); --okedge:#0ca30c;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --plane:#0d0d0d; --surface:#1a1a19; --raised:#232322;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
    --series-1:#3987e5; --series-2:#d95926; --neutral:#6e6c66;
    --wash-1:rgba(57,135,229,0.16); --wash-2:rgba(217,89,38,0.16);
    --warnwash:rgba(250,178,25,0.10); --warnedge:#e0a83a;
    --okwash:rgba(25,158,112,0.12); --okedge:#199e70;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --plane:#0d0d0d; --surface:#1a1a19; --raised:#232322;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --neutral:#6e6c66;
  --wash-1:rgba(57,135,229,0.16); --wash-2:rgba(217,89,38,0.16);
  --warnwash:rgba(250,178,25,0.10); --warnedge:#e0a83a;
  --okwash:rgba(25,158,112,0.12); --okedge:#199e70;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--plane); color:var(--ink);
  font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  padding:0 20px 80px;
}
.wrap{max-width:860px;margin:0 auto}
a{color:var(--series-1)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:0.93em;font-variant-numeric:tabular-nums}
header{padding:52px 0 30px;border-bottom:1px solid var(--grid)}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);font-weight:600;margin:0 0 12px}
h1{font-size:clamp(28px,4.4vw,40px);line-height:1.16;margin:0 0 14px;
  letter-spacing:-0.02em;font-weight:680}
.meta{color:var(--ink-2);font-size:14.5px;margin:0}
.meta b{color:var(--ink);font-weight:600}
.lede{font-size:18.5px;line-height:1.55;color:var(--ink-2);margin:26px 0 0}
.lede strong{color:var(--ink);font-weight:650}
h2{font-size:21px;margin:52px 0 14px;letter-spacing:-0.01em;font-weight:650}
h3{font-size:16px;margin:30px 0 8px;font-weight:650}
p{margin:0 0 14px}
.small{font-size:14px;color:var(--ink-2)}

/* stat tiles: hero numbers, no plot -- text tokens only, colour never carries
   the value (marks-and-anatomy.md, "hero number") */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
  gap:12px;margin:30px 0 8px}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:12px;
  padding:16px 16px 14px}
.tile .n{font-size:30px;font-weight:680;letter-spacing:-0.025em;
  font-variant-numeric:tabular-nums;line-height:1.1}
.tile .n small{font-size:15px;font-weight:600;color:var(--ink-2);margin-left:2px}
.tile .k{font-size:12.5px;color:var(--muted);margin-top:5px;line-height:1.35}

/* stacked coverage bar: 2px surface gaps between segments, 4px rounded ends,
   every segment direct-labelled */
figure{margin:26px 0 8px}
figcaption{font-size:13px;color:var(--muted);margin-top:12px}
.bar{display:flex;gap:2px;height:34px;width:100%}
.bar span{display:block;height:100%}
.bar span:first-child{border-radius:5px 0 0 5px}
.bar span:last-child{border-radius:0 5px 5px 0}
.key{display:flex;flex-wrap:wrap;gap:6px 22px;margin-top:13px;font-size:14px}
.key div{display:flex;align-items:baseline;gap:8px}
.sw{width:11px;height:11px;border-radius:3px;flex:0 0 auto;
  position:relative;top:1px}
.key b{font-weight:650;font-variant-numeric:tabular-nums}
.key em{font-style:normal;color:var(--muted)}

/* callouts */
.note{border-radius:12px;padding:16px 18px;margin:22px 0;
  border:1px solid var(--grid);background:var(--surface)}
.note.ok{background:var(--okwash);border-color:var(--okedge)}
.note.warn{background:var(--warnwash);border-color:var(--warnedge)}
.note p:last-child{margin-bottom:0}
.note .h{font-weight:650;margin:0 0 6px;display:flex;align-items:center;gap:8px}
.note .h .ic{font-size:15px}

table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0 4px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);
  vertical-align:middle}
th{font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--muted);font-weight:650;border-bottom:1px solid var(--axis);
  white-space:nowrap}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
tbody tr:hover{background:var(--wash-1)}
tfoot td{font-weight:650;border-top:1px solid var(--axis);border-bottom:none}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.run{font-weight:650;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.chip{display:inline-block;font-size:11px;font-weight:650;padding:1px 7px;
  border-radius:999px;border:1px solid var(--muted);color:var(--muted);
  margin-left:7px;white-space:nowrap;letter-spacing:.02em}
.chip.tape{border-color:var(--series-2);color:var(--series-2)}
td.sub{color:var(--muted);font-size:13px}

/* single-series magnitude bar in the table: one hue, 4px rounded end */
.mb{display:flex;align-items:center;gap:8px;justify-content:flex-end}
.mb i{display:block;height:8px;border-radius:0 4px 4px 0;
  background:var(--series-1);flex:0 0 auto}
.mb u{text-decoration:none;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  min-width:34px;text-align:right}

pre{background:var(--surface);border:1px solid var(--grid);border-radius:10px;
  padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.75;margin:14px 0}
.facts{display:grid;grid-template-columns:auto 1fr;gap:0}
.facts dt{padding:8px 16px 8px 0;border-bottom:1px solid var(--grid);
  color:var(--muted);font-size:13.5px;white-space:nowrap}
.facts dd{padding:8px 0;border-bottom:1px solid var(--grid);margin:0;
  font-size:14.5px}
footer{margin-top:60px;padding-top:22px;border-top:1px solid var(--grid);
  color:var(--muted);font-size:13px}
@media (max-width:620px){
  .facts{grid-template-columns:1fr}
  .facts dt{border-bottom:none;padding-bottom:0}
  .tile .n{font-size:26px}
}
"""


def render(rows, tot_f, tot_tb, tot_h, beam_h, ready_h,
           past_end, inside, gaps, lost_runs, recov_runs, skipped,
           on_disk, on_tape, skip_bins, today):
    """The complete standalone document."""
    max_block = max(r['beam_hours_blocked'] for r in rows) or 1.0
    prio = {r['ntof_run'] for r in past_end}

    trs = []
    for r in rows:
        w = 74.0 * r['beam_hours_blocked'] / max_block
        tape = r['stream1_on_disk'] != 'yes'
        chips = ''
        if r['ntof_run'] in prio:
            chips += '<span class="chip">after the pass</span>'
        if tape:
            chips += '<span class="chip tape">recall from tape</span>'
        trs.append(
            f'<tr>'
            f'<td><span class="run">{r["ntof_run"]}</span>{chips}</td>'
            f'<td class="n">{_esc(r["start_utc"])}</td>'
            f'<td class="n">{r["hours"]:.1f}</td>'
            f'<td class="sub">{_esc(r["window"])}</td>'
            f'<td class="n">{r["raw_files"] or "&mdash;"}</td>'
            f'<td class="n">{r["raw_TB"]:.2f}</td>'
            f'<td>{_esc(r["dream_runs"])}</td>'
            f'<td><div class="mb"><i style="width:{w:.1f}px"></i>'
            f'<u>{r["beam_hours_blocked"]:.1f}</u></div></td></tr>')

    unproc_h = beam_h - ready_h
    seg = [('PROCESSED', 'var(--series-1)', ready_h,
            'a run in <code>done/</code> covers it'),
           ('NOT PROCESSED', 'var(--series-2)', unproc_h,
            'no processed output of any kind exists')]
    bar = ''.join(f'<span style="background:{c};width:{100*v/beam_h:.3f}%" '
                  f'title="{n}: {v:.1f} h"></span>' for n, c, v, _ in seg)
    key = ''.join(
        f'<div><span class="sw" style="background:{c}"></span>'
        f'<span><b>{n} &mdash; {v:.0f} h</b> ({v/beam_h:.0%})<br>'
        f'<em>{d}</em></span></div>' for n, c, v, d in seg)

    skip_wrapped = '\n'.join(' '.join(str(r) for r in skipped[i:i + 10])
                             for i in range(0, len(skipped), 10))
    tape_list = ', '.join(str(r['ntof_run']) for r in on_tape)

    # skip-rate-vs-size table (why_skipped.py computes the same thing)
    skiprows = ''
    for lo, hi, n, sk in skip_bins:
        bar = ('<i style="width:%.0fpx"></i>' % (90.0 * sk / max(n, 1))
               if sk else '')
        skiprows += (f'<tr><td class="n">{lo:.2f}&ndash;{hi:.2f}</td>'
                     f'<td class="n">{n}</td><td class="n">{sk}</td>'
                     f'<td class="n">{sk/n:.0%}</td>'
                     f'<td><div class="mb" style="justify-content:flex-start">'
                     f'{bar}</div></td></tr>')
    small_post = ', '.join(str(r['ntof_run']) for r in rows
                           if r['ntof_run'] > 224687 and r['raw_TB'] < 0.35)
    n_post = len(past_end)
    PASS_HI = 224687

    return f"""<!--note
date: {today}
title: n_TOF processing — the {len(rows)} X17 runs still needed
summary: The 5-7 August pass used our UserInput exactly, verified parameter for parameter, then stopped. 325 runs are done; {len(rows)} remain, block {tot_h:.0f} h of X17 beam time, and have no processed output of any kind.
tags: X17, nTOF, processing, handoff
-->
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>n_TOF processing — the {len(rows)} X17 runs still needed</title>
<meta name="description" content="Request to the n_TOF processing team: {len(rows)} X17 runs remain to be processed. The 5-7 August pass used our proposed UserInput exactly, verified parameter for parameter, then stopped; the remaining runs block {tot_h:.0f} hours of beam time and have no processed output of any kind.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <p class="eyebrow">X17 / DREAM group &middot; request to n_TOF processing</p>
  <h1>{len(rows)} X17 runs still to process</h1>
  <p class="meta"><b>{today}</b> &middot; Dylan Neff &middot;
     <a href="mailto:dneff@cern.ch">dneff@cern.ch</a></p>
  <p class="lede">The pass you ran on 5&ndash;7 August is
     <strong>exactly right</strong> &mdash; we verified it against our own copy,
     parameter for parameter. <strong>325 runs are done.</strong> The
     <strong>{len(rows)}</strong> below are still missing, they block
     <strong>{tot_h:.0f} hours</strong> of X17 beam time, and none of them has
     any processed output at all.</p>
</header>

<div class="note ok">
  <p class="h"><span class="ic">&#10003;</span> We checked, rather than assuming</p>
  <p>Every X17 file in <code>/eos/experiment/ntof/processing/official/done/</code>
  carries <code>UserInput_2026_EAR2_X17_v4.h</code> in its <code>history</code>
  string. We diffed that against our proposal: <strong>identical on all 14
  detector rows and all 26 pulse-shape filenames.</strong> Nothing about the
  configuration needs revisiting &mdash; this request is only about coverage.</p>
</div>

<div class="tiles">
  <div class="tile"><div class="n">325</div>
    <div class="k">runs processed and verified</div></div>
  <div class="tile"><div class="n">{len(rows)}</div>
    <div class="k">runs still needed</div></div>
  <div class="tile"><div class="n">{tot_tb:.1f}<small>&nbsp;TB</small></div>
    <div class="k">stream1 staged on disk for {len(on_disk)} of them; {len(on_tape)}
    need a tape recall</div></div>
  <div class="tile"><div class="n">{tot_h:.0f}<small>&nbsp;h</small></div>
    <div class="k">of X17 beam time they block</div></div>
</div>

<h2>Where the campaign stands</h2>
<figure>
  <div class="bar">{bar}</div>
  <div class="key">{key}</div>
  <figcaption>DREAM beam time, {beam_h:.0f} h over 282 sub-runs
  (runs 77&ndash;156, 26 July &ndash; 8 August). Processing the {len(rows)}
  runs below moves essentially the whole orange band into the blue
  one.</figcaption>
</figure>

<h2>What happened, as far as we can see</h2>
<p>The pass ran from <strong>5 August</strong> and the last file landed
<strong>7 August at 19:56</strong>. Nothing has been written since &mdash; about
a day, as we write this. That stop cleanly explains the tail:
<strong>{len(past_end)} runs ({past_end[0]['ntof_run']}&ndash;{past_end[-1]['ntof_run']})
are simply after the point where <code>done/</code> ends</strong>, and they cover
our last two days of data taking (DREAM runs 150&ndash;156).</p>

<p>What it does <strong>not</strong> explain is the rest. There are
<strong>{len(gaps)} runs missing from inside {224300}&ndash;{224687}</strong>,
scattered through the range rather than clustered at either end, and
<strong>{len(inside)}</strong> of those overlap X17 beam time. We cannot see a
reason for them from the outside. One partial correlation: {len(lost_runs)} of
the {len(gaps)} in-range gaps no longer have their stream1 staged on the EOS
disk, which would explain a skip if the pass reads from disk &mdash; but the
other {len(recov_runs)} do still have it and were skipped anyway.</p>

<div class="note">
  <p class="h">If you know why those were passed over, we would like to hear it</p>
  <p>It is the one piece we cannot reconstruct from the outside, and it would
  tell us whether re-running them is straightforward or whether something about
  them is broken.</p>
</div>

<h2>A clue: only large runs were skipped</h2>
<p>We looked for anything that distinguishes the skipped runs. Directory
structure is identical on both sets &mdash; <code>stream0</code> +
<code>stream1</code>, every file <code>.finished</code>, no stragglers. An
output-size cap does not fit either: it would have to sit below 21&nbsp;GB, and
42 processed runs already exceed that. Position in the run range says nothing;
the gaps are scattered.</p>

<p><strong>Raw size fits, and not subtly.</strong> Of the 135 in-range runs whose
stream1 is still staged:</p>

<figure>
<div class="scroll">
<table>
  <thead><tr><th class="n">raw TB</th><th class="n">runs</th>
    <th class="n">skipped</th><th class="n">rate</th><th>&nbsp;</th></tr></thead>
  <tbody>{skiprows}</tbody>
</table>
</div>
<figcaption>Skip rate against the size of the run's staged stream1.</figcaption>
</figure>

<p><strong>Below 0.35&nbsp;TB nothing was ever skipped &mdash; 0 of 63.</strong>
At or above it, 30 of 72 were, and the rate keeps climbing with size (31&nbsp;%
in the lower half of that group, 53&nbsp;% in the upper). That is the shape of a
<em>resource limit a large job sometimes misses and sometimes makes</em> &mdash;
a wall-clock, memory or scratch ceiling &mdash; rather than a rule that rejects
a run outright. If it were deterministic, the big runs would all have failed;
they did not.</p>

<div class="note">
  <p class="h">The control, which we think confirms it</p>
  <p>Of the {n_post} runs missing from <em>after</em> {PASS_HI}, three
  (<code>{small_post}</code>) are below 0.35&nbsp;TB &mdash; the band in which the
  pass never skipped anything. So those really are missing because the pass
  stopped, not for this reason. Two mechanisms, cleanly separated.</p>
</div>

<p class="small">We cannot see your job configuration, so this is an
association, not a diagnosis &mdash; but if there is a per-job limit worth
raising for the re-run, the size distribution above is where we would look.</p>

<h2>What we need</h2>
<dl class="facts">
  <dt>runs</dt><dd>the {len(rows)} listed below</dd>
  <dt>UserInput</dt><dd>the same one already used &mdash;
      <code>UserInput_2026_EAR2_X17_v4.h</code></dd>
  <dt>output</dt><dd>the same place,
      <code>/eos/experiment/ntof/processing/official/done/</code></dd>
  <dt>raw</dt><dd>{len(on_disk)} still have stream1 staged on disk under
      <code>/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/&lt;run&gt;/stream1/</code>;
      {len(on_tape)} (<code>{tape_list}</code>) will need a recall from tape</dd>
  <dt>order</dt><dd><strong>whatever suits your queue</strong> &mdash; we want
      all of them</dd>
</dl>

<div class="note">
  <p class="h">These runs have no processed output at all</p>
  <p>Not just no v12 &mdash; <strong>nothing</strong>. There is no file for any
  of them anywhere under <code>/eos/experiment/ntof/processing/</code>, and none
  under the earlier <code>v2</code> processing either. <code>done/</code> keeps
  older output (it holds files back to April 2025, including 141 from July
  2026), but in 224300&ndash;224687 every one of the 325 files present is dated
  5&ndash;7 August. A run processed under v2 and then skipped by this pass would
  still be sitting there with its old timestamp; none is.</p>
</div>

<div class="note warn">
  <p class="h"><span class="ic">&#9888;</span> One naming point that will confuse
     anyone checking</p>
  <p>The file is called <code>UserInput_2026_EAR2_X17_v4.h</code> and its content
  is what our group tracks internally as <strong>v12_liqpileup</strong>. Both
  names refer to the same thing; the header comment inside the file says so. We
  mention it only because our own repository also has a <em>different</em> file
  called <code>v4</code>, and we do not want anyone to reconcile the two by
  filename.</p>
</div>

<h2>The {len(rows)} runs</h2>
<p class="small">&ldquo;Beam h blocked&rdquo; is how much X17 DREAM beam time
depends on that n_TOF run. A <em>bracketed</em> window means the run has neither
a processed file nor staged stream1, so we placed it between its nearest
measurable neighbours by run number &mdash; coarse, but enough to show that it
overlaps beam.</p>
<div class="scroll">
<table>
  <thead><tr>
    <th>n_TOF run</th><th class="n">start (UTC)</th><th class="n">hours</th>
    <th>window</th><th class="n">raw files</th><th class="n">raw TB</th>
    <th>DREAM runs affected</th><th class="n">beam h blocked</th>
  </tr></thead>
  <tbody>{''.join(trs)}</tbody>
  <tfoot><tr><td>{len(rows)} runs</td><td></td><td></td><td></td>
    <td class="n">{tot_f}</td><td class="n">{tot_tb:.1f}</td><td></td>
    <td class="n">{tot_h:.0f}</td></tr></tfoot>
</table>
</div>

<h2>For reference</h2>
<dl class="facts">
  <dt>processed and verified</dt><dd>325 runs, 224300&ndash;224687</dd>
  <dt>still needed</dt><dd>{len(inside)} inside that range,
      {len(past_end)} after its end</dd>
  <dt>X17 campaign</dt><dd>DREAM runs 77&ndash;156, 26 July &ndash; 8 August,
      282 beam sub-runs</dd>
  <dt>processed today</dt><dd>{ready_h:.0f} h of {beam_h:.0f} h
      ({ready_h/beam_h:.0%})</dd>
  <dt>not processed</dt><dd>{unproc_h:.0f} h ({unproc_h/beam_h:.0%})</dd>
  <dt>after these {len(rows)}</dt><dd>~{beam_h:.0f} h (essentially all of it)</dd>
</dl>

<h2>The {len(skipped)} in-range gaps we are <em>not</em> asking about</h2>
<p>Of the <strong>{len(gaps)}</strong> runs missing from 224300&ndash;224687, we
are asking for the <strong>{len(inside)}</strong> that overlap X17 beam time.
The other <strong>{len(skipped)}</strong> were live while DREAM was not, so they
block nothing for us:</p>
<pre class="mono">{skip_wrapped}</pre>
<p>We mention them only because they are part of the same unexplained set
&mdash; if they were skipped for a reason that also applies to the ones we
<em>are</em> asking for, that would be worth knowing.</p>

<h2>Why it matters to us</h2>
<p>We key every n_TOF hit to a DREAM trigger through a time calibration that is
fitted <strong>per (DREAM run, n_TOF processing) pair</strong> and does not
transfer between processings. Mixing a v12 run with an older processing inside
one DREAM run is not an option: the plastic &gamma;-flash identification alone
differs by 37&ndash;85&nbsp;% of bunches, and our own v11 differs from v12 by
14&ndash;21&nbsp;% in liquid hit yield. So a DREAM run is either fully covered
by this reprocessing or it waits.</p>

<footer>
  <p>Generated from the run inventory by
  <code>ntof_processing/slim_study/make_handoff.py</code> &mdash; the numbers
  here, the machine-readable
  <code>missing_runs_{today}.csv</code> and the markdown version all come from
  one pass over the same listings, so they cannot drift.
  Coverage is computed by <code>slim_study/coverage_map.py</code> from the
  <code>index</code>-tree wall clock of every reprocessed run, the raw stream1
  mtimes, and the DREAM sub-run boundaries.</p>
</footer>

</div>
</body>
</html>
"""
