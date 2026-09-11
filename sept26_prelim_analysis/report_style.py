#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_style.py -- the one stylesheet every report in this package wears.

Until now the CSS lived inside ``make_funnel_report`` and everything else
imported it from there.  That worked while the funnel report was the only one;
it stopped working the moment the newer reports grew their own markup.  Five of
the reports written on 2026-09-10 rendered as unstyled documents:

  * they wrap their body in ``<main>``, and nothing styled ``main`` -- so the
    text ran the full width of the monitor with no column at all;
  * they emit bare ``<table>``, and only ``table.t`` was styled -- so all 41
    tables were browser defaults, unaligned, hairline-ruled, left-ragged;
  * ``.deck``, ``.verdict``, ``.prov``, ``.foot`` and pandas' ``.dataframe``
    had no rules anywhere.

So the stylesheet moved here and became **element-first**: a report gets the
layout by using ordinary tags, and the classes are refinements on top rather
than the price of admission.  Nothing that used to be styled stopped being
styled -- every class the older reports emit (``.wrap``, ``.t``, ``.cards``,
``.caution``, ``.panel``, ``.legend``, ``.tile*``, ``.eyebrow``, ``.badge``)
is still here and unchanged in meaning.

:data:`SCRIPT` is the other half.  It is ~2 kB of dependency-free JavaScript
that repairs the markup in the browser rather than in fourteen generators:
wraps bare tables so they scroll instead of overflowing, right-aligns numeric
cells, gives every heading an id and a quiet anchor, builds a sticky contents
rail out of the ``<h2>``s, and carries a light/dark/auto toggle.  A report that
is opened with JavaScript off loses the rail and the toggle and keeps
everything else, which is the right way round.

Design, deliberately: the figures are document figures (see ``figstyle``), and
so is this -- one measured column, a real type scale, restrained rules, no
slide-deck contrast.  Plex Sans and Plex Mono, tabular numerals everywhere
digits stack, and a palette with proper dark-mode counterparts, because these
pages are read in a control room at three in the morning as often as not.

Use :data:`HEAD` -- fonts, stylesheet and script in one -- in the ``<head>`` of
a report:

    from sept26_prelim_analysis.report_style import HEAD
    ...
    f'<title>...</title>{HEAD}</head><body><main>'
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
# IBM Plex Sans / Plex Mono -- an instrumentation face, drawn for technical
# documentation; the mono carries every count, unit and channel name.  Real
# fallback stacks, because the control-room browser may be offline.
FONT_LINK = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500;600&'
             'family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">')

# --------------------------------------------------------------------------- #
# The stylesheet
# --------------------------------------------------------------------------- #
CSS = """
:root{
  color-scheme:light dark;
  --bg:#fbfafc; --bg-2:#f3f1f6; --panel:#ffffff; --panel-2:#faf9fc;
  --ink:#1c1a24; --ink-2:#565064; --ink-3:#8b8599;
  --line:#e5e1ec; --line-2:#d3cddf;
  --accent:#7d3a86; --accent-2:#a15fa8; --accent-wash:#7d3a860e;
  --warn:#9c6414; --warn-wash:#9c641412; --warn-line:#e2c795;
  --good:#0d6a8c; --good-wash:#0d6a8c12;
  --shadow:0 1px 2px #1c1a240a, 0 6px 20px -12px #1c1a2426;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --col:74ch;         /* the reading column */
  --page:1120px;      /* the widest a figure or table may go */
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#131118; --bg-2:#1a1822; --panel:#1b1924; --panel-2:#201d2b;
  --ink:#eceaf3; --ink-2:#a9a3ba; --ink-3:#7c7591;
  --line:#2d2939; --line-2:#3b3650;
  --accent:#c98fce; --accent-2:#d7a9db; --accent-wash:#c98fce14;
  --warn:#e0a95f; --warn-wash:#e0a95f14; --warn-line:#5a4526;
  --good:#67b6d8; --good-wash:#67b6d814;
  --shadow:0 1px 2px #0006, 0 8px 24px -14px #000a;
}}
:root[data-theme="dark"]{
  --bg:#131118; --bg-2:#1a1822; --panel:#1b1924; --panel-2:#201d2b;
  --ink:#eceaf3; --ink-2:#a9a3ba; --ink-3:#7c7591;
  --line:#2d2939; --line-2:#3b3650;
  --accent:#c98fce; --accent-2:#d7a9db; --accent-wash:#c98fce14;
  --warn:#e0a95f; --warn-wash:#e0a95f14; --warn-line:#5a4526;
  --good:#67b6d8; --good-wash:#67b6d814;
  --shadow:0 1px 2px #0006, 0 8px 24px -14px #000a;
}

*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:74px}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *{transition:none!important;animation:none!important}
}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15.5px;line-height:1.65;margin:0;padding:0;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
  font-variant-numeric:tabular-nums}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:4px}

/* --- the page column ---------------------------------------------------- */
/* main and .wrap are the same thing; older reports use one, newer the other. */
main,.wrap{max-width:var(--page);margin:0 auto;padding:0 24px 96px;
  display:block}
main>*,.wrap>*{max-width:var(--col)}
/* the wide things -- anything with its own frame -- get the full page */
main>figure,main>table,main>.scroll,main>.cards,main>.tiles,main>.panel,
main>.grid,main>.figrow,main>.tbl-wrap,main>hr,main>footer,main>header,
.wrap>figure,.wrap>table,.wrap>.scroll,.wrap>.cards,.wrap>.tiles,.wrap>.panel,
.wrap>.grid,.wrap>.figrow,.wrap>.tbl-wrap,.wrap>hr,.wrap>footer,.wrap>header{
  max-width:none}

/* --- the sticky bar and the contents rail (both built by SCRIPT) --------- */
.topbar{position:sticky;top:0;z-index:40;background:color-mix(in srgb,var(--bg) 86%,transparent);
  backdrop-filter:saturate(1.6) blur(10px);-webkit-backdrop-filter:saturate(1.6) blur(10px);
  border-bottom:1px solid var(--line)}
.topbar-in{max-width:var(--page);margin:0 auto;padding:9px 24px;
  display:flex;align-items:center;gap:14px}
.topbar .who{font-family:var(--mono);font-size:11px;font-weight:500;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);
  white-space:nowrap}
.topbar .now{font-size:13px;color:var(--ink-2);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
.topbar .now b{color:var(--ink);font-weight:600}
.themer{appearance:none;background:var(--panel);color:var(--ink-2);
  border:1px solid var(--line-2);border-radius:999px;cursor:pointer;
  font-family:var(--mono);font-size:11px;letter-spacing:.06em;padding:4px 11px;
  white-space:nowrap;transition:color .15s,border-color .15s}
.themer:hover{color:var(--accent);border-color:var(--accent-2)}

.toc{display:none}
@media (min-width:1560px){
  /* Sized from the gutter that is actually there, so it never runs off the
     right edge on a 1600-wide window. */
  .toc{display:block;position:fixed;top:96px;left:calc(50% + var(--page)/2 + 28px);
    width:min(220px,calc(50vw - var(--page)/2 - 46px));
    max-height:calc(100vh - 140px);overflow-y:auto;
    font-size:12.5px;line-height:1.45;border-left:1px solid var(--line);
    padding-left:14px}
  .toc h6{margin:0 0 9px;font-family:var(--mono);font-size:10px;font-weight:500;
    letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3)}
  .toc a{display:block;padding:4px 0;color:var(--ink-2);text-decoration:none;
    border-left:2px solid transparent;margin-left:-16px;padding-left:14px;
    transition:color .15s,border-color .15s}
  .toc a:hover{color:var(--ink)}
  .toc a.on{color:var(--accent);border-left-color:var(--accent);font-weight:500}
}

/* --- headings ----------------------------------------------------------- */
h1{font-size:clamp(28px,3.4vw,36px);line-height:1.14;margin:46px 0 14px;
  font-weight:600;letter-spacing:-.024em;text-wrap:balance;max-width:26ch}
h2{font-size:21.5px;margin:64px 0 4px;letter-spacing:-.016em;font-weight:600;
  text-wrap:balance;padding-top:16px;border-top:1px solid var(--line);
  scroll-margin-top:78px;display:flex;align-items:baseline;gap:13px;
  position:relative}
h2 .n{font-family:var(--mono);font-size:13px;font-weight:500;color:var(--accent);
  letter-spacing:.06em;flex:none}
h3{font-size:16.5px;margin:34px 0 4px;font-weight:600;letter-spacing:-.008em;
  scroll-margin-top:78px;position:relative}
h4{font-size:13px;margin:26px 0 4px;font-family:var(--mono);font-weight:500;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
/* the quiet anchor SCRIPT hangs off every heading */
.anch{position:absolute;left:-.95em;width:.95em;color:var(--ink-3);
  text-decoration:none;opacity:0;font-weight:400;transition:opacity .15s;
  line-height:inherit}
h2:hover .anch,h3:hover .anch,.anch:focus{opacity:.65}
.anch:hover{color:var(--accent)}

/* --- prose -------------------------------------------------------------- */
p{margin:13px 0}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
a:hover{color:var(--accent-2)}
ul,ol{margin:13px 0;padding-left:22px}
li{margin:7px 0}
li::marker{color:var(--ink-3)}
b,strong{font-weight:600}
i,em{font-style:italic}
hr{border:0;border-top:1px solid var(--line);margin:44px 0}
small{font-size:12.5px;color:var(--ink-2)}
sub,sup{font-size:.72em}
code{font-family:var(--mono);font-size:.86em;background:var(--bg-2);
  border:1px solid var(--line);border-radius:4px;padding:.08em .28em;
  word-break:break-word}
a code{color:inherit}
pre{background:var(--panel);border:1px solid var(--line);border-radius:9px;
  padding:15px 17px;overflow-x:auto;font-size:12.5px;line-height:1.6}
pre code{background:none;border:0;padding:0;font-size:inherit}

/* The opening paragraph: .lede and .deck are the same promise, and both
   reports' authors reached for a different word.  Style both. */
.lede,.deck{font-size:18px;line-height:1.56;margin:22px 0 6px;color:var(--ink);
  max-width:66ch;letter-spacing:-.004em}
.note,.prov{color:var(--ink-2);font-size:13.5px;line-height:1.62}
.prov{font-family:var(--mono);font-size:12px}

/* --- the answer, up top ------------------------------------------------- */
/* .verdict is the newer reports' name, .caution the older one's warning box. */
.verdict{background:linear-gradient(var(--accent-wash),var(--accent-wash)),var(--panel);
  border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:4px 10px 10px 4px;padding:20px 24px;margin:26px 0 34px;
  box-shadow:var(--shadow);max-width:84ch}
.verdict>:first-child{margin-top:0}.verdict>:last-child{margin-bottom:0}
.verdict p{max-width:72ch}
.verdict b{font-weight:600}
.caution,.warn{background:linear-gradient(var(--warn-wash),var(--warn-wash)),var(--panel);
  border:1px solid var(--warn-line);border-left:3px solid var(--warn);
  border-radius:4px 10px 10px 4px;padding:16px 20px;margin:24px 0;font-size:14.5px}
.caution b,.warn b{color:var(--warn);font-weight:600}
.caution>:first-child,.warn>:first-child{margin-top:0}
.caution>:last-child,.warn>:last-child{margin-bottom:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:20px 22px;margin:20px 0;box-shadow:var(--shadow)}
.panel>:first-child{margin-top:0}.panel>:last-child{margin-bottom:0}

/* --- number cards ------------------------------------------------------- */
.cards,.tiles{display:grid;
  grid-template-columns:repeat(auto-fit,minmax(196px,1fr));
  gap:1px;margin:26px 0;background:var(--line);border:1px solid var(--line);
  border-radius:10px;overflow:hidden;box-shadow:var(--shadow)}
.card,.tile{background:var(--panel);padding:17px 19px;border:0}
.card .v,.tile-v{font-family:var(--mono);font-size:26px;font-weight:600;
  letter-spacing:-.032em;line-height:1.12;font-variant-numeric:tabular-nums;
  display:block}
.card .l,.tile-k{font-size:12.5px;color:var(--ink-2);margin-top:6px;
  line-height:1.42;display:block}
.tile-s{font-size:11.5px;color:var(--ink-3);margin-top:3px;display:block;
  font-family:var(--mono)}

/* --- tables ------------------------------------------------------------- */
/* Bare <table> is styled: the generators emit it and should not have to
   remember a class.  .t and .num stay as no-ops for the older reports. */
.scroll,.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:20px 0;
  border:1px solid var(--line);border-radius:10px;background:var(--panel);
  box-shadow:var(--shadow)}
.scroll>table,.tbl-wrap>table{border:0;border-radius:0;margin:0;box-shadow:none;
  min-width:100%}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:20px 0;
  background:var(--panel);border:1px solid var(--line);border-radius:10px}
caption{caption-side:top;text-align:left;font-size:12.5px;color:var(--ink-2);
  padding:0 0 9px;font-family:var(--mono)}
th,td{padding:9px 14px;text-align:left;vertical-align:top;
  border-bottom:1px solid var(--line)}
thead th{position:sticky;top:0;z-index:1;background:var(--panel-2);
  font-family:var(--mono);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.11em;color:var(--ink-3);font-weight:500;white-space:nowrap;
  border-bottom:1.5px solid var(--line-2)}
thead th .u{font-family:var(--sans);font-weight:400;text-transform:none;
  letter-spacing:0;font-size:10.5px}
tbody tr:nth-child(even) td,tbody tr:nth-child(even)>th{background:var(--panel-2)}
tbody tr:hover td,tbody tr:hover>th{background:var(--accent-wash)}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
tbody th{font-weight:500}
td.n,th.n,td.num,thead th.n{text-align:right;font-family:var(--mono);
  font-variant-numeric:tabular-nums;font-weight:500;white-space:nowrap}
table.num td+td,table.num th+th{text-align:right;
  font-variant-numeric:tabular-nums}
th.s{font-weight:500;white-space:normal;max-width:34ch}
.why{display:block;font-weight:400;color:var(--ink-3);font-size:11.5px;
  line-height:1.42;margin-top:3px}
tfoot td,tfoot th{font-size:12px;color:var(--ink-2);border-top:1.5px solid var(--line-2);
  border-bottom:0}
/* pandas .to_html() output, when a report takes the shortcut */
table.dataframe{font-size:13px}
table.dataframe th{text-align:left}
table.dataframe tbody th{font-family:var(--mono);font-size:12px;color:var(--ink-2)}
table.dataframe td{text-align:right;font-family:var(--mono);
  font-variant-numeric:tabular-nums}

.legend{margin-top:15px;font-family:var(--mono);font-size:11.5px;
  color:var(--ink-2);display:flex;flex-wrap:wrap;gap:7px 22px}
.legend .k{white-space:nowrap;display:inline-flex;align-items:center}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;
  margin-right:7px}

/* --- figures ------------------------------------------------------------ */
figure{margin:30px 0;padding:0}
figure a{display:block;border-radius:10px;overflow:hidden;
  border:1px solid var(--line);background:#fff;box-shadow:var(--shadow);
  transition:border-color .16s,box-shadow .16s}
figure a:hover{border-color:var(--line-2);
  box-shadow:0 2px 4px #1c1a2412,0 14px 34px -18px #1c1a2440}
figure img{display:block;width:100%;height:auto;background:#fff}
figure>img{border:1px solid var(--line);border-radius:10px;
  box-shadow:var(--shadow)}
figcaption{font-size:13px;color:var(--ink-2);margin-top:11px;max-width:80ch;
  line-height:1.58}
figcaption b{color:var(--ink);font-weight:600}
figcaption .src{font-family:var(--mono);font-size:11px;color:var(--ink-3);
  text-decoration:none;white-space:nowrap;border:1px solid var(--line);
  border-radius:999px;padding:1px 8px;margin-left:5px;
  display:inline-block;transition:color .15s,border-color .15s}
figcaption .src:hover{color:var(--accent);border-color:var(--accent-2)}
.figrow{display:grid;gap:22px;margin:30px 0;
  grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.figrow figure{margin:0}
svg text{font-family:var(--mono);font-variant-numeric:tabular-nums}

/* --- odds and ends the reports reach for -------------------------------- */
/* a headline number standing alone on its own line */
.big{font-size:1.32em;font-weight:600;margin:1.05em 0;letter-spacing:-.015em;
  font-variant-numeric:tabular-nums}
.badge{display:inline-block;background:var(--accent);color:#fff;font-size:10.5px;
  font-weight:600;letter-spacing:.13em;padding:3px 9px;border-radius:999px;
  font-family:var(--mono);text-transform:uppercase}
.eyebrow{font-family:var(--mono);font-size:11.5px;font-weight:500;
  text-transform:uppercase;letter-spacing:.14em;color:var(--ink-3);
  display:flex;flex-wrap:wrap;gap:8px 18px;margin-bottom:14px;align-items:center}
/* .sub means two things in this package: the mono strap line under the title
   (funnel, tracking QA) and a small prose aside under a table (fold).  The
   first is the one that wants the mono, and it is the one next to the h1. */
.sub{color:var(--ink-2);font-size:13.5px;line-height:1.6;margin:10px 0}
h1+.sub,header .sub{font-family:var(--mono);font-size:13px;line-height:1.75;
  margin:0}
header{padding:44px 0 16px;border-bottom:2px solid var(--ink);margin-bottom:30px}
header h1{margin-top:0}

/* --- the colophon ------------------------------------------------------- */
footer,.foot{margin-top:64px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--ink-3);font-size:12px;font-family:var(--mono);line-height:1.85;
  max-width:none}
footer code,.foot code{background:none;border:0;padding:0;color:var(--ink-2);
  word-break:break-all}

/* --- narrow ------------------------------------------------------------- */
@media (max-width:640px){
  body{font-size:15px}
  main,.wrap{padding:0 16px 72px}
  .topbar-in{padding:8px 16px}
  h1{margin-top:30px}
  h2{margin-top:46px}
  .lede,.deck{font-size:16.5px}
  .verdict{padding:16px 18px}
  th,td{padding:8px 11px}
}

/* --- print -------------------------------------------------------------- */
@media print{
  :root{--bg:#fff;--panel:#fff;--panel-2:#fff;--ink:#000;--ink-2:#333;
    --ink-3:#666;--line:#ccc;--line-2:#999;--shadow:none;--accent-wash:transparent}
  .topbar,.toc,.anch{display:none!important}
  body{font-size:10.5pt}
  main,.wrap{max-width:none;padding:0}
  figure,table,.verdict,.panel,.cards{break-inside:avoid;box-shadow:none}
  h2,h3{break-after:avoid}
  a{color:inherit;text-decoration:none}
}
"""

# --------------------------------------------------------------------------- #
# The progressive-enhancement layer
# --------------------------------------------------------------------------- #
# Everything here is a repair or an affordance, never content: with the script
# stripped the report still reads top to bottom.  It is deliberately one
# self-contained block with no dependency -- these pages are opened from a file
# path, from the DAQ's Analysis tab and from the CERN web site, and only the
# last of those can be assumed to reach a CDN.
SCRIPT = """<script>
(function(){
  'use strict';
  var D = document;

  function ready(fn){
    if (D.readyState !== 'loading') fn();
    else D.addEventListener('DOMContentLoaded', fn);
  }

  // A cell is numeric if what it holds is a number, possibly signed, with the
  // usual decorations: thousands commas, a unit, a +-, a percent, an en dash
  // for "no value".  Deliberately strict -- a run name like run_145 must not
  // be dragged to the right.
  var NUMISH = /^[+\\u2212\\u00b1\\-]?[\\d,]+(\\.\\d+)?([\\s\\u00a0\\u2009]*(%|\\u00b1[\\s\\u00a0\\u2009]*[\\d.,]+|[a-z\\u00b5\\u03c3\\u00b0/]{1,8}))?$/i;
  function numeric(s){
    s = (s || '').trim();
    if (!s || s === '\\u2014' || s === '\\u2013') return false;
    return NUMISH.test(s);
  }

  function fixTables(){
    var tables = D.querySelectorAll('main table, .wrap table');
    Array.prototype.forEach.call(tables, function(t){
      // Give it a scroll frame so a wide table never widens the page.
      var p = t.parentNode;
      if (!p || !(p.classList.contains('scroll') || p.classList.contains('tbl-wrap'))){
        var box = D.createElement('div');
        box.className = 'scroll';
        p.insertBefore(box, t);
        box.appendChild(t);
      }
      // Right-align the columns that are actually numbers.  Only touch cells
      // that carry no class of their own -- a report that said td.n or th.s
      // has already decided.
      var rows = t.tBodies.length ? t.tBodies[0].rows : [];
      if (!rows.length) return;
      var ncol = 0;
      Array.prototype.forEach.call(rows, function(r){
        ncol = Math.max(ncol, r.cells.length);
      });
      for (var c = 0; c < ncol; c++){
        var seen = 0, num = 0, free = [];
        for (var i = 0; i < rows.length; i++){
          var cell = rows[i].cells[c];
          if (!cell) continue;
          var txt = cell.textContent;
          if (!txt.trim()) continue;
          seen++;
          // A cell the generator already called .n counts as a number without
          // being re-read; only unclassed cells are ours to relabel.
          if (cell.classList.contains('n') || cell.classList.contains('num')) num++;
          else if (!cell.className){ free.push(cell); if (numeric(txt)) num++; }
        }
        // A column is a number column when essentially all of it is numbers.
        if (seen >= 2 && num >= seen - 1 && num > 0){
          free.forEach(function(cell){ cell.classList.add('n'); });
          // The header has to move with the column, or the label sits over
          // empty space while the digits hug the far edge.
          if (t.tHead) Array.prototype.forEach.call(t.tHead.rows, function(hr){
            var h = hr.cells[c];
            if (h && !h.className && h.colSpan === 1) h.classList.add('n');
          });
        }
      }
    });
  }

  function slug(s){
    return (s || '').toLowerCase().replace(/[^\\w\\s-]/g, '')
                    .trim().replace(/\\s+/g, '-').slice(0, 48) || 'section';
  }

  // Headings get a stable id and a quiet anchor, so a number in a report can
  // be linked to from the board or from a message.
  function fixHeadings(){
    var seen = {};
    var hs = D.querySelectorAll('main h2, main h3, .wrap h2, .wrap h3');
    Array.prototype.forEach.call(hs, function(h){
      if (!h.id){
        var base = slug(h.textContent), id = base, k = 2;
        while (seen[id] || D.getElementById(id)) { id = base + '-' + (k++); }
        h.id = id;
      }
      seen[h.id] = 1;
      if (!h.querySelector('.anch')){
        var a = D.createElement('a');
        a.className = 'anch';
        a.href = '#' + h.id;
        a.setAttribute('aria-hidden', 'true');
        a.tabIndex = -1;
        a.textContent = '\\u00a7';
        h.insertBefore(a, h.firstChild);
      }
    });
    return hs;
  }

  function theme(){
    var KEY = 'x17-report-theme';
    var order = ['auto', 'light', 'dark'];
    var cur = 'auto';
    try { cur = localStorage.getItem(KEY) || 'auto'; } catch (e) {}
    var btn = D.createElement('button');
    btn.className = 'themer';
    btn.type = 'button';
    function paint(){
      if (cur === 'auto') D.documentElement.removeAttribute('data-theme');
      else D.documentElement.setAttribute('data-theme', cur);
      btn.textContent = ({auto: 'auto', light: 'light', dark: 'dark'})[cur];
      btn.setAttribute('aria-label', 'colour theme: ' + cur);
    }
    btn.addEventListener('click', function(){
      cur = order[(order.indexOf(cur) + 1) % order.length];
      try { localStorage.setItem(KEY, cur); } catch (e) {}
      paint();
    });
    paint();
    return btn;
  }

  function topbar(hs){
    var h1 = D.querySelector('h1');
    var bar = D.createElement('div');
    bar.className = 'topbar';
    var inner = D.createElement('div');
    inner.className = 'topbar-in';
    var who = D.createElement('span');
    who.className = 'who';
    who.textContent = 'x17 \\u00b7 n_TOF 2026';
    var now = D.createElement('span');
    now.className = 'now';
    now.innerHTML = '<b>' + (h1 ? h1.textContent.trim() : D.title) + '</b>';
    inner.appendChild(who);
    inner.appendChild(now);
    inner.appendChild(theme());
    bar.appendChild(inner);
    D.body.insertBefore(bar, D.body.firstChild);

    // Once past the title, the bar says which section you are in.
    if (!hs.length) return;
    var title = now.innerHTML;
    var last = null;
    var io = ('IntersectionObserver' in window) ? null : undefined;
    window.addEventListener('scroll', function(){
      var y = window.scrollY + 90, at = null;
      Array.prototype.forEach.call(hs, function(h){
        if (h.tagName === 'H2' && h.getBoundingClientRect().top + window.scrollY <= y) at = h;
      });
      var txt = at ? '<b>' + at.textContent.replace('\\u00a7', '').trim() + '</b>' : title;
      if (txt !== last){ now.innerHTML = txt; last = txt; }
      var id = at ? at.id : null;
      Array.prototype.forEach.call(D.querySelectorAll('.toc a'), function(a){
        a.classList.toggle('on', a.getAttribute('href') === '#' + id);
      });
    }, {passive: true});
  }

  function toc(hs){
    var h2 = Array.prototype.filter.call(hs, function(h){ return h.tagName === 'H2'; });
    if (h2.length < 3) return;
    var nav = D.createElement('nav');
    nav.className = 'toc';
    nav.setAttribute('aria-label', 'Contents');
    var html = '<h6>Contents</h6>';
    h2.forEach(function(h){
      html += '<a href="#' + h.id + '">' +
              h.textContent.replace('\\u00a7', '').trim() + '</a>';
    });
    nav.innerHTML = html;
    D.body.appendChild(nav);
  }

  ready(function(){
    try { fixTables(); } catch (e) {}
    var hs = [];
    try { hs = fixHeadings(); } catch (e) {}
    try { toc(hs); } catch (e) {}
    try { topbar(hs); } catch (e) {}
  });
})();
</script>"""

#: Everything a report needs in its ``<head>``, in one substitution.
HEAD = f'{FONT_LINK}<style>{CSS}</style>{SCRIPT}'


def head(title: str) -> str:
    """A complete ``<head>`` for a standalone report, title included.

    >>> head('Tracking QA').startswith('<meta charset')
    True
    """
    return ('<meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light dark">'
            f'<title>{title}</title>{HEAD}')


if __name__ == '__main__':
    print(f'CSS    {len(CSS):,} chars')
    print(f'SCRIPT {len(SCRIPT):,} chars')
    print(f'HEAD   {len(HEAD):,} chars')
