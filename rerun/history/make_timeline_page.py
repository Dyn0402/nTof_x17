#!/usr/bin/env python3
"""make_timeline_page.py [--out DIR] -- the interactive analysis timeline (one self-contained HTML page) from
timeline_data.py: zoom / pan, filters by status and area, search, a detail panel, an overview strip and a list.

    python rerun/history/make_timeline_page.py
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py rerun/history/out/mx17-analysis-timeline-explorer.html \
        --slug mx17-analysis-timeline-explorer --force --deploy

State lives in the URL hash (#t0=2026-10-20&t1=2026-10-29&sel=97&q=bow&group=status), so a view can be linked.
"""
import argparse
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import timeline_data as td                # noqa: E402

STATUS = {  # key: (label, light colour, dark colour, meaning)
    'chain': ('in the chain', '#2f8a5b', '#5cc28d', 'built into the code / registry by default (no switch)'),
    'down': ('downstream step', '#1f5fa8', '#6aa6e8', 'a step after the pass (S3-S6): the pass tables are enough'),
    'open': ('open: tonight', '#c63d4f', '#e8798a', 'not yet in the chain: a step of PASS_PLAN_2026-10-28'),
    'fixed': ('fixed today', '#b8860b', '#e0b84a', 'was left behind; fixed while compiling the history'),
    'lim': ('stated limitation', '#6f7886', '#a3abb9', 'cannot be answered from the existing data'),
    'doc': ('rule / record', '#7a52b8', '#b195e3', 'a rule or decision that lives in a document'),
    'sup': ('superseded', '#a39c90', '#8a857c', 'replaced by a later lesson'),
}
ORDER = ['chain', 'down', 'open', 'fixed', 'lim', 'doc', 'sup']


def data():
    les = [dict(id=i, d=d, a=a, t=t, w=w, s=s) for i, (d, a, t, w, s) in enumerate(sorted(td.LESSONS, key=lambda L: L[0]))]
    return dict(
        lessons=les,
        milestones=[dict(d=d, t=t, a=a) for d, t, a in td.MILESTONES],
        wrong=[dict(d0=a, t=c, d1=b, by=w) for a, c, b, w in td.WITHDRAWN],
        found=[dict(id=i, t=t, v=v, act=a) for i, t, v, a in td.FOUND],
        periods=[dict(d0=a, d1=b, t=t) for a, b, t in td.PERIODS],
        areas=td.AREAS,
        status={k: dict(label=v[0], c=v[1], cd=v[2], m=v[3]) for k, v in STATUS.items()},
        order=[k for k in ORDER if any(L[4] == k for L in td.LESSONS)],
        built=dt.date.today().isoformat(),
    )


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MX17 analysis timeline</title>
<meta name="description" content="Interactive timeline of the MX17 bench + SPS analysis: every dated lesson and how the reprocessing chain carries it.">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>
:root{--bg:#f5f3ee;--card:#fffdf9;--ink:#1c2230;--mut:#5a6170;--rule:#d8d3c8;--band:#ebe8e1;--hi:#1c2230;--sel:#c8601a;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151a24;--card:#1d2330;--ink:#e9ecf1;--mut:#a3abb9;--rule:#333b4a;--band:#1a202b;--hi:#e9ecf1;--sel:#f0a060;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#151a24;--card:#1d2330;--ink:#e9ecf1;--mut:#a3abb9;--rule:#333b4a;--band:#1a202b;--hi:#e9ecf1;--sel:#f0a060;color-scheme:dark}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--ink);font:15px/1.45 'IBM Plex Sans',Helvetica,Arial,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:26px;font-weight:600;letter-spacing:-.3px}
.sub{color:var(--mut);margin-top:4px}
.stats{display:flex;flex-wrap:wrap;gap:10px 26px;margin:14px 0 6px}
.stat b{font-size:22px;display:block;line-height:1.1}
.stat span{color:var(--mut);font-size:13px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}
.bar .lab{color:var(--mut);font-size:13px;margin-right:2px;min-width:52px}
.chip{border:1px solid var(--rule);background:var(--card);color:var(--ink);border-radius:14px;padding:3px 11px;font:inherit;font-size:13px;display:inline-flex;align-items:center;gap:6px}
.chip .dot{width:10px;height:10px;border-radius:50%}
.chip.off{opacity:.38}
.chip.on{border-color:var(--ink)}
input[type=search]{font:inherit;padding:5px 10px;border:1px solid var(--rule);border-radius:8px;background:var(--card);color:var(--ink);width:260px;max-width:100%}
.main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;margin-top:14px}
@media (max-width:980px){.main{grid-template-columns:minmax(0,1fr)}}
.panel{background:var(--card);border:1px solid var(--rule);border-radius:12px}
#tl{position:relative;overflow:hidden;touch-action:pan-y}
#tl svg{display:block;user-select:none}
#ov{border-top:1px solid var(--rule)}
#ov svg{display:block}
.hint{color:var(--mut);font-size:12px;padding:6px 12px;border-top:1px solid var(--rule);display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px}
#detail{padding:16px;align-self:start;position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto}
#detail h3{font-size:13px;color:var(--mut);font-weight:400;text-transform:uppercase;letter-spacing:1.5px}
#detail .big{font-size:18px;font-weight:600;margin:6px 0 10px;line-height:1.35}
#detail .kv{margin-top:8px}
#detail .kv .k{display:block;color:var(--mut);font-size:12px}
#detail .nav{display:flex;gap:8px;margin-top:14px}
.badge{display:inline-block;border-radius:10px;padding:1px 9px;color:#fff;font-size:12px;font-weight:600}
code{font-family:'IBM Plex Mono',monospace;font-size:12.5px;word-break:break-word}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:7px 10px;border-radius:7px;font-size:13px;max-width:380px;display:none;z-index:9;white-space:pre-line;line-height:1.35}
h2{font-size:19px;margin:28px 0 8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--card)}
tr.row:hover{background:var(--band)}
tr.row.sel{outline:2px solid var(--sel);outline-offset:-2px}
.list{max-height:520px;overflow:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
.card{padding:12px 14px}
.card b{display:block;margin-bottom:4px}
.foot{color:var(--mut);font-size:12px;margin-top:30px}
a{color:inherit}
button{cursor:default}
</style></head><body><div class="wrap">
<h1>MX17 bench + SPS analysis: an interactive timeline</h1>
<p class="sub">Every dated lesson from Nov 2025 to the from-scratch reprocessing, and how the chain carries it. Scroll or pinch-zoom the timeline, drag to pan, click a dot.
Static version: <a href="mx17-analysis-timeline.html">slide note</a> · run sheet <code>rerun/PASS_PLAN_2026-10-28.md</code></p>
<div class="stats" id="stats"></div>
<div class="bar"><span class="lab">status</span><span id="fs"></span></div>
<div class="bar"><span class="lab">area</span><span id="fa"></span></div>
<div class="bar"><span class="lab">view</span><span id="fp"></span>
  <button class="chip" id="zin">＋</button><button class="chip" id="zout">－</button>
  <span class="lab" style="margin-left:10px">lanes</span><button class="chip" id="grp"></button>
  <button class="chip" id="wrongbtn"></button>
  <input type="search" id="q" placeholder="search lessons (e.g. bow, det7, M3)…">
  <button class="chip" id="theme">◐ theme</button></div>
<div class="main">
  <div class="panel"><div id="tl"></div><div id="ov"></div>
    <div class="hint"><span id="range"></span><span>wheel / pinch: zoom · drag: pan · click: details · ← →: previous / next lesson · Esc: clear</span></div></div>
  <div class="panel" id="detail"></div>
</div>
<h2 id="lh"></h2>
<div class="panel list"><table><thead><tr><th style="width:92px">date</th><th style="width:150px">area</th><th>lesson</th><th style="width:130px">status</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Left behind, found on 28 October</h2>
<div class="cards" id="found"></div>
<h2>Claims that were wrong</h2>
<div class="panel list"><table><thead><tr><th>believed</th><th>claim</th><th>corrected</th><th>by</th><th>days</th></tr></thead><tbody id="wrongrows"></tbody></table></div>
<p class="foot" id="foot"></p>
</div><div id="tip"></div>
<script>
const DATA = __DATA__;
const DAY = 864e5, $ = s => document.querySelector(s);
const T = s => Date.parse(s + 'T12:00:00Z');
const fmt = (t, o) => new Date(t).toLocaleDateString('en-GB', Object.assign({timeZone: 'UTC'}, o || {day: '2-digit', month: 'short', year: 'numeric'}));
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const L = DATA.lessons.map(l => Object.assign(l, {x: T(l.d)}));
const M = DATA.milestones.map(m => Object.assign(m, {x: T(m.d)}));
const W = DATA.wrong.map(w => Object.assign(w, {x0: T(w.d0), x1: T(w.d1)}));
const AREAS = Object.keys(DATA.areas);
const T0 = T('2025-11-20'), T1 = T('2026-11-05');
const dark = () => document.documentElement.dataset.theme === 'dark' ||
  (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
const col = s => dark() ? DATA.status[s].cd : DATA.status[s].c;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const S = {t0: T0, t1: T1, st: new Set(DATA.order), ar: new Set(AREAS), q: '', sel: null, group: 'area', wrong: true};
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get('t0')) S.t0 = T(h.get('t0')); if (h.get('t1')) S.t1 = T(h.get('t1'));
  if (h.get('q')) S.q = h.get('q'); if (h.get('group')) S.group = h.get('group');
  if (h.get('sel') !== null && h.get('sel') !== '') S.sel = +h.get('sel');
  if (h.get('status')) S.st = new Set(h.get('status').split(','));
  if (h.get('wrong') === '0') S.wrong = false;
}
let hashTimer;
function writeHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const h = new URLSearchParams();
    h.set('t0', new Date(S.t0).toISOString().slice(0, 10)); h.set('t1', new Date(S.t1).toISOString().slice(0, 10));
    if (S.q) h.set('q', S.q); if (S.group !== 'area') h.set('group', S.group);
    if (S.sel !== null) h.set('sel', S.sel);
    if (S.st.size !== DATA.order.length) h.set('status', [...S.st].join(','));
    if (!S.wrong) h.set('wrong', '0');
    history.replaceState(null, '', '#' + h.toString());
  }, 150);
}

const match = l => S.st.has(l.s) && S.ar.has(l.a) &&
  (!S.q || (l.t + ' ' + l.w + ' ' + DATA.areas[l.a] + ' ' + l.d).toLowerCase().includes(S.q.toLowerCase()));

// ------------------------------------------------------------------ controls
function chips() {
  $('#fs').innerHTML = DATA.order.map(k => {
    const n = L.filter(l => l.s === k).length;
    return `<button class="chip ${S.st.has(k) ? '' : 'off'}" data-st="${k}" data-tip="${esc(DATA.status[k].m)}"><span class="dot" style="background:${col(k)}"></span>${DATA.status[k].label} <b>${n}</b></button>`;
  }).join(' ');
  $('#fa').innerHTML = AREAS.map(k => `<button class="chip ${S.ar.has(k) ? '' : 'off'}" data-ar="${k}">${DATA.areas[k]} <b>${L.filter(l => l.a === k).length}</b></button>`).join(' ')
    + ` <button class="chip" data-ar="*">all</button>`;
  $('#fp').innerHTML = `<button class="chip" data-p="all">all</button> ` + DATA.periods.map((p, i) =>
    `<button class="chip" data-p="${i}">${p.t}</button>`).join(' ');
  $('#grp').textContent = S.group === 'area' ? 'by area' : 'by status';
  $('#wrongbtn').textContent = (S.wrong ? '☑' : '☐') + ' wrong claims';
  $('#wrongbtn').className = 'chip' + (S.wrong ? ' on' : '');
}
document.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.st) { const k = b.dataset.st;
    if (e.shiftKey || e.altKey) S.st = new Set([k]); else S.st.has(k) ? S.st.delete(k) : S.st.add(k);
    if (!S.st.size) S.st = new Set(DATA.order); }
  else if (b.dataset.ar) { const k = b.dataset.ar;
    if (k === '*') S.ar = new Set(AREAS);
    else if (S.ar.size === AREAS.length) S.ar = new Set([k]);
    else { S.ar.has(k) ? S.ar.delete(k) : S.ar.add(k); if (!S.ar.size) S.ar = new Set(AREAS); } }
  else if (b.dataset.p) { if (b.dataset.p === 'all') { S.t0 = T0; S.t1 = T1; }
    else { const p = DATA.periods[+b.dataset.p], a = T(p.d0), z = T(p.d1), pad = Math.max((z - a) * .04, DAY);
      S.t0 = a - pad; S.t1 = z + pad; } }
  else if (b.id === 'zin') zoom(.6, (S.t0 + S.t1) / 2);
  else if (b.id === 'zout') zoom(1 / .6, (S.t0 + S.t1) / 2);
  else if (b.id === 'grp') S.group = S.group === 'area' ? 'status' : 'area';
  else if (b.id === 'wrongbtn') S.wrong = !S.wrong;
  else if (b.id === 'theme') { const d = dark(); document.documentElement.dataset.theme = d ? 'light' : 'dark';
    try { localStorage.setItem('tl-theme', document.documentElement.dataset.theme); } catch (_) {} statics(); }
  else if (b.dataset.go) { select(+b.dataset.go, true); return; }
  else return;
  render();
});
$('#q').addEventListener('input', e => { S.q = e.target.value; render(); });

function zoom(f, c) {
  let span = Math.min(Math.max((S.t1 - S.t0) * f, 2 * DAY), T1 - T0);
  let a = c - (c - S.t0) * span / (S.t1 - S.t0);
  a = Math.max(T0, Math.min(a, T1 - span)); S.t0 = a; S.t1 = a + span;
}
function pan(dt) { const span = S.t1 - S.t0; let a = Math.max(T0, Math.min(S.t0 + dt, T1 - span)); S.t0 = a; S.t1 = a + span; }

// ------------------------------------------------------------------ timeline
let PADL = 178; const PADR = 18;
let geo = null;
function ticks(t0, t1, w) {
  const span = (t1 - t0) / DAY, out = [];
  const d0 = new Date(t0), y = d0.getUTCFullYear(), m = d0.getUTCMonth();
  if (span > 70) {                      // months
    const step = span > 250 ? (w < 700 ? 2 : 1) : 1;
    for (let k = 0; ; k += step) { const t = Date.UTC(y, m + k, 1); if (t > t1) break;
      if (t >= t0) out.push([t, fmt(t, {month: 'short'}) + (new Date(t).getUTCMonth() === 0 || out.length === 0 ? ' ' + new Date(t).getUTCFullYear() : '')]); }
  } else if (span > 16) {               // weeks (Mondays)
    let t = Date.UTC(y, m, d0.getUTCDate()); while (new Date(t).getUTCDay() !== 1) t += DAY;
    for (; t <= t1; t += 7 * DAY) out.push([t, fmt(t, {day: 'numeric', month: 'short'})]);
  } else {                               // days
    let t = Date.UTC(y, m, d0.getUTCDate()) + DAY;
    const step = w / span < 60 ? 2 : 1;
    for (; t <= t1; t += step * DAY) out.push([t, fmt(t, {weekday: 'short', day: 'numeric', month: 'short'})]);
  }
  return out;
}
function render() {
  chips();
  const box = $('#tl'), w = box.clientWidth, narrow = w < 640;
  PADL = narrow ? 92 : 178;
  const X = t => PADL + (w - PADL - PADR) * (t - S.t0) / (S.t1 - S.t0);
  const inView = t => t >= S.t0 - DAY && t <= S.t1 + DAY;
  const ink = css('--ink'), mut = css('--mut'), rule = css('--rule'), band = css('--band'), sel = css('--sel');
  const g = [];
  // milestones: greedy rows, labels that do not fit are dropped (dot stays)
  const ms = M.filter(m => inView(m.x)).sort((a, b) => a.x - b.x), ends = [], MR = narrow ? 2 : 5, placed = [];
  for (const m of ms) { const x = X(m.x), wd = 7.2 * (m.t.length + 9);
    let anc = 'start', lo = x - 3, hi = x + wd; if (hi > w - 4) { anc = 'end'; lo = x - wd; hi = x + 3; }
    if (lo < 4) { placed.push({m, x, r: -1, anc}); continue; }   // no room on either side: dot only
    let r = ends.findIndex(e => e < lo - 10); if (r < 0 && ends.length < MR) { ends.push(-1e9); r = ends.length - 1; }
    if (r >= 0) ends[r] = hi; placed.push({m, x, r, anc}); }
  const nr = Math.max(ends.length, 1), AX = 26 + 20 * nr;
  // lanes
  const keys = S.group === 'area' ? AREAS.filter(k => S.ar.has(k)) : DATA.order.filter(k => S.st.has(k));
  const vis = L.filter(l => inView(l.x));
  const laneOf = l => S.group === 'area' ? l.a : l.s;
  const R = 7, SUB = 3, LH = 2 * R * SUB + 12;
  const top = AX + 36;
  const used = keys.filter(k => vis.some(l => laneOf(l) === k && match(l)) || S.group === 'status');
  let y = top;
  const laneY = {};
  for (const k of used) { laneY[k] = y; y += LH; }
  // wrong claims band
  let wy = y + 10, wrows = [];
  const wv = S.wrong ? W.filter(c => c.x1 >= S.t0 && c.x0 <= S.t1).sort((a, b) => a.x0 - b.x0) : [];
  for (const c of wv) { const a = Math.max(X(c.x0), PADL), z = Math.max(X(c.x1), a + 6), lab = 6.6 * c.t.length;
    c._end = a + lab > w - PADR; c._hide = narrow || (c._end && z - lab < PADL);                       // label drawn right-aligned to the bar end
    const lo = c._hide ? a : c._end ? Math.min(a, z - lab) : a, hi = c._hide ? z : c._end ? z : Math.max(z, a + lab);
    let r = wrows.findIndex(e => e < lo - 8); if (r < 0) { wrows.push(0); r = wrows.length - 1; }
    wrows[r] = hi; c._r = r; }
  const WR = narrow ? 14 : 22, H = (S.wrong && wv.length ? wy + 26 + WR * wrows.length : y) + 10;
  // background: periods
  DATA.periods.forEach((p, i) => { const a = Math.max(X(T(p.d0)), PADL), z = Math.min(X(T(p.d1) + DAY), w - PADR);
    if (z <= a) return;
    if (i % 2 === 0) g.push(`<rect x="${a}" y="0" width="${z - a}" height="${H}" fill="${band}"/>`);
    if (z - a > 7.4 * p.t.length + 12) g.push(`<text x="${a + 6}" y="14" font-size="11" fill="${mut}" letter-spacing="1">${esc(p.t.toUpperCase())}</text>`); });
  // axis + ticks
  g.push(`<line x1="${PADL}" x2="${w - PADR}" y1="${AX}" y2="${AX}" stroke="${ink}" stroke-width="2"/>`);
  for (const [t, lab] of ticks(S.t0, S.t1, w)) { const x = X(t); if (x < PADL || x > w - PADR) continue;
    g.push(`<line x1="${x}" x2="${x}" y1="${AX}" y2="${H - 4}" stroke="${rule}" stroke-width="1"/>`,
      `<text x="${x + 3}" y="${AX + 16}" font-size="11.5" fill="${mut}">${esc(lab)}</text>`); }
  // milestones
  for (const p of placed) { const ly = AX - 14 - 20 * p.r;
    if (p.r >= 0) g.push(`<line x1="${p.x}" x2="${p.x}" y1="${ly + 4}" y2="${AX}" stroke="${mut}" stroke-dasharray="2 3"/>`,
      `<text x="${p.x + (p.anc === 'end' ? 4 : -2)}" y="${ly}" font-size="12.5" fill="${ink}" text-anchor="${p.anc}" data-tip="${esc(p.m.d + ' · ' + DATA.areas[p.m.a] + '\n' + p.m.t)}">${esc(fmt(p.m.x, {day: '2-digit', month: 'short'}) + ' · ' + p.m.t)}</text>`);
    g.push(`<rect x="${p.x - 5}" y="${AX - 5}" width="10" height="10" transform="rotate(45 ${p.x} ${AX})" fill="${ink}" data-tip="${esc(p.m.d + ' · milestone\n' + p.m.t)}"/>`); }
  // lane labels
  for (const k of used) { const yy = laneY[k] + LH / 2;
    g.push(`<line x1="${PADL}" x2="${w - PADR}" y1="${laneY[k] + LH}" y2="${laneY[k] + LH}" stroke="${rule}"/>`);
    const lab = S.group === 'area' ? DATA.areas[k] : DATA.status[k].label;
    const c = S.group === 'area' ? ink : col(k);
    const short = narrow && lab.length > 13 ? lab.slice(0, 12) + '…' : lab;
    g.push(`<text x="${PADL - 8}" y="${yy + 4}" font-size="${narrow ? 11 : 12.5}" text-anchor="end" fill="${c}" font-weight="600" data-tip="${esc(lab)}">${esc(short)}</text>`); }
  // dots: per lane, greedy sub-rows; beyond SUB they overlap (still clickable)
  const pts = [];
  for (const k of used) { const subs = Array(SUB).fill(-1e9);
    for (const l of vis.filter(l => laneOf(l) === k).sort((a, b) => a.x - b.x)) {
      const x = X(l.x); let r = subs.findIndex(e => e < x - 2 * R - 1); if (r < 0) r = subs.indexOf(Math.min(...subs));
      subs[r] = x; pts.push({l, x, y: laneY[k] + 6 + R + r * 2 * R}); } }
  for (const p of pts) { const on = match(p.l), s = S.sel === p.l.id;
    g.push(`<circle cx="${p.x}" cy="${p.y}" r="${s ? R + 2 : R - .5}" fill="${col(p.l.s)}" fill-opacity="${on ? 1 : .12}" stroke="${s ? sel : css('--card')}" stroke-width="${s ? 3 : 1.2}" data-id="${p.l.id}" data-tip="${esc(p.l.d + ' · ' + DATA.areas[p.l.a] + ' · ' + DATA.status[p.l.s].label + '\n' + p.l.t)}"/>`); }
  // wrong claims
  if (S.wrong && wv.length) {
    g.push(`<text x="${PADL - 8}" y="${wy + 14}" font-size="12" text-anchor="end" fill="${ink}" font-weight="600">wrong claims</text>`,
      narrow ? '' : `<text x="${PADL - 10}" y="${wy + 29}" font-size="11" text-anchor="end" fill="${mut}">believed → withdrawn</text>`);
    for (const c of wv) { const a = Math.max(X(c.x0), PADL), z = Math.max(X(c.x1), a + 6), yy = wy + 8 + WR * c._r;
      const days = Math.round((c.x1 - c.x0) / DAY), cc = days > 30 ? col('open') : '#c8601a';
      const tip = `Believed ${c.d0}: ${c.t}\nWithdrawn ${c.d1} (${days} d): ${c.by}`;
      g.push(`<rect x="${a}" y="${yy}" width="${z - a}" height="10" rx="4" fill="${cc}" data-tip="${esc(tip)}"/>`,
        c._hide ? '' : `<text x="${c._end ? z : a}" y="${yy + 22 - 2}" font-size="11.5" fill="${mut}" text-anchor="${c._end ? 'end' : 'start'}" data-tip="${esc(tip)}">${esc(c.t)}</text>`); } }
  box.innerHTML = `<svg width="${w}" height="${H}" viewBox="0 0 ${w} ${H}" font-family="IBM Plex Sans, Helvetica, Arial, sans-serif">${g.join('')}</svg>`;
  geo = {w, X};
  overview(); list(); detail(); writeHash();
  $('#range').textContent = fmt(S.t0) + ' – ' + fmt(S.t1) + ' · ' + vis.filter(match).length + ' lessons in view';
}

// ------------------------------------------------------------------ overview strip (whole range, brush = view)
function overview() {
  const box = $('#ov'), w = box.clientWidth, H = 54, X = t => PADL + (w - PADL - PADR) * (t - T0) / (T1 - T0);
  const g = [`<text x="${PADL - 10}" y="30" font-size="12" text-anchor="end" fill="${css('--mut')}">overview</text>`];
  for (const p of DATA.periods) g.push(`<line x1="${X(T(p.d0))}" x2="${X(T(p.d0))}" y1="6" y2="${H - 6}" stroke="${css('--rule')}"/>`);
  for (const l of L) if (match(l)) g.push(`<rect x="${X(l.x) - 1}" y="${12 + 30 * DATA.order.indexOf(l.s) / DATA.order.length}" width="2.4" height="6" fill="${col(l.s)}"/>`);
  const a = X(S.t0), z = X(S.t1);
  g.push(`<rect id="brush" x="${a}" y="4" width="${Math.max(z - a, 4)}" height="${H - 8}" fill="${css('--sel')}" fill-opacity=".14" stroke="${css('--sel')}" stroke-width="1.5" rx="3"/>`);
  box.innerHTML = `<svg width="${w}" height="${H}">${g.join('')}</svg>`;
  box._X = X; box._w = w;
}
(function () {
  const ov = $('#ov'); let drag = null;
  const tAt = e => { const r = ov.getBoundingClientRect(); return T0 + (T1 - T0) * (e.clientX - r.left - PADL) / (ov._w - PADL - PADR); };
  ov.addEventListener('pointerdown', e => { const t = tAt(e); const span = S.t1 - S.t0;
    if (t < S.t0 || t > S.t1) { pan(t - span / 2 - S.t0); render(); }
    drag = {t, t0: S.t0}; ov.setPointerCapture(e.pointerId); });
  ov.addEventListener('pointermove', e => { if (!drag) return; const span = S.t1 - S.t0;
    let a = Math.max(T0, Math.min(drag.t0 + tAt(e) - drag.t, T1 - span)); S.t0 = a; S.t1 = a + span; render(); });
  ov.addEventListener('pointerup', () => drag = null);
})();

// ------------------------------------------------------------------ timeline interaction
(function () {
  const tl = $('#tl'); let drag = null, moved = false; const ptrs = new Map(); let pinch = null;
  const tAt = x => S.t0 + (S.t1 - S.t0) * (x - PADL) / (geo.w - PADL - PADR);
  tl.addEventListener('wheel', e => { if (!e.ctrlKey && Math.abs(e.deltaX) > Math.abs(e.deltaY)) { pan((S.t1 - S.t0) * e.deltaX / geo.w); render(); e.preventDefault(); return; }
    const r = tl.getBoundingClientRect(); zoom(Math.exp(e.deltaY * .0016), tAt(e.clientX - r.left)); render(); e.preventDefault(); }, {passive: false});
  tl.addEventListener('pointerdown', e => { ptrs.set(e.pointerId, e.clientX); moved = false;
    if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = {d: Math.abs(a - b), t0: S.t0, t1: S.t1}; drag = null; }
    else drag = {x: e.clientX, t0: S.t0}; });
  tl.addEventListener('pointermove', e => { if (!ptrs.has(e.pointerId)) return; ptrs.set(e.pointerId, e.clientX);
    if (pinch && ptrs.size === 2) { const [a, b] = [...ptrs.values()], r = tl.getBoundingClientRect();
      const c = tAt((a + b) / 2 - r.left); S.t0 = pinch.t0; S.t1 = pinch.t1; zoom(pinch.d / Math.max(Math.abs(a - b), 10), c); moved = true; render(); return; }
    if (!drag) return; const dx = e.clientX - drag.x; if (Math.abs(dx) > 3) { moved = true; tl.setPointerCapture(e.pointerId); }
    if (moved) { const span = S.t1 - S.t0; let a = Math.max(T0, Math.min(drag.t0 - dx * span / (geo.w - PADL - PADR), T1 - span)); S.t0 = a; S.t1 = a + span; render(); } });
  const up = e => { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = null;
    if (!moved) { const c = e.target.closest && e.target.closest('[data-id]'); if (c) select(+c.dataset.id, false); }
    drag = null; };
  tl.addEventListener('pointerup', up); tl.addEventListener('pointercancel', up);
})();
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'Escape') { S.sel = null; render(); return; }
  if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
  const vis = L.filter(match); if (!vis.length) return;
  let i = vis.findIndex(l => l.id === S.sel);
  i = i < 0 ? (e.key === 'ArrowRight' ? 0 : vis.length - 1) : Math.max(0, Math.min(vis.length - 1, i + (e.key === 'ArrowRight' ? 1 : -1)));
  select(vis[i].id, true); e.preventDefault();
});
function select(id, center) {
  S.sel = id; const l = L[id];
  if (center && (l.x < S.t0 || l.x > S.t1)) pan(l.x - (S.t0 + S.t1) / 2);
  render();
  const tr = document.querySelector(`tr[data-id="${id}"]`); if (tr) tr.scrollIntoView({block: 'nearest'});
}

// ------------------------------------------------------------------ detail, list, found, wrong
const badge = s => `<span class="badge" style="background:${col(s)}">${DATA.status[s].label}</span>`;
function detail() {
  const d = $('#detail');
  if (S.sel === null) {
    const c = {}; L.forEach(l => c[l.s] = (c[l.s] || 0) + 1);
    d.innerHTML = `<h3>How to read it</h3><p class="big">Each dot is one dated lesson, coloured by how tonight's reprocessing chain carries it.</p>`
      + DATA.order.map(k => `<div class="kv">${badge(k)} <b>${c[k]}</b><span class="k">${esc(DATA.status[k].m)}</span></div>`).join('')
      + `<div class="kv"><span class="k">Diamonds on the axis are milestones; bars under the lanes are claims that were later withdrawn (red: believed for over a month). Shift-click a status chip to show only that status.</span></div>`;
    return; }
  const l = L[S.sel], vis = L.filter(match), i = vis.findIndex(v => v.id === l.id);
  const sameDay = L.filter(v => v.d === l.d && v.id !== l.id);
  d.innerHTML = `<h3>${fmt(l.x, {weekday: 'short', day: 'numeric', month: 'long', year: 'numeric'})} · ${esc(DATA.areas[l.a])}</h3>
    <p class="big">${esc(l.t)}</p>${badge(l.s)}
    <div class="kv"><span class="k">what the chain does / where it lives</span>${esc(l.w).replace(/([\w./-]+\.(py|md|json|csv|sh)(:[\d,-]+)?)/g, '<code>$1</code>')}</div>
    <div class="kv"><span class="k">status means</span>${esc(DATA.status[l.s].m)}</div>
    ${sameDay.length ? `<div class="kv"><span class="k">same day</span>${sameDay.map(v => `<a href="#" data-go2="${v.id}">${esc(v.t.slice(0, 70))}${v.t.length > 70 ? '…' : ''}</a>`).join('<br>')}</div>` : ''}
    <div class="nav"><button class="chip" data-go="${vis[Math.max(i - 1, 0)] ? vis[Math.max(i - 1, 0)].id : l.id}">← previous</button>
    <button class="chip" data-go="${vis[Math.min(i + 1, vis.length - 1)] ? vis[Math.min(i + 1, vis.length - 1)].id : l.id}">next →</button>
    <button class="chip" id="clr">clear</button></div>`;
  $('#clr').onclick = () => { S.sel = null; render(); };
  d.querySelectorAll('[data-go2]').forEach(a => a.onclick = ev => { ev.preventDefault(); select(+a.dataset.go2, true); });
}
function list() {
  const v = L.filter(match);
  $('#lh').textContent = `Lessons (${v.length} of ${L.length} match the filters)`;
  $('#rows').innerHTML = v.map(l => `<tr class="row ${S.sel === l.id ? 'sel' : ''}" data-id="${l.id}"><td>${l.d}</td><td>${esc(DATA.areas[l.a])}</td>
    <td>${esc(l.t)}<div style="color:var(--mut);font-size:12.5px;margin-top:2px">${esc(l.w)}</div></td><td>${badge(l.s)}</td></tr>`).join('');
}
$('#rows').addEventListener('click', e => { const tr = e.target.closest('tr[data-id]'); if (tr) select(+tr.dataset.id, true); });
function statics() {
  const vc = {fixed: ['fixed today', 'fixed'], plan: ['in the plan', 'open'], doc: ['wording', 'doc'], lim: ['limitation', 'lim']};
  $('#found').innerHTML = DATA.found.map(f => `<div class="panel card"><b>${f.id} · <span class="badge" style="background:${col(vc[f.v][1])}">${vc[f.v][0]}</span></b>${esc(f.t)}
    <div style="color:var(--mut);font-size:13px;margin-top:6px">→ ${esc(f.act)}</div></div>`).join('');
  $('#wrongrows').innerHTML = W.slice().sort((a, b) => a.x0 - b.x0).map(c => { const days = Math.round((c.x1 - c.x0) / DAY);
    return `<tr><td>${c.d0}</td><td>${esc(c.t)}</td><td>${c.d1}</td><td>${esc(c.by)}</td><td style="color:${days > 30 ? col('open') : 'inherit'}">${days}</td></tr>`; }).join('');
  const c = {}; L.forEach(l => c[l.s] = (c[l.s] || 0) + 1);
  $('#stats').innerHTML = [[L.length, 'dated lessons'], [c.chain || 0, 'built into the chain'], [c.open || 0, 'open: tonight\'s plan'],
    [W.length, 'claims withdrawn'], [DATA.found.length, 'left-behind items found'], [M.length, 'milestones']]
    .map(([n, t]) => `<div class="stat"><b>${n}</b><span>${t}</span></div>`).join('');
  $('#foot').innerHTML = `Generated ${DATA.built} by <code>nTof_x17/rerun/history/make_timeline_page.py</code> from <code>timeline_data.py</code> (the same data as LESSONS.md and the slide note). The view is in the URL: copy it to share a zoom, filter or selected lesson.`;
}

// ------------------------------------------------------------------ tooltip
(function () {
  const tip = $('#tip');
  document.addEventListener('pointerover', e => { const el = e.target.closest('[data-tip]'); if (!el) return; tip.textContent = el.dataset.tip; tip.style.display = 'block'; });
  document.addEventListener('pointermove', e => { if (tip.style.display !== 'block') return;
    const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8), y = e.clientY + 16 + tip.offsetHeight > innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 16;
    tip.style.left = x + 'px'; tip.style.top = y + 'px'; });
  document.addEventListener('pointerout', e => { if (e.target.closest('[data-tip]')) tip.style.display = 'none'; });
})();

try { const t = localStorage.getItem('tl-theme'); if (t) document.documentElement.dataset.theme = t; } catch (_) {}
readHash(); $('#q').value = S.q; statics(); render();
addEventListener('resize', () => render());
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { statics(); render(); });
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = ('<!--note\ntitle: MX17 analysis timeline (interactive)\n'
            f'summary: Zoomable timeline of {len(td.LESSONS)} dated lessons of the MX17 bench + SPS analysis, filterable by '
            'status and area, with withdrawn claims and the items left behind before the pass.\n'
            f'tags: x17, reconstruction, history, interactive\ndate: {dt.date.today().isoformat()}\n-->\n')
    p = os.path.join(a.out, 'mx17-analysis-timeline-explorer.html')
    open(p, 'w').write(meta + PAGE.replace('__DATA__', json.dumps(data(), separators=(',', ':'))))
    print('wrote', p)


if __name__ == '__main__':
    main()
