#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_clock_dashboard.py -- one page that makes a bad DREAM->n_TOF clock obvious.

    python make_clock_dashboard.py <slim_root> [-o clock_dashboard.html]

`clock_qa.py` judges one segment against absolute thresholds. This adds the
thing a single segment cannot know: whether it is odd COMPARED TO ITS PEERS.
A segment can pass every absolute check and still be the only one in the
campaign whose T0 sits 300 ns from its neighbours, and that is exactly the
failure mode that produced a silently mis-timed slim in the first place.

So there are two independent layers, and the page shows both:

  absolute    thresholds from measurement (clock_qa.TH) -- PASS/WARN/FAIL
  population  robust z against the fleet median (MAD-scaled) -- OUTLIER

Population outliers are reported separately and never silently upgraded into a
FAIL: being unusual is a reason to look, not proof of being wrong. Several
genuinely different DREAM runs would each look like outliers to each other, so
T0 is compared only within its own DREAM run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ntof_processing.slim_pipeline import clock_qa as Q          # noqa: E402
from ntof_processing.slim_pipeline.dashboard import charts as CH  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
FAMILIES = ('WAL', 'PSS', 'LIQ')
Z_OUTLIER = 5.0        # robust z beyond which a segment is called out
MIN_SPREAD = dict(K=2e-7, T0_ns=25.0, efficiency=0.01, accidental=0.0005,
                  da_rms=2.0, dk_rms_ppm=0.3, residual_rms=1.0,
                  arm=2.0)


# ------------------------------------------------------------------ collect
def collect(root: Path, use_cache=True):
    out = []
    for f in sorted(root.rglob('ntof_hits_*.root')):
        d = f.parent
        cache = d / 'clock_qa.json'
        if use_cache and cache.is_file() and \
                cache.stat().st_mtime >= f.stat().st_mtime:
            out.append(json.loads(cache.read_text()))
            continue
        try:
            q = Q.analyse(d)
        except Exception as e:                                  # noqa: BLE001
            print(f'  !! {d}: {type(e).__name__}: {e}')
            continue
        from dataclasses import asdict
        rec = asdict(q)
        cache.write_text(json.dumps(rec, indent=1))
        out.append(rec)
    return out


def _robust_z(vals):
    v = np.asarray([x if x is not None and np.isfinite(x) else np.nan
                    for x in vals], float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    return v, med, float(mad * 1.4826)


def population(recs):
    """Attach OUTLIER notes by comparing each segment to the fleet."""
    notes = {i: [] for i in range(len(recs))}

    def scan(key, getter, floor, group=None, label=None):
        groups = {}
        for i, r in enumerate(recs):
            g = group(r) if group else '_'
            groups.setdefault(g, []).append(i)
        for g, idx in groups.items():
            if len(idx) < 4:
                continue                      # too few peers to call anything
            v, med, sig = _robust_z([getter(recs[i]) for i in idx])
            sig = max(sig, floor)
            for j, i in enumerate(idx):
                if not np.isfinite(v[j]):
                    continue
                z = abs(v[j] - med) / sig
                if z >= Z_OUTLIER:
                    notes[i].append(dict(
                        metric=label or key, value=float(v[j]),
                        median=float(med), z=float(z), group=str(g)))

    scan('K', lambda r: r['clock']['K'], MIN_SPREAD['K'])
    scan('T0_ns', lambda r: r['clock']['T0_ns'], MIN_SPREAD['T0_ns'],
         group=lambda r: r['segment']['dream_run'],
         label='T0 (within DREAM run)')
    scan('efficiency', lambda r: r['match']['efficiency'],
         MIN_SPREAD['efficiency'])
    scan('accidental', lambda r: r['match']['accidental'],
         MIN_SPREAD['accidental'])
    scan('residual_rms', lambda r: r['match']['residual_rms'],
         MIN_SPREAD['residual_rms'])
    scan('da_rms', lambda r: (r['perbunch'] or {}).get('da_rms'),
         MIN_SPREAD['da_rms'])
    scan('dk_rms_ppm', lambda r: (r['perbunch'] or {}).get('dk_rms_ppm'),
         MIN_SPREAD['dk_rms_ppm'])
    for a in ARMS:
        scan(f'arm {a}',
             lambda r, a=a: r['match']['per_arm'][a]['offset_ns'],
             MIN_SPREAD['arm'], label=f'arm {a} offset')
    return notes


# ------------------------------------------------------------------- charts
def _lab(r):
    s = r['segment']
    return f'{s["dream_run"]}/{s["dream_subrun"]}x{s["ntof_run"]}'


def chart_metric(recs, getter, title, ylabel, *, log=False, pct=False,
                 ref=None, band=None, fmt='{:.4g}', outliers=None):
    xs = list(range(len(recs)))
    ys = [getter(r) for r in recs]
    good = [y for y in ys if y is not None and np.isfinite(y)]
    if not good:
        return ''
    lo, hi = min(good), max(good)
    if band:
        lo, hi = min(lo, band[0]), max(hi, band[1])
    if ref is not None:
        lo, hi = min(lo, ref), max(hi, ref)
    f = CH.Frame(title=title, ylabel=ylabel, xlabel='segment')
    f.xlim(-0.5, len(recs) - 0.5, ticks=[])
    if log:
        # Pad multiplicatively. Additive padding drove the lower bound negative,
        # which clamped to the 1e-12 floor and produced a 15-decade axis with
        # every point squashed onto the top line.
        f.ylim(max(lo, 1e-12) / 3, hi * 3, log=True)
    else:
        pad = (hi - lo) * 0.12 or abs(hi) * 0.1 or 1
        f.ylim(lo - pad, hi + pad)
    if band:
        f.band(band[0], band[1], f'expected {fmt.format(band[0])}'
                                 f'..{fmt.format(band[1])}')
    if ref is not None:
        f.hline(ref, label=f'reference {fmt.format(ref)}')
    # Separate the DREAM runs. T0 in particular is only comparable inside one,
    # so a reader must be able to see where one ends.
    for a_, b_, lab in _groups(recs)[:-1]:
        x = f.px(b_ + 0.5)
        f.body.append(f'<line class="grid" x1="{x:.1f}" y1="{f.t}" '
                      f'x2="{x:.1f}" y2="{f.h-f.b}" stroke-dasharray="2 3"/>')
    # Colour marks by whether THIS metric is odd -- never by the segment's
    # overall verdict. Colouring by verdict paints a segment red in the K chart
    # because its efficiency was low, which reads as "K is wrong" and is how a
    # dashboard teaches people to distrust it.
    cols, tips = [], []
    for i, r in enumerate(recs):
        is_out = bool(outliers and outliers.get(i))
        v = ys[i]
        cols.append('var(--bad)' if is_out else 'var(--series1)')
        vs = fmt.format(v) if v is not None and np.isfinite(v) else 'n/a'
        tips.append(f'{_lab(r)} — {vs}'
                    + ('  [fleet outlier on this metric]' if is_out else ''))
    f.points(xs, ys, colours=cols, tips=tips)
    return f'<figure>{f.svg()}{_group_key(recs)}</figure>'


def _groups(recs):
    """(start, end, label) spans of consecutive segments in one DREAM run."""
    out, start = [], 0
    for i in range(1, len(recs) + 1):
        if i == len(recs) or (recs[i]['segment']['dream_run']
                              != recs[start]['segment']['dream_run']):
            out.append((start, i - 1, recs[start]['segment']['dream_run']))
            start = i
    return out


def _group_key(recs):
    g = _groups(recs)
    if len(g) < 2:
        return ''
    return ('<div class="legend">' + ''.join(
        f'<span class="lg" style="color:var(--muted)">segments '
        f'{a+1}–{b+1}: {CH.esc(lab)}</span>' for a, b, lab in g)
        + '</div>')


def chart_arms(recs):
    f = CH.Frame(title='Per-arm trigger offsets',
                 ylabel='offset (ns)', xlabel='segment')
    allv = [r['match']['per_arm'][a]['offset_ns'] for r in recs for a in ARMS]
    allv = [v for v in allv if np.isfinite(v)]
    if not allv:
        return ''
    f.xlim(-0.5, len(recs) - 0.5, ticks=[])
    f.ylim(min(allv) - 3, max(allv) + 3)
    for a in ARMS:
        ys = [r['match']['per_arm'][a]['offset_ns'] for r in recs]
        f.points(range(len(recs)), ys, colours=CH.ARM_COLOUR[a],
                 tips=[f'{_lab(r)} — arm {a} {y:+.2f} ns' for r, y in
                       zip(recs, ys)], r=2.8)
        f.hline(Q.REF_ARM[a], cls='ref faint',
                label=f'arm {a} reference {Q.REF_ARM[a]:+.2f} ns')
    return (f'<figure>{f.svg()}'
            + CH.legend([(f'arm {a}', CH.ARM_COLOUR[a]) for a in ARMS])
            + '</figure>')


def chart_resid_overlay(recs):
    f = CH.Frame(title='Matched residual, every segment overlaid',
                 ylabel='fraction of matches / bin', xlabel='residual (ns)')
    hs = [r['match']['residual_hist'] for r in recs if r['match'].get(
        'residual_hist')]
    if not hs:
        return ''
    peak = 0.0
    for h in hs:
        tot = sum(h['counts']) or 1
        peak = max(peak, max(h['counts']) / tot)
    f.xlim(hs[0]['lo'], hs[0]['hi'])
    f.ylim(0, peak * 1.1)
    for r, h in zip(recs, hs):
        tot = sum(h['counts']) or 1
        f.step_hist(h['lo'], h['bin'], [c / tot for c in h['counts']],
                    colour='var(--series1)', width=1,
                    tip=f'{_lab(r)} — RMS {r["match"]["residual_rms"]:.2f} ns')
    return (f'<figure>{f.svg()}<figcaption>Every segment on one frame. A '
            f'segment whose core sits off zero, or is visibly wider, is a '
            f'clock that did not converge to the same place as the '
            f'rest.</figcaption></figure>')


def chart_drift(recs):
    f = CH.Frame(title='Residual vs time since flash (per-bunch fit should '
                       'flatten this)',
                 ylabel='median residual (ns)', xlabel='time since flash (ms)')
    prof = [(r, r['match'].get('residual_vs_t')) for r in recs]
    prof = [(r, p) for r, p in prof if p and p['centres']]
    if not prof:
        return ''
    allm = [v for _, p in prof for v in p['median']]
    f.xlim(0, 85)
    f.ylim(min(allm) - 1, max(allm) + 1)
    f.hline(0, cls='ref')
    for r, p in prof:
        f.line(p['centres'], p['median'], colour='var(--series1)', width=1,
               tip=_lab(r))
    return (f'<figure>{f.svg()}<figcaption>Flat is correct. A slope means the '
            f'per-bunch correction did not take, and hits late in the 80 ms '
            f'burst are attached with a growing error.</figcaption></figure>')


def chart_family_dt(recs):
    """Signal and +100 us control, per detector family, summed over segments."""
    out = []
    for fam in FAMILIES:
        sig = None
        ctl = None
        meta = None
        for r in recs:
            fs = r['hits']['families'].get(fam)
            if not fs:
                continue
            s = np.array(fs['dt_signal']['counts'], float)
            c = np.array(fs['dt_control']['counts'], float)
            sig = s if sig is None else sig + s
            ctl = c if ctl is None else ctl + c
            meta = fs['dt_signal']
        if sig is None:
            continue
        f = CH.Frame(h=230, title=f'{fam}: hit time vs prediction',
                     ylabel='hits / bin', xlabel='dt (ns)')
        f.xlim(meta['lo'], meta['hi'])
        f.ylim(0, max(sig.max(), 1) * 1.1)
        f.step_hist(meta['lo'], meta['bin'], sig, colour=CH.PALETTE[0],
                    tip=f'{fam} signal')
        f.step_hist(meta['lo'], meta['bin'], ctl, colour=CH.PALETTE[1],
                    tip=f'{fam} +100 us control (accidental floor)')
        edge = int(len(sig) * 0.05)
        frac = float((sig[:edge].sum() + sig[-edge:].sum() -
                      ctl[:edge].sum() - ctl[-edge:].sum())
                     / max(sig.sum() - ctl.sum(), 1))
        out.append(
            f'<figure>{f.svg()}'
            + CH.legend([('signal', CH.PALETTE[0]),
                         ('+100 µs control', CH.PALETTE[1])])
            + f'<figcaption>{fam}: {frac:.1%} of the background-subtracted '
              f'excess sits in the outer 10 % of the window. Near zero means '
              f'the window contains the coincidence; a large value means it '
              f'is being cut.</figcaption></figure>')
    return ''.join(out)


# --------------------------------------------------------------------- page
CSS = """
:root{--bg:#fcfcfb;--surface:#fff;--ink:#1a1a19;--ink2:#4a4a48;--muted:#8a8a86;
--line:#e4e4e0;--good:#15803d;--warn:#b45309;--bad:#b91c1c;
--series1:#2563eb;--series2:#ea580c;--band:#2563eb14;--code:#f4f4f1}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#1a1a19;--surface:#232322;--ink:#f0efec;--ink2:#c9c8c3;--muted:#8f8e89;
--line:#35352f;--good:#4ade80;--warn:#fbbf24;--bad:#f87171;
--series1:#3b82f6;--series2:#e8690b;--band:#3b82f622;--code:#26261f}}
:root[data-theme="dark"]{--bg:#1a1a19;--surface:#232322;--ink:#f0efec;
--ink2:#c9c8c3;--muted:#8f8e89;--line:#35352f;--good:#4ade80;--warn:#fbbf24;
--bad:#f87171;--series1:#3b82f6;--series2:#e8690b;--band:#3b82f622;--code:#26261f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:18px;margin:38px 0 6px;letter-spacing:-.01em}
h3{font-size:15px;margin:22px 0 4px;color:var(--ink2)}
p{margin:8px 0;color:var(--ink2);max-width:74ch}
.sub{color:var(--muted);font-size:13px;margin-bottom:22px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
gap:10px;margin:18px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;
padding:12px 14px}
.tile .v{font-size:23px;font-weight:600;letter-spacing:-.02em}
.tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;
letter-spacing:.06em;margin-top:2px}
.covbar{display:flex;height:26px;border-radius:6px;overflow:hidden;
border:1px solid var(--line);margin:6px 0 4px}
.covbar span{display:flex;align-items:center;justify-content:center;
font-size:11px;color:#fff;font-weight:600}
figure{margin:14px 0 6px;background:var(--surface);border:1px solid var(--line);
border-radius:8px;padding:10px 12px 6px;overflow-x:auto}
figcaption{font-size:12.5px;color:var(--muted);padding:2px 2px 8px;max-width:78ch}
svg{width:100%;height:auto;display:block;min-width:520px}
.grid{stroke:var(--line);stroke-width:1}
.axis{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:10px}
.axlabel{fill:var(--muted);font-size:11px}
.ctitle{fill:var(--ink);font-size:12.5px;font-weight:600}
.band{fill:var(--band)}
.ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 3}
.ref.faint{opacity:.45}
.ln{opacity:.5}
.pt{opacity:.9}
.legend{display:flex;flex-wrap:wrap;gap:14px;padding:2px 2px 8px;font-size:12px;
color:var(--ink2)}
.lg{display:inline-flex;align-items:center;gap:6px}
.lg i{width:11px;height:11px;border-radius:2px;display:inline-block}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
font-weight:600}
.PASS{background:#15803d1f;color:var(--good)}
.WARN{background:#b4530922;color:var(--warn)}
.FAIL{background:#b91c1c1f;color:var(--bad)}
.NA{background:#8a8a8618;color:var(--muted)}
details{background:var(--surface);border:1px solid var(--line);
border-radius:8px;padding:8px 12px;margin:8px 0}
summary{cursor:pointer;font-size:13.5px;font-weight:600}
.chk{font-size:12.5px;padding:3px 0;border-bottom:1px dotted var(--line)}
.callout{background:var(--surface);border-left:3px solid var(--warn);
border-radius:0 8px 8px 0;padding:10px 14px;margin:14px 0}
.callout.bad{border-left-color:var(--bad)}
.callout.ok{border-left-color:var(--good)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.5px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);
font-size:12px;padding:5px 9px;border-radius:6px;opacity:0;transition:opacity
.1s;z-index:9;max-width:320px}
"""

JS = """
(function(){
 var t=document.getElementById('tip');
 document.addEventListener('mouseover',function(e){
   var m=e.target.closest('[data-tip]'); if(!m)return;
   t.textContent=m.getAttribute('data-tip'); t.style.opacity=1;});
 document.addEventListener('mousemove',function(e){
   if(t.style.opacity=='1'){var x=e.clientX+14,y=e.clientY+16;
     if(x+t.offsetWidth>innerWidth)x=e.clientX-t.offsetWidth-10;
     t.style.left=x+'px'; t.style.top=y+'px';}});
 document.addEventListener('mouseout',function(e){
   if(e.target.closest('[data-tip]'))t.style.opacity=0;});
})();
"""


def build(recs, notes, title, source):
    n = len(recs)
    v = {k: sum(1 for r in recs if r['verdict'] == k)
         for k in ('PASS', 'WARN', 'FAIL')}
    nout = sum(1 for i in notes if notes[i])
    eff = [r['match']['efficiency'] for r in recs]
    acc = [r['match']['accidental'] for r in recs]
    nev = sum(r['match']['n_physics'] for r in recs)
    nhit = sum(r['hits']['n_total'] for r in recs)
    size = sum(r['segment']['size_mb'] for r in recs)
    runs = sorted({r['segment']['ntof_run'] for r in recs})

    H = [f'<div class="wrap"><h1>{CH.esc(title)}</h1>',
         f'<div class="sub">{n} segment(s) over {len(runs)} n_TOF run(s) · '
         f'generated {dt.datetime.now():%Y-%m-%d %H:%M} · source '
         f'<code>{CH.esc(source)}</code></div>']

    H.append('<div class="tiles">')
    for k, val in (('segments', f'{n}'),
                   ('pass', f'{v["PASS"]}'), ('warn', f'{v["WARN"]}'),
                   ('fail', f'{v["FAIL"]}'),
                   ('fleet outliers', f'{nout}'),
                   ('median efficiency',
                    f'{np.median(eff):.2%}' if eff else '—'),
                   ('DREAM triggers', f'{nev:,}'),
                   ('n_TOF hits kept', f'{nhit:,}'),
                   ('total size', f'{size/1000:.1f} GB' if size >= 1000
                    else f'{size:.0f} MB')):
        H.append(f'<div class="tile"><div class="v">{val}</div>'
                 f'<div class="k">{k}</div></div>')
    H.append('</div>')

    if n:
        seg = []
        for k, c in (('PASS', 'var(--good)'), ('WARN', 'var(--warn)'),
                     ('FAIL', 'var(--bad)')):
            if v[k]:
                seg.append(f'<span style="width:{100*v[k]/n:.2f}%;'
                           f'background:{c}" data-tip="{v[k]} {k}">'
                           f'{v[k] if 100*v[k]/n > 6 else ""}</span>')
        H.append(f'<div class="covbar">{"".join(seg)}</div>')

    # ------------------------------------------------ what to look at first
    trouble = [(i, r) for i, r in enumerate(recs)
               if r['verdict'] != 'PASS' or notes[i]]
    if not trouble:
        H.append('<div class="callout ok"><strong>Nothing to look at.</strong> '
                 'Every segment passed every absolute check and none is a '
                 'fleet outlier.</div>')
    else:
        H.append(f'<h2>Look at these first</h2><p>{len(trouble)} of {n} '
                 f'segment(s) either failed an absolute threshold or sit far '
                 f'from their peers. Population outliers are a reason to look, '
                 f'not proof of a fault.</p>')
        H.append('<table><tr><th>segment</th><th>verdict</th>'
                 '<th>what</th></tr>')
        for i, r in trouble:
            why = [f'{c["name"]} ({c["level"]})' for c in r['checks']
                   if c['level'] in ('WARN', 'FAIL')]
            why += [f'{o["metric"]} {o["value"]:.4g} vs fleet '
                    f'{o["median"]:.4g} (z={o["z"]:.1f})' for o in notes[i]]
            H.append(f'<tr><td>{CH.esc(_lab(r))}</td>'
                     f'<td><span class="pill {r["verdict"]}">{r["verdict"]}'
                     f'</span></td><td>{CH.esc("; ".join(why))}</td></tr>')
        H.append('</table>')

    # -------------------------------------------------------------- charts
    H.append('<h2>Clock constants</h2><p>The map is '
             '<code>t_nTOF = t_DREAM(1+K) + T0 + a_arm</code>. Nothing here '
             'transfers between pairs, so every segment fits its own. K is a '
             'property of the two clocks and should barely move; T0 is per '
             'pair and is only comparable inside one DREAM run.</p>')
    H.append(chart_metric(recs, lambda r: r['clock']['K'],
                          'K — clock rate ratio', 'K', fmt='{:.6e}',
                          band=(Q.TH['k_lo'], Q.TH['k_hi']),
                          outliers={i: any(o['metric'] == 'K' for o in notes[i])
                                    for i in notes}))
    H.append(chart_metric(recs, lambda r: r['clock']['T0_ns'],
                          'T0 — offset (compare only within a DREAM run)',
                          'T0 (ns)', fmt='{:+.1f}',
                          outliers={i: any('T0' in o['metric']
                                           for o in notes[i]) for i in notes}))
    H.append(chart_arms(recs))

    H.append('<h2>Match quality</h2>')
    H.append(chart_metric(recs, lambda r: r['match']['efficiency'],
                          'Efficiency at the ±25 ns accept window',
                          'matched fraction', fmt='{:.2%}', ref=0.9584,
                          outliers={i: any(o['metric'] == 'efficiency'
                                           for o in notes[i]) for i in notes}))
    H.append(chart_metric(recs, lambda r: max(r['match']['accidental'], 1e-6),
                          'Accidental rate from the +100 µs control',
                          'accidental', log=True, fmt='{:.4%}',
                          outliers={i: any(o['metric'] == 'accidental'
                                           for o in notes[i]) for i in notes}))
    H.append(chart_metric(recs, lambda r: r['match']['cv_gap'] * 100,
                          'In-sample minus held-out efficiency '
                          '(overfitting of the per-bunch fit)',
                          'points', fmt='{:+.3f}', ref=0.0))

    H.append('<h2>Fit health</h2><p>The coarse search is what stops the fit '
             'locking onto the accidental floor. Its signal-to-noise is the '
             'single most diagnostic number on this page: below ~6 the fit '
             'refuses to run at all.</p>')
    H.append(chart_metric(
        recs, lambda r: (r['bootstrap'] or {}).get('snr'),
        'Coarse-search peak over the accidental floor', 'S/N', log=True,
        ref=Q.TH['boot_snr_warn'], fmt='{:.0f}'))
    H.append(chart_metric(
        recs, lambda r: (r['perbunch'] or {}).get('da_rms'),
        'Per-bunch offset scatter', 'da RMS (ns)', fmt='{:.2f}',
        outliers={i: any(o['metric'] == 'da_rms' for o in notes[i])
                  for i in notes}))
    H.append(chart_metric(
        recs, lambda r: (r['perbunch'] or {}).get('dk_rms_ppm'),
        'Per-bunch rate scatter', 'dk RMS (ppm)', fmt='{:.2f}'))
    H.append(chart_metric(
        recs, lambda r: (r['perbunch'] or {}).get('frac_fitted'),
        'Fraction of bunches that got their own correction', 'fraction',
        fmt='{:.1%}'))

    H.append('<h2>Residuals</h2>')
    H.append(chart_resid_overlay(recs))
    H.append(chart_drift(recs))

    H.append('<h2>What the slim kept</h2><p>Signal against the +100 µs '
             'control, summed over every segment. The control is the '
             'accidental floor measured the same way on the same events, so '
             'the difference is the real coincidence.</p>')
    H.append(chart_family_dt(recs))

    # --------------------------------------------------------- the segments
    H.append('<h2>Every segment</h2><table><tr><th>segment</th>'
             '<th>verdict</th><th class="num">K</th><th class="num">T0 (ns)</th>'
             '<th class="num">eff</th><th class="num">acc</th>'
             '<th class="num">resid RMS</th><th class="num">S/N</th>'
             '<th class="num">MB</th></tr>')
    for i, r in enumerate(recs):
        b = r['bootstrap'] or {}
        H.append(
            f'<tr><td>{CH.esc(_lab(r))}</td>'
            f'<td><span class="pill {r["verdict"]}">{r["verdict"]}</span></td>'
            f'<td class="num">{r["clock"]["K"]:.4e}</td>'
            f'<td class="num">{r["clock"]["T0_ns"]:+.1f}</td>'
            f'<td class="num">{r["match"]["efficiency"]:.2%}</td>'
            f'<td class="num">{r["match"]["accidental"]:.4%}</td>'
            f'<td class="num">{r["match"]["residual_rms"]:.2f}</td>'
            f'<td class="num">{b.get("snr", float("nan")):.0f}</td>'
            f'<td class="num">{r["segment"]["size_mb"]:.0f}</td></tr>')
    H.append('</table>')

    for i, r in enumerate(recs):
        rows = ''.join(
            f'<div class="chk"><span class="pill {c["level"]}">{c["level"]}'
            f'</span> <strong>{CH.esc(c["name"])}</strong> — '
            f'{CH.esc(c["detail"])}'
            + (f' <span style="color:var(--muted)">[{CH.esc(c["threshold"])}]'
               f'</span>' if c['threshold'] else '')
            + '</div>' for c in r['checks'])
        out = ''.join(
            f'<div class="chk"><span class="pill WARN">OUTLIER</span> '
            f'{CH.esc(o["metric"])} = {o["value"]:.4g} vs fleet median '
            f'{o["median"]:.4g} (robust z {o["z"]:.1f})</div>'
            for o in notes[i])
        H.append(f'<details><summary>{CH.esc(_lab(r))} — '
                 f'<span class="pill {r["verdict"]}">{r["verdict"]}</span>'
                 f'</summary>{rows}{out}</details>')

    H.append('</div><div id="tip"></div>')
    return (f'<title>{CH.esc(title)}</title>'
            f'<meta name="description" content="Per-segment QA for the '
            f'DREAM to n_TOF clock fit behind the slimmed ntof_hits files.">'
            f'<style>{CSS}</style>{"".join(H)}<script>{JS}</script>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('-o', '--out', type=Path, default=Path('clock_dashboard.html'))
    ap.add_argument('--title', default='DREAM ↔ n_TOF clock QA')
    ap.add_argument('--no-cache', action='store_true')
    a = ap.parse_args()

    print(f'collecting from {a.root} ...')
    recs = collect(a.root, use_cache=not a.no_cache)
    if not recs:
        print('no slim files found')
        return 2
    notes = population(recs)
    a.out.write_text(build(recs, notes, a.title, str(a.root)))
    v = {k: sum(1 for r in recs if r['verdict'] == k)
         for k in ('PASS', 'WARN', 'FAIL')}
    print(f'{len(recs)} segment(s): {v["PASS"]} pass, {v["WARN"]} warn, '
          f'{v["FAIL"]} fail, {sum(1 for i in notes if notes[i])} outlier(s)')
    print(f'-> {a.out}  ({a.out.stat().st_size/1000:.0f} kB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
