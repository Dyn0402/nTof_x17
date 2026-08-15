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
import collections
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
    roots = sorted(root.rglob('ntof_hits_*.root'))
    if not roots:
        # No ROOT files here: this is a records-only tree, rsynced from the
        # cluster so the page can be built without moving 8 GB of hits. The
        # records are complete -- clock_qa derives everything at write time.
        recs = sorted(root.rglob('clock_qa.json'))
        if recs:
            print(f'  no .root files; building from {len(recs)} '
                  f'clock_qa.json record(s)')
            return [json.loads(p.read_text()) for p in recs]
        return out
    for f in roots:
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
        scan(f'arm {a} residual mean',
             lambda r, a=a: r['match']['per_arm'][a].get('residual_mean'),
             1.0, label=f'arm {a} residual mean')
    scan('pss_primary',
         lambda r: (r['match'].get('pss_primary') or {}).get('within_core'),
         0.02, label='plastic primary within accept')
    scan('ringing_removed',
         lambda r: (r['hits'].get('pss_ringing') or {}).get('late_removed'),
         0.02, label='PSS late tail removed by ringing flag')
    return notes


# -------------------------------------------------------------------- time
def load_times(data_dir: Path | None = None):
    """{(dream_run, subrun): epoch seconds} for the x axis.

    Segment index tells a reader nothing; wall-clock time is what makes a
    time-localised problem visible -- the 2026-08-09 campaign's failures sat in
    three contiguous blocks, which is invisible when the axis is an index.
    Times come from the same cached listing the segment proposal uses.
    """
    d = data_dir or (Path(__file__).resolve().parents[3]
                     / 'ntof_processing' / 'slim_study' / 'coverage_inputs')
    try:
        sys.path.insert(0, str(d.parent))
        import coverage_map as cm                              # noqa: E402
        subs, _ = cm.load_dream(d / 'dream_eos_subruns.txt',
                                d / 'dream_daq_subruns.txt')
        return {k: v[0] for k, v in subs.items()}
    except Exception as e:                                     # noqa: BLE001
        print(f'  (no sub-run times, x axis falls back to index: {e})')
        return {}


def _date_ticks(lo, hi, max_ticks=12):
    """Midnight ticks, thinned to at most `max_ticks`, labelled MM-DD."""
    day = 86400.0
    start = (lo // day) * day
    step = day * max(1, int(np.ceil((hi - lo) / day / max_ticks)))
    out, t = [], start
    while t <= hi + step:
        if t >= lo - step:
            out.append((t, dt.datetime.utcfromtimestamp(t).strftime('%m-%d')))
        t += step
    return out


def _when(r):
    t = r.get('t_start')
    return (dt.datetime.utcfromtimestamp(t).strftime('%m-%d %H:%M')
            if t else '?')


# ------------------------------------------------------------------- charts
def _lab(r):
    s = r['segment']
    return f'{s["dream_run"]}/{s["dream_subrun"]}x{s["ntof_run"]}'


def _tip(r, value=''):
    """What a reader needs to act on a point: when, which DREAM sub-run, which
    n_TOF run, and where it sits in the ordering."""
    s = r['segment']
    head = f'{_when(r)}  ·  seg {r.get("index", "?")}'
    body = (f'DREAM {s["dream_run"]} / {s["dream_subrun"]}'
            f'  ·  n_TOF {s["ntof_run"]}')
    return f'{head}\n{body}' + (f'\n{value}' if value else '')


def chart_metric(recs, getter, title, ylabel, *, log=False, pct=False,
                 ref=None, band=None, fmt='{:.4g}', outliers=None):
    xs, xlab, dated = _xaxis(recs)
    ys = [getter(r) for r in recs]
    good = [y for y in ys if y is not None and np.isfinite(y)]
    if not good:
        return ''
    lo, hi = min(good), max(good)
    if band:
        lo, hi = min(lo, band[0]), max(hi, band[1])
    if ref is not None:
        lo, hi = min(lo, ref), max(hi, ref)
    f = CH.Frame(title=title, ylabel=ylabel,
                 xlabel='date (UTC)' if dated else 'segment')
    if dated:
        pad = (max(xs) - min(xs)) * 0.02 or 3600
        f.xlim(min(xs) - pad, max(xs) + pad, labels=xlab, exact=True)
    else:
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
    # On an index axis the DREAM runs need marking, because nothing else says
    # where one ends. On a date axis they are contiguous blocks separated by
    # real gaps, and 22 dashed lines plus a 22-entry key is noise -- the axis
    # and the hover already carry it.
    if not dated:
        for a_, b_, lab in _groups(recs)[:-1]:
            x = f.px(b_ + 0.5)
            f.body.append(f'<line class="grid" x1="{x:.1f}" y1="{f.t}" '
                          f'x2="{x:.1f}" y2="{f.h-f.b}" '
                          f'stroke-dasharray="2 3"/>')
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
        tips.append(_tip(r, f'{ylabel or "value"} = {vs}'
                         + ('   [fleet outlier on this metric]'
                            if is_out else '')))
    f.points(xs, ys, colours=cols, tips=tips)
    return (f'<figure>{f.svg()}'
            f'{"" if dated else _group_key(recs)}</figure>')


def _xaxis(recs):
    """(x values, tick labels, is_a_date_axis)."""
    ts = [r.get('t_start') for r in recs]
    if all(t for t in ts) and len(set(ts)) > 1:
        return ts, _date_ticks(min(ts), max(ts)), True
    return list(range(len(recs))), None, False


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
    xs, xlab, dated = _xaxis(recs)
    if dated:
        pad = (max(xs) - min(xs)) * 0.02 or 3600
        f.xlim(min(xs) - pad, max(xs) + pad, labels=xlab, exact=True)
        f.xlabel = 'date (UTC)'
    else:
        f.xlim(-0.5, len(recs) - 0.5, ticks=[])
    f.ylim(min(allv) - 3, max(allv) + 3)
    for a in ARMS:
        ys = [r['match']['per_arm'][a]['offset_ns'] for r in recs]
        f.points(xs, ys, colours=CH.ARM_COLOUR[a],
                 tips=[_tip(r, f'arm {a} offset = {y:+.2f} ns')
                       for r, y in zip(recs, ys)], r=2.8)
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
    """Signal, +100 us control and ringing-cut signal, per detector family.

    The third series is the point of the plot on PSS. The plastics ring: every
    large pulse is trailed by real secondary pulses out to ~1 us plus a fixed
    81-82 ns cable echo (`../../pss_ringing/`), and they are most of what the
    slim window holds -- 6.1x the coincident core. `clock_qa` flags them with
    `amp_0 < 0.05 x shadow_amp` on the same channel within 1 us and stores the
    surviving histogram, so the raw and cleaned shapes can be read against each
    other. What it should show: the PSS late tail collapsing onto the
    accidental floor, and WAL/LIQ barely moving -- the latter is the control
    that says the flag removes ringing rather than hits in general.
    """
    out = []
    for fam in FAMILIES:
        sig = ctl = cut = cutc = None
        meta = None
        for r in recs:
            fs = r['hits']['families'].get(fam)
            if not fs:
                continue
            s = np.array(fs['dt_signal']['counts'], float)
            c = np.array(fs['dt_control']['counts'], float)
            sig = s if sig is None else sig + s
            ctl = c if ctl is None else ctl + c
            if 'dt_signal_ring_cut' in fs:
                k = np.array(fs['dt_signal_ring_cut']['counts'], float)
                kc = np.array(fs['dt_control_ring_cut']['counts'], float)
                cut = k if cut is None else cut + k
                cutc = kc if cutc is None else cutc + kc
            meta = fs['dt_signal']
        if sig is None:
            continue
        f = CH.Frame(h=230, title=f'{fam}: hit time vs prediction',
                     ylabel='hits / bin', xlabel='dt (ns)')
        f.xlim(meta['lo'], meta['hi'])
        # Log y. The four series span three decades -- the coincidence peak,
        # the ringing tail, and the accidental floor the cleaned tail has to
        # fall onto -- and on a linear axis the only question the plot is asked
        # ("is the late tail gone?") is decided in a band two pixels tall.
        floor = min([x[x > 0].min() for x in (sig, ctl, cut, cutc)
                     if x is not None and (x > 0).any()] or [1.0])
        f.ylim(max(floor, 1.0) / 3, max(sig.max(), 1) * 3, log=True)
        f.step_hist(meta['lo'], meta['bin'], sig, colour=CH.PALETTE[0],
                    tip=f'{fam} signal')
        f.step_hist(meta['lo'], meta['bin'], ctl, colour=CH.PALETTE[1],
                    tip=f'{fam} +100 us control (accidental floor)')
        keys = [('signal', CH.PALETTE[0]), ('+100 µs control', CH.PALETTE[1])]
        cap_cut = ''
        if cut is not None and cut.sum():
            # BOTH sides of the subtraction get the flag. The +100 us control
            # is plastic singles too, so it rings exactly as hard (measured on
            # the reference segment: 62.7 % of signal hits flagged, 34.9 % of
            # control ones), and a cleaned signal against a raw control
            # subtracts a floor that is no longer there -- it reads 122 %
            # removed and puts the late excess NEGATIVE.
            f.step_hist(meta['lo'], meta['bin'], cut, colour=CH.PALETTE[2],
                        tip=f'{fam} signal, ringing flagged out')
            f.step_hist(meta['lo'], meta['bin'], cutc, colour=CH.PALETTE[3],
                        tip=f'{fam} +100 us control, ringing flagged out')
            keys += [('signal, ringing removed', CH.PALETTE[2]),
                     ('control, ringing removed', CH.PALETTE[3])]
            late = slice(int(len(sig) * 0.55), None)     # the tail, +100 ns on
            ex0 = sig[late].sum() - ctl[late].sum()
            ex1 = cut[late].sum() - cutc[late].sum()
            cap_cut = (f' Against the equally-flagged control, the ringing cut '
                       f'takes the late (dt &gt; +100 ns) excess from '
                       f'{ex0:,.0f} to {ex1:,.0f} hits '
                       f'({1 - ex1/max(ex0, 1):.0%}), for '
                       f'{1 - cut.sum()/max(sig.sum(), 1):.0%} of all {fam} '
                       f'hits in the window.')
        edge = int(len(sig) * 0.05)
        frac = float((sig[:edge].sum() + sig[-edge:].sum() -
                      ctl[:edge].sum() - ctl[-edge:].sum())
                     / max(sig.sum() - ctl.sum(), 1))
        out.append(
            f'<figure>{f.svg()}'
            + CH.legend(keys)
            + f'<figcaption>{fam}: {frac:.1%} of the background-subtracted '
              f'excess sits in the outer 10 % of the window. Near zero means '
              f'the window contains the coincidence; a large value means it '
              f'is being cut.{cap_cut}</figcaption></figure>')
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
font-size:12px;padding:6px 10px;border-radius:6px;opacity:0;transition:opacity
.1s;z-index:9;max-width:340px;white-space:pre-line;line-height:1.45}
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


def build(recs, notes, title, source, coverage=None, ledger_dir=None):
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

    # ------------------------------------------- UNMATCHED PULSES, FIRST
    #
    # The follow-up list, and the reason the page exists now: for every
    # (sub-run x n_TOF run), how many beam pulses failed the coincidence test
    # -- under `pulse_min_frac` of their triggers having a wall AND plastic hit
    # on the same arm in the accept window. This is the number that has to
    # reach zero, so it goes above everything else; the clock diagnostics below
    # explain WHY a segment misses, but this says which ones do.
    #
    # Not the same as 'bunches fitted', which asks whether a bunch got its own
    # per-bunch clock correction. A pulse can be perfectly matched and still
    # lack its own correction, and vice versa.
    # ------------------------------------- every pulse accounted for, FIRST
    # The pulse ledger's own section, delivered by pulse_ledger as a function
    # so this file stays single-owner (contract 2026-08-13). It goes ABOVE the
    # follow-up list below because it carries the DENOMINATOR: this file can
    # only ever count pulses that reached a product, while the ledger counts
    # every censused burst -- including the segments that failed, the inter-run
    # gaps and the spans never attempted, which are invisible from here and are
    # exactly where the campaign's real losses live.
    #
    # Absent ledger -> empty string -> no section. The dashboard must still
    # build from products alone, because it did for the whole campaign before
    # the ledger existed.
    have_ledger = False
    if ledger_dir:
        try:
            from ntof_processing.slim_pipeline import pulse_ledger
            sec = pulse_ledger.build_dashboard_section(Path(ledger_dir))
            if sec:
                H.append(sec)
                have_ledger = True
            else:
                H.append('<div class="sub warn">Pulse ledger directory given '
                         f'({CH.esc(str(ledger_dir))}) but it holds no '
                         'campaign_ledger.json -- run pulse_ledger campaign '
                         'first. Counts below are PRODUCT-side only.</div>')
        except Exception as e:                                   # noqa: BLE001
            # A broken ledger must not cost the clock dashboard, but it must
            # not vanish either -- silence here would read as "no losses".
            H.append('<div class="sub bad">Pulse ledger section failed to '
                     f'build ({CH.esc(type(e).__name__)}: {CH.esc(str(e))}). '
                     'Counts below are PRODUCT-side only and do NOT include '
                     'failed segments, gaps or unattempted spans.</div>')

    pu = [(r, (r.get('pulses') or {})) for r in recs]
    have = [(r, d) for r, d in pu if d.get('n')]
    if have:
        tot_p = sum(d['n'] for _, d in have)
        tot_u = sum(d['unmatched'] for _, d in have)
        nseg_bad = sum(1 for _, d in have if d['unmatched'])
        H.append('<h2>Unmatched pulses &mdash; the follow-up list</h2>')
        # Point at the ledger only when there IS one above. Saying "see the
        # pulse ledger above" on a page that has no ledger section sends the
        # reader looking for something that is not there -- and worse, implies
        # the denominator has been accounted for somewhere when it has not.
        H.append('<div class="sub">Product-side: pulses that reached a slim. '
                 + ('For the campaign denominator &mdash; including segments '
                    'that failed, inter-run gaps and spans never attempted '
                    '&mdash; see the pulse ledger above.'
                    if have_ledger else
                    'This does NOT include segments that failed, inter-run '
                    'gaps or spans never attempted &mdash; those are invisible '
                    'from the products alone. Build with --ledger for the '
                    'campaign denominator.')
                 + '</div>')
        H.append(f'<div class="sub">{tot_u:,} of {tot_p:,} beam pulses failed '
                 f'the coincidence test, across {nseg_bad} of {len(have)} '
                 f'segments. A pulse counts as matched when at least '
                 f'{100*Q.TH["pulse_min_frac"]:.0f}% of its triggers have a '
                 f'wall+plastic coincidence on the same arm inside the accept '
                 f'window.</div>')
        worst = sorted(have, key=lambda rd: -rd[1]['unmatched'])
        rows = ['<table class="tbl"><tr><th>segment</th><th>n_TOF</th>'
                '<th>pulses</th><th>unmatched</th><th>median pulse</th></tr>']
        for r, d in worst:
            if not d['unmatched']:
                continue
            s = r['segment']
            cls = ('bad' if d['unmatched'] > Q.TH['pulse_unmatched_warn']
                   else 'warn')
            rows.append(
                f'<tr><td>{CH.esc(s["dream_run"])}/'
                f'{CH.esc(s["dream_subrun"])}</td>'
                f'<td>{s["ntof_run"]}</td><td>{d["n"]:,}</td>'
                f'<td class="{cls}">{d["unmatched"]:,}</td>'
                f'<td>{d.get("median_frac", 0):.1%}</td></tr>')
        rows.append('</table>')
        if nseg_bad:
            H.append(''.join(rows))
        else:
            H.append('<div class="sub good">Every pulse in every segment '
                     'matched. Nothing to follow up.</div>')

    # ------------------------------------------------ what to look at first
    #
    # A check that fires on most of the campaign is describing the DATASET, not
    # picking out a segment. Left in the triage table it drowns the handful of
    # genuinely unusual segments -- 45 identical "plastic tail" rows buried the
    # one real outlier on the first real run of this page. So: anything firing
    # on more than half the segments is reported once, as a campaign property,
    # and excluded from the per-segment list.
    # A quarter, not a half: the plastic-tail check fires on 39 % of the July
    # campaign, which is plainly a dataset property, and at a 50 % cut it still
    # buried the one real outlier.
    SYSTEMIC = 0.25
    fired = collections.Counter()
    for r in recs:
        for c in r['checks']:
            if c['level'] in ('WARN', 'FAIL'):
                fired[c['name']] += 1
    systemic = {name for name, k in fired.items()
                if len(recs) and k >= SYSTEMIC * len(recs)}
    if systemic:
        H.append('<h2>Systematic across the campaign</h2><p>These fire on most '
                 'segments, so they are properties of the dataset rather than '
                 'of any one segment. They are excluded from the triage list '
                 'below.</p><table><tr><th>check</th><th class="num">segments'
                 '</th><th>what it means</th></tr>')
        for cname in sorted(systemic, key=lambda x: -fired[x]):
            ex = next(c for r in recs for c in r['checks']
                      if c['name'] == cname and c['level'] in ('WARN', 'FAIL'))
            H.append(f'<tr><td>{CH.esc(cname)}</td>'
                     f'<td class="num">{fired[cname]} / {len(recs)}</td>'
                     f'<td>{CH.esc(ex["detail"])}</td></tr>')
        H.append('</table>')

    trouble = [(i, r) for i, r in enumerate(recs)
               if notes[i] or any(c['level'] in ('WARN', 'FAIL')
                                  and c['name'] not in systemic
                                  for c in r['checks'])]
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
                   if c['level'] in ('WARN', 'FAIL')
                   and c['name'] not in systemic]
            why += [f'{o["metric"]} {o["value"]:.4g} vs fleet '
                    f'{o["median"]:.4g} (z={o["z"]:.1f})' for o in notes[i]]
            H.append(f'<tr><td>{CH.esc(_lab(r))}</td>'
                     f'<td><span class="pill {r["verdict"]}">{r["verdict"]}'
                     f'</span></td><td>{CH.esc("; ".join(why))}</td></tr>')
        H.append('</table>')

    # -------------------------------------------------------------- charts
    if coverage:
        cov, tot = coverage['covered'], coverage['total']
        H.append('<h2>Coverage — what is NOT on this page</h2>')
        H.append(f'<div class="callout bad"><strong>{tot - cov} of {tot} DREAM '
                 f'sub-runs produced no slim at all</strong> '
                 f'({coverage.get("uncovered_pct", 0):.0f} % of beam minutes) '
                 f'and are therefore absent from every chart below. A page '
                 f'showing only what succeeded would read as a clean bill of '
                 f'health for the campaign; it is a clean bill of health for '
                 f'{cov} sub-runs.</div>')
        if coverage.get('note'):
            H.append(f'<p>{coverage["note"]}</p>')
        seg = []
        for label, k, c in (('slimmed', cov, 'var(--good)'),
                            ('no coincidence found', tot - cov, 'var(--bad)')):
            if k:
                seg.append(f'<span style="width:{100*k/tot:.2f}%;'
                           f'background:{c}" data-tip="{k} sub-runs {label}">'
                           f'{k}</span>')
        H.append(f'<div class="covbar">{"".join(seg)}</div>')

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
    H.append(chart_metric(
        recs, lambda r: max((abs(r['match']['per_arm'][a].get(
            'residual_mean', float('nan'))) for a in ARMS
            if np.isfinite(r['match']['per_arm'][a].get(
                'residual_mean', float('nan')))), default=None),
        'Worst per-arm residual centring — a wrong arm offset hides from '
        'every global check', '|mean| (ns)', fmt='{:.2f}',
        ref=Q.TH['arm_resid_warn'],
        outliers={i: any('residual mean' in o['metric'] for o in notes[i])
                  for i in notes}))

    H.append('<h2>The plastics ring</h2><p>Every large plastic pulse is '
             'followed by a train of real secondary pulses out to ~1 µs '
             '(<code>pss_ringing/</code>), and they are most of the PSS '
             'content of the slim window. Two questions decide whether the '
             '±25 ns analysis slice is safe: per matched trigger, does the '
             '<strong>largest</strong> plastic pulse on the trigger arm land '
             'inside it (the earliest almost never does — unrelated singles '
             'at 720 kHz); and is the late tail fully explained by the '
             'shadow flag (a hit under 5 % of a bigger hit on the same '
             'channel in the previous µs), i.e. ringing rather than '
             'mis-handled coincidence yield.</p>')
    H.append(chart_metric(
        recs, lambda r: (r['match'].get('pss_primary') or {}).get('within_core'),
        'Largest plastic pulse on the trigger arm within ±25 ns',
        'fraction of matched triggers', fmt='{:.1%}',
        ref=Q.TH['pss_primary_warn'],
        outliers={i: any(o['metric'] == 'plastic primary within accept'
                         for o in notes[i]) for i in notes}))
    H.append(chart_metric(
        recs, lambda r: (r['hits'].get('pss_ringing') or {}).get('late_removed'),
        'PSS 150–1000 ns excess removed by the shadow flag',
        'fraction of late excess', fmt='{:.1%}',
        ref=Q.TH['ringing_removed_warn'],
        outliers={i: any('ringing' in o['metric'] for o in notes[i])
                  for i in notes}))
    H.append(chart_metric(
        recs, lambda r: (r['hits'].get('pss_ringing') or {}).get('core_cost'),
        'Core cost of the flag (small-amplitude hits inside ±25 ns it also '
        'removes)', 'fraction of core excess', fmt='{:.1%}'))

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
        'Fraction of bunches WITH BEAM that got their own correction',
        'fraction', fmt='{:.1%}'))

    # The beam, reported and never judged. Availability is the PS's business:
    # a segment can be perfect and still sit at 0.86, and every one of the
    # first campaign's four 'bunches fitted' WARNs was exactly that
    # (FINDINGS_2026-08-10_unfitted_bunches.md). The parasitic mix belongs
    # beside it because it is most of the efficiency spread above -- r = -0.82
    # over the campaign, 97.7 % dedicated against 91.2 % parasitic.
    if any((r['perbunch'] or {}).get('has_beam_column') for r in recs):
        H.append('<h2>The beam</h2><p>Neither of these is a quality metric. '
                 'Empty pulses — PS pulses that delivered no protons, '
                 'intensity below 10e10 with no gamma flash — are filtered out '
                 'of the slim, and the fraction of them is the accelerator '
                 'talking, not the fit. The parasitic mix is here because it '
                 'sets the match efficiency: 97.7 % on dedicated pulses '
                 'against 91.2 % on parasitic.</p>')
        H.append(chart_metric(
            recs, lambda r: (r['perbunch'] or {}).get('beam_availability'),
            'Beam availability (pulses that delivered protons)', 'fraction',
            fmt='{:.1%}'))
        H.append(chart_metric(
            recs, lambda r: (r['perbunch'] or {}).get('parasitic_fraction'),
            'Parasitic fraction of the delivered pulses', 'fraction',
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
    H.append('<h2>Every segment</h2><table><tr><th class="num">#</th>'
             '<th>when (UTC)</th><th>segment</th>'
             '<th>verdict</th><th class="num">K</th><th class="num">T0 (ns)</th>'
             '<th class="num">eff</th><th class="num">acc</th>'
             '<th class="num">resid RMS</th><th class="num">S/N</th>'
             '<th class="num">MB</th></tr>')
    for i, r in enumerate(recs):
        b = r['bootstrap'] or {}
        H.append(
            f'<tr><td class="num">{r.get("index", i+1)}</td>'
            f'<td>{_when(r)}</td><td>{CH.esc(_lab(r))}</td>'
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
        H.append(f'<details><summary>{r.get("index", i+1)}. {_when(r)} — '
                 f'{CH.esc(_lab(r))} — '
                 f'<span class="pill {r["verdict"]}">{r["verdict"]}</span>'
                 f'</summary>{rows}{out}</details>')

    H.append('</div><div id="tip"></div>')
    # A COMPLETE document, not a fragment: this file has to render identically
    # from the file system, from the DAQ web page's analysis tab, and as a
    # published note, none of which wrap it in anything.
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,'
            f'initial-scale=1">'
            f'<title>{CH.esc(title)}</title>'
            f'<meta name="description" content="Per-segment QA for the DREAM '
            f'to n_TOF clock fit behind the slimmed ntof_hits files: '
            f'{len(recs)} segments, absolute checks and fleet outliers.">'
            f'<style>{CSS}</style></head><body>'
            f'{"".join(H)}<script>{JS}</script></body></html>')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('-o', '--out', type=Path, default=Path('clock_dashboard.html'))
    ap.add_argument('--title', default='DREAM ↔ n_TOF clock QA')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--ledger', type=Path, default=None,
                    help='pulse_ledger directory (<qa-root>/pulse_ledger); '
                         'adds the every-pulse-accounted-for section')
    ap.add_argument('--coverage', type=Path, default=None,
                    help='JSON with {covered,total,uncovered_pct,note} -- what '
                         'is missing from this tree, so the page cannot read '
                         'as a clean bill of health for the whole campaign')
    a = ap.parse_args()

    print(f'collecting from {a.root} ...')
    recs = collect(a.root, use_cache=not a.no_cache)
    if not recs:
        print('no slim files found')
        return 2

    # Order by wall clock, not by filename. Sorting by path interleaves DREAM
    # runs (run_100 sorts before run_77) and scatters a time-localised problem
    # across the whole axis.
    times = load_times()
    for r in recs:
        s = r['segment']
        r['t_start'] = times.get((s['dream_run'], s['dream_subrun']))
    recs.sort(key=lambda r: (r['t_start'] is None, r['t_start'] or 0,
                             r['segment']['ntof_run']))
    for i, r in enumerate(recs, 1):
        r['index'] = i
    dated = sum(1 for r in recs if r['t_start'])
    print(f'  {dated}/{len(recs)} segment(s) have a wall-clock time')

    notes = population(recs)
    cov = json.loads(a.coverage.read_text()) if a.coverage else None
    a.out.write_text(build(recs, notes, a.title, str(a.root), cov,
                           ledger_dir=a.ledger))
    v = {k: sum(1 for r in recs if r['verdict'] == k)
         for k in ('PASS', 'WARN', 'FAIL')}
    print(f'{len(recs)} segment(s): {v["PASS"]} pass, {v["WARN"]} warn, '
          f'{v["FAIL"]} fail, {sum(1 for i in notes if notes[i])} outlier(s)')
    print(f'-> {a.out}  ({a.out.stat().st_size/1000:.0f} kB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
