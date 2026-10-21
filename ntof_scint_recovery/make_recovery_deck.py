#!/usr/bin/env python3
"""
make_recovery_deck.py -- the slide note of the quick pass: example and average
waveforms after the gamma flash, with the recovery time drawn on them and
quoted per scintillator, always split by beam-pulse intensity.

    .venv/bin/python -m ntof_scint_recovery.make_recovery_deck
    python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
        <out>/deck/scint-flash-recovery.html --slug scint-flash-recovery --force --deploy

Reads only the quick pass's outputs (out_dir('quick'): curves.npz, recovery.csv,
recovery_by_run.csv, bunches.csv, examples.npz), so rerunning quick.py after more
runs land moves every number and figure here.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
import sys
from pathlib import Path

import numpy as np

from ntof_scint_recovery.common import REPO, out_dir
from ntof_scint_recovery.pulse_analysis import AMP_FLOOR as AMPF
from ntof_scint_recovery.quick import DETS, DIVERT_OFF, EPOCHS, PRODUCTION_FROM
from ntof_scint_recovery.trig_analysis import WALL_THR

sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd  # noqa: E402

Q = out_dir('quick')
#: categorical slots 1-4 for the four arms (slidedoc palette)
ARM_COL_SD = (sd.BLUE, sd.ORANGE, sd.GREEN, sd.PURPLE)
FAM = {'WAL': 'SiPM wall', 'PSS': 'plastic', 'LIQ': 'liquid'}
CC = {'par': sd.BLUE, 'ded': sd.ORANGE}
CN = {'par': 'parasitic', 'ded': 'dedicated'}
TOL = 3.0
INVENTORY = REPO / 'ntof_processing/flash_pull/inputs/run_inventory_2026-10-20.csv'
#: configuration changes worth marking on a time axis (CLAUDE.md, flash_timing/)
EVENTS = [(dt.date(2026, 7, 22), 'wall gate delay 200→1000 ns'),
          (dt.date(2026, 7, 23), 'DREAM clock 6→4, noise ×2')]
D0 = dt.date(2026, 7, 1)


# ------------------------------------------------------------------ data
def rows(name):
    return list(csv.DictReader(open(Q / name)))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def recovery():
    R = {}
    for r in rows('recovery.csv'):
        R[(r['det'], r['ch'], r['cls'])] = {k: fnum(r[k]) for k in ('n', 't3mV_us', 't1mV_us')}
    return R


def by_run():
    out = {}
    for r in rows('recovery_by_run.csv'):
        out.setdefault((r['det'], r['cls']), []).append((int(r['run']), fnum(r['t3mV_us']), fnum(r['t1mV_us'])))
    return out


def run_dates():
    inv = list(csv.DictReader(open(INVENTORY)))
    known = {}
    for r in inv:
        d = r['date']
        if d and d not in ('', '-1') and len(d) == 7:
            known[int(r['run'])] = dt.date(2000 + int(d[1:3]), int(d[3:5]), int(d[5:7]))
    ks = sorted(known)

    def date(run):
        if run in known:
            return known[run]
        j = np.searchsorted(ks, run)
        return known[ks[min(max(j - 1, 0), len(ks) - 1)]]
    return date


def fmt_t(us):
    if not np.isfinite(us):
        return '—'
    if us < 10:
        return f'{us:.1f} µs'
    if us < 1000:
        return f'{us:.0f} µs'
    return f'{us / 1e3:.1f} ms'


def fam_t(R, fam, cls, key='t3mV_us'):
    v = [R[(d, 'all', cls)][key] for d in DETS if d.startswith(fam) and (d, 'all', cls) in R]
    v = [x for x in v if np.isfinite(x)]
    return (min(v), float(np.median(v)), max(v)) if v else (math.nan,) * 3


# ------------------------------------------------------------------ plotting helpers
def tlog_ticks(lo, hi):
    labs = {0.1: '0.1 µs', 1: '1 µs', 10: '10 µs', 100: '100 µs', 1e3: '1 ms', 1e4: '10 ms', 3e4: '30 ms'}
    return [(v, s) for v, s in labs.items() if lo <= v <= hi]


def clip(v, lo, hi):
    return [min(max(float(x), lo), hi) if np.isfinite(x) else None for x in v]


def env_band(P, xs, lo, hi, color, ylo, yhi, alpha=0.55, tip=None):
    P.band(xs, clip(lo, ylo, yhi), clip(hi, ylo, yhi), color, alpha=alpha, tip=tip)


# ------------------------------------------------------------------ slides
def cover(D, R, B):
    live = [b for b in B if b['cls'] in ('par', 'ded') and int(b['n_live']) > 0]
    nrun = len({int(b['run']) for b in live})
    npar = sum(b['cls'] == 'par' for b in live)
    nded = sum(b['cls'] == 'ded' for b in live)
    tl = fam_t(R, 'LIQ', 'ded')[1]
    tp = fam_t(R, 'PSS', 'ded')[1]
    tw = fam_t(R, 'WAL', 'ded')[1]
    rat = np.nanmedian([R[(d, 'all', 'ded')]['t3mV_us'] / R[(d, 'all', 'par')]['t3mV_us']
                        for d in DETS if (d, 'all', 'ded') in R and (d, 'all', 'par') in R])
    head = (f'Back within 3 mV of baseline after ~{fmt_t(tl)} (liquids), ~{fmt_t(tp)} (plastics) '
            f'and ~{fmt_t(tw)} (SiPM walls), for dedicated pulses')
    sub = (f'Dedicated pulses (twice the protons) take ×{rat:.1f} longer than parasitic; median over the 12 detectors, '
           f'production configuration (runs ≥ {PRODUCTION_FROM})')
    body = (sd.kicker('n_TOF EAR2 · X17 scintillators · γ-flash recovery, quick pass')
            + sd.title(f'<span style="color:{sd.DINK}">{head}</span>', f'<span style="color:{sd.DMUT}">{sub}</span>')
            + sd.row(sd.bignum(f'{nrun}', 'runs with scintillators', sd.DBLUE,
                               'with beam and at least one live channel', tip='flash_pull plan, 2026-10-20'),
                     sd.bignum(f'{npar + nded:,}', 'bunches of raw waveforms', sd.DGREEN,
                               f'{npar:,} parasitic · {nded:,} dedicated'),
                     sd.bignum('44', 'channels', sd.DGREY, '32 wall ends · 8 plastic ends · 4 liquids'),
                     sd.bignum('0–10 ms', 'kept after every flash', sd.DRED, '+ 30–32 ms as a reference'), gap=56))
    notes = ('<p>Quick pass: the baseline of every channel, measured on the raw samples, against time since the '
             'flash. "Recovered" here means the median baseline is back within ±3 mV of its pre-flash level and '
             'stays there to 10 ms. This is the electronics coming back. It does not yet say a pulse has its '
             'full amplitude again; that is the full analysis (last slides).</p>'
             '<p>Every number is split by beam-pulse intensity: parasitic (~4×10¹² protons) and dedicated '
             '(~8.5×10¹²). Pooling them would quote a time that belongs to neither.</p>'
             f'<p>Dataset: <code>/eos/experiment/ntof/data/x17/flash_tail</code> (<code>ntof_processing/flash_pull</code>). '
             'Summaries: <code>ntof_scint_recovery/flash_summary.py</code>; this pass: <code>quick.py</code>.</p>')
    D.slide('cover', body, notes, dark=True, short='cover')


def timeline(D):
    P = sd.Plot(1664, 520, x=(0.05, 4e4, 'log'), y=(0, 6), xlabel='time since the γ-flash at the detector', margin=(24, 40, 92, 40))
    P.xticks(tlog_ticks(0.05, 4e4))
    bands = [
        (0.05, 0.7, 5.0, sd.GREY, 'wall blanking ~0.7 µs', 'The SiPM walls are switched off (divert gate) around the flash in every run except 224356–360, 224464, 224466.'),
        (0.05, 18, 4.0, sd.PURPLE, 'unsuppressed head: the flash itself', 'The DAQ always keeps 30 µs from t = 0 un-zero-suppressed; the flash arrives ~11.6 µs in, so ~18 µs of it is after the flash.'),
        (0.05, 1e4, 3.0, sd.GREEN, 'this dataset: every block, 0–10 ms', 'flash_pull keeps every zero-suppressed block of all 44 channels and PKUP.'),
        (3e4, 3.2e4, 3.0, sd.GREEN, 'ref.', '30–32 ms: same bunch, far from the flash -- the full-amplitude reference.'),
        (500, 4e4, 2.0, sd.BLUE, 'DREAM (MM) triggers', 'DREAM accepts triggers from ~0.5 ms; the MM tracks exist only there.'),
        (1e3, 4e4, 1.0, sd.ORANGE, 'X17 physics: neutrons ≲ 2 eV', 'E ≈ (72.3 × 19.5 m / t[µs])² eV: 1 ms ↔ 2 eV.'),
    ]
    for lo, hi, y, c, lab, tp in bands:
        P.rect(lo, hi, y - 0.35, y + 0.35, color=c, w=0, fill=c, tip=tp)
        inside = P.X(hi) - P.X(lo) > 420
        P.text(lo * 1.15 if inside else hi * 1.15, y - 0.12, lab, 22, sd.CARD if inside else c, 'start', 600)
    for t_us, e in ((1, '2 MeV'), (10, '20 keV'), (100, '200 eV'), (1e3, '2 eV'), (1e4, '20 meV')):
        P.text(t_us, 5.75, f'Eₙ {e}', 20, sd.MUT, 'middle')
    P.vline(0.05, sd.INK, None, 3)
    body = (sd.title('After the flash the detectors are blind, then recover: this dataset covers it all',
                     'What happens when, on the bunch timeline (log scale); the neutron energy at each time is on top')
            + P.svg('timeline')
            + sd.p('Hover any dotted word, band or point for its definition and numbers.', 24, sd.MUT))
    notes = ('<p>The flash time at each detector is t<sub>PKUP</sub> + C[detector], with C from '
             '<code>ntof_processing/flash_timing</code> (walls, divert-off calibration) and run 224572 (plastics, '
             'liquids). That is good to ~10 ns, which no result here resolves.</p>'
             '<p>The pulse-shape (PSA) output and the earlier waveform pull both start too late for this: the PSA '
             'gives hits, not baselines, and the waveform pull is ±5 µs around DREAM triggers, which start at ~0.5 ms.</p>')
    D.slide('timeline', body, notes, short='timeline')


def intensity(D, B):
    ppp = np.array([fnum(b['ppp']) for b in B])
    pk = np.array([fnum(b['pk_amp_mV']) for b in B])
    cls = np.array([b['cls'] for b in B])
    has = np.array([int(b['has_official']) for b in B]) == 1
    P1 = sd.Plot(810, 560, x=(0, 10), y=(0, 1), xlabel='protons per pulse (×10¹²)', ylabel='bunches (fraction of max)',
                 title='PulseIntensity, processed runs')
    e = np.arange(0, 10.01, 0.25)
    h, _ = np.histogram(ppp[has] / 1e12, e)
    for i, v in enumerate(h):
        c = CC['par'] if e[i] < 6.5 and e[i] >= 0.1 else CC['ded'] if e[i] >= 6.5 else sd.GREY
        P1.vbar(e[i] + 0.125, v / max(h.max(), 1), 22, c, tip=f'{e[i]:.2f}–{e[i + 1]:.2f}×10¹²: {v} bunches')
    P1.vline(6.5, sd.INK, '6 5', 2, label='cut 6.5')
    P1.xticks([(v, str(v)) for v in range(0, 11, 2)]).yticks([(0, '0'), (0.5, '0.5'), (1, '1')])
    P2 = sd.Plot(810, 560, x=(0, 1200), y=(0, 1), xlabel='PKUP flash amplitude (mV)', ylabel='bunches (fraction of max)',
                 title='the same split seen by PKUP')
    e2 = np.arange(0, 1201, 25)
    tops = []
    for c in ('par', 'ded'):
        hh, _ = np.histogram(pk[has & (cls == c)], e2)
        tops.append((c, hh))
    mx = max(t[1].max() for t in tops) or 1
    for c, hh in tops:
        P2.line((e2[:-1] + 12.5).tolist(), (hh / mx).tolist(), CC[c], 3, markers=False,
                tip=f'{CN[c]}: {int(hh.sum())} processed bunches')
    hn, _ = np.histogram(pk[~has], e2)
    if hn.sum():
        P2.line((e2[:-1] + 12.5).tolist(), (hn / max(hn.max(), 1) * 0.5).tolist(), sd.GREY, 2.5, dash='6 5', markers=False,
                tip=f'unprocessed runs: {int(hn.sum())} bunches, classified on this amplitude (scaled ×0.5)')
    P2.xticks([(v, str(v)) for v in range(0, 1201, 300)]).yticks([(0, '0'), (0.5, '0.5'), (1, '1')])
    n = {c: int((cls == c).sum()) for c in ('par', 'ded', 'none')}
    body = (sd.title('Two beam intensities, cleanly separated: every result is split on them',
                     f'{n["par"]:,} parasitic and {n["ded"]:,} dedicated bunches; {n["none"]} with no protons are dropped')
            + sd.row(P1.svg('intensity histogram'), P2.svg('PKUP amplitude'), gap=44)
            + sd.legend([('parasitic', CC['par'], 'box'), ('dedicated', CC['ded'], 'box'),
                         ('unprocessed runs (PKUP only)', sd.GREY, 'dash')]))
    notes = ('<p>Class: PulseIntensity from the official processed file (&lt; 10¹¹ none, &lt; 6.5×10¹² parasitic, '
             'otherwise dedicated). The 31 runs n_TOF never processed have no PulseIntensity; their bunches are '
             'classed on the PKUP flash amplitude in the product, cut midway between the two classes\' medians on '
             'the processed bunches. The right panel shows that this separates the classes as cleanly as the '
             'intensity does.</p>')
    D.slide('intensity', body, notes, short='intensity split')


def examples(D, R, X, fam):
    run = int(X['run'])
    dets = [d for d in DETS if d.startswith(fam)]
    det = dets[0]
    yr = {'WAL': (-25, 60), 'PSS': (-15, 60), 'LIQ': (-15, 60)}[fam]
    lin = X['lin_edges_us']
    lc = 0.5 * (lin[:-1] + lin[1:])
    lg = X['log_edges_us']
    gc = np.sqrt(lg[:-1] * lg[1:])
    ymax = max(float(np.max(X[f'{c}_{det}_lin_hi'])) for c in ('par', 'ded') if f'{c}_{det}_lin_hi' in X)
    ytop = 250 * math.ceil(ymax * 1.05 / 250)
    P1 = sd.Plot(810, 580, x=(-3, 45), y=(-50, ytop), xlabel='time since the flash (µs)', ylabel='signal (mV, pulse direction up)',
                 title='the flash')
    P2 = sd.Plot(810, 580, x=(0.3, 1e4, 'log'), y=yr, xlabel='time since the flash', ylabel='signal (mV)',
                 title='the recovery, zoomed to a few × threshold')
    for c in ('ded', 'par'):
        k = f'{c}_{det}'
        if k + '_lin_lo' not in X:
            continue
        tip = f'{CN[c]} bunch {int(X[c + "_bunch"])}, {X[c + "_ppp"] / 1e12:.1f}×10¹² protons'
        env_band(P1, lc.tolist(), X[k + '_lin_lo'], X[k + '_lin_hi'], CC[c], -50, ytop, alpha=0.35, tip=tip)
        env_band(P2, gc.tolist(), X[k + '_log_lo'], X[k + '_log_hi'], CC[c], *yr, alpha=0.22, tip=tip)
        if k + '_log_med' in X:
            P2.line(gc.tolist(), clip(X[k + '_log_med'], *yr), CC[c], 3.5, markers=False,
                    tip=f'{tip}: median of the samples in each column')
        t = R.get((det, 'all', c), {}).get('t3mV_us', math.nan)
        if np.isfinite(t):
            P2.vline(t, CC[c], '8 5', 3, tip=f'{det} {CN[c]}: baseline within ±3 mV after {fmt_t(t)} (median over all bunches)')
            P2.text(t * (1.12 if c == 'ded' else 0.89), yr[1] - 0.08 * (yr[1] - yr[0]), f'{fmt_t(t)}', 24, CC[c],
                    'start' if c == 'ded' else 'end', 600)
    for P in (P1, P2):
        P.hline(0, sd.INK, None, 1.5)
    P2.hline(3, sd.MUT, '4 4', 1.5).hline(-3, sd.MUT, '4 4', 1.5, label='±3 mV', where='below')
    P1.xticks([(v, str(v)) for v in (0, 10, 20, 30, 40)]).yticks([(v, str(v)) for v in range(0, int(ytop) + 1, 250)])
    P2.xticks(tlog_ticks(0.3, 1e4)).yticks([(v, str(v)) for v in range(-10, int(yr[1]) + 1, 10) if v >= yr[0]])
    t_d = R.get((det, 'all', 'ded'), {}).get('t3mV_us', math.nan)
    t_p = R.get((det, 'all', 'par'), {}).get('t3mV_us', math.nan)
    what = {'WAL': 'the diverted flash clamps, decays, then undershoots the baseline',
            'PSS': 'the PMT recovers in µs; a pulse train from the beam rides on it',
            'LIQ': 'the PMT recovers fastest; pulses are back on a flat baseline in µs'}[fam]
    body = (sd.title(f'{FAM[fam].capitalize()} {det[3]}: back within 3 mV after {fmt_t(t_p)} (parasitic), {fmt_t(t_d)} (dedicated)',
                     f'One bunch of each intensity, raw samples, run {run}: {what}')
            + sd.row(P1.svg('flash'), P2.svg('recovery'), gap=44)
            + sd.legend([(f'parasitic bunch {int(X["par_bunch"])}', CC['par'], 'box'),
                         (f'dedicated bunch {int(X["ded_bunch"])}', CC['ded'], 'box'),
                         ('median of the samples', sd.INK, 'line'),
                         ('recovery time (all bunches, this detector)', sd.MUT, 'dash')]))
    notes = (f'<p>The band is the min-to-max envelope of the samples in each display column; a zero-suppression '
             f'gap is drawn at 0 (the signal was inside the ZS band). Channel 1 of {det}. The vertical lines are '
             f'the recovery time of {det} from every bunch of that class, not from this bunch.</p>'
             '<p>Sign: up is the pulse direction for every family (walls are positive-going, plastics and liquids '
             'negative-going on the ADC).</p>')
    D.slide(f'ex_{fam}', body, notes, foot=f'Raw stream1 via flash_pull, run {run}; flash time from PKUP + C[detector]',
            short=f'example {FAM[fam]}')


def averages(D, C, fam_list=('WAL', 'PSS', 'LIQ')):
    lin = C['lin_edges_us']
    lcv = 0.5 * (lin[:-1] + lin[1:])
    xm = (lcv >= -2) & (lcv <= 40)
    lc = lcv[xm].tolist()
    panels = []
    for fam in fam_list:
        dets = [d for d in DETS if d.startswith(fam)]
        ys = {}
        for c in ('par', 'ded'):
            arr = [C[f'{d}_{c}_lin'] for d in dets if f'{d}_{c}_lin' in C]
            if arr:
                ys[c] = np.nanmean(arr, axis=0)[xm]
        top = max(float(np.nanmax(v)) for v in ys.values()) if ys else 100
        top = 100 * math.ceil(top * 1.1 / 100)
        P = sd.Plot(530, 620, x=(-2, 40), y=(-30, top), xlabel='time since the flash (µs)',
                    ylabel='mean signal (mV)' if fam == 'WAL' else '', title=FAM[fam], margin=(24, 20, 92, 96))
        for c, y in ys.items():
            n = int(sum(C.get(f'{d}_{c}_n', 0) for d in dets))
            P.line(lc, clip(y, -30, top), CC[c], 3, markers=False, tip=f'{CN[c]}: mean of {n:,} channel-bunches')
        P.hline(0, sd.INK, None, 1.2)
        P.xticks([(v, str(v)) for v in (0, 10, 20, 30, 40)])
        step = 100 if top <= 500 else 200 if top <= 1000 else 500
        P.yticks([(v, str(v)) for v in range(0, int(top) + 1, step)])
        panels.append(P.svg(fam))
    tail = (lcv >= 3) & (lcv <= 20)
    rat = {}
    for fam in fam_list:
        v = {}
        for c in ('par', 'ded'):
            arr = [C[f'{d}_{c}_lin'] for d in DETS if d.startswith(fam) and f'{d}_{c}_lin' in C]
            v[c] = float(np.nanmean(np.nanmean(arr, axis=0)[tail])) if arr else np.nan
        rat[fam] = v['ded'] / v['par']
    rs = ', '.join(f'{FAM[f]} ×{rat[f]:.1f}' for f in fam_list)
    body = (sd.title('The flash peak is clipped at the rail; its tail scales with the protons',
                     f'Mean signal over every bunch and channel of each family, first 40 µs. Tail (3–20 µs), dedicated/parasitic: {rs}')
            + sd.row(*panels, gap=36)
            + sd.legend([('parasitic', CC['par'], 'line'), ('dedicated', CC['ded'], 'line')]))
    notes = ('<p>Mean of the signal in 100 ns bins, ZS gaps counted as zero, over all channels of the family and all '
             'bunches of the class. This is the average waveform: the flash, its tail and the pile-up current of '
             'the beam-induced pulses on top.</p>'
             '<p>Walls: the clamp of the diverted signal sits at a fixed level, so its height does not scale with '
             'intensity; the tail after it does.</p>')
    D.slide('averages', body, notes, short='average flash')


def baselines(D, C, R):
    tc = C['tc_us']
    m = tc < 1e4
    xs = tc[m].tolist()
    panels = []
    for fam in ('WAL', 'PSS', 'LIQ'):
        dets = [d for d in DETS if d.startswith(fam)]
        yr = (-20, 20)
        P = sd.Plot(530, 640, x=(0.3, 1e4, 'log'), y=yr, xlabel='time since the flash',
                    ylabel='baseline shift (mV)' if fam == 'WAL' else '', title=FAM[fam], margin=(24, 20, 92, 96))
        P.band([0.3, 1e4], [-3, -3], [3, 3], sd.GREEN, alpha=0.10, tip='±3 mV: the recovery tolerance')
        for c in ('par', 'ded'):
            for d in dets:
                k = f'{d}_{c}_shift'
                if k not in C:
                    continue
                y = C[k][1][m]
                t = R.get((d, 'all', c), {}).get('t3mV_us', math.nan)
                P.line(xs, clip(y, *yr), CC[c], 2.2, markers=False,
                       tip=f'{d} {CN[c]}: median baseline; within ±3 mV after {fmt_t(t)}')
        P.hline(0, sd.INK, None, 1.2)
        P.xticks(tlog_ticks(0.3, 1e4)[::1]).yticks([(v, str(v)) for v in (-20, -10, -3, 0, 3, 10, 20)])
        panels.append(P.svg(fam))
    body = (sd.title('The baseline comes back in µs on the liquids, 0.1 ms on the plastics, ~1 ms on the walls',
                     'Median baseline shift of each detector against time since the flash; one line per detector')
            + sd.row(*panels, gap=36)
            + sd.legend([('parasitic', CC['par'], 'line'), ('dedicated', CC['ded'], 'line'),
                         ('±3 mV', sd.GREEN, 'box')]))
    notes = ('<p>Baseline shift: inside the unsuppressed flash block, the median of the samples in each time bin; '
             'after it, the median of each zero-suppressed block\'s 200 leading samples (each block carries 259 '
             'samples before its trigger). Medians over every bunch and channel of the detector. Clipped at ±20 mV '
             'for display; the flash itself runs off the top.</p>'
             '<p>On the plastics in the first ~100 µs the median includes pile-up of real pulses, so it is the '
             'baseline a discriminator would see, not the amplifier\'s own. The walls\' negative excursion is a real '
             'undershoot after the flash (checked on single traces in the first look).</p>')
    D.slide('baselines', body, notes, short='baselines')


def summary(D, R, BR):
    P = sd.Plot(1100, 680, x=(1, 3e3, 'log'), y=(-0.6, 11.6), xlabel='recovery time: baseline back within ±3 mV',
                margin=(24, 30, 92, 130))
    P.xticks(tlog_ticks(1, 3e3) + [(3, '3 µs'), (30, '30 µs'), (300, '300 µs'), (3e3, '3 ms')])
    P.yticks([(11 - i, d) for i, d in enumerate(DETS)])
    for i, d in enumerate(DETS):
        y = 11 - i
        for c, dy, mk in (('par', 0.17, 'open'), ('ded', -0.17, 'circle')):
            r = R.get((d, 'all', c))
            if not r:
                continue
            per = [x[1] for x in BR.get((d, c), []) if np.isfinite(x[1])]
            if len(per) >= 3:
                lo, hi = np.percentile(per, [16, 84])
                P.raw(sd.line(P.X(max(lo, 1)), P.Y(y + dy), P.X(min(hi, 3e3)), P.Y(y + dy), CC[c], 3))
            tip = (f'{d} {CN[c]}: {fmt_t(r["t3mV_us"])} pooled ({int(r["n"]):,} channel-bunches); '
                   f'±1 mV after {fmt_t(r["t1mV_us"])}' + (f'\nrun to run 16–84 %: {fmt_t(lo)} – {fmt_t(hi)} over {len(per)} runs' if len(per) >= 3 else ''))
            P.points([r['t3mV_us']], [y + dy], CC[c], r=9, tips=[tip], marker=mk)
    for yb in (3.5, 7.5):
        P.raw(sd.line(P.x0, P.Y(yb), P.x0 + P.pw, P.Y(yb), sd.RULE, 2))
    tab = [['', 'parasitic', 'dedicated']]
    for fam in ('LIQ', 'PSS', 'WAL'):
        a, b = fam_t(R, fam, 'par'), fam_t(R, fam, 'ded')
        tab.append([FAM[fam], f'{fmt_t(a[0])} – {fmt_t(a[2])}', f'{fmt_t(b[0])} – {fmt_t(b[2])}'])
    side = sd.col(sd.p('<b>Range over the four detectors</b>', 26),
                  sd.table(tab[0], tab[1:], size=24),
                  sd.p('Bars: run-to-run spread (16–84 %). Hover a point for the ±1 mV time.', 22, sd.MUT), w=520)
    rr = np.nanmedian([R[(d, 'all', 'ded')]['t3mV_us'] / R[(d, 'all', 'par')]['t3mV_us']
                       for d in DETS if (d, 'all', 'ded') in R and (d, 'all', 'par') in R])
    body = (sd.title(f'Every scintillator recovers later at dedicated intensity: ×{rr:.1f} (median)',
                     f'Baseline back within ±3 mV of its pre-flash level; production runs (≥ {PRODUCTION_FROM}), all bunches and channels')
            + sd.row(P.svg('summary'), side, gap=44)
            + sd.legend([('parasitic', CC['par'], 'open'), ('dedicated', CC['ded'], 'dot')]))
    rows_ = ''.join(f'<tr><td>{d}</td>' + ''.join(
        f'<td>{fmt_t(R.get((d, "all", c), {}).get(k, math.nan))}</td>' for c in ('par', 'ded') for k in ('t3mV_us', 't1mV_us'))
        + '</tr>' for d in DETS)
    notes = ('<table><tr><th>detector</th><th>par ±3 mV</th><th>par ±1 mV</th><th>ded ±3 mV</th><th>ded ±1 mV</th></tr>'
             + rows_ + '</table>'
             '<p>Pooled: the median baseline over every bunch and channel of the detector and class, then the first time '
             'after which it stays within the tolerance out to 10 ms. Per run: the same on that run\'s ~20 bunches of '
             'the class. 3 mV is a tenth of the lowest wall trigger threshold (26 mV) and ~5× the rms noise; 1 mV is '
             '~1.5× the noise. Table: <code>recovery.csv</code>, per run: <code>recovery_by_run.csv</code>.</p>')
    D.slide('summary', body, notes, short='recovery times')


def stability(D, BR):
    date = run_dates()
    panels = []
    for fam in ('WAL', 'PSS', 'LIQ'):
        dets = [d for d in DETS if d.startswith(fam)]
        P = sd.Plot(530, 640, x=(5, 42), y=(1, 3e3, 'log'), xlabel='day of July–Aug 2026 (from 1 July)',
                    ylabel='recovery time (±3 mV)' if fam == 'WAL' else '', title=FAM[fam], margin=(24, 20, 92, 110))
        for c in ('par', 'ded'):
            per = {}
            for d in dets:
                for run, t3, _ in BR.get((d, c), []):
                    if np.isfinite(t3):
                        per.setdefault(run, []).append(t3)
            xs, ys, tips = [], [], []
            for run in sorted(per):
                day = (date(run) - D0).days + 1
                v = float(np.median(per[run]))
                xs.append(day)
                ys.append(min(max(v, 1), 3e3))
                tips.append(f'run {run} ({date(run)}), {CN[c]}: {fmt_t(v)} (median of {len(per[run])} detectors)'
                            + (' — divert OFF' if run in DIVERT_OFF else ''))
            P.points(xs, ys, CC[c], r=5, tips=tips, marker='open' if c == 'par' else 'circle')
            off = [(x, y) for x, y, r in zip(xs, ys, sorted(per)) if r in DIVERT_OFF]
            if off and fam == 'WAL':
                P.points([o[0] for o in off], [o[1] for o in off], sd.RED, r=9, marker='diamond')
        for name, r0 in EPOCHS[1:]:
            P.vline((date(r0) - D0).days + 1, sd.MUT, '5 5', 1.5, tip=f'{name} from run {r0} ({date(r0)})')
        P.xticks([(v, str(v)) for v in (10, 20, 30, 40)]).yticks([(v, s) for v, s in tlog_ticks(1, 3e3)] + [(1, '1 µs')])
        panels.append(P.svg(fam))
    body = (sd.title(f'Flat from run {PRODUCTION_FROM} ({date(PRODUCTION_FROM):%-d %B}) to the end; commissioning steps before it',
                     'Recovery time per run (median of the family\'s detectors) against date; headline numbers use the flat part only')
            + sd.row(*panels, gap=36)
            + sd.legend([('parasitic', CC['par'], 'open'), ('dedicated', CC['ded'], 'dot'),
                         ('wall divert off', sd.RED, 'dot'), ('step in the data (hover)', sd.MUT, 'dash')]))
    notes = ('<p>Dashed lines: steps found in this data, every detector of a family moving at the same run. 224355: '
             'walls 0.25 → 0.55 ms. 224401: walls → 0.83 ms, plastics 12 → 31 µs, the liquids first appear. 224450: '
             'plastics 31 → 114 µs, liquids 20 → 8 µs, the walls enter the DAQ configuration. 224450–224487: walls and '
             'liquids flat, the plastics still moving run to run (114 → 43 → 20 → 126 µs, dedicated). 224401 also moved '
             'every plastic ZS threshold by 5 mV; 224355 changed no DAQ setting (so hardware: HV, gain, gate). '
             'From 224488 nothing moves, including across 22 July (wall gate delay 200 → '
             '1000 ns) and 23 July (DREAM clock divider), so those are not marked.</p>'
             '<p>Dates of the runs n_TOF never processed are taken from the nearest processed run.</p>')
    D.slide('stability', body, notes, short='over the campaign')


def divert(D, C, R):
    tc = C['tc_us']
    m = tc < 1e4
    lin = C['lin_edges_us']
    lcv = 0.5 * (lin[:-1] + lin[1:])
    xm = (lcv >= -2) & (lcv <= 40)
    lc = lcv[xm].tolist()
    ep = 'prod'
    have = f'WALA_ded_divoff_{ep}_lin' in C or f'WALA_par_divoff_{ep}_lin' in C
    c = 'ded' if f'WALA_ded_divoff_{ep}_lin' in C else 'par'
    P1 = sd.Plot(810, 580, x=(-2, 40), y=(-50, 1500), xlabel='time since the flash (µs)', ylabel='mean signal (mV)',
                 title=f'the average wall flash ({CN[c]})')
    P2 = sd.Plot(810, 580, x=(0.3, 1e4, 'log'), y=(-20, 20), xlabel='time since the flash', ylabel='baseline shift (mV)',
                 title='the baseline recovery')
    leg = []
    if have:
        for tag, col, name in (('divon', sd.GREY, 'divert on (production runs)'), ('divoff', sd.RED, 'divert off (224464, 224466)')):
            arr = [C[f'{d}_{c}_{tag}_{ep}_lin'] for d in DETS[:4] if f'{d}_{c}_{tag}_{ep}_lin' in C]
            sh = [C[f'{d}_{c}_{tag}_{ep}_shift'][1] for d in DETS[:4] if f'{d}_{c}_{tag}_{ep}_shift' in C]
            if not arr:
                continue
            n = int(sum(C.get(f'{d}_{c}_{tag}_{ep}_n', 0) for d in DETS[:4]))
            P1.line(lc, clip(np.nanmean(arr, axis=0)[xm], -50, 1500), col, 3, markers=False, tip=f'{name}: {n:,} channel-bunches')
            ts = [R.get((d, f'{tag}_{ep}', c), {}).get('t3mV_us', math.nan) for d in DETS[:4]]
            P2.line(tc[m].tolist(), clip(np.nanmedian(sh, axis=0)[m], -20, 20), col, 3, markers=False,
                    tip=f'{name}: ±3 mV after {fmt_t(np.nanmedian(ts))} (median of the four walls)')
            leg.append((name, col, 'line'))
        t_on = np.nanmedian([R.get((d, f'divon_{ep}', c), {}).get('t3mV_us', math.nan) for d in DETS[:4]])
        t_off = np.nanmedian([R.get((d, f'divoff_{ep}', c), {}).get('t3mV_us', math.nan) for d in DETS[:4]])
        title_ = (f'The divert gate does not set the wall recovery: {fmt_t(t_off)} with it off, '
                  f'{fmt_t(t_on)} with it on')
    else:
        title_ = 'Divert-off runs: pending (on tape)'
    P1.hline(0, sd.INK, None, 1.2).xticks([(v, str(v)) for v in (0, 10, 20, 30, 40)]).yticks([(v, str(v)) for v in (0, 500, 1000, 1500)])
    P2.band([0.3, 1e4], [-3, -3], [3, 3], sd.GREEN, alpha=0.10).hline(0, sd.INK, None, 1.2)
    P2.xticks(tlog_ticks(0.3, 1e4)).yticks([(v, str(v)) for v in (-20, -10, -3, 0, 3, 10, 20)])
    e2 = [(R.get((d, 'divoff_e2', c), {}).get('t3mV_us', math.nan), R.get((d, 'divon_e2', c), {}).get('t3mV_us', math.nan))
          for d in DETS[:4]]
    e2s = ', '.join(f'{d}: {fmt_t(a)} off / {fmt_t(b)} on' for d, (a, b) in zip(DETS[:4], e2))
    body = (sd.title(title_, f'SiPM walls, all four, {CN[c]}: divert-off runs 224464, 224466 against the other runs from 224450 (walls flat there)')
            + sd.row(P1.svg('divert flash'), P2.svg('divert baseline'), gap=44) + sd.legend(leg))
    notes = (f'<p>The 10–11 July divert-off runs 224356–360 sit in a commissioning epoch, so they are compared with '
             f'that epoch\'s divert-on runs (224355, 224361–224400) instead, {CN[c]}: {e2s}.</p>'
             '<p>The divert (blanking) gate switches the wall SiPM signal off for ~1 µs around the flash. In seven '
             'runs it was off (N1081B module 6 down on 10–11 July; again 16 July), so the walls recorded the '
             'undiverted flash (<code>ntof_processing/FLASH_DIVERT_OFF_RUNS.md</code>). Comparing them says how '
             'much of the wall recovery is the gate and how much the SiPMs and their amplifiers.</p>')
    D.slide('divert', body, notes, short='wall divert off')


# ------------------------------------------------------------------ the pulse layer
def pulse_curves():
    f = out_dir('pulses') / 'pulse_curves.npz'
    return dict(np.load(f)) if f.exists() else None


def pulse_rows():
    f = out_dir('pulses') / 'pulse_recovery.csv'
    return {(r['det'], r['cls']): {k: fnum(v) for k, v in r.items() if k not in ('det', 'cls')}
            for r in csv.DictReader(open(f))} if f.exists() else {}


def fam_mean(PC, fam, c, key):
    arr = [PC[f'{d}_{c}_{key}'] for d in DETS if d.startswith(fam) and f'{d}_{c}_{key}' in PC]
    return np.nanmean(arr, axis=0) if arr else None


def shape(D, PC, PR):
    tc = PC['tc_us']
    m = (tc >= 0.3) & (tc < 1e4)
    panels = []
    for fam, yr in (('WAL', (0, 120)), ('PSS', (0, 30))):
        P = sd.Plot(810, 560, x=(0.3, 1e4, 'log'), y=yr, xlabel='time since the flash',
                    ylabel='median pulse width (ns, above half height)', title=FAM[fam])
        for c in ('par', 'ded'):
            y = fam_mean(PC, fam, c, 'w50')
            t = np.nanmedian([PR.get((d, c), {}).get('t_width_us', math.nan) for d in DETS if d.startswith(fam)])
            P.line(tc[m].tolist(), clip(y[m], *yr), CC[c], 3, markers=False,
                   tip=f'{FAM[fam]} {CN[c]}: width within ±15 % of its 1–10 ms value after {fmt_t(t)} (median of 4 detectors)')
            if np.isfinite(t):
                P.vline(t, CC[c], '8 5', 2.5)
        P.xticks(tlog_ticks(0.3, 1e4)).yticks([(v, str(v)) for v in range(yr[0], yr[1] + 1, 20 if fam == 'WAL' else 5)])
        panels.append(P.svg(fam))
    tw = np.nanmedian([PR.get((d, 'ded'), {}).get('t_width_us', math.nan) for d in DETS[:4]])
    body = (sd.title(f'Wall pulses are narrow for the first ~{fmt_t(tw)}; plastic pulses have their shape within µs',
                     'Median width of the pulses found, against time since the flash; production runs, mean of the four detectors')
            + sd.row(*panels, gap=44)
            + sd.legend([('parasitic', CC['par'], 'line'), ('dedicated', CC['ded'], 'line'),
                         ('back within ±15 %', sd.MUT, 'dash')]))
    notes = ('<p>Pulses from <code>pulses.py</code>: baseline from each ZS block\'s pre-samples, or a running median '
             '(200–400 ns chunks) inside the unsuppressed flash block; peaks above 4 mV after smoothing. Width = samples '
             'above half height. The liquids (~4 ns wide) are not shown: their width is at the 3 ns bin resolution of '
             'the histograms.</p><p>The narrow early wall pulses could be the chunked baseline cutting their tails '
             'on the steep flash decay, as much as a real shape change; a template fit with a local exponential '
             'baseline will tell.</p>')
    D.slide('pulse_shape', body, notes, short='pulse shape')


def amplitude(D, PC, PR):
    tc = PC['tc_us']
    m = (tc >= 0.5) & (tc < 1e4)
    panels = []
    early = {}
    for fam in ('WAL', 'PSS', 'LIQ'):
        P = sd.Plot(530, 600, x=(0.5, 1e4, 'log'), y=(0, 1.6), xlabel='time since the flash',
                    ylabel='amplitude scale / 1–10 ms value' if fam == 'WAL' else '', title=FAM[fam], margin=(24, 20, 92, 96))
        P.band([0.5, 1e4], [0.9, 0.9], [1.1, 1.1], sd.GREEN, alpha=0.10, tip='±10 % of the 1–10 ms value')
        for c in ('par', 'ded'):
            y = fam_mean(PC, fam, c, 'gain')
            ref = float(np.nanmean([PC[f'{d}_{c}_gain_ref'] for d in DETS if d.startswith(fam) and f'{d}_{c}_gain_ref' in PC]))
            P.line(tc[m].tolist(), clip(y[m], 0, 1.6), CC[c], 3, markers=False,
                   tip=f'{FAM[fam]} {CN[c]}: 90 % quantile of the pulses above {AMPF[fam]:.0f} mV, over its 1–10 ms value')
            P.hline(min(ref, 1.6), CC[c], '3 5', 2, tip=f'{CN[c]}: the same at 30–32 ms (cosmics dominate): {ref:.2f}')
            if c == 'ded':
                early[fam] = float(np.nanmean(y[(tc > 1.5) & (tc < 3)]))
        P.hline(1, sd.INK, None, 1.2)
        P.xticks(tlog_ticks(0.5, 1e4)).yticks([(v, f'{v:.1f}') for v in (0, 0.5, 1.0, 1.5)])
        panels.append(P.svg(fam))
    body = (sd.title(f'Plastic and liquid pulses are ×{1 / early["PSS"]:.1f} and ×{1 / early["LIQ"]:.1f} smaller 2 µs after the flash; wall pulses are not',
                     'The amplitude scale of the pulses found against time (dedicated 1.5–3 µs in the title). Gain, or a different particle mix: not yet separated')
            + sd.row(*panels, gap=36)
            + sd.legend([('parasitic', CC['par'], 'line'), ('dedicated', CC['ded'], 'line'),
                         ('same scale at 30–32 ms', sd.MUT, 'dot'), ('±10 %', sd.GREEN, 'box')]))
    rows_ = ''.join(f'<tr><td>{d}</td>' + ''.join(f'<td>{fmt_t(PR.get((d, c), {}).get(k, math.nan))}</td>'
                                                  for c in ('par', 'ded') for k in ('t_amp90_us', 't_amp97_us')) + '</tr>'
                    for d in DETS)
    notes = ('<p>Scale = the 90 % quantile of the amplitudes above a floor (walls and liquids 30 mV, plastics 50 mV), '
             'clear of the threshold and of the small-pulse population, interpolated in log amplitude; divided by its '
             'own median over 1–10 ms. Production runs, live channels, mean of the four detectors.</p>'
             '<p><b>Why this is not yet a gain measurement.</b> The spectrum\'s shape follows the particle mix: fast-'
             'neutron recoils (quenched) in the first µs, capture γs from ~0.1 ms, cosmic muons at 30 ms. The dotted '
             'lines show the same scale at 30–32 ms differing from 1–10 ms by −15 % (walls) to +35 % (plastics) with '
             'no flash nearby -- that is population, not gain. So the early dip is an upper bound on the gain sag. '
             'The fixed reference that separates them is next: plastic pulses in coincidence with the wall of the '
             'same arm (charged particles crossing the stack, a fixed energy loss per cm), and for the walls the ratio '
             'of the two ends of a bar at fixed position.</p>'
             '<table><tr><th>detector</th><th>par 90 %</th><th>par 97 %</th><th>ded 90 %</th><th>ded 97 %</th></tr>'
             + rows_ + '</table><p>Times after which the scale stays above 90 % / 97 % of its 1–10 ms value.</p>')
    D.slide('amplitude', body, notes, short='pulse amplitude')


def wall_edge(D, PC):
    tc = PC['tc_us']
    ae = PC['a_edges']
    ac = np.sqrt(ae[1:] * ae[:-1])
    dla = np.diff(np.log(ae))
    sl = [(3, 10), (10, 100), (100, 1000), (1000, 1e4)]
    cols = [sd.PURPLE, sd.RED, sd.GOLD, sd.GREEN]
    panels = []
    frac = {}
    for c in ('par', 'ded'):
        h = sum(PC[f'{d}_{c}_ha'] for d in DETS[:4])
        P = sd.Plot(810, 560, x=(2, 1000, 'log'), y=(1e-3, 3, 'log'), xlabel='pulse amplitude (mV)',
                    ylabel='pulses per log-amplitude (normalised above 30 mV)', title=f'SiPM walls, {CN[c]}')
        norm_m = ac > 30
        for (lo, hi), col in zip(sl, cols):
            mm = (tc > lo) & (tc < hi)
            sp = h[:100][mm].sum(axis=0)
            y = sp / max(sp[norm_m].sum(), 1) / dla
            P.line(ac.tolist(), [max(v, 1e-3) if v > 0 else None for v in y], col, 3, markers=False,
                   tip=f'{fmt_t(lo)}–{fmt_t(hi)}: {int(sp.sum()):,} pulses')
            frac[(c, lo)] = sp[(ac > 4) & (ac < 15)].sum() / max(sp[norm_m].sum(), 1)
        sp = h[-1]
        P.line(ac.tolist(), [max(v, 1e-3) if v > 0 else None for v in sp / max(sp[norm_m].sum(), 1) / dla], sd.INK, 2.5,
               dash='7 5', markers=False, tip=f'30–32 ms: {int(sp.sum()):,} pulses')
        P.xticks([(v, str(v)) for v in (2, 5, 10, 20, 50, 100, 200, 500, 1000)]).yticks(sd.log_ticks(-3, 0))
        panels.append(P.svg(c))
    r = frac[('ded', 1000)] / max(frac[('ded', 10)], 1e-9)
    body = (sd.title(f'While the baseline undershoots, the walls lose their small pulses: ×{r:.1f} fewer 4–15 mV pulses at 10–100 µs',
                     'Wall pulse-amplitude spectra in slices of time since the flash, normalised to the pulses above 30 mV')
            + sd.row(*panels, gap=44)
            + sd.legend([(f'{fmt_t(lo)}–{fmt_t(hi)}', col, 'line') for (lo, hi), col in zip(sl, cols)]
                        + [('30–32 ms', sd.INK, 'dash')]))
    notes = ('<p>A zero-suppressed block opens only when the signal crosses a fixed threshold relative to the '
             'ORIGINAL baseline. While the baseline is 10–19 mV low (10 µs – 1 ms), a pulse must be that much larger to be '
             'recorded at all: the effective threshold rises and the small pulses vanish. Above ~30 mV the spectra '
             'agree. Ratio in the title: 4–15 mV pulses per pulse above 30 mV at 1–10 ms over the same at 10–100 µs, '
             'dedicated.</p><p>If the hardware trigger (the sum of a bar\'s two ends, 26–38 mV) sees the same '
             'undershoot, the wall trigger threshold rises by the same amount until ~1 ms.</p>')
    D.slide('wall_edge', body, notes, short='wall small pulses')


# ------------------------------------------------------------------ the fixed-reference layer
def coinc_curves():
    f = out_dir('coinc') / 'coinc_curves.npz'
    return dict(np.load(f)) if f.exists() else None


def coinc_rows():
    f = out_dir('coinc') / 'coinc_recovery.csv'
    return {(r['what'], r['fam'], r['cls']): fnum(r['t_us']) for r in csv.DictReader(open(f))} if f.exists() else {}


def fit_shape(D, K, KR):
    tc = K['tc_us']
    m = (tc >= 0.3) & (tc < 1e4)
    panels = []
    for fam in ('WAL', 'PSS', 'LIQ'):
        ys = {c: K.get(f'{fam}_{c}_resfrac') for c in ('par', 'ded')}
        top = 0.6 if fam == 'WAL' else 1.0
        P = sd.Plot(530, 600, x=(0.3, 1e4, 'log'), y=(0, top), xlabel='time since the flash',
                    ylabel='fit residual rms / amplitude' if fam == 'WAL' else '', title=FAM[fam], margin=(24, 20, 92, 96))
        for c, y in ys.items():
            if y is None:
                continue
            t = KR.get(('shape80', fam, c), math.nan)
            t25 = KR.get(('shape', fam, c), math.nan)
            P.line(tc[m].tolist(), clip(y[m], 0, top), CC[c], 3, markers=False,
                   tip=f'{FAM[fam]} {CN[c]}: 80 % of the early excess misfit gone after {fmt_t(t)}; within +25 % of the 1–10 ms value after {fmt_t(t25)}')
            if np.isfinite(t):
                P.vline(t, CC[c], '8 5', 2.5)
        P.xticks(tlog_ticks(0.3, 1e4)).yticks([(v, f'{v:.1f}') for v in np.arange(0, top + 1e-9, 0.2)])
        panels.append(P.svg(fam))
    tw = KR.get(('shape80', 'WAL', 'ded'), math.nan)
    tp = KR.get(('shape80', 'PSS', 'ded'), math.nan)
    tl = KR.get(('shape80', 'LIQ', 'ded'), math.nan)
    sw = KR.get(('shape', 'WAL', 'ded'), math.nan)
    body = (sd.title(f'Fitted on a local baseline, 80 % of the shape is back by ~{fmt_t(tp)} (plastics), ~{fmt_t(tw)} (walls), ~{fmt_t(tl)} (liquids)',
                     f'Template fit of isolated pulses: median residual rms over amplitude, dedicated. The walls\' last 20 % takes until ~{fmt_t(sw)}')
            + sd.row(*panels, gap=36)
            + sd.legend([('parasitic', CC['par'], 'line'), ('dedicated', CC['ded'], 'line'),
                         ('80 % of the early excess gone', sd.MUT, 'dash')]))
    notes = ('<p>Recovered shape: the median misfit has lost 80 % of its early excess over the 1–10 ms value (robust to the '
             'small bump the plastics show at 0.1–1 ms, which follows the capture-γ population). The tooltips also give the '
             'stricter "within +25 % of the 1–10 ms value".</p><p>Templates: the median normalised shape of clean pulses 1–10 ms after the flash (one pulse in its ZS block, '
             '30–300 mV), per family (<code>templates.py</code>, <code>templates.npz</code>). Every found pulse is refitted with '
             'A·template + b₀ + b₁·t over [−150, +350] ns (walls), [−40, +120] (plastics), [−30, +100] (liquids), so the '
             'steep flash tail enters as a local slope instead of biasing the amplitude. Isolated = no other pulse in the '
             'window. The misfit floor of the plastics and liquids (~0.1) is the template aligned to the peak sample: '
             'a 4–13 ns pulse is not resolved in 1 ns steps. This is the answer to "how far back into the tail can a '
             'pulse be reconstructed": from the time its shape is the late-time shape.</p>')
    D.slide('fit_shape', body, notes, short='template fit')


def plastic_wall(D, K, KR):
    tcc = K['tcc_us']
    m = (tcc >= 1) & (tcc < 1e4)
    panels = []
    for c in ('par', 'ded'):
        P = sd.Plot(810, 560, x=(1, 1e4, 'log'), y=(0, 1.6), xlabel='time since the flash',
                    ylabel='plastic amplitude / 1–10 ms value', title=f'plastic pulses behind a wall hit, {CN[c]}')
        P.band([1, 1e4], [0.9, 0.9], [1.1, 1.1], sd.GREEN, alpha=0.10)
        for nm, col, lab in (('median', CC[c], 'median'), ('edge95', sd.PURPLE, 'upper edge (95 %)')):
            y = K.get(f'pw_{c}_{nm}')
            if y is None:
                continue
            mv = float(K[f'pw_{c}_{nm}_mV'])
            ref = float(K[f'pw_{c}_{nm}_ref'])
            P.line(tcc[m].tolist(), clip(y[m], 0, 1.6), col, 3.5, markers=True, r=5,
                   tips=[f'{lab}: {v:.2f} × {mv:.0f} mV at {fmt_t(t)}' for t, v in zip(tcc[m], y[m])])
            if np.isfinite(ref):
                P.hline(min(ref, 1.58), col, '3 5', 2, tip=f'{lab} at 30–32 ms (cosmic muons): {ref:.2f}')
        P.hline(1, sd.INK, None, 1.2)
        P.xticks(tlog_ticks(1, 1e4)).yticks([(v, f'{v:.1f}') for v in (0, 0.5, 1.0, 1.5)])
        panels.append(P.svg(c))
    pur = K.get('pw_ded_purity')
    tm = KR.get(('plastic_given_wall_median', 'PSS', 'ded'), math.nan)
    te_ = KR.get(('plastic_given_wall_edge95', 'PSS', 'ded'), math.nan)
    body = (sd.title(f'Behind a wall hit: the plastic median is back by {fmt_t(tm)}, its upper edge only by {fmt_t(te_)}',
                     'Plastic pulses 5–25 ns after a wall pulse of the same arm, accidentals subtracted; median and upper edge over their 1–10 ms value')
            + sd.row(*panels, gap=44)
            + sd.legend([('median', sd.INK, 'line'), ('upper edge (95 %)', sd.PURPLE, 'line'),
                         ('30–32 ms (cosmics)', sd.MUT, 'dash'), ('±10 %', sd.GREEN, 'box')]))
    notes = ('<p>The plastic minus wall time difference peaks at −15…−5 ns in every arm. Signal window −25…+5 ns; '
             'accidentals from +60…+210 ns, scaled by the width ratio. Requiring the wall selects particles that '
             'crossed the wall first, which takes out most of the neutron recoils and capture γs that hit the '
             'plastic alone. The upper edge is set by the light, not the electronics: over each bar\'s own pulse scale it '
             'is 1.19–1.36 across the 8 bars while the scales span 176–278 mV, so it follows each bar\'s gain. '
             'The population is still not fixed (no MIP peak in the beam window: the median is '
             'far below the cosmic muons at 30–32 ms, dotted), so this is a cleaner, not a clean, reference.</p>'
             + (f'<p>Signal purity (net / all in the window), dedicated, 1–10 ms: {np.nanmean(pur[:len(tcc)][(tcc > 1e3) & (tcc < 1e4)]):.0%}.</p>' if pur is not None else ''))
    D.slide('plastic_wall', body, notes, short='plastic behind wall')


def plastic_edge_bars(D, K, KR):
    tcc = K['tcc_us']
    m = (tcc >= 3) & (tcc < 1e4)
    panels = []
    for c in ('par', 'ded'):
        per = K.get(f'pw_{c}_edge_per_bar')
        if per is None:
            return
        P = sd.Plot(810, 560, x=(3, 1e4, 'log'), y=(0.4, 1.2), xlabel='time since the flash',
                    ylabel='upper edge / its 1–10 ms value', title=f'each plastic bar behind a wall hit, {CN[c]}')
        P.band([3, 1e4], [0.95, 0.95], [1.05, 1.05], sd.GREEN, alpha=0.12)
        for arm in range(4):
            for bar in range(2):
                P.line(tcc[m].tolist(), clip(per[arm, bar][m], 0.4, 1.2), ARM_COL_SD[arm], 1.6, markers=False,
                       tip=f'PSS{"ABCD"[arm]} bar {bar + 1}')
        P.line(tcc[m].tolist(), clip(K[f'pw_{c}_edge_bar_mean'][m], 0.4, 1.2), sd.INK, 4, markers=True, r=5,
               tips=[f'mean of 8 bars, {fmt_t(t)}: {v:.2f}' for t, v in zip(tcc[m], K[f'pw_{c}_edge_bar_mean'][m])])
        P.hline(1, sd.INK, None, 1.2)
        P.xticks(tlog_ticks(3, 1e4)).yticks([(v, f'{v:.1f}') for v in (0.4, 0.6, 0.8, 1.0, 1.2)])
        panels.append(P.svg(c))
    e10 = KR.get(('plastic_edge_per_bar_at_10us', 'PSS', 'ded'), math.nan)
    e100 = KR.get(('plastic_edge_per_bar_at_100us', 'PSS', 'ded'), math.nan)
    e300 = KR.get(('plastic_edge_per_bar_at_300us', 'PSS', 'ded'), math.nan)
    body = (sd.title(f'All 8 plastic bars follow one curve: their light edge is at {e10:.2f}, {e100:.2f}, {e300:.2f} of full at 10, 100, 300 µs',
                     'Upper edge (95 %) of the plastic pulses behind a wall hit, each bar over its own 1–10 ms value; dedicated in the title')
            + sd.row(*panels, gap=44)
            + sd.legend([('arm A', ARM_COL_SD[0], 'line'), ('arm B', ARM_COL_SD[1], 'line'), ('arm C', ARM_COL_SD[2], 'line'),
                         ('arm D', ARM_COL_SD[3], 'line'), ('mean of 8 bars', sd.INK, 'line'), ('±5 %', sd.GREEN, 'box')]))
    notes = ('<p>The edge is set by the light, not the electronics: over each bar\'s own pulse scale it is 1.19–1.36 across the '
             'bars while the scales span 176–278 mV. So each bar\'s curve is that bar\'s light-to-signal gain against time, '
             'as far as the coincident particles\' energy distribution holds. Eight bars with different gains on four PMT '
             'pairs following one curve within a few % is what a common recovery of the PMTs looks like; a change of particle '
             'mix would give the same curve too, so the two are still not fully separated. Bar-to-bar spread (rms) at 10 / '
             '100 / 300 µs is in <code>coinc_recovery.csv</code>.</p>')
    D.slide('plastic_edge_bars', body, notes, short='plastic edge per bar')


def bar_ends(D, K, KR):
    tcc = K['tcc_us']
    m = (tcc >= 0.3) & (tcc < 1e4)
    P = sd.Plot(1100, 600, x=(0.3, 1e4, 'log'), y=(-0.3, 0.3), xlabel='time since the flash',
                ylabel='Δ ln(A₁/A₂) from its 1–10 ms value')
    P.band([0.3, 1e4], [-0.05, -0.05], [0.05, 0.05], sd.GREEN, alpha=0.12, tip='±5 % gain imbalance between the two ends')
    for c in ('par', 'ded'):
        dr = K.get(f'bar_{c}_lnratio_drift')
        if dr is None:
            continue
        for w in range(4):
            for b in range(4):
                P.line(tcc[m].tolist(), clip(dr[w, b][m], -0.3, 0.3), CC[c], 1.2, markers=False,
                       tip=f'{DETS[w]} bar {b + 1}, {CN[c]}')
        P.line(tcc[m].tolist(), clip(K[f'bar_{c}_lnratio_drift_all'][m], -0.3, 0.3), CC[c], 4, markers=False,
               tip=f'median of the 16 bars, {CN[c]}')
    P.hline(0, sd.INK, None, 1.2)
    P.xticks(tlog_ticks(0.3, 1e4)).yticks([(v, f'{v:+.1f}' if v else '0') for v in (-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3)])
    dmax = KR.get(('bar_end_ratio_max_drift_after_2us', 'WAL', 'ded'), math.nan)
    side = sd.col(sd.p('<b>Why this is a gain reference</b>', 26),
                  sd.p('The two ends of a bar see the same particle. At a fixed position (fixed time difference '
                       'between the ends) their amplitude ratio depends only on the two gains — not on what particle '
                       'hit or how much light it made.', 23, sd.MUT),
                  sd.p('A gain loss on <i>one</i> end moves this ratio. A loss common to both ends does not: that '
                       'half is the next slide.', 23, sd.MUT), w=500)
    body = (sd.title(f'The two ends of a wall bar recover together: their ratio moves ≤ {100 * (math.exp(dmax) - 1):.0f} % after 2 µs (median of 16 bars)',
                     'Median ln(A₁/A₂) of coincident end pairs at fixed position (|Δt| ≤ 10 ns), against its 1–10 ms value; thin: each of the 16 bars')
            + sd.row(P.svg('bar ends'), side, gap=44)
            + sd.legend([('parasitic', CC['par'], 'line'), ('dedicated', CC['ded'], 'line'), ('±5 %', sd.GREEN, 'box')]))
    notes = ('<p>End pairs: the nearest end-2 pulse within ±30 ns of each end-1 pulse (both above 10 mV); accidentals '
             'from the same search with end 2 shifted by +500 ns, subtracted bin by bin. Ratio median per 2 ns bin of '
             'Δt, minus its own 1–10 ms value, averaged over the bins with |Δt| ≤ 10 ns.</p>')
    D.slide('bar_ends', body, notes, short='bar ends')


def along_bar(D, K, KR):
    tcc = K['tcc_us']
    m = (tcc >= 0.3) & (tcc < 1e4)
    panels = []
    for c in ('par', 'ded'):
        P = sd.Plot(810, 560, x=(2, 1e4, 'log'), y=(0.6, 1.3), xlabel='time since the flash',
                    ylabel='√(A₁A₂) scale / 1–10 ms value', title=f'along the wall bars, {CN[c]}')
        m = (tcc >= 2) & (tcc < 1e4)
        P.band([2, 1e4], [0.9, 0.9], [1.1, 1.1], sd.GREEN, alpha=0.10)
        for bn, col in (('end 1', sd.BLUE), ('centre', sd.INK), ('end 2', sd.RED)):
            y = K.get(f'bar_{c}_gm_{bn}')
            if y is None:
                continue
            t = KR.get((f'bar_gm90_{bn}', 'WAL', c), math.nan)
            nn = K.get(f'bar_{c}_gm_{bn}_n')
            P.line(tcc[m].tolist(), clip(y[m], 0.6, 1.3), col, 3, markers=True, r=4,
                   tips=[f'{bn}, {fmt_t(tt)}: {v:.3f}' + (f' ({int(k):,} pairs)' if nn is not None else '')
                         for tt, v, k in zip(tcc[m], y[m], (nn[m] if nn is not None else [0] * int(m.sum())))])
        P.hline(1, sd.INK, None, 1.2)
        P.xticks(tlog_ticks(2, 1e4)).yticks([(v, f'{v:.1f}') for v in (0.6, 0.8, 1.0, 1.2)])
        panels.append(P.svg(c))
    spr = KR.get(('bar_band_spread_after_3us', 'WAL', 'ded'), math.nan)
    gmin = KR.get(('bar_gm_min_after_3us', 'WAL', 'ded'), math.nan)
    body = (sd.title(f'Along the bar the recovery is the same within {100 * spr:.0f} %; all positions dip together to {gmin:.2f}',
                     'Geometric mean of the two end amplitudes (position-independent light), 90 % quantile above 30 mV, in three position bands; after 3 µs, dedicated, in the title')
            + sd.row(*panels, gap=44)
            + sd.legend([('v < −300 mm', sd.BLUE, 'line'), ('|v| < 150 mm (centre)', sd.INK, 'line'),
                         ('v > +300 mm', sd.RED, 'line')]))
    notes = ('<p>Position v along the bar from the time difference between its two ends, converted per arm and bar with '
             'the MM-track calibration of <code>ntof_scint_stack</code> (Δt = a + b·v, b ≈ −0.0136 ns/mm, i.e. ~74 mm/ns; '
             'offsets a of −7…+6 ns, so raw Δt bins would mix positions between bars). √(A₁A₂) cancels the attenuation to first order, '
             'so the three bands should agree when nothing depends on position. The scale still follows the particle '
             'mix, as on the plastic slide; what this slide tests is whether the recovery differs ALONG the bar.</p>')
    D.slide('along_bar', body, notes, short='along the bar')


# ------------------------------------------------------------------ per channel
def channel_rows():
    f = out_dir('channels') / 'channel_maps.csv'
    return list(csv.DictReader(open(f))) if f.exists() else None


def channel_map(D, CH):
    by = {(r['det'], int(r['ch']), r['cls']): r for r in CH}
    order = [(d, ch) for d in DETS for ch in sorted({int(r['ch']) for r in CH if r['det'] == d})]
    cols = [('t_base3', 'par', 'baseline ±3 mV\nparasitic'), ('t_base3', 'ded', 'baseline ±3 mV\ndedicated'),
            ('t_shape80', 'ded', 'pulse shape\n(80 %), dedicated'), ('t_amp90', 'ded', 'amplitude scale\n≥ 90 %, dedicated')]
    n = len(order)
    lo, hi = 0, 4                                   # log10 us: 1 us .. 10 ms
    P = sd.Plot(1180, 700, x=(0, len(cols)), y=(n, 0), margin=(96, 20, 16, 120))
    cells = []
    for j, (k, c, lab) in enumerate(cols):
        for i, (d, ch) in enumerate(order):
            v = fnum(by.get((d, ch, c), {}).get(k))
            lv = math.log10(v) if np.isfinite(v) and v > 0 else math.nan
            cells.append((j + 0.04, j + 0.96, i + 0.06, i + 0.94, lv,
                          f'{d}{ch} — {lab.replace(chr(10), " ")}: {fmt_t(v)}'))
    P.cells(cells, lo, hi, missing=sd.RULE)
    P.yticks([(i + 0.5, f'{d}{ch}' if not d.startswith('LIQ') else 'LIQ A–D') for i, (d, ch) in enumerate(order) if (ch == 1 and not d.startswith('LIQ')) or d == 'LIQA'])
    for j, (_, _, lab) in enumerate(cols):
        for li, line_ in enumerate(lab.split('\n')):
            P.text(j + 0.5, -3.4 + 1.6 * li, line_, 20, sd.INK, 'middle')
    for yb in (32, 40):
        P.raw(sd.line(P.x0, P.Y(yb), P.x0 + P.pw, P.Y(yb), sd.INK, 2))
    cb = sd.colorbar(140, 600, lo, hi, [(0, '1 µs'), (1, '10 µs'), (2, '100 µs'), (3, '1 ms'), (4, '10 ms')], label='time')
    slow = sorted(((fnum(r['t_base3']), r['det'] + r['ch']) for r in CH if r['cls'] == 'ded' and r['det'].startswith('PSS')),
                  reverse=True)[0]
    body = (sd.title(f'Walls uniform, plastics not: slowest {slow[1]} at {fmt_t(slow[0])}',
                     'Every recovery time of each of the 44 channels; production runs. Rows: 32 wall ends (A1–D8), 8 plastic bars, 4 liquids')
            + sd.row(P.svg('channel map'), cb, gap=40, align='center'))
    rows_ = ''.join(f"<tr><td>{r['det']}{r['ch']}</td><td>{r['cls']}</td><td>{fmt_t(fnum(r['t_base3']))}</td>"
                    f"<td>{fnum(r['under_mV']):.1f}</td><td>{fmt_t(fnum(r['t_shape80']))}</td><td>{fmt_t(fnum(r['t_amp90']))}</td>"
                    f"<td>{fnum(r['scale_mV']):.0f}</td></tr>" for r in CH)
    notes = ('<p>Grey: not measurable (too few pulses). Definitions as on the earlier slides; per channel instead of per '
             'detector. The amplitude-scale column inherits that measure\'s population caveat, which is why it scatters '
             'on the plastics and liquids.</p><table><tr><th>channel</th><th>class</th><th>baseline ±3 mV</th>'
             '<th>undershoot (mV)</th><th>shape 80 %</th><th>scale ≥ 90 %</th><th>scale (mV)</th></tr>' + rows_ + '</table>'
             '<p><code>channel_maps.csv</code></p>')
    D.slide('channels', body, notes, short='per channel')


def undershoot_gain(D, CH):
    panels = []
    corr = {}
    for c in ('par', 'ded'):
        w = [r for r in CH if r['det'].startswith('WAL') and r['cls'] == c]
        x = np.array([fnum(r['scale_mV']) for r in w])
        y = np.array([-fnum(r['under_mV']) for r in w])
        t = np.array([fnum(r['t_base3']) for r in w])
        ok = np.isfinite(x) & np.isfinite(y)
        corr[c] = (np.corrcoef(x[ok], y[ok])[0, 1], np.corrcoef(y[ok], t[ok])[0, 1], float(np.median(y[ok] / x[ok])))
        P = sd.Plot(810, 560, x=(70, 130), y=(5, 25), xlabel="channel's pulse scale at 1–10 ms (mV)",
                    ylabel='baseline undershoot depth (mV)', title=f'32 wall ends, {CN[c]}')
        P.points(x[ok].tolist(), y[ok].tolist(), CC[c], r=9,
                 tips=[f"{r['det']}{r['ch']}: scale {xi:.0f} mV, undershoot {yi:.1f} mV, baseline back after {fmt_t(ti)}"
                       for r, xi, yi, ti in zip(w, x, y, t)])
        b1, b0 = np.polyfit(x[ok], y[ok], 1)
        P.line([70, 130], [b0 + b1 * 70, b0 + b1 * 130], sd.MUT, 2, dash='6 5', markers=False,
               tip=f'fit: {b1:.3f} mV per mV of scale')
        P.xticks([(v, str(v)) for v in range(70, 131, 10)]).yticks([(v, str(v)) for v in range(5, 26, 5)])
        panels.append(P.svg(c))
    body = (sd.title(f'The undershoot follows the channel\'s gain: ~{100 * corr["ded"][2]:.0f} % of its pulse scale, the same for every wall end',
                     f'Wall baseline undershoot depth against the channel\'s pulse scale; correlation {corr["par"][0]:.2f} (parasitic), '
                     f'{corr["ded"][0]:.2f} (dedicated). Deeper undershoots also recover later: correlation {corr["ded"][1]:.2f}')
            + sd.row(*panels, gap=44)
            + sd.legend([('one wall end (hover)', sd.INK, 'dot'), ('linear fit', sd.MUT, 'dash')]))
    notes = ('<p>The undershoot is the AC-coupled response to the flash charge, and the flash charge goes through the '
             'same gain as the pulses. So in units of each channel\'s own pulse size the undershoot is the same everywhere '
             'and every wall end loses the same FRACTION of its small pulses (the effective threshold rises by ~10–15 % '
             'of a typical pulse). Pulse scale: 90 % quantile above 30 mV at 1–10 ms; undershoot: the most negative '
             'median baseline after 20 µs.</p>')
    D.slide('undershoot_gain', body, notes, short='undershoot vs gain')


# ------------------------------------------------------------------ the trigger path
def trigger_slide(D):
    f = out_dir('trigger_ana') / 'trig_curves.npz'
    if not f.exists():
        return
    T = dict(np.load(f))
    tq = T['tcq_us']
    m = (tq >= 100) & (tq < 1e4)
    thr = float(np.mean(list(WALL_THR.values())))
    P = sd.Plot(1100, 600, x=(100, 1e4, 'log'), y=(-5, 40), xlabel='time since the flash',
                ylabel='wall-sum undershoot (mV)')
    P.band([100, 1e4], [26, 26], [38, 38], sd.GREY, alpha=0.18, tip='wall trigger thresholds, 26–38 mV (sum of a bar\'s two ends)')
    for c in ('par', 'ded'):
        P.line(tq[m].tolist(), clip(T[f'sum_undershoot_{c}_mV'][m], -5, 40), CC[c], 3.5, markers=False,
               tip=f'{CN[c]}: 2 × the median per-end baseline undershoot')
    ft = T['first_trigger_us']
    q = np.percentile(ft, [1, 50, 99])
    P.band([q[0], q[2]], [-5, -5], [40, 40], sd.BLUE, alpha=0.12,
           tip=f'first DREAM trigger of a bunch: 1–99 % at {fmt_t(q[0])}–{fmt_t(q[2])} ({len(ft):,} bunches)')
    P.vline(q[1], sd.BLUE, '6 5', 2.5, label='DREAM gate opens', tip=f'median first trigger {fmt_t(q[1])}')
    P.hline(0, sd.INK, None, 1.2)
    P.xticks(tlog_ticks(100, 1e4)).yticks([(v, str(v)) for v in (0, 10, 20, 30, 40)])
    at = {c: float(np.interp(q[1], tq, T[f'sum_undershoot_{c}_mV'])) for c in ('par', 'ded')}
    pe = T['prev_edges_us']
    side = sd.col(sd.p('<b>Why it cannot bias the recorded triggers</b>', 26),
                  sd.p(f'When the gate opens the sum is {at["ded"]:.1f} mV low (dedicated), {100 * at["ded"] / thr:.0f} % of '
                       f'the mean threshold. Before that the undershoot is large, but DREAM records nothing.', 23, sd.MUT),
                  sd.p('A direct turn-on measurement was tried (Details): DREAM\'s busy time after a trigger varies up to several '
                       'ms, so live time cannot be reconstructed well enough for a mV-level test.', 23, sd.MUT), w=500)
    body = (sd.title(f'DREAM opens at ~{fmt_t(q[1])}, when the wall-sum undershoot is down to {at["ded"]:.1f} mV: '
                     f'any trigger bias is ≤ {100 * at["ded"] / thr:.0f} % of threshold',
                     'Expected undershoot of the wall trigger sum (2 × the per-end median baseline) against the time DREAM records its first trigger')
            + sd.row(P.svg('trigger'), side, gap=44)
            + sd.legend([('parasitic', CC['par'], 'line'), ('dedicated', CC['ded'], 'line'),
                         ('first DREAM trigger, 1–99 %', sd.BLUE, 'box'), ('thresholds 26–38 mV', sd.GREY, 'box')]))
    rows_ = ''.join(f'<tr><td>{fmt_t(a)}–{fmt_t(b)}</td><td>{int(n):,}</td><td>{e:.2f}</td></tr>'
                    for a, b, n, e in zip(pe[:-1], pe[1:], T['n_vs_prev'], T['eff_vs_prev']))
    notes = ('<p>The question: do the analog sums feeding the wall discriminators see the 10–19 mV undershoot of the digitized '
             'signal? It only matters while DREAM records, and DREAM\'s first trigger in a bunch comes at the time above. '
             'Whatever the analog path does, the bias of the recorded triggers is bounded by the curve at that time.</p>'
             '<p>Direct test, attempted: emulated wall-sum + plastic coincidences in the raw waveforms (both ends of a bar '
             'within ±30 ns, template-fit amplitudes; plastic within −60…+100 ns), matched to DREAM triggers (t_pred − t in '
             '−100…+40 ns; 95 % name the same arm). Efficiency of clearly-above-threshold candidates (wall and plastic '
             '> 1.5× threshold) against time since the previous DREAM trigger:</p>'
             '<table><tr><th>since previous trigger</th><th>candidates</th><th>efficiency</th></tr>' + rows_ + '</table>'
             f'<p>Before a bunch\'s first trigger (gate just open, &lt; 1.05 ms): {float(T["eff_first_window"]):.2f} '
             f'(n = {int(T["n_first_window"]):,}). DREAM is dead ~205 µs after a trigger and partly busy for several ms '
             'after that, and some bunches are not recorded at all, so a threshold shift of 1–2 mV cannot be measured this '
             'way. Code: <code>trigger_cand.py</code>, <code>trig_analysis.py</code>.</p>')
    D.slide('trigger', body, notes, short='trigger path')

# ------------------------------------------------------------------ single photoelectrons
def spe_slide(D):
    f = out_dir('spe') / 'spe_curves.npz'
    if not f.exists():
        return
    Z = dict(np.load(f))
    R = list(csv.DictReader(open(out_dir('spe') / 'spe_channels.csv')))
    acc = [r for r in R if r['accepted'] == 'True' and r['cls'] == 'ded' and r['det'].startswith('PSS')]
    x = Z['x_mV']
    tcc = Z['tcc_us']
    P1 = sd.Plot(810, 560, x=(1, 9), y=(0, 1.05), xlabel='small-peak height (mV)', ylabel='peaks (fraction of max)',
                 title='1–10 ms, quiet tails of ZS blocks, dedicated')
    cols = (sd.BLUE, sd.ORANGE, sd.GREEN, sd.PURPLE, sd.RED, sd.GOLD, sd.GREY, sd.INK)
    shown = [r for r in R if r['det'].startswith('PSS') and r['cls'] == 'ded']
    for r, col in zip(shown, cols):
        h = Z.get(f"{r['det']}{r['ch']}_ded_late")
        if h is None:
            continue
        mm = (x >= 1) & (x <= 9)
        y = h / max(h[mm].max(), 1)
        lab = f"{r['det']}{r['ch']}" + (f": peak {fnum(r['mu']):.1f} mV" if r['accepted'] == 'True' else ': no clear peak')
        P1.line(x[mm].tolist(), y[mm].tolist(), col, 2.5 if r['accepted'] == 'True' else 1.2,
                dash=None if r['accepted'] == 'True' else '4 4', markers=False, tip=lab)
    P1.xticks([(v, str(v)) for v in range(1, 10)]).yticks([(v, f'{v:.1f}') for v in (0, 0.5, 1.0)])
    m = (tcc >= 3) & (tcc < 1e4)
    P2 = sd.Plot(810, 560, x=(3, 1e4, 'log'), y=(0.6, 1.8), xlabel='time since the flash',
                 ylabel='peak position / 1–10 ms value', title='the candidate peak against time')
    P2.band([3, 1e4], [0.95, 0.95], [1.05, 1.05], sd.GREEN, alpha=0.12)
    curves = []
    for r, col in zip(acc, cols):
        mu = Z.get(f"{r['det']}{r['ch']}_ded_mu")
        if mu is None:
            continue
        mu = np.where((mu > 0.5) & (mu < 2.0), mu, np.nan)          # failed fits
        curves.append(mu)
        P2.line(tcc[m].tolist(), clip(mu[m], 0.6, 1.8), col, 2, markers=True, r=4,
                tips=[f"{r['det']}{r['ch']}, {fmt_t(t)}: {v:.2f}" for t, v in zip(tcc[m], mu[m])])
    if curves:
        mean = np.nanmean(curves, axis=0)
        P2.line(tcc[m].tolist(), clip(mean[m], 0.6, 1.8), sd.INK, 4, markers=False, tip='mean of the accepted channels')
    P2.hline(1, sd.INK, None, 1.2)
    P2.xticks(tlog_ticks(3, 1e4)).yticks([(v, f'{v:.1f}') for v in (0.6, 1.0, 1.4, 1.8)])
    if curves:
        mean = np.nanmean(curves, axis=0)
        j100 = int(np.argmin(np.abs(tcc - 100)))
        v100 = mean[j100]
        if v100 > 1.05:
            ttl = (f'The small-pulse peak sits HIGHER early (×{v100:.2f} at 100 µs) on {len(acc)} plastic bars: '
                   f'rate-contaminated, not a clean gain reference')
        else:
            ttl = (f'A population-free reference on {len(acc)} plastic bars: the small-pulse peak sits at {v100:.2f} '
                   f'of its late position at 100 µs')
    else:
        ttl = 'No usable single-photoelectron peak: the population-free reference is not available'
    body = (sd.title(ttl, 'Small peaks in the quiet tails of zero-suppressed blocks: left, the late spectra; right, the fitted peak '
                          'position against time (accepted channels, dedicated)')
            + sd.row(P1.svg('spe spectra'), P2.svg('spe vs time'), gap=44)
            + sd.legend([('accepted channel', sd.INK, 'line'), ('no clear peak', sd.MUT, 'dash'), ('±5 %', sd.GREEN, 'box')]))
    rows_ = ''.join(f"<tr><td>{r['det']}{r['ch']}</td><td>{r['cls']}</td><td>{int(fnum(r['n_late'])):,}</td><td>{fnum(r['mu']):.2f} ± {fnum(r['mu_err']):.2f}</td>"
                    f"<td>{fnum(r['sigma']):.2f}</td><td>{fnum(r['signif']):.1f}</td><td>{r['accepted']}</td></tr>"
                    for r in R if r['det'][:3] in ('PSS', 'LIQ'))
    notes = ('<p>Dark counts and single photoelectrons are the same population at every time since the flash, so the '
             'position of their peak is a pure gain reference, unlike every spectrum of real pulses. Peaks are taken '
             'from SPE_SKIP = 700 samples into each ZS block (the pulse that opened the block is ~260 samples in), smoothed '
             'like the pulse finder, 0.5–10 mV. Fit: Gaussian + two falling exponentials over 1–9 mV. Accepted: peak at '
             '1.5–6 mV, narrower than its position, > 5σ. Per time bin the width and the exponential slopes are fixed '
             'to the late fit. Result: early the peak moves UP, not down, settling by ~1 ms: while the rate is high the '
             'quiet tails are not quiet (multi-photoelectron and small real pulses pile into the fit), so this cannot '
             'measure a gain sag; a sag would push the peak DOWN, so it is at least not evidence for a large one. The '
             'walls show no peak (a SiPM photoelectron is below their noise; one wall end passing the cuts is excluded).</p><table><tr><th>channel</th><th>class</th><th>peaks 1–10 ms</th><th>peak (mV)</th>'
             '<th>σ (mV)</th><th>significance</th><th>accepted</th></tr>' + rows_ + '</table>')
    D.slide('spe', body, notes, short='photoelectrons')


def plan(D):
    steps = [('done', 'Template fit with sub-ns shift, plastic edge per bar, bar ends, along the bar, per channel, '
              'undershoot ∝ gain, trigger bound, photoelectron test.'),
             ('separate gain from mix', 'The plastic light edge (0.71 / 0.87 / 0.93 at 10 / 100 / 300 µs, all 8 bars together) '
              'is the best common-mode reference; Geant4 of the coincident population would tell how much is mix.'),
             ('MM tracks (deferred)', 'From ~1 ms only (DREAM), after nearly everything has recovered. Needs the scint_stack '
              'track tables rebuilt from stage 3 (11 GB).'),
             ('per-channel templates', 'The liquids\' shape metric is soft (60 → 133 µs between template versions): fit their '
              'pulses with per-channel templates and a PSD-aware model.')]
    body = (sd.title('Next: separate gain from particle mix on the plastics; the MM tracks; the liquids\' shape',
                     'The full analysis, on the same dataset')
            + sd.flow([dict(label=s, sub=t, color=sd.BLUE) for s, t in steps], size=24)
            + sd.row(sd.callout('The baseline is a necessary condition. A PMT or SiPM can be back on baseline with its gain '
                                'still low; only the pulses can tell.', sd.BLUE, 26),
                     sd.callout('Intensity stays a split, never a pool. So do the wall divert state and the 22 July '
                                'gate change.', sd.ORANGE, 26), gap=32))
    notes = ('<p>Inputs: the flash_tail products (raw blocks, 0–10 ms and 30–32 ms), the official PSA hits for '
             'comparison, and the stage-3 MM tracks / scint_stack tables for the position step. The MM tracks exist '
             'only on DREAM triggers, so position dependence can be studied from ~0.5 ms on, not in the first µs.</p>')
    D.slide('plan', body, notes, short='next')


def limits(D):
    items = [
        ('A common gain sag is not separated from the particle mix.', 'A loss on one end of a wall bar is ruled out '
         '(the end ratio moves ≤ 4 % after 2 µs). A loss common to both ends, and the plastics\' slow last ~15 % to '
         '~0.5 ms, can still be gain or a changing particle mix: upper bounds, not measurements.'),
        ('A sample of the campaign.', '~40 bunches per run (2 raw files), 6 files for the divert-off runs and 224572. Ample '
         'for medians; thin for rare configurations.'),
        ('Pile-up on the plastics.', 'In the first ~100 µs the plastics are a continuous pulse train; their "baseline" there '
         'is what a discriminator sees, not the amplifier\'s.'),
        ('The trigger path is bounded, not measured.', 'DREAM opens at ~1 ms, when the wall sum is ≤ ~1.6 mV low, so '
         'a bias is at most ~5 % of threshold. DREAM\'s variable busy time prevents measuring it directly.'),
        ('Time base to ~10 ns.', 'Per-detector flash offsets, not per channel: irrelevant at these times, not good enough '
         'for anything inside the first 100 ns.'),
    ]
    cards = ''.join(sd.card(sd.p(f'<b>{a}</b>', 26, sd.DINK) + sd.p(b, 23, sd.DMUT), bg='#1e2430') for a, b in items[:3])
    cards2 = ''.join(sd.card(sd.p(f'<b>{a}</b>', 26, sd.DINK) + sd.p(b, 23, sd.DMUT), bg='#1e2430') for a, b in items[3:])
    body = (sd.title(f'<span style="color:{sd.DINK}">What this does not rule out</span>')
            + f'<div style="display:flex;gap:28px">{cards}</div><div style="display:flex;gap:28px">{cards2}</div>')
    D.slide('limits', body, '', dark=True, short='limits')


def main() -> int:
    C = dict(np.load(Q / 'curves.npz'))
    R = recovery()
    BR = by_run()
    B = rows('bunches.csv')
    X = dict(np.load(Q / 'examples.npz'))
    D = sd.Deck('Scintillator flash recovery', 'How long the X17 scintillators take to recover after the n_TOF γ-flash')
    cover(D, R, B)
    timeline(D)
    intensity(D, B)
    for fam in ('WAL', 'PSS', 'LIQ'):
        examples(D, R, X, fam)
    averages(D, C)
    baselines(D, C, R)
    summary(D, R, BR)
    stability(D, BR)
    divert(D, C, R)
    PC = pulse_curves()
    if PC is not None:
        PR = pulse_rows()
        shape(D, PC, PR)
        amplitude(D, PC, PR)
        wall_edge(D, PC)
    K = coinc_curves()
    if K is not None:
        KR = coinc_rows()
        fit_shape(D, K, KR)
        plastic_wall(D, K, KR)
        plastic_edge_bars(D, K, KR)
        bar_ends(D, K, KR)
        along_bar(D, K, KR)
    CH = channel_rows()
    if CH:
        channel_map(D, CH)
        undershoot_gain(D, CH)
    trigger_slide(D)
    spe_slide(D)
    plan(D)
    limits(D)
    path = D.write(out_dir('deck') / 'scint-flash-recovery.html', note_meta=dict(
        title='Scintillator recovery after the γ-flash',
        summary='Quick pass: example and average waveforms of the 44 X17 scintillator channels after the n_TOF flash, '
                'and the time each one takes to come back to baseline, split by beam intensity.',
        tags='X17, n_TOF, scintillators', date=dt.date.today().isoformat()))
    print(f'-> {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
