#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_flash_slides.py -- the DAQ-saturation figures, rebuilt for the reworked
Status slides 28-30 (2026-08-20).

    ../.venv/bin/python make_flash_slides.py            # everything
    ../.venv/bin/python make_flash_slides.py --only railing,two_chains
    ../.venv/bin/python make_flash_slides.py --numbers  # the arithmetic only

WHY A SECOND FILE AND NOT make_status_plots.py.  That module's ``render()`` is
imported by ``ntof_july_analysis/flash_charge/make_report.py``, which calls it
with no name list and therefore builds every figure in its ``FIGURES`` table
into the report's own directory.  Adding five deck-only figures there would
make the analysis report re-render them -- including the two that open a
154 MB waveform archive.  The old figures are unchanged and still belong to
that package; these belong to the talk.

WHAT CHANGED, PER FIGURE (Dylan, 2026-08-20).

railing
    The left panel of ``status_flash_waveform.png`` on its own, in a column
    aspect, with the burned-in title stripped -- slide 28 is now a two-frame
    build whose right-hand column pops this trace up beside the DREAM
    introduction.  The noise panel that was beside it stayed a pair and went
    to backup.

two_readouts_op
    The interval plot, rebuilt at the PRODUCTION OPERATING POINT.  The version
    in the deck until today compared run 224302 (July, Ar/CF4/iso 88/10/2, and
    a chamber whose identity is not recoverable from the data) against DREAM
    det A at 540 V -- three caveats, and Dylan was right to be suspicious of
    them.  Run 224709 removes all three: its MMA channel *is* strip 32 of
    detector A on cable Y8, the gas is Ar/iso 90/10, and it sits on a
    700 / 540 V plateau, which is the same chamber at the same amplification
    voltage as the run_57 recovery point.  Both rows are now one chamber at
    one setpoint, and the number the slide quotes is their ratio.

two_chains
    New.  "Can we include an actual waveform near the flash from ntof daq
    (ideally at operating voltage) and show that it goes back to baseline
    normally there?  Can we actually superimpose a dream waveform on the same
    plot?"  Both, on one time axis, each aligned on its own flash peak.

charge_ladder
    ``status_charge_scale.png`` cut from eight rows to three: the largest
    DREAM range, and the two independent determinations of detector A's flash
    charge.  The four CSA ranges were four rows saying one thing, and the
    second chamber was not the comparison being made.

deadtime_detA
    ``status_deadtime_vs_charge.png`` for detector A alone, with the
    amplification voltage written on the points, the production and
    gain-optimum voltages marked, and the MeV window drawn three decades below
    anything measured.  The three-chamber version is unchanged, in backup.

eff_recovery
    New, for backup.  "We had a very nice plot of recovery time curves
    superimposed with efficiency curves from the cosmic bench, can you find
    that and put in backup."  It does not exist -- searched the repo, the run
    report and both flash packages -- so it is built here from the two
    committed reductions.  The scans barely overlap in voltage and the figure
    says so on its face.

SOURCES
-------
n_TOF digitiser   /media/dylan/data/x17/ntof_mm_flash/mm_224709.npz  (+ the
                  plateau table ntof_processing/mm_flash/hv_plateaus_224709.csv)
                  and results_709.json / results_board.json for the reduced
                  numbers, which are read rather than recomputed.
DREAM waveform    ~/.cache/mpgd26_status/wf/*flashOff*A500*.root  (run_32)
recovery          ~/.cache/mpgd26_status/metrics_run_57_perdet.csv
charge vs HV      ntof_july_analysis/flash_charge/results/flash_charge_subruns.csv
bench efficiency  <cosmic_bench>/Analysis/mx17_det2_det3_overnight_6-22-26/
                  hv_scan/mx17_{2,3}/efficiency_vs_hv.csv

Anything missing is skipped with a message rather than crashing the set.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import plotstyle as P                       # noqa: E402
import make_status_plots as S               # noqa: E402

OUT = os.path.join(HERE, 'slides', 'assets', 'img')
FIGDIR = os.path.join(HERE, 'figures')
CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'mpgd26_status')

MM_NPZ = '/media/dylan/data/x17/ntof_mm_flash/mm_224709.npz'
MM_PLATEAUS = os.path.join(REPO, 'ntof_processing', 'mm_flash',
                           'hv_plateaus_224709.csv')
MM_RESULTS = os.path.join(REPO, 'ntof_processing', 'mm_flash',
                          'results_709.json')
MM_BOARD = os.path.join(REPO, 'ntof_processing', 'mm_flash',
                        'results_board.json')
BENCH = ('/media/dylan/data/x17/cosmic_bench/Analysis/'
         'mx17_det2_det3_overnight_6-22-26/hv_scan/mx17_{n}/efficiency_vs_hv.csv')
BENCH_DET = {'3': 'A', '2': 'B'}

# run 224709's own calibration, from the MODH channel record (analyse_709.py)
MV_PER_COUNT = 5043.7915 / 65536
ZS_FILL = -32768
SETTLE_S = 45                   # discard this much after every HV change
PKUP_SPLIT = 20875.5            # dedicated / parasitic, results_709.json

# the working point everything on slides 29-30 is quoted at
OP_DRIFT, OP_RESIST = 700, 540
GAIN_RESIST = 560               # where the chamber would run for efficiency
MM_THRESH_MV = 4.0

# the two windows, in ms since the flash
THERMAL_LO, THERMAL_HI = S.THERMAL_LO, S.THERMAL_HI      # 3-8 ms
MEV_LO_MS, MEV_HI_MS = 0.449e-3, 4.459e-3                # make_x17_rate's decades

CSA_FULL_SCALE_PC = 0.600       # the largest DREAM range, 600 fC


def save(fig, name):
    """Write the figure to the deck's asset directory AND to figures/.

    The deck reads assets/img; mpgd26/make_report.py's ``plain_fig`` reads
    figures/.  Writing both from one call is what stops the report showing a
    stale copy of a figure the slide has already moved on from.
    """
    P.save(fig, os.path.join(OUT, f'{name}.png'))
    os.makedirs(FIGDIR, exist_ok=True)
    import shutil
    shutil.copyfile(os.path.join(OUT, f'{name}.png'),
                    os.path.join(FIGDIR, f'{name}.png'))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _plateaus():
    rows = []
    with open(MM_PLATEAUS) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(start=int(r['start_s']), end=int(r['end_s']),
                             drift=int(r['A_drift_V']),
                             resist=int(r['A_resist_V'])))
    return rows


def mm_mean_trace(drift=OP_DRIFT, resist=OP_RESIST, cls='dedicated'):
    """Bunch-mean MMA trace [mV vs ns] on one HV plateau of run 224709.

    Reproduces analyse_709.py exactly: baseline-subtracted with the per-bunch
    baseline from `stats`, zero-suppressed samples zeroed, beam class split on
    the PKUP flash peak at the same threshold that file recorded.
    """
    if not os.path.exists(MM_NPZ):
        return None
    d = np.load(MM_NPZ)
    wall = d['wall']
    secs = ((wall[:, 0] // 10000) * 3600 + ((wall[:, 0] // 100) % 100) * 60
            + wall[:, 0] % 100)
    stats = d['stats']
    ev = stats[:, 0].astype(int)
    base = stats[:, 1]

    pk = np.zeros(len(d['bunch']))
    pkup = d['pkup']
    pk[pkup[:, 0].astype(int)] = pkup[:, 1]
    pk_ev = pk[ev]
    dedicated = (pk_ev > 1000) & (pk_ev > PKUP_SPLIT)
    want = dedicated if cls == 'dedicated' else ((pk_ev > 1000) & ~dedicated)

    hit = None
    for p in _plateaus():
        if p['drift'] == drift and p['resist'] == resist:
            hit = p
            break
    if hit is None:
        return None
    m = ((secs[ev] >= hit['start'] + SETTLE_S) & (secs[ev] <= hit['end'])
         & want)
    if m.sum() < 5:
        return None

    # only the selected events are materialised -- the whole archive is 160 MB
    sel = ev[m]
    raw = d['flash'][sel].astype(np.float64)
    dev = base[m][:, None] - raw
    dev[raw <= ZS_FILL] = 0.0
    mt = dev.mean(axis=0) * MV_PER_COUNT
    return dict(t_ns=np.arange(mt.size, dtype=float), mv=mt, n=int(m.sum()))


def mm_recovery_ns(drift=OP_DRIFT, resist=OP_RESIST, cls='dedicated'):
    """Recovery of the bunch-mean trace below 4 mV, relative to its own peak.

    Read from results_709.json rather than recomputed, so the slide and the
    analysis package cannot disagree about it.
    """
    if not os.path.exists(MM_RESULTS):
        return np.nan
    r = json.load(open(MM_RESULTS))
    for row in r['scan']:
        if (row['drift'] == drift and row['resist'] == resist
                and row['cls'] == cls):
            return float(row['recovery']['4mV'])
    return np.nan


def dream_flash_event():
    """One run_32 flash event: time [us since its own peak] and 24 hot strips."""
    cands = glob.glob(os.path.join(CACHE, 'wf', '*flashOff*A500*.root'))
    if not cands:
        return None
    import uproot
    t = uproot.open(cands[0])['nt']
    n = min(t.num_entries, 40)
    amp = t['amplitude'].array(entry_stop=n, library='np')
    nch, dt_ns = 512, 20.0
    W = np.stack([np.asarray(x, float).reshape(-1, nch) for x in amp])
    ev = W[0]
    hot = np.argsort(-np.ptp(ev, axis=0))[:24]
    tt = np.arange(ev.shape[0]) * dt_ns / 1e3          # us
    return dict(t_us=tt, traces=ev[:, hot], all_events=W, dt_ns=dt_ns)


def dream_recovery_ms(det='A', volts=OP_RESIST):
    """run_57 post-flash recovery of one chamber at one amplification voltage."""
    rec = S.load_recovery(CACHE)
    rows = S.load_charge('run_57')
    sub_v = {(r['det'], r['subrun']): r['resist_v'] for r in rows}
    out = np.nan
    for s, ms in rec.get(det, {}).items():
        if abs(sub_v.get((det, s), -1) - volts) <= 1:
            out = ms
    return out


def detA_charge_recovery():
    """run_57 detector A: (charge nC, recovery ms, resist V) per sub-run."""
    rows = S.load_charge('run_57')
    rec = S.load_recovery(CACHE)
    q_by_sub = {r['subrun']: (r['q_per_pulse_nc'], r['resist_v'])
                for r in rows if r['det'] == 'A'}
    pts = []
    for s, ms in rec.get('A', {}).items():
        if s not in q_by_sub:
            continue
        q, v = q_by_sub[s]
        if np.isfinite(q) and q > 0:
            pts.append((q, max(ms, 0.2), v))
    pts.sort()
    return pts


def bench_efficiency():
    """{letter: (hv, eff, err, spark_frac)} from the June cosmic-bench scans.

    The spark fraction comes back with it because it is what the efficiency
    curve is DOING above ~485 V: on this bench, at drift 1000 V, both chambers
    start discharging and the matched-hit efficiency falls with the spark rate.
    Plotting the efficiency without it invites the reading that more gain is
    simply worse, which is not what the n_TOF scan says.
    """
    out = {}
    for num, letter in BENCH_DET.items():
        path = BENCH.format(n=num)
        if not os.path.exists(path):
            continue
        hv, eff, err, spk = [], [], [], []
        for r in csv.DictReader(open(path)):
            try:
                hv.append(float(r['hv']))
                eff.append(float(r['eff_reco']))
                err.append(float(r['eff_reco_err']))
                spk.append(float(r.get('spark_frac', 'nan')))
            except (KeyError, ValueError):
                continue
        if hv:
            out[letter] = (np.array(hv), np.array(eff), np.array(err),
                           np.array(spk))
    return out


def board_numbers():
    """The two determinations of detector A's flash charge, at 700 / 540 V."""
    if not os.path.exists(MM_BOARD):
        return None
    b = json.load(open(MM_BOARD))['working_point']
    strip_ded = np.nan
    if os.path.exists(MM_RESULTS):
        for row in json.load(open(MM_RESULTS))['scan']:
            if (row['drift'] == OP_DRIFT and row['resist'] == OP_RESIST
                    and row['cls'] == 'dedicated'):
                strip_ded = float(row['charge_pC'])
    return dict(uniform_pC=float(b['expected_uniform_pC']),
                strip_mix_pC=float(b['q_strip_mix_pC']),
                strip_ded_pC=strip_ded,
                chamber_nC=float(b['q_imon_nC']),
                residual=float(b['residual']))


# --------------------------------------------------------------------------- #
# 1. The railing trace, alone
# --------------------------------------------------------------------------- #

def fig_railing():
    d = dream_flash_event()
    if d is None:
        print('  .. railing: run_32 decoded_root missing, skipped')
        return
    # 1.216:1 -- the MEASURED aspect of slide 28's right-hand imgwrap
    fig, ax = plt.subplots(figsize=(6.6, 5.43))
    tt = d['t_us']
    for c in range(d['traces'].shape[1]):
        ax.plot(tt, d['traces'][:, c], color=P.DET_COLOR['A'], lw=0.9,
                alpha=0.35)
    for y, lab, va in ((4095, ' +rail 4095', 'bottom'), (0, ' −rail 0', 'top')):
        ax.axhline(y, color=P.BAND_DEAD, lw=1.4, ls='--')
        ax.text(tt[-1], y, lab, color=P.BAND_DEAD, fontsize=10.5, va=va,
                ha='right', fontweight='bold')
    # the point of the panel is the CONTRAST between the two baselines, so it
    # is drawn: ripple before, none after
    ax.annotate('', xy=(0.05, 250), xytext=(1.45, 250),
                arrowprops=dict(arrowstyle='<->', color=P.MUTED, lw=1.0))
    ax.text(0.75, 195, 'noise', fontsize=10, color=P.MUTED, ha='center',
            va='top')
    ax.annotate('', xy=(3.2, 250), xytext=(7.95, 250),
                arrowprops=dict(arrowstyle='<->', color=P.MUTED, lw=1.0))
    ax.text(5.6, 195, 'none', fontsize=10, color=P.MUTED, ha='center',
            va='top')
    ax.set_xlabel('time in the read-out window  [µs]')
    ax.set_ylabel('ADC code')
    ax.set_ylim(-350, 4450)
    ax.set_xlim(tt[0], tt[-1])
    P.strip(ax)
    fig.tight_layout()
    save(fig, 'status_flash_railing')


# --------------------------------------------------------------------------- #
# 2. The interval plot, at the production operating point
# --------------------------------------------------------------------------- #

def fig_two_readouts_op():
    mm_ns = mm_recovery_ns()
    dream_ms = dream_recovery_ms('A', OP_RESIST)
    if not np.isfinite(mm_ns) or not np.isfinite(dream_ms):
        print('  .. two_readouts_op: inputs missing, skipped')
        return
    mm_ms = mm_ns / 1e6

    # 1.505:1 -- slide 29's left imgwrap
    fig, ax = plt.subplots(figsize=(8.4, 5.58))
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 1e2)

    ax.axvspan(THERMAL_LO, THERMAL_HI, color=P.BAND_SIGNAL, alpha=0.10,
               zorder=0)
    ax.text(np.sqrt(THERMAL_LO * THERMAL_HI), 2.68,
            'thermal neutrons\narrive here', color=P.BAND_SIGNAL, fontsize=10.5,
            fontweight='bold', ha='center', va='bottom')

    rows = [
        (1.95, 'the chamber — digitised directly, 1 GS/s, no charge amplifier',
         mm_ms, P.DET_COLOR['C'],
         f'usable again {mm_ms * 1e3:.1f} µs after the flash peak'),
        (0.60, 'the read-out — the same chamber, same 540 V, through DREAM',
         dream_ms, P.DET_COLOR['B'],
         f'front-end noise back after {dream_ms:.1f} ms'),
    ]
    for y, label, t_end, col, ann in rows:
        ax.plot([1e-4, t_end], [y, y], color=P.BAND_DEAD, lw=15, alpha=0.30,
                solid_capstyle='butt', zorder=2)
        ax.plot([t_end, 1e2], [y, y], color=col, lw=15, alpha=0.80,
                solid_capstyle='butt', zorder=2)
        ax.plot([t_end], [y], marker='|', ms=30, color=P.INK, mew=2.2, zorder=4)
        ax.text(1.4e-4, y + 0.30, label, fontsize=11.0, fontweight='bold',
                color=P.INK, va='bottom')
        ax.text(1.4e-4, y - 0.30, ann, fontsize=10.5, color=col,
                fontweight='bold', va='top')

    ax.annotate(f'×{dream_ms / mm_ms:,.0f}', xy=(np.sqrt(mm_ms * dream_ms), 1.30),
                ha='center', va='center', fontsize=15, fontweight='bold',
                color=P.INK, zorder=6,
                bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=2.0))
    ax.annotate('', xy=(mm_ms * 1.15, 1.82), xytext=(mm_ms * 1.15, 0.73),
                arrowprops=dict(arrowstyle='-', color=P.MUTED, lw=1.0,
                                ls=(0, (2, 2))), zorder=3)
    ax.annotate('', xy=(dream_ms / 1.15, 1.82), xytext=(dream_ms / 1.15, 0.73),
                arrowprops=dict(arrowstyle='-', color=P.MUTED, lw=1.0,
                                ls=(0, (2, 2))), zorder=3)

    ax.set_yticks([])
    ax.set_ylim(0.05, 3.20)
    ax.set_xlabel('time since the γ flash  [ms, log scale]')
    ax.grid(axis='y', visible=False)
    P.strip(ax, left=False)
    ax.text(0.0, 1.015, 'red = blind · colour = usable · one chamber, one '
            'voltage, two read-out chains', transform=ax.transAxes, ha='left',
            va='bottom', fontsize=11, color=P.MUTED)
    fig.tight_layout()
    save(fig, 'status_two_readouts_op')


# --------------------------------------------------------------------------- #
# 3. The same flash down both chains
# --------------------------------------------------------------------------- #

def fig_two_chains():
    mm = mm_mean_trace()
    dr = dream_flash_event()
    if mm is None or dr is None:
        print('  .. two_chains: inputs missing, skipped')
        return
    rec_ns = mm_recovery_ns()

    # both aligned on their OWN flash peak: the two chains were triggered by
    # different systems and neither t = 0 is the other's
    i_pk = int(np.argmax(mm['mv']))
    t_mm = (mm['t_ns'] - mm['t_ns'][i_pk]) / 1e3               # us
    trace = dr['traces']
    j_pk = int(np.argmax(np.median(trace, axis=1)))
    t_dr = dr['t_us'] - dr['t_us'][j_pk]

    # 1.446:1 -- slide 29's right imgwrap
    fig, ax = plt.subplots(figsize=(8.1, 5.60))
    ax.set_xlim(-0.6, 6.0)

    axd = ax.twinx()
    for c in range(trace.shape[1]):
        axd.plot(t_dr, trace[:, c], color=P.DET_COLOR['A'], lw=0.9, alpha=0.30,
                 zorder=2)
    axd.set_ylim(-350, 4450)
    axd.set_ylabel('DREAM, ADC code', color=P.DET_COLOR['A'])
    axd.tick_params(axis='y', colors=P.DET_COLOR['A'])
    axd.grid(False)
    for side in ('top', 'left', 'bottom'):
        axd.spines[side].set_visible(False)
    axd.spines['right'].set_color(P.LINE)
    axd.axhline(4095, color=P.BAND_DEAD, lw=1.1, ls='--', zorder=1)
    axd.text(5.95, 4095, ' +rail 4095 ', color=P.BAND_DEAD, fontsize=9.5,
             va='bottom', ha='right', fontweight='bold')
    axd.text(5.95, 0, ' −rail 0 ', color=P.BAND_DEAD, fontsize=9.5,
             va='bottom', ha='right', fontweight='bold')

    ax.plot(t_mm, mm['mv'], color=P.DET_COLOR['C'], lw=2.0, zorder=5)
    ax.set_ylim(-12, 62)
    ax.set_ylabel('n_TOF digitiser, strip signal  [mV]', color=P.DET_COLOR['C'])
    ax.tick_params(axis='y', colors=P.DET_COLOR['C'])
    ax.axhline(MM_THRESH_MV, color=P.DET_COLOR['C'], lw=1.0, ls=':', zorder=4)
    ax.text(-0.55, MM_THRESH_MV, ' 4 mV threshold', color=P.DET_COLOR['C'],
            fontsize=9.5, va='bottom', ha='left')
    ax.set_zorder(axd.get_zorder() + 1)
    ax.patch.set_visible(False)

    if np.isfinite(rec_ns):
        x = rec_ns / 1e3
        ax.plot([x], [MM_THRESH_MV], marker='v', ms=10,
                color=P.DET_COLOR['C'], zorder=6)
        ax.annotate(f'back under threshold\n{x:.1f} µs after its own peak',
                    xy=(x, MM_THRESH_MV), xytext=(x + 0.45, 27.0),
                    fontsize=10.5, color=P.DET_COLOR['C'], fontweight='bold',
                    va='center', zorder=6,
                    arrowprops=dict(arrowstyle='->', color=P.DET_COLOR['C'],
                                    lw=1.2))
    ax.annotate('this baseline carries no noise —\nthe channel is still dead,'
                ' for another 5 ms',
                xy=(4.6, 1.0), xytext=(2.35, 55.0), fontsize=10.5,
                color=P.DET_COLOR['A'], fontweight='bold', va='top', zorder=6,
                arrowprops=dict(arrowstyle='->', color=P.DET_COLOR['A'],
                                lw=1.2))

    ax.set_xlabel('time since each chain’s own flash peak  [µs]')
    P.strip(ax)
    ax.text(0.0, 1.015, 'green: det A, 700 / 540 V, mean of '
            f'{mm["n"]} dedicated pulses  ·  blue: DREAM, 24 hottest strips, '
            'one event', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=11, color=P.MUTED)
    fig.tight_layout()
    save(fig, 'status_flash_two_chains')


# --------------------------------------------------------------------------- #
# 4. The charge ladder -- three rows, one axis
# --------------------------------------------------------------------------- #

def fig_charge_ladder():
    b = board_numbers()
    if b is None:
        print('  .. charge_ladder: results_board.json missing, skipped')
        return
    # the strip row is the PULSE MIX, not the dedicated-pulse median: the mix
    # is what results_board.json divides by the board's uniform expectation to
    # get the 4.1x charge-density residual, and quoting the dedicated 662 pC
    # here would put a 5.0x on the slide against a 4.1x in the note it cites.
    strip = b['strip_mix_pC']
    rows = [
        ('DREAM CSA full scale\n(largest range, 600 fC)', CSA_FULL_SCALE_PC,
         P.MUTED, None),
        ('γ flash, chamber average per channel\nfrom the HV supply current',
         b['uniform_pC'], P.DET_COLOR['B'], b['uniform_pC'] / CSA_FULL_SCALE_PC),
        ('γ flash, the one strip measured directly\nn_TOF digitiser, 1 GS/s',
         strip, P.DET_COLOR['C'], strip / CSA_FULL_SCALE_PC),
    ]
    # 1.488:1 -- slide 30's left imgwrap
    fig, ax = plt.subplots(figsize=(8.3, 5.58))
    y = np.arange(len(rows))[::-1]
    xmin = 0.12
    for yy, (lab, v, col, ratio) in zip(y, rows):
        ax.plot([xmin, v], [yy, yy], color=col, lw=1.4, alpha=0.35, zorder=2)
        ax.plot([v], [yy], marker='o', ms=13, color=col, zorder=3,
                markeredgecolor=P.SURFACE, markeredgewidth=1.2)
        ax.text(v * 1.32, yy, f'{v:.1f} pC' if v < 10 else f'{v:,.0f} pC',
                va='center', fontsize=13.0, color=col, fontweight='bold')
        # the ratio rides ON the leader line, in the empty half of the axis --
        # to the right of the dot it ran off the canvas
        if ratio:
            ax.text(np.sqrt(xmin * v), yy + 0.13,
                    f'×{ratio:,.0f} the front end’s full scale', ha='center',
                    va='bottom', fontsize=11.5, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.5)
    ax.set_xscale('log')
    ax.set_xlim(xmin, 6e3)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel('charge presented to ONE DREAM input  [pC, log scale]')
    ax.grid(axis='y', visible=False)
    P.strip(ax, left=False)

    # the gap between the two determinations is a measured charge-density
    # ratio, not a disagreement -- draw it as one
    lo, hi = b['uniform_pC'], strip
    ax.annotate('', xy=(lo, 0.5), xytext=(hi, 0.5),
                arrowprops=dict(arrowstyle='<->', color=P.MUTED, lw=1.2))
    ax.text(lo / 1.35, 0.5, f'×{hi / lo:.1f} — that strip’s own charge density ',
            ha='right', va='center', fontsize=11.0, color=P.MUTED)
    fig.tight_layout()
    save(fig, 'status_charge_ladder')


# --------------------------------------------------------------------------- #
# 5. Dead time against charge, detector A alone
# --------------------------------------------------------------------------- #

def fig_deadtime_detA():
    pts = detA_charge_recovery()
    if len(pts) < 5:
        print('  .. deadtime_detA: inputs missing, skipped')
        return
    q = np.array([p[0] for p in pts])
    m = np.array([p[1] for p in pts])
    v = np.array([p[2] for p in pts])

    # 1.320:1 -- slide 30's right imgwrap
    fig, ax = plt.subplots(figsize=(7.4, 5.61))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(2.5e-4, 70.0)
    ax.set_xlim(q.min() / 2.6, q.max() * 2.6)

    ax.axhspan(THERMAL_LO, THERMAL_HI, color=P.BAND_SIGNAL, alpha=0.11,
               zorder=0)
    ax.text(0.985, np.sqrt(THERMAL_LO * THERMAL_HI), 'thermal window ',
            transform=ax.get_yaxis_transform(), color=P.BAND_SIGNAL,
            fontsize=10.5, fontweight='bold', va='center', ha='right')
    ax.axhspan(MEV_LO_MS, MEV_HI_MS, color=P.BAND_DEAD, alpha=0.18, zorder=0)
    ax.text(0.015, np.sqrt(MEV_LO_MS * MEV_HI_MS),
            'the MeV window — 79 % of the rate', color=P.BAND_DEAD,
            transform=ax.get_yaxis_transform(), fontsize=10.5,
            fontweight='bold', va='center', ha='left')

    ok = m > 0.25                       # the prompt floor is a limit, not a point
    c = np.polyfit(np.log10(q[ok]), np.log10(m[ok]), 1)
    xs = np.logspace(np.log10(q.min() / 1.6), np.log10(q.max() * 2.2), 50)
    ax.plot(xs, 10 ** np.polyval(c, np.log10(xs)), color=P.MUTED, lw=1.4,
            ls='--', zorder=2)
    ax.text(0.985, 0.40, f't ∝ Q$^{{{c[0]:.2f}}}$', transform=ax.transAxes,
            ha='right', color=P.MUTED, fontsize=13.0)

    ax.plot(q, m, marker=P.DET_MARKER['A'], color=P.DET_COLOR['A'], lw=0,
            ms=7.5, markeredgecolor=P.SURFACE, markeredgewidth=0.8, zorder=3)

    # the amplification voltage, on the three points that have room for it
    for volts, off in ((520, (-6, -16)), (550, (10, 3)), (580, (2, -16))):
        k = int(np.argmin(np.abs(v - volts)))
        if abs(v[k] - volts) > 1:
            continue
        ax.annotate(f'{volts:.0f} V', xy=(q[k], m[k]), xytext=off,
                    textcoords='offset points', fontsize=10.5, color=P.MUTED)

    # the two that carry the sentence, called out into the empty quadrants:
    # 540 above-left of its point, 560 below-right of its own
    for volts, col, lab, tx, ha, va in (
            (OP_RESIST, P.INK, 'where we ran', (0.30, 0.90), 'right', 'center'),
            (GAIN_RESIST, P.BAND_DEAD, 'where the gain wants to be',
             (0.97, 0.62), 'right', 'center')):
        k = int(np.argmin(np.abs(v - volts)))
        if abs(v[k] - volts) > 1:
            continue
        ax.plot(q[k], m[k], marker='*', ms=21, color=P.DET_COLOR['A'],
                markeredgecolor=col, markeredgewidth=1.5, zorder=5)
        ax.annotate(f'{volts} V — {lab}\n{m[k]:.1f} ms', xy=(q[k], m[k]),
                    xytext=tx, textcoords='axes fraction',
                    fontsize=11.5, color=col, fontweight='bold', ha=ha, va=va,
                    zorder=6,
                    bbox=dict(facecolor=P.SURFACE, edgecolor='none', pad=1.6),
                    arrowprops=dict(arrowstyle='-', color=col, lw=1.0,
                                    shrinkB=9))

    # the empty middle of the axis IS the result -- measure it rather than
    # leaving it blank
    x_gap = q.min() / 1.7
    k_op = int(np.argmin(np.abs(v - OP_RESIST)))
    y_lo, y_hi = MEV_HI_MS, m[k_op]
    ax.annotate('', xy=(x_gap, y_lo), xytext=(x_gap, y_hi),
                arrowprops=dict(arrowstyle='<->', color=P.MUTED, lw=1.1))
    ax.text(x_gap * 1.12, np.sqrt(y_lo * y_hi),
            f'{np.log10(y_hi / y_lo):.0f} decades below where\nwe ran — no '
            'voltage closes it', fontsize=10.5, color=P.MUTED, ha='left',
            va='center')

    ax.set_xlabel('avalanche charge delivered per beam pulse  [nC]')
    ax.set_ylabel('post-flash recovery time  [ms]')
    P.strip(ax)
    ax.text(0.0, 1.015, 'detector A · run_57 · both axes measured on the same '
            'sub-runs', transform=ax.transAxes, ha='left', va='bottom',
            fontsize=11, color=P.MUTED)
    fig.tight_layout()
    save(fig, 'status_deadtime_detA')


# --------------------------------------------------------------------------- #
# 6. Backup -- efficiency and blindness against the same knob
# --------------------------------------------------------------------------- #

def fig_eff_recovery():
    eff = bench_efficiency()
    rec = S.load_recovery(CACHE)
    rows = S.load_charge('run_57')
    if not eff or not rec or not rows:
        print('  .. eff_recovery: inputs missing, skipped')
        return
    sub_v = {(r['det'], r['subrun']): r['resist_v'] for r in rows}

    # 2.469:1 -- the backup slide's figure-solo hole
    fig, ax = plt.subplots(figsize=(12.5, 5.06))
    axr = ax.twinx()

    axr.axhspan(THERMAL_LO, THERMAL_HI, color=P.BAND_SIGNAL, alpha=0.11,
                zorder=0)
    axr.set_yscale('log')
    axr.set_ylim(0.3, 60)
    axr.set_ylabel('post-flash recovery at n_TOF  [ms, log]', color=P.MUTED)
    axr.tick_params(axis='y', colors=P.MUTED)
    axr.grid(False)

    for det in 'ABC':
        d = rec.get(det, {})
        pts = sorted((sub_v.get((det, s), np.nan), ms) for s, ms in d.items())
        pts = [(v, max(m, 0.2)) for v, m in pts if np.isfinite(v)]
        if len(pts) < 3:
            continue
        axr.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.8, ls='--',
                 color=P.DET_COLOR[det], alpha=0.85, zorder=3)
    axr.text(0.985, np.sqrt(THERMAL_LO * THERMAL_HI),
             'thermal window ', transform=axr.get_yaxis_transform(),
             color=P.BAND_SIGNAL, fontsize=10.5, fontweight='bold',
             va='center', ha='right')

    hv_a, _, _, spk_a = eff.get('A', (None, None, None, None))
    if hv_a is not None and np.isfinite(spk_a).any():
        ax.fill_between(hv_a, 0, spk_a * 100, color=P.BAND_DEAD, alpha=0.13,
                        zorder=1, lw=0)
        ax.plot(hv_a, spk_a * 100, color=P.BAND_DEAD, lw=1.4, ls='-',
                alpha=0.7, zorder=2)
        ax.text(498.0, 30.0, 'det A spark fraction\non the bench',
                color=P.BAND_DEAD, fontsize=10.5, fontweight='bold',
                ha='right', va='bottom', zorder=6)

    for det, (hv, e, err, _spk) in sorted(eff.items()):
        ax.errorbar(hv, e * 100, yerr=err * 100, marker=P.DET_MARKER[det],
                    color=P.DET_COLOR[det], lw=2.0, capsize=0, elinewidth=1.0,
                    markeredgecolor=P.SURFACE, markeredgewidth=0.8, zorder=5,
                    label=f'det {det}')
    ax.set_ylim(0, 100)
    ax.set_xlim(445, 585)
    ax.set_ylabel('cosmic-bench efficiency  [%]')
    ax.set_xlabel('amplification (resistive-layer) voltage  [V]')
    P.strip(ax)
    ax.set_zorder(axr.get_zorder() + 1)
    ax.patch.set_visible(False)

    for volts, lab in ((525, 'bench scan ends'), (OP_RESIST, 'we ran here')):
        ax.axvline(volts, color=P.INK, lw=1.0, ls=':', zorder=4)
        ax.text(volts + 1.6, 99, lab, fontsize=10, color=P.INK, rotation=90,
                va='top', ha='left')

    # det C exists only on the recovery axis, so it needs a proxy or its
    # green dashes are an unexplained fourth colour
    ax.plot([], [], ls='--', lw=1.8, color=P.DET_COLOR['C'],
            label='det C (recovery only)')
    ax.legend(loc='lower left', ncol=3, columnspacing=1.4, handletextpad=0.5,
              bbox_to_anchor=(0.0, 0.0))
    ax.text(0.0, 1.015, 'solid + markers, left axis: June cosmic bench, no '
            'flash  ·  dashed, right axis: run_57 at n_TOF',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=11,
            color=P.MUTED)
    fig.tight_layout()
    save(fig, 'status_eff_recovery')


# --------------------------------------------------------------------------- #

FIGURES = {
    'railing': fig_railing,
    'two_readouts_op': fig_two_readouts_op,
    'two_chains': fig_two_chains,
    'charge_ladder': fig_charge_ladder,
    'deadtime_detA': fig_deadtime_detA,
    'eff_recovery': fig_eff_recovery,
}


def numbers():
    mm_ns = mm_recovery_ns()
    dream = dream_recovery_ms('A', OP_RESIST)
    gain = dream_recovery_ms('A', GAIN_RESIST)
    b = board_numbers()
    print(f'  det A, {OP_DRIFT}/{OP_RESIST} V')
    print(f'    n_TOF digitiser back under 4 mV   {mm_ns / 1e3:6.2f} us '
          '(mean trace, after its own peak)')
    print(f'    DREAM noise back                  {dream:6.2f} ms')
    if np.isfinite(mm_ns) and np.isfinite(dream):
        print(f'    ratio                             x{dream / (mm_ns / 1e6):,.0f}')
    print(f'    DREAM at {GAIN_RESIST} V                    {gain:6.2f} ms')
    if b:
        print(f'    chamber charge per pulse          {b["chamber_nC"]:.1f} nC')
        print(f'    per channel, board accounting     {b["uniform_pC"]:.0f} pC'
              f'   x{b["uniform_pC"] / CSA_FULL_SCALE_PC:,.0f} full scale')
        print(f'    strip 32, dedicated pulses        {b["strip_ded_pC"]:.0f} pC'
              f'   x{b["strip_ded_pC"] / CSA_FULL_SCALE_PC:,.0f} full scale')
        print(f'    density residual                  x{b["residual"]:.1f}')
    for det, (hv, e, _err, _spk) in sorted(bench_efficiency().items()):
        print(f'  bench det {det}: {hv.min():.0f}-{hv.max():.0f} V, '
              f'eff {e.min() * 100:.0f}-{e.max() * 100:.0f} %')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='comma list of figure names')
    ap.add_argument('--numbers', action='store_true')
    args = ap.parse_args()
    numbers()
    if args.numbers:
        return 0
    P.use()
    for n in (args.only.split(',') if args.only else list(FIGURES)):
        if n not in FIGURES:
            print(f'  ?? unknown figure {n!r}')
            continue
        print(n)
        FIGURES[n]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
