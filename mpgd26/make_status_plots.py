#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_status_plots.py -- the data figures for the Status section of the MPGD2026
talk.

    ../.venv/bin/python make_status_plots.py                # everything
    ../.venv/bin/python make_status_plots.py --only deadtime_vs_charge

Every figure is built from a committed reduction, never from bulk data, so the
deck refreshes by re-running the upstream analysis and then this:

  charge/HV      ntof_july_analysis/flash_charge/results/flash_charge_subruns.csv
                 (regenerate: flash_charge/analyze.py --src <mirror>)
  recovery       flash_recovery/run57/metrics_run_57_perdet.csv  (from the DAQ)
  track rate     wft run_79 merged_prelim.parquet                (local mirror)
  flash waveform one decoded_root file from run_32                (local mirror)

Anything missing is skipped with a message rather than crashing the set.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'ntof_july_analysis', 'flash_charge'))

import plotstyle as P  # noqa: E402

OUT = os.path.join(HERE, 'slides', 'assets', 'img')

CHARGE_CSV = os.path.join(REPO, 'ntof_july_analysis', 'flash_charge',
                          'results', 'flash_charge_subruns.csv')

# Inputs that live outside the repo.  Override with --data.
DEFAULT_DATA = os.path.join(os.path.expanduser('~'), '.cache', 'mpgd26_status')
RECOVERY_CSV = 'metrics_run_57_perdet.csv'
TRACKS_PARQUET = ('/media/dylan/data/x17/beam_july/analysis/wft/run_79/'
                  'stat090_0000/mx17_A/merged_prelim.parquet')

# The production operating point (run_158 run_config.json).
OP_POINT = {'A': 540, 'B': 540, 'C': 525, 'D': 520}

# Thermal-neutron arrival: Geant4 timedist_2cm peaks at 5.3 ms; the
# reconstructed-track rate on run_79 peaks at 4-7 ms.
THERMAL_LO, THERMAL_HI = 3.0, 8.0

# DREAM CSA input ranges [fC], manual Table 1.
CSA_RANGES = (50.0, 100.0, 200.0, 600.0)
CHANNELS_PER_DET = 1024


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_charge(run: str) -> list[dict]:
    if not os.path.exists(CHARGE_CSV):
        return []
    out = []
    with open(CHARGE_CSV) as fh:
        for r in csv.DictReader(fh):
            if r['run'] != run:
                continue
            for k in ('resist_v', 'drift_v', 'q_per_pulse_nc', 'q_err_nc',
                      'di_ua', 'i_median_ua', 'pulse_rate_hz'):
                try:
                    r[k] = float(r[k])
                except (KeyError, ValueError):
                    r[k] = np.nan
            r['leak_ok'] = r.get('leak_ok') == 'True'
            out.append(r)
    return out


def load_recovery(data_dir: str, cls='all') -> dict[str, dict[str, float]]:
    """{det: {subrun: recovery_ms}}"""
    path = os.path.join(data_dir, RECOVERY_CSV)
    if not os.path.exists(path):
        return {}
    out: dict[str, dict[str, float]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get('cls', 'all') != cls:
                continue
            try:
                out.setdefault(r['det'], {})[r['subrun']] = float(r['recovery_ms'])
            except (KeyError, ValueError):
                continue
    return out


def _group_by_hv(rows: list[dict], det: str, min_sig=2.0):
    """resist setpoint -> pulse-weighted mean charge, sorted in V.

    Sub-runs whose charge is not significant (or is negative -- the estimator is
    unbiased, so a quiet sub-run can land below zero) are dropped rather than
    plotted with an error bar that runs off a log axis.
    """
    g: dict[int, list[dict]] = {}
    for r in rows:
        if r['det'] != det or not np.isfinite(r['q_per_pulse_nc']):
            continue
        if r['q_per_pulse_nc'] <= 0 or not np.isfinite(r['q_err_nc']):
            continue
        if r['q_err_nc'] > 0 and r['q_per_pulse_nc'] / r['q_err_nc'] < min_sig:
            continue
        g.setdefault(int(round(r['resist_v'])), []).append(r)
    v, q, e = [], [], []
    for key in sorted(g):
        rs = g[key]
        w = np.array([max(float(x['n_pulses']), 1) for x in rs])
        qq = np.array([x['q_per_pulse_nc'] for x in rs])
        ee = np.array([x['q_err_nc'] for x in rs])
        v.append(key)
        q.append(float(np.average(qq, weights=w)))
        e.append(float(np.sqrt(np.sum((w * ee) ** 2)) / np.sum(w)))
    return np.array(v, float), np.array(q), np.array(e)


# --------------------------------------------------------------------------- #
# 1. Recovery time vs resist HV
# --------------------------------------------------------------------------- #

def fig_recovery_vs_hv(data_dir: str):
    rows = load_charge('run_57')
    rec = load_recovery(data_dir)
    if not rows or not rec:
        print('  .. recovery_vs_hv: inputs missing, skipped')
        return
    sub_v = {(r['det'], r['subrun']): r['resist_v'] for r in rows}

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.axhspan(THERMAL_LO, THERMAL_HI, color=P.BAND_SIGNAL, alpha=0.10, zorder=0)
    ax.text(0.012, np.sqrt(THERMAL_LO * THERMAL_HI), 'thermal-neutron arrival\n(3–8 ms, peak 5.3 ms)',
            transform=ax.get_yaxis_transform(), color=P.BAND_SIGNAL,
            fontsize=10.5, fontweight='bold', va='center')

    for det in 'ABCD':
        d = rec.get(det, {})
        pts = sorted(((sub_v.get((det, s), np.nan), ms) for s, ms in d.items()),
                     key=lambda t: t[0])
        pts = [(v, m) for v, m in pts if np.isfinite(v)]
        if len(pts) < 3:
            continue
        v = np.array([p[0] for p in pts])
        m = np.array([max(p[1], 0.2) for p in pts])
        ax.plot(v, m, marker=P.DET_MARKER[det], color=P.DET_COLOR[det],
                label=f'det {det}', zorder=3, markeredgecolor=P.SURFACE,
                markeredgewidth=0.8)
        # the operating point actually used in production
        op = OP_POINT[det]
        k = np.argmin(np.abs(v - op))
        if abs(v[k] - op) <= 3:
            ax.plot(v[k], m[k], marker='*', ms=17, color=P.DET_COLOR[det],
                    markeredgecolor=P.INK, markeredgewidth=0.9, zorder=5)

    ax.set_yscale('log')
    ax.set_xlabel('resistive-layer HV  [V]')
    ax.set_ylabel('post-flash recovery time  [ms]')
    ax.set_ylim(0.3, 55)
    P.strip(ax)
    P.title(ax, 'Every 10 V of gain costs milliseconds of blindness',
            'run_57 · flash-random probe · Ar/iC₄H₁₀ 90/10 · ★ = production operating point')
    ax.legend(loc='lower right', ncol=2, columnspacing=1.4, handletextpad=0.5)
    P.note(fig, 'Recovery = time since flash at which per-channel baseline noise returns and stays back '
                '(dead ≈ 2 ADC, alive ≈ 50–130 ADC, threshold 15). Quantised to the analysis log-time bins, so '
                'points land on a discrete ladder; the top bin is “does not recover inside the ~30 ms gate”. '
                'Det D sits on its own HV grid, 10 V below A/B/C, and carries the standing noise caveat.')
    P.save(fig, os.path.join(OUT, 'status_recovery_vs_hv.png'))


# --------------------------------------------------------------------------- #
# 2. Charge per pulse vs resist HV
# --------------------------------------------------------------------------- #

def fig_charge_vs_hv(data_dir: str):
    rows = load_charge('run_57')
    if not rows:
        print('  .. charge_vs_hv: inputs missing, skipped')
        return
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for det in 'ABC':                       # D's run_57 curve is not usable
        v, q, e = _group_by_hv(rows, det)
        if v.size < 3:
            continue
        ax.errorbar(v, q, yerr=e, marker=P.DET_MARKER[det], color=P.DET_COLOR[det],
                    label=f'det {det}', capsize=0, elinewidth=1.2,
                    markeredgecolor=P.SURFACE, markeredgewidth=0.8, zorder=3)

    ax.set_yscale('log')
    ax.set_xlabel('resistive-layer HV  [V]')
    ax.set_ylabel('avalanche charge delivered per beam pulse  [nC]')
    ax.set_ylim(15, 1400)
    P.strip(ax)
    P.title(ax, 'The flash delivers 30–800 nC per chamber, per pulse',
            'run_57 · from the resistive-layer supply current, outside the saturated readout')
    ax.legend(loc='upper left', ncol=3, columnspacing=1.4, handletextpad=0.5)

    # Second scale for the same quantity -- NOT a second measure. It restates the
    # left axis per DREAM channel, in units of the front end's full-scale input.
    ax2 = ax.secondary_yaxis(
        'right',
        functions=(lambda q: q * 1e3 / CHANNELS_PER_DET * 1e3 / 600.0,
                   lambda m: m * 600.0 / 1e3 * CHANNELS_PER_DET / 1e3))
    ax2.set_ylabel('same charge, per channel:  × CSA full scale (600 fC)',
                   color=P.MUTED, labelpad=8)
    ax2.tick_params(colors=P.MUTED)

    P.note(fig, 'Q = (mean − median) of the resist-supply current ÷ the beam-pulse rate, per sub-run; the median is the '
                'standing leakage at that HV and the pulse rate comes from the beam-intensity log. Validated three ways: '
                'it returns zero on a beam-off run (run_159, 0.000 Hz), it gives the same charge per pulse at a 10× '
                'different beam rate (run_157 vs run_158), and its HV dependence is the gas-gain curve. Det D is '
                'excluded — its run_57 curve falls with HV and is not understood. '
                'The right-hand axis divides by the chamber’s 1 024 DREAM channels, so it is an average — the beam spot is worse.')
    P.save(fig, os.path.join(OUT, 'status_charge_vs_hv.png'))


# --------------------------------------------------------------------------- #
# 3. The join -- dead time vs charge
# --------------------------------------------------------------------------- #

def fig_deadtime_vs_charge(data_dir: str):
    rows = load_charge('run_57')
    rec = load_recovery(data_dir)
    if not rows or not rec:
        print('  .. deadtime_vs_charge: inputs missing, skipped')
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.axhspan(THERMAL_LO, THERMAL_HI, color=P.BAND_SIGNAL, alpha=0.10, zorder=0)
    ax.text(0.985, np.sqrt(THERMAL_LO * THERMAL_HI), 'thermal window ',
            transform=ax.get_yaxis_transform(), color=P.BAND_SIGNAL,
            fontsize=10.5, fontweight='bold', va='center', ha='right')

    allq, allm = [], []
    for det in 'ABC':
        q_by_sub = {r['subrun']: (r['q_per_pulse_nc'], r['resist_v'])
                    for r in rows if r['det'] == det}
        pts = [(q_by_sub[s][0], ms) for s, ms in rec.get(det, {}).items()
               if s in q_by_sub and np.isfinite(q_by_sub[s][0]) and q_by_sub[s][0] > 0]
        if len(pts) < 3:
            continue
        pts.sort()
        q = np.array([p[0] for p in pts])
        m = np.array([max(p[1], 0.2) for p in pts])
        allq.append(q)
        allm.append(m)
        ax.plot(q, m, marker=P.DET_MARKER[det], color=P.DET_COLOR[det], lw=0,
                label=f'det {det}', markeredgecolor=P.SURFACE, markeredgewidth=0.8,
                zorder=3, ms=7)

    if allq:
        q = np.concatenate(allq)
        m = np.concatenate(allm)
        ok = (q > 0) & (m > 0.25)          # drop the prompt floor, it is a limit
        c = np.polyfit(np.log10(q[ok]), np.log10(m[ok]), 1)
        xs = np.logspace(np.log10(q.min()), np.log10(q.max()), 50)
        ax.plot(xs, 10 ** np.polyval(c, np.log10(xs)), color=P.MUTED, lw=1.4,
                ls='--', zorder=2)
        ax.text(0.97, 0.06, f'common slope  t ∝ Q$^{{{c[0]:.2f}}}$',
                transform=ax.transAxes, ha='right', color=P.MUTED, fontsize=11.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('avalanche charge delivered per beam pulse  [nC]')
    ax.set_ylabel('post-flash recovery time  [ms]')
    P.strip(ax)
    P.title(ax, 'Dead time is set by charge, not by voltage',
            'run_57 · both axes measured on the SAME sub-runs · three chambers, one curve')
    ax.legend(loc='lower right', ncol=1, columnspacing=1.4, handletextpad=0.5,
              bbox_to_anchor=(1.0, 0.10))
    P.note(fig, 'Charge from the HV supply current, recovery from the flash-random probe — two independent '
                'measurements of the same 31 sub-runs, joined on the sub-run name. Three chambers at three '
                'different gains falling on one curve is the statement: what the front end cannot survive is the '
                'charge, and HV is only the knob that sets it. Recovery is quantised to log-time bins, which is '
                'the vertical stepping; the dashed line is an unweighted power-law fit, drawn to show the trend.')
    P.save(fig, os.path.join(OUT, 'status_deadtime_vs_charge.png'))


# --------------------------------------------------------------------------- #
# 4. Reconstructed-track rate vs time since flash
# --------------------------------------------------------------------------- #

def fig_track_rate(data_dir: str):
    path = TRACKS_PARQUET
    if not os.path.exists(path):
        print('  .. track_rate: parquet missing, skipped')
        return
    import pandas as pd
    d = pd.read_parquet(path, columns=['t_since_flash_ns', 'x_ok', 'y_ok',
                                       'wal_hit_A', 'x_quality_ok'])
    t = d['t_since_flash_ns'].to_numpy(dtype=float) / 1e6      # ms
    good = d['x_ok'].to_numpy(bool)
    tagged = good & d['wal_hit_A'].to_numpy(bool)
    t = np.where(np.isfinite(t), t, -1)

    bins = np.linspace(0, 80, 81)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    h_all, _ = np.histogram(t[good], bins=bins)
    h_tag, _ = np.histogram(t[tagged], bins=bins)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.axvspan(THERMAL_LO, THERMAL_HI, color=P.BAND_SIGNAL, alpha=0.12, zorder=0)
    ax.step(ctr, h_all, where='mid', color=P.DET_COLOR['A'], lw=2.0,
            label='reconstructed tracks, det A')
    ax.step(ctr, h_tag, where='mid', color=P.DET_COLOR['B'], lw=2.0,
            label='+ arm-A scintillator tag')

    first = float(np.min(t[good & (t > 0)]))
    ax.set_xlim(0, 80)
    ax.set_ylim(0, h_all.max() * 1.22)
    ax.axvspan(0, first, color=P.BAND_DEAD, alpha=0.13, zorder=1)
    ax.axvline(first, color=P.BAND_DEAD, lw=1.6, ls='--', zorder=4)
    ax.annotate(f'no events at all before {first:.2f} ms',
                xy=(first, h_all.max() * 1.02),
                xytext=(9.0, h_all.max() * 1.14),
                color=P.BAND_DEAD, fontsize=10.5, fontweight='bold', va='center',
                arrowprops=dict(arrowstyle='->', color=P.BAND_DEAD, lw=1.3))
    frac = float(np.sum(h_all[(ctr > THERMAL_LO) & (ctr < THERMAL_HI)])) / max(h_all.sum(), 1)
    ax.text(THERMAL_HI + 1.2, h_all.max() * 0.62,
            f'{frac * 100:.0f} % of everything we record\nlands in 3–8 ms',
            color=P.BAND_SIGNAL, fontsize=10.5, fontweight='bold', va='center')

    ax.set_xlabel('time since γ flash  [ms]')
    ax.set_ylabel('reconstructed tracks per 1 ms')
    P.strip(ax)
    P.title(ax, 'Everything we record sits where the front end is still coming back',
            'run_79 · waveform-first reconstruction, joined to the n_TOF stream · PRELIMINARY')
    ax.legend(loc='upper right')
    P.note(fig, 'Shaded purple = the thermal-neutron arrival window (Geant4 timedist_2cm, peak 5.3 ms); shaded red = '
                'before the earliest event in the run. The distribution turns on sharply at 1 ms — that is the DREAM '
                'gate, not a reconstruction effect — and its bulk sits in the first ~10 ms, which is exactly the '
                'interval the recovery map says the chambers are still blind or only partly alive in. Tagged and '
                'untagged keep the same shape, so the scintillator tag is an acceptance and not a time-dependent '
                'selection. PRELIMINARY: transferred bench calibration, no in-situ calibration '
                '(RUN79_PRELIM_2026-07-30.md).')
    P.save(fig, os.path.join(OUT, 'status_track_rate.png'))


# --------------------------------------------------------------------------- #
# 5. What the flash does to a channel
# --------------------------------------------------------------------------- #

def fig_flash_waveform(data_dir: str):
    import glob
    cands = glob.glob(os.path.join(data_dir, 'wf', '*flashOff*A500*.root'))
    if not cands:
        print('  .. flash_waveform: run_32 decoded_root missing, skipped')
        return
    import uproot
    f = uproot.open(cands[0])
    t = f['nt']
    n = min(t.num_entries, 40)
    amp = t['amplitude'].array(entry_stop=n, library='np')

    # sample-major flat [nsample * nchan]; run_32 is 400 samples x 512 ch, 20 ns.
    a0 = np.asarray(amp[0], dtype=float)
    nch = 512
    nsamp = a0.size // nch
    dt_ns = 20.0
    W = np.stack([np.asarray(x, dtype=float).reshape(nsamp, nch) for x in amp])
    tt = np.arange(nsamp) * dt_ns / 1e3        # us

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    ax = axes[0]
    ev = W[0]
    hot = np.argsort(-np.ptp(ev, axis=0))[:24]
    for c in hot:
        ax.plot(tt, ev[:, c], color=P.DET_COLOR['A'], lw=0.9, alpha=0.35)
    ax.axhline(4095, color=P.BAND_DEAD, lw=1.4, ls='--')
    ax.axhline(0, color=P.BAND_DEAD, lw=1.4, ls='--')
    ax.text(tt[-1], 4095, ' +rail 4095', color=P.BAND_DEAD, fontsize=10.5,
            va='bottom', ha='right', fontweight='bold')
    ax.text(tt[-1], 0, ' −rail 0', color=P.BAND_DEAD, fontsize=10.5,
            va='top', ha='right', fontweight='bold')
    ax.set_xlabel('time in the readout window  [µs]')
    ax.set_ylabel('ADC code')
    ax.set_ylim(-350, 4450)
    P.strip(ax)
    P.title(ax, 'Rail to rail, both ways', '24 hottest strips, one flash event')

    ax = axes[1]
    # Per-channel TEMPORAL noise: successive-sample difference removes coherent
    # drift, /sqrt(2) undoes the differencing, median over channels and events so
    # a handful of hot strips cannot carry it.
    dW = np.diff(W, axis=1) / np.sqrt(2.0)
    noise = np.median(np.abs(dW), axis=(0, 2)) * 1.4826
    ax.plot(tt[1:], noise, color=P.DET_COLOR['C'], lw=2.0)
    ax.set_xlabel('time in the readout window  [µs]')
    ax.set_ylabel('per-channel noise  [ADC]')
    P.strip(ax)
    P.title(ax, 'The baseline comes back long before the channel does',
            'per-channel sample-to-sample scatter, median over 40 events and 512 channels')
    ax.annotate('flat baseline here is the shaper AC-coupling,\n'
                'not a recovered front end',
                xy=(tt[-1] * 0.72, noise[int(len(noise) * 0.72)]),
                xytext=(tt[-1] * 0.30, max(noise) * 0.75),
                color=P.INK, fontsize=10.5,
                arrowprops=dict(arrowstyle='->', color=P.MUTED, lw=1.2))

    fig.subplots_adjust(wspace=0.28)
    P.note(fig, 'run_32, flash-compensation OFF, resist 500 V, 400 samples × 20 ns. The ADC returns to a flat '
                'baseline within ~3 µs, but the CSA underneath stays pinned against its rail while the input '
                'current exceeds its ~9–90 nA feedback limit — and a pinned CSA has no small-signal gain, so it '
                'shows neither tracks NOR noise. The absence of noise is the only tell at ADC level.', y=-0.04)
    P.save(fig, os.path.join(OUT, 'status_flash_waveform.png'))


# --------------------------------------------------------------------------- #
# 6. Where the charge sits against the front end -- a scale figure, not a chart
# --------------------------------------------------------------------------- #

def fig_charge_scale(data_dir: str):
    rows = load_charge('run_158')
    if not rows:
        print('  .. charge_scale: inputs missing, skipped')
        return
    clean = [r for r in rows if r['leak_ok'] and r['det'] in ('A', 'C')]
    if not clean:
        print('  .. charge_scale: no clean detectors, skipped')
        return
    per_det: dict[str, float] = {}
    for det in ('A', 'C'):
        q = [r['q_per_pulse_nc'] for r in clean if r['det'] == det]
        if q:
            per_det[det] = float(np.mean(q))

    # Dots on a log axis, not bars: a bar encodes length from zero, and a log
    # axis has no zero, so bar length would be meaningless here. Position does
    # the encoding and the connector is a leader line, not a magnitude.
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    labels, vals, cols = [], [], []
    for fs in CSA_RANGES:
        labels.append(f'CSA full scale, {fs:.0f} fC range')
        vals.append(fs / 1e3)                       # pC
        cols.append(P.MUTED)
    for det, q in per_det.items():
        labels.append(f'γ flash, det {det} — per channel')
        vals.append(q * 1e3 / CHANNELS_PER_DET)     # pC
        cols.append(P.DET_COLOR[det])

    y = np.arange(len(vals))[::-1]
    xmin = 0.02
    for yy, v, c in zip(y, vals, cols):
        ax.plot([xmin, v], [yy, yy], color=c, lw=1.2, alpha=0.35, zorder=2)
        ax.plot([v], [yy], marker='o', ms=11, color=c, zorder=3,
                markeredgecolor=P.SURFACE, markeredgewidth=1.2)
        ax.text(v * 1.35, yy, f'{v:,.3g} pC' if v < 1 else f'{v:,.0f} pC',
                va='center', fontsize=11.5, color=c, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xscale('log')
    ax.set_xlabel('charge presented to one DREAM input  [pC, log scale]')
    ax.set_xlim(xmin, 3e3)
    ax.set_ylim(-0.7, len(vals) - 0.3)
    ax.grid(axis='y', visible=False)
    P.strip(ax, left=False)
    P.title(ax, 'The front end is asked to swallow ~200–3 000 × its full scale',
            'production operating point (run_158), averaged over all 1 024 channels of a chamber')
    P.note(fig, 'The flash points divide the measured charge per pulse evenly over the chamber’s channels, so they '
                'are an average and the illuminated region is worse. Which CSA range we actually run is a loose '
                'end — the DREAM state1 register decode is unverified.', y=-0.10)
    P.save(fig, os.path.join(OUT, 'status_charge_scale.png'))


# --------------------------------------------------------------------------- #
# 7. The same detector, two readouts -- where the failure actually lives
# --------------------------------------------------------------------------- #

MM_FLASH_JSON = os.path.join(REPO, 'ntof_processing', 'mm_flash', 'results.json')


def fig_two_readouts(data_dir: str):
    """Recovery of the same chamber seen by a direct analog channel and by DREAM.

    The form is an interval plot on a log time axis, not a bar chart: what is
    being compared is *when* each readout is usable again, so position and
    extent along time are the encoding.
    """
    import json
    if not os.path.exists(MM_FLASH_JSON):
        print('  .. two_readouts: mm_flash/results.json missing, skipped')
        return
    mm = json.load(open(MM_FLASH_JSON))
    r = mm['runs'].get('224302', {})
    peak_ns = float(r.get('peak_time_ns', np.nan))
    rec_ns = float(r.get('recovery_to_4mV_ns', {}).get('p50', np.nan))
    zs_ns = float(r.get('zs_first_block_ns', np.nan))
    q_pc = float(r.get('charge_pC', {}).get('p50', np.nan))
    if not np.isfinite(rec_ns - peak_ns):
        print('  .. two_readouts: mm_flash numbers incomplete, skipped')
        return
    mm_rec_ms = (rec_ns - peak_ns) / 1e6
    mm_zs_ms = (zs_ns - peak_ns) / 1e6

    # DREAM at the production point, from the run_57 map (det A at 540 V).
    rec = load_recovery(data_dir)
    rows = load_charge('run_57')
    sub_v = {(x['det'], x['subrun']): x['resist_v'] for x in rows}
    dream_ms = np.nan
    for s, ms in rec.get('A', {}).items():
        if abs(sub_v.get(('A', s), -1) - OP_POINT['A']) <= 1:
            dream_ms = ms
    dream_hi = max([m for d in 'ABC' for m in rec.get(d, {}).values()] or [np.nan])

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 1e2)

    ax.axvspan(THERMAL_LO, THERMAL_HI, color=P.BAND_SIGNAL, alpha=0.10, zorder=0)
    ax.text(np.sqrt(THERMAL_LO * THERMAL_HI), 2.50, 'thermal neutrons\narrive here',
            color=P.BAND_SIGNAL, fontsize=10.5, fontweight='bold',
            ha='center', va='bottom')

    rows_spec = [
        (1.75, 'the chamber — digitised directly, 1 GS/s, no charge amplifier', mm_rec_ms,
         P.DET_COLOR['C'], f'usable again {mm_rec_ms * 1e3:.2f} µs after the flash peak'),
        (0.75, 'the readout — the same chamber through DREAM', dream_ms,
         P.DET_COLOR['B'], f'front-end noise back after {dream_ms:.0f} ms'),
    ]
    for y, label, t_end, col, ann in rows_spec:
        if not np.isfinite(t_end):
            continue
        ax.plot([1e-4, t_end], [y, y], color=P.BAND_DEAD, lw=15, alpha=0.30,
                solid_capstyle='butt', zorder=2)
        ax.plot([t_end, 1e2], [y, y], color=col, lw=15, alpha=0.80,
                solid_capstyle='butt', zorder=2)
        ax.plot([t_end], [y], marker='|', ms=30, color=P.INK, mew=2.2, zorder=4)
        ax.text(1.4e-4, y + 0.30, label, fontsize=11.5, fontweight='bold',
                color=P.INK, va='bottom')
        ax.text(1.4e-4, y - 0.30, ann, fontsize=10.5, color=col,
                fontweight='bold', va='top')

    if np.isfinite(mm_zs_ms):
        ax.plot([mm_zs_ms], [1.75], marker='v', ms=10, color=P.INK, zorder=5)
        ax.text(mm_zs_ms * 1.35, 1.75, f'  first hit the DAQ allows, {mm_zs_ms * 1e3:.0f} µs',
                fontsize=10, color=P.INK, ha='left', va='center')

    ax.set_yticks([])
    ax.set_ylim(0.15, 2.95)
    ax.set_xlabel('time since the γ flash  [ms, log scale]')
    ax.grid(axis='y', visible=False)
    P.strip(ax, left=False)
    P.title(ax, 'The chamber is fine. The front end is not.',
            'red = blind · colour = usable · same detector, two readout chains')
    P.note(fig, f'Left row: an MX17 micromegas digitised directly at 1 GS/s by the n_TOF DAQ, no charge-sensitive '
                f'preamplifier in the chain (run 224302; median flash charge {q_pc:.0f} pC into 50 Ω). Its signal is '
                f'back below the 4 mV threshold {mm_rec_ms * 1e3:.2f} µs after its own peak, and it delivers hits from '
                f'{mm_zs_ms * 1e3:.0f} µs after the peak — the first instant that DAQ permits one — at the highest rate of the whole '
                f'20 ms cycle. Right row: the same style of chamber through DREAM at the production operating point. '
                f'Caveats: the two are not the same run, gas or amplification voltage, and which of the four chambers '
                f'the n_TOF channel was on cannot be determined from the data — only the cabling record can say. '
                f'The conclusion does not depend on either: nothing in the gas, the field or the amplification stage '
                f'has a millisecond time constant.', y=-0.06)
    P.save(fig, os.path.join(OUT, 'status_two_readouts.png'))


FIGURES = {
    'recovery_vs_hv': fig_recovery_vs_hv,
    'two_readouts': fig_two_readouts,
    'charge_vs_hv': fig_charge_vs_hv,
    'deadtime_vs_charge': fig_deadtime_vs_charge,
    'track_rate': fig_track_rate,
    'flash_waveform': fig_flash_waveform,
    'charge_scale': fig_charge_scale,
}


def render(names=None, data_dir=DEFAULT_DATA, out_dir=None) -> list[str]:
    """Render figures into `out_dir` (default: the deck's asset directory).

    Importable so the analysis package's own report can build the same figures
    from the same code — `ntof_july_analysis/flash_charge/make_report.py` calls
    this rather than re-plotting, so the talk and the report cannot drift.
    """
    global OUT
    prev, OUT = OUT, (out_dir or OUT)
    try:
        P.use()
        made = []
        for n in (names or list(FIGURES)):
            if n not in FIGURES:
                print(f'  ?? unknown figure {n!r}')
                continue
            print(n)
            FIGURES[n](data_dir)
            made.append(n)
        return made
    finally:
        OUT = prev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='comma list of figure names')
    ap.add_argument('--data', default=DEFAULT_DATA,
                    help='mirror holding metrics_run_57_perdet.csv and wf/')
    ap.add_argument('--out', default=None,
                    help='output directory (default: the deck asset dir)')
    args = ap.parse_args()
    render(args.only.split(',') if args.only else None, args.data, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
