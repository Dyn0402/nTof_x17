#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
charge_lib.py -- avalanche charge per beam pulse from the resistive-layer HV
supply current.

The resistive-layer supply carries the avalanche ion current, so its average
current IS the charge delivered to the amplification stage of a chamber,
integrated over everything one beam pulse does to it (gamma flash + the whole
neutron tail).  That makes it the one handle on "how much charge is DREAM being
asked to swallow" that does not go through the saturated front end.

    Q_per_pulse = (I_beam - I_leak) / f_pulse

Two things make this measurable per sub-run with no dedicated beam-off run:

1. **The baseline is in the data.**  The CAEN readback is sampled at ~1 Hz while
   beam pulses arrive every ~3.3 s, so most samples sit at the standing leakage
   current.  The per-sub-run MEDIAN is therefore the leakage AT THAT HV, and
   mean - median is the beam-induced part.  Verified on 2026-08-09 against real
   beam-off runs at the identical setpoint (run_157/159 vs run_158):

       det A   median(beam-on) 0.018 uA   vs  median(beam-off) 0.016 uA
       det C   median(beam-on) 0.009 uA   vs  median(beam-off) 0.008 uA

   and the resulting dI agrees to 5 % (A: 0.041 vs 0.043 uA).  Self-calibrating
   per HV point, which is what makes an HV scan usable.

2. **The pulse rate is logged**, in slow_control/beam_intensity/, so each
   sub-run gets its own rate over its own time window rather than a run average.

CAVEAT that bounds everything downstream: this assumes the supply's imon
readback preserves the time-average of a current burst much shorter than the
sample spacing.  27.8 % of run_158's samples sit above baseline on BOTH clean
detectors, which is consistent with the monitor smoothing each burst over ~1 s
(so the mean is right).  If instead it reports a short instantaneous average,
every Q here is a LOWER BOUND.  See HANDOFF_FLASH_CHARGE_2026-08-09.md sec 4.
"""
from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Resistive-layer (amplification) HV: card 5, one channel per detector.
RESIST_CH = {'A': '5:1', 'B': '5:2', 'C': '5:3', 'D': '5:4'}
# Drift cathode: card 9.
DRIFT_CH = {'A': '9:0', 'B': '9:1', 'C': '9:2', 'D': '9:3'}

# A beam pulse, for rate counting.  July/August n_TOF pulses are bimodal at
# ~410e10 and ~850e10 protons; anything above this is a real pulse and anything
# below is the between-pulse floor.  Same constant as
# track_rate_hv_time_intensity/intensity_split.py uses to split the bands.
PULSE_E10_MIN = 100.0

# Channels per detector across its two FEUs (512 each), i.e. the number of DREAM
# inputs the charge is shared over.  X and Y strips share the same avalanche.
CHANNELS_PER_DET = 1024

# DREAM CSA input range settings [fC] (manual Table 1).
CSA_FULL_SCALE_FC = (50.0, 100.0, 200.0, 600.0)

# Standing leakage above which a detector's dI is not trustworthy.  Det B sits
# at ~2.1 uA and drifts by ~0.7 uA run to run, which swamps a ~0.04 uA signal.
MAX_LEAK_UA = 0.30


@dataclass
class SubrunCharge:
    run: str
    subrun: str
    det: str
    resist_v: float
    drift_v: float
    n_samples: int
    t_start: float
    t_end: float
    i_median_ua: float          # leakage at this HV
    i_mean_ua: float
    di_ua: float                # beam-induced, = mean - median
    di_err_ua: float            # bootstrap 1 sigma on di
    frac_elevated: float        # fraction of samples > median + 0.02 uA
    n_pulses: int
    pulse_rate_hz: float
    q_per_pulse_nc: float
    q_err_nc: float
    q_per_channel_pc: float
    leak_ok: bool
    notes: str = ''


# --------------------------------------------------------------------------- #
# Beam-pulse rate
# --------------------------------------------------------------------------- #

class PulseLog:
    """Beam pulses from slow_control/beam_intensity/beam_intensity_<date>.csv."""

    def __init__(self, intensity_dir: str):
        self.dir = intensity_dir
        self._cache: dict[str, np.ndarray] = {}

    def _day(self, date: str) -> np.ndarray:
        if date not in self._cache:
            path = os.path.join(self.dir, f'beam_intensity_{date}.csv')
            ts, val = [], []
            if os.path.exists(path):
                with open(path) as fh:
                    for row in csv.DictReader(fh):
                        try:
                            ts.append(float(row['unix_ts']))
                            val.append(float(row['intensity_e10']))
                        except (KeyError, ValueError):
                            continue
            self._cache[date] = np.array([ts, val]) if ts else np.zeros((2, 0))
        return self._cache[date]

    def rate(self, t_start: float, t_end: float, dates: list[str]) -> tuple[int, float]:
        """Pulses and mean rate [Hz] in [t_start, t_end]."""
        n = 0
        for d in dates:
            a = self._day(d)
            if a.shape[1] == 0:
                continue
            m = (a[0] >= t_start) & (a[0] <= t_end) & (a[1] > PULSE_E10_MIN)
            n += int(m.sum())
        span = max(t_end - t_start, 1.0)
        return n, n / span


# --------------------------------------------------------------------------- #
# Per-sub-run reduction
# --------------------------------------------------------------------------- #

def _parse_ts(s: str) -> float:
    """'2026-07-19 17:35:59' -> unix seconds (local time, as the DAQ writes it)."""
    import datetime as _dt
    return _dt.datetime.strptime(s.strip()[:19], '%Y-%m-%d %H:%M:%S').timestamp()


def read_hv_monitor(path: str) -> dict[str, np.ndarray]:
    """Column name -> float array, plus 't' (unix seconds)."""
    cols: dict[str, list] = {}
    times: list[float] = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                times.append(_parse_ts(row['timestamp']))
            except (KeyError, ValueError):
                continue
            for k, v in row.items():
                if k == 'timestamp' or v is None:
                    continue
                try:
                    cols.setdefault(k, []).append(float(v))
                except ValueError:
                    cols.setdefault(k, []).append(np.nan)
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    out['t'] = np.asarray(times, dtype=float)
    return out


def _bootstrap_di(i: np.ndarray, n_boot=400, seed=17) -> float:
    """1 sigma on (mean - median), resampling the sub-run's own samples.

    Honest under a leaky channel: readback noise on a 2 uA baseline inflates
    this term without biasing the estimator, which is exactly what we want it
    to say.
    """
    rng = np.random.default_rng(seed)
    n = i.size
    idx = rng.integers(0, n, size=(n_boot, n))
    s = i[idx]
    return float(np.std(np.mean(s, axis=1) - np.median(s, axis=1), ddof=1))


def subrun_charges(run: str, subrun_dir: str, pulses: PulseLog,
                   dets=('A', 'B', 'C', 'D')) -> list[SubrunCharge]:
    path = os.path.join(subrun_dir, 'hv_monitor.csv')
    if not os.path.exists(path):
        return []
    d = read_hv_monitor(path)
    t = d['t']
    if t.size < 20:
        return []
    t0, t1 = float(t.min()), float(t.max())
    import datetime as _dt
    dates = sorted({_dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d') for x in (t0, t1)})
    n_pulse, rate = pulses.rate(t0, t1, dates)

    out = []
    for det in dets:
        ich = RESIST_CH[det] + ' imon'
        vch = RESIST_CH[det] + ' vmon'
        dch = DRIFT_CH[det] + ' vmon'
        if ich not in d or vch not in d:
            continue
        i = d[ich][np.isfinite(d[ich])]
        v = d[vch][np.isfinite(d[vch])]
        if i.size < 20 or v.size == 0:
            continue
        # Guard against a sub-run that contains an HV ramp: keep only samples at
        # the modal setpoint, else the "leakage" median mixes two voltages.
        v_set = float(np.median(v))
        keep = np.abs(d[vch] - v_set) < 1.0
        i = d[ich][keep & np.isfinite(d[ich])]
        if i.size < 20:
            continue
        med = float(np.median(i))
        mean = float(np.mean(i))
        di = mean - med
        di_err = _bootstrap_di(i)
        frac = float(np.mean(i > med + 0.02))
        q_nc = di * 1e-6 / rate * 1e9 if rate > 0 else np.nan
        q_err = di_err * 1e-6 / rate * 1e9 if rate > 0 else np.nan
        out.append(SubrunCharge(
            run=run, subrun=os.path.basename(subrun_dir.rstrip('/')), det=det,
            resist_v=v_set,
            drift_v=float(np.median(d[dch][np.isfinite(d[dch])])) if dch in d else np.nan,
            n_samples=int(i.size), t_start=t0, t_end=t1,
            i_median_ua=med, i_mean_ua=mean, di_ua=di, di_err_ua=di_err,
            frac_elevated=frac,
            n_pulses=n_pulse, pulse_rate_hz=rate,
            q_per_pulse_nc=q_nc, q_err_nc=q_err,
            q_per_channel_pc=q_nc * 1e3 / CHANNELS_PER_DET if np.isfinite(q_nc) else np.nan,
            leak_ok=med < MAX_LEAK_UA,
        ))
    return out


def run_charges(run_dir: str, pulses: PulseLog, **kw) -> list[SubrunCharge]:
    run = os.path.basename(run_dir.rstrip('/'))
    rows = []
    for sd in sorted(glob.glob(os.path.join(run_dir, '*/'))):
        rows.extend(subrun_charges(run, sd, pulses, **kw))
    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def by_hv(rows: list[SubrunCharge], det: str, clean_only=True):
    """Sub-runs of one detector grouped by resist setpoint -> arrays sorted in V.

    Sub-runs at the same setpoint are averaged weighted by their pulse count,
    which is the right weight for a charge-per-pulse average.
    """
    sel = [r for r in rows if r.det == det and np.isfinite(r.q_per_pulse_nc)]
    if clean_only:
        sel = [r for r in sel if r.leak_ok]
    groups: dict[float, list[SubrunCharge]] = {}
    for r in sel:
        groups.setdefault(round(r.resist_v), []).append(r)
    v, q, dq, n = [], [], [], []
    for key in sorted(groups):
        g = groups[key]
        w = np.array([max(x.n_pulses, 1) for x in g], dtype=float)
        qq = np.array([x.q_per_pulse_nc for x in g])
        v.append(float(key))
        q.append(float(np.average(qq, weights=w)))
        # bootstrap error, combined in quadrature over repeats; where there ARE
        # repeats take the larger of that and their observed spread, so a
        # sub-run-to-sub-run beam variation is not hidden by a tight bootstrap.
        e = np.array([x.q_err_nc for x in g])
        boot = float(np.sqrt(np.sum((w * e) ** 2)) / np.sum(w))
        if len(g) > 1:
            boot = max(boot, float(np.std(qq, ddof=1) / np.sqrt(len(g))))
        dq.append(boot)
        n.append(int(sum(x.n_pulses for x in g)))
    return np.array(v), np.array(q), np.array(dq), np.array(n)


def csa_full_scale_multiple(q_per_channel_pc: float, fs_fc: float) -> float:
    """How many times the CSA's full-scale input charge one channel is handed."""
    return q_per_channel_pc * 1e3 / fs_fc


def write_csv(rows: list[SubrunCharge], path: str) -> None:
    import dataclasses
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fields = [f.name for f in dataclasses.fields(SubrunCharge)]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dataclasses.asdict(r))


# --------------------------------------------------------------------------- #
# The recovery-time map, for joining charge to dead time
# --------------------------------------------------------------------------- #

def read_recovery_csv(path: str, cls='all') -> dict[str, dict[float, float]]:
    """metrics_run_57_perdet.csv -> {det: {resist_v: recovery_ms}}."""
    out: dict[str, dict[float, float]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get('cls', 'all') != cls:
                continue
            det = row['det']
            try:
                v = float(row['resist'])
                r = float(row['recovery_ms'])
            except (KeyError, ValueError):
                continue
            out.setdefault(det, {})[round(v)] = r
    return out
