#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clock_match.py -- put beam-off DREAM cosmic triggers on the n_TOF clock.

WHY THIS IS DIFFERENT FROM THE BEAM-ON JOIN.  With beam, DREAM triggers come
in bursts that start on a PS pulse, so `bunch_join` locks bursts to n_TOF
bunches through the pulse stream and the fine map (DREAM_NTOF_CALIBRATION.md)
works in time-since-flash.  With no beam there are no bursts and no flash:
DREAM triggers on scintillator singles at ~25 Hz, continuously, and n_TOF free-
runs (period 0.5009 s, 80 ms window, ~16 % duty) with its own trigger.  The
only common object is the particle: a DREAM trigger that falls inside an n_TOF
window should have a wall x plastic SINGLES there at the same instant.

THE TWO CLOCKS, absolute:

  DREAM   t_abs = t_go + timestamp * 10 ns
          t_go is unknown; the DAQ log's "Subrun started" line (PC clock, ms)
          is the anchor, and t_go = log + L.  Beam-on sub-runs give L+0.83 s
          (the NXCALS lag) at 6-11 s (run_79/86/96, 2026-10-02).
  n_TOF   t_abs = PKUP.psTime(bunch) + tof
          psTime is filled per bunch even with no protons (float64 ns, so
          quantised to 256 ns at 1.8e18); tof is ns since the n_TOF trigger.
          The stored tflash is meaningless with no beam and is NOT used.

THE MAP, measured on run_149/cos_0000 <-> 224678 (2026-10-02):

  off = t_DREAM_on_ntof - psTime_b = delta_b + tof * (1 - kappa)
  t_DREAM_on_ntof = t_go_log + S + td * (1 + k)

  S        translation, log anchor -> n_TOF               +6.450 s
  k        DREAM oscillator vs psTime, between bunches    -3.8 ppm
  kappa    DREAM vs n_TOF digitizer, inside a bunch       116.6 ppm
           (beam-on run_79<->224572 had K = 110.4 ppm; per run pair)
  delta_b  per-bunch offset of the window start from psTime, MAD ~37 us --
           the beam-off analogue of the beam-on delta_a_b, ~1000x larger,
           and fitted from the bunch's own matched triggers

Stages, each opening a window only as wide as the previous one's error:
  1 coarse   S on a 60 s slice, 1 ms -> 10 us -> 100 ns histograms
  2 drift    S + k*td, robust line, windows 5 ms -> 200 us (jitter floor)
  3 kappa    from bunches holding two matched triggers (delta_b cancels)
  4 bunch    delta_b per bunch; every residual is LEAVE-ONE-OUT (delta_b
             from the bunch's other triggers), so nothing validates itself.
             A trigger alone in its bunch cannot be validated at ns level.
  control    stage 4 rerun with DREAM shifted +5 ms: the accidental rate.

    .venv/bin/python ntof_cosmics/clock_match.py --run run_149 \\
        --subrun cosbounce_cos_0000 --ntof 224678
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ntof_dream_merge import fast_singles as FS  # noqa: E402
from ntof_dream_merge.dream_trigger import (load_thresholds,  # noqa: E402
                                            load_adc_mv)

RUNS = Path('/media/dylan/data/x17/beam_july/runs')
NTOF = Path('/media/dylan/data/x17/beam_july/ntof_data')
DREAM_TS = Path('/media/dylan/data/x17/beam_july/dream_ts')
OUT = HERE / 'results' / 'clock_match'

TICK_NS = 10            # DREAM timestamp granularity
SEARCH_S = 60.0         # +- around the log-anchored guess
SPAN_PAD_S = 15.0       # straddling sub-runs: DREAM kept within this of the n_TOF run


# --------------------------------------------------------------------------- #
# DREAM side
# --------------------------------------------------------------------------- #
def log_start(run: str, subrun: str) -> float:
    """Unix seconds of the DAQ log's 'Subrun started' line (local PC clock)."""
    log = (RUNS / run / 'dream_daq.log').read_text()
    m = re.search(rf'^(\S+ \S+) INFO: Subrun started: {re.escape(subrun)}\b',
                  log, re.M)
    if not m:
        raise LookupError(f'{subrun} not in {run}/dream_daq.log')
    return dt.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f').timestamp()


def dream_triggers(run: str, subrun: str, feu: str = '01') -> dict:
    """Every trigger of the sub-run from one FEU's decoded tree.

    decoded_root, not combined_hits: combined_hits only holds events with
    Micromegas activity, and the clock match wants every trigger.
    """
    files = sorted(glob.glob(str(RUNS / run / subrun / 'decoded_root'
                                 / f'*_{feu}.root')))
    if not files:
        # timestamps-only extract made on lxplus (scratchpad dream_ts.py:
        # eventId + timestamp of every FEU-01 entry), so the 330 MB decoded
        # files need not be pulled just for the clock
        z = DREAM_TS / run / f'ts_{subrun}.npz'
        if not z.exists():
            raise FileNotFoundError(f'no decoded FEU {feu} or {z} for {run}/{subrun}')
        a = np.load(z)
        eid = a['eventId'].astype(np.int64)
        t_ns = a['timestamp'].astype(np.int64) * TICK_NS
        o = np.argsort(t_ns, kind='stable')
        return dict(eventId=eid[o], t_ns=t_ns[o])
    eid, ts = [], []
    for f in files:
        a = uproot.open(f)['nt'].arrays(['eventId', 'timestamp'], library='np')
        eid.append(a['eventId'])
        ts.append(a['timestamp'])
    eid = np.concatenate(eid).astype(np.int64)
    t_ns = np.concatenate(ts).astype(np.int64) * TICK_NS
    o = np.argsort(t_ns, kind='stable')
    return dict(eventId=eid[o], t_ns=t_ns[o])


# --------------------------------------------------------------------------- #
# n_TOF side
# --------------------------------------------------------------------------- #
def _parts(ntof_run: int) -> list[Path]:
    p = sorted((NTOF / f'run{ntof_run}.parts').glob(f'run{ntof_run}_*.root'))
    if not p:
        raise FileNotFoundError(f'no partials for n_TOF {ntof_run} under {NTOF}')
    return p


def ntof_bunches(ntof_run: int) -> dict:
    """{BunchNumber: psTime ns} from PKUP, which keeps one row per bunch."""
    bn, ps = [], []
    for f in _parts(ntof_run):
        a = uproot.open(f)['PKUP'].arrays(['BunchNumber', 'psTime'],
                                          library='np')
        bn.append(a['BunchNumber'])
        ps.append(a['psTime'])
    bn, ps = np.concatenate(bn).astype(np.int64), np.concatenate(ps)
    bn, i = np.unique(bn, return_index=True)
    return dict(bunch=bn, psTime_ns=ps[i])


def _raw_tof_reader(ntof_run: int, pss_shift: dict | None = None):
    """A drop-in for fast_singles.read_bunches that returns the raw tof.

    fast_singles asks for 't_since_flash_ns' = tof - tflash; with no beam the
    stored tflash is noise, so hand it tof itself.  Raw tof is NOT on a common
    zero across trees: with beam each tree's own tflash absorbs its cable
    delay, and without it the plastic sits ~35 ns after the wall -- outside
    the 20 ns coincidence.  `pss_shift[arm]` is subtracted from PSS<arm>.
    Whole trees are read once and cached (a quiet run is ~250 MB).
    """
    cache: dict[str, dict] = {}
    shift = pss_shift or {}

    def read(run, tree, bunches, branches):
        if tree not in cache:
            want = ['BunchNumber', 'amp', 'detn', 'tof']
            parts = [uproot.open(f)[tree].arrays(want, library='np')
                     for f in _parts(ntof_run)]
            cache[tree] = {k: np.concatenate([p[k] for p in parts]) for k in want}
        d = cache[tree]
        m = np.isin(d['BunchNumber'], bunches)
        out = {k: v[m] for k, v in d.items() if k in set(branches) | {'tof'}}
        out['t_since_flash_ns'] = out['tof'] - (
            shift.get(tree[3], 0.0) if tree.startswith('PSS') else 0.0)
        return out
    return read


#: Beam-on plastic - wall peak per arm (DREAM_NTOF_CALIBRATION.md sec 2b).
#: The in-situ shift puts the quiet-run peak back here, so the 20 ns AND sees
#: the same relative timing the hardware coincidence did.
PSS_MINUS_WALL_BEAM = {'A': -6.8, 'B': -3.8, 'C': -6.3, 'D': -8.8}


def measure_pss_shift(ntof_run: int, bunches, thr: dict, adc: dict) -> dict:
    """Per arm, how far raw PSS tof sits from where the beam-on timing had it.

    Peak of (plastic over threshold) - (wall sum over threshold) within +-2 us,
    then the median within +-30 ns of that peak.  Real coincidences (cosmics,
    room background) make a peak on an essentially empty background.
    """
    FS.read_bunches = _raw_tof_reader(ntof_run)
    out = {}
    for arm in 'ABCD':
        w = FS.singles_candidates(ntof_run, bunches, arm, thr, adc,
                                  require_plastic=False)
        p = FS.read_bunches(ntof_run, f'PSS{arm}', bunches,
                            ('BunchNumber', 'detn', 'amp'))
        pmv = p['amp'] * adc[f'PSS{arm}'][(p['detn'] - 1).astype(int)]
        s = np.isin(p['detn'], thr['pmts'][arm]) & (pmv > thr['plastic'][arm])
        kp = FS._pack(p['BunchNumber'][s], p['tof'][s])
        o = np.argsort(kp)
        kp, pt = kp[o], p['tof'][s][o]
        ka = FS._pack(w['bunch'], w['t'])
        lo, hi = np.searchsorted(kp, ka - 2000), np.searchsorted(kp, ka + 2000)
        d = np.concatenate([pt[a:b] - t for a, b, t in zip(lo, hi, w['t'])])
        h, e = np.histogram(d, bins=400, range=(-2000, 2000))
        pk = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
        core = d[np.abs(d - pk) < 30]
        out[arm] = float(np.median(core)) - PSS_MINUS_WALL_BEAM[arm]
    return out


def ntof_singles(ntof_run: int, run: str, subrun: str) -> tuple[dict, dict]:
    """The DREAM trigger rebuilt from n_TOF hits, per arm, on raw tof."""
    bunches = ntof_bunches(ntof_run)['bunch']
    thr, adc = load_thresholds(run, subrun), load_adc_mv()
    shift = measure_pss_shift(ntof_run, bunches, thr, adc)
    FS.read_bunches = _raw_tof_reader(ntof_run, shift)
    return FS.all_arms(ntof_run, bunches, thr, adc), shift


# --------------------------------------------------------------------------- #
# Stage 1: coarse translation
# --------------------------------------------------------------------------- #
def pair_diffs(t_d: np.ndarray, t_n: np.ndarray, half: float) -> np.ndarray:
    """All t_n - t_d with |.| < half, both sorted, as float64 ns."""
    lo = np.searchsorted(t_n, t_d - half)
    hi = np.searchsorted(t_n, t_d + half)
    n = hi - lo
    idx = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    return t_n[idx] - np.repeat(t_d, n)


def coarse(t_d_abs: np.ndarray, t_n_abs: np.ndarray, slice_s: float) -> dict:
    """Find S on the first `slice_s` of DREAM triggers, zooming in three steps."""
    d0 = t_d_abs[t_d_abs < t_d_abs[0] + slice_s * 1e9]
    out = {'n_dream_slice': int(d0.size), 'steps': []}
    centre, half = 0.0, SEARCH_S * 1e9
    for width in (1e6, 1e4, 100.0):          # 1 ms, 10 us, 100 ns bins
        diffs = pair_diffs(d0 + centre, t_n_abs, half) + centre
        nb = int(round(2 * half / width))
        h, e = np.histogram(diffs, bins=nb, range=(centre - half, centre + half))
        k = int(h.argmax())
        bg = float(np.median(h))
        peak = 0.5 * (e[k] + e[k + 1])
        out['steps'].append(dict(bin_ns=width, peak_ns=peak, peak=int(h[k]),
                                 median_bin=bg, n_diffs=int(diffs.size)))
        if width == 1e6:                      # keep the 1 ms scan for the figure
            z = slice(max(k - 20, 0), k + 21)
            out['hist_1ms'] = dict(
                full_100ms=h.reshape(-1, 100).sum(1).tolist(), full_lo_ns=float(e[0]),
                zoom=h[z].tolist(), zoom_lo_ns=float(e[z.start]))
        centre, half = peak, 20 * width
    out['S_ns'] = centre
    return out


# --------------------------------------------------------------------------- #
# Stage 2: translation + rate across the sub-run
# --------------------------------------------------------------------------- #
def _robust_line(x: np.ndarray, y: np.ndarray, it: int = 4) -> tuple:
    ok = np.ones(x.size, bool)
    for _ in range(it):
        c = np.polyfit(x[ok], y[ok], 1)
        r = y - np.polyval(c, x)
        sig = 1.4826 * np.median(np.abs(r[ok]))
        ok = np.abs(r) < 3 * sig
    return c, r, ok, sig


def drift(td: np.ndarray, t_n: np.ndarray, S: float) -> dict:
    """Carry S across the sub-run: windows shrink as the line improves."""
    a, k, steps = S, 0.0, []
    for half in (5e6, 1e6, 2e5):
        p = td + a + k * td
        lo, hi = np.searchsorted(t_n, p - half), np.searchsorted(t_n, p + half)
        n = hi - lo
        i_d = np.repeat(np.arange(td.size), n)
        i_n = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        r = t_n[i_n] - p[i_d]
        c, res, ok, sig = _robust_line(td[i_d], r)
        a, k = a + c[1], k + c[0]
        steps.append(dict(half_ns=half, pairs=int(r.size), S_ns=a, k=k,
                          core_sigma_ns=float(sig), n_core=int(ok.sum())))
    return dict(S_ns=a, k=k, steps=steps)


# --------------------------------------------------------------------------- #
# Stages 3-4: inside the bunch
# --------------------------------------------------------------------------- #
def bunch_pairs(td, psb, cand, S, k, half=3e5, shift=0.0):
    """Every (DREAM trigger, candidate) pair within +-half of the drift map.

    Returns a frame with off = DREAM time since the bunch's psTime on the
    n_TOF clock, and the candidate's tof.  The nearest candidate per trigger
    (in off - tof) is kept; `shift` moves DREAM for the accidental control.
    """
    import pandas as pd
    T = td + S + k * td + shift
    tabs = psb[cand['bi']] + cand['t']
    o = np.argsort(tabs)
    tabs = tabs[o]
    lo, hi = np.searchsorted(tabs, T - half), np.searchsorted(tabs, T + half)
    n = hi - lo
    i_d = np.repeat(np.arange(td.size), n)
    i_n = o[np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))]
    df = pd.DataFrame(dict(d=i_d, c=i_n, bi=cand['bi'][i_n], arm=cand['arm'][i_n],
                           tof=cand['t'][i_n], off=T[i_d] - psb[cand['bi'][i_n]]))
    df['u'] = df.off - df.tof
    df = df.loc[(df.u - df.u.median()).abs().groupby(df.d).idxmin()]
    return df.reset_index(drop=True)


def fit_kappa(df) -> dict:
    """In-bunch rate from bunches with exactly two matched triggers."""
    two = df[df.groupby('bi').bi.transform('size') == 2].sort_values(['bi', 'tof'])
    g = two.groupby('bi')
    f1, f2 = g.nth(0).set_index('bi'), g.nth(1).set_index('bi')
    du, dtof = (f2.u - f1.u).to_numpy(), (f2.tof - f1.tof).to_numpy()
    c, r, ok, sig = _robust_line(dtof, du)
    return dict(kappa=float(-c[0]), sigma_pair_ns=float(sig),
                n_bunches=int(ok.sum()), n_two=int(du.size),
                mad_by_sep={f'{lo:g}-{hi:g}ms': float(np.median(np.abs(
                    r[ok & (np.abs(dtof) >= lo * 1e6) & (np.abs(dtof) < hi * 1e6)])))
                    for lo, hi in ((0, 10), (10, 30), (30, 80))})


def per_bunch(df, kappa: float):
    """delta_b and the leave-one-out residual of every trigger."""
    df = df.copy()
    df['v'] = df.off - df.tof * (1 - kappa)          # = delta_b if real
    g = df.groupby('bi').v
    df['n_b'] = g.transform('size')
    # leave-one-out MEDIAN: for each row, the median of the bunch's others
    loo = np.full(len(df), np.nan)
    for _, idx in df.groupby('bi').groups.items():
        if len(idx) < 2:
            continue
        v = df.loc[idx, 'v'].to_numpy()
        for j, i in enumerate(idx):
            loo[df.index.get_loc(i)] = np.median(np.delete(v, j))
    df['res'] = df.v - loo
    df['delta_b'] = g.transform('median')
    return df


SMOOTH_S = 30.0        # running-median span for the slow part of delta_b
WINDOW_NS = 8.0e7      # n_TOF acquisition length (max tof seen: 79.999 ms)
EDGE_NS = 1.0e5        # stay clear of the window edges by 100 us


def efficiency(td, psb, df, drift_fit, kappa, W) -> tuple[float, int]:
    """Fraction of in-window DREAM triggers with a validated match.

    Denominator: every trigger whose time, on the drift map, falls inside a
    bunch window whose delta_b is known from triggers OTHER than itself.
    """
    T = td + drift_fit['S_ns'] + drift_fit['k'] * td
    bi = np.searchsorted(psb, T, side='right') - 1
    ok = bi >= 0
    stats = df.groupby('bi').v.agg(['sum', 'size', 'median'])
    matched = dict(zip(df.d, df.bi))
    num = den = 0
    res_by_d = dict(zip(df.d, df.res))
    for d in np.flatnonzero(ok):
        b = bi[d]
        if b not in stats.index:
            continue
        own = matched.get(d) == b
        if own and stats.at[b, 'size'] < 2:
            continue                          # only itself: no delta_b to test
        delta = stats.at[b, 'median']         # LOO handled through res below
        off = (T[d] - psb[b] - delta) / (1 - kappa)
        if not (EDGE_NS < off < WINDOW_NS - EDGE_NS):
            continue
        den += 1
        if own and abs(res_by_d.get(d, np.inf)) < W:
            num += 1
    return num / max(den, 1), den


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--run', default='run_149')
    ap.add_argument('--subrun', default='cosbounce_cos_0000')
    ap.add_argument('--ntof', type=int, default=224678)
    ap.add_argument('--slice-s', type=float, default=60.0)
    ap.add_argument('--window-ns', type=float, default=50.0,
                    help='final accept on the leave-one-out residual')
    a = ap.parse_args()

    t0 = log_start(a.run, a.subrun)
    REF = np.int64(round(t0 * 1e9))           # everything below is ns from REF
    dr = dream_triggers(a.run, a.subrun)
    td = dr['t_ns'].astype(np.float64)

    nb = ntof_bunches(a.ntof)
    sg, pss_shift = ntof_singles(a.ntof, a.run, a.subrun)
    # 7 % of quiet-run bunches carry an unfilled psTime (0); dropped for now
    good = nb['psTime_ns'] > 1e18
    bunches = nb['bunch'][good]
    psb = (nb['psTime_ns'][good] - REF).astype(np.float64)
    keep = np.isin(sg['bunch'], bunches)
    cand = dict(bi=np.searchsorted(bunches, sg['bunch'][keep]),
                t=sg['t'][keep], arm=sg['arm'][keep])
    t_n = np.sort(psb[cand['bi']] + cand['t'])

    print(f'DREAM {a.run}/{a.subrun}: {td.size} triggers over {td[-1] / 1e9:.1f} s; '
          f'log start {dt.datetime.fromtimestamp(t0)}')
    print(f'n_TOF {a.ntof}: {good.sum()} of {good.size} bunches with psTime, '
          f'{cand["t"].size} singles; PSS shift '
          f'{ {k: round(v, 1) for k, v in pss_shift.items()} } ns')

    # a sub-run can straddle two n_TOF runs (cos_0001: 224678 then 224679):
    # keep only the DREAM triggers that can fall inside THIS run's bunches
    # (S, the log -> n_TOF offset, is 6-11 s), so the coarse slice is not
    # taken from the part the other n_TOF run recorded
    span = (td > psb.min() - SPAN_PAD_S * 1e9) & (td < psb.max() + SPAN_PAD_S * 1e9)
    if not span.all():
        print(f'           {span.sum()} of {td.size} DREAM triggers inside this '
              f'n_TOF run\'s span (+-{SPAN_PAD_S:g} s)')
        td = td[span]
        dr = {k: v[span] for k, v in dr.items()}
    c1 = coarse(td, t_n, a.slice_s)
    print(f"1 coarse   S = {c1['S_ns'] / 1e9:+.7f} s  (1 ms peak "
          f"{c1['steps'][0]['peak']} over median {c1['steps'][0]['median_bin']:.0f})")
    c2 = drift(td, t_n, c1['S_ns'])
    for st in c2['steps']:
        print(f"2 drift    +-{st['half_ns'] / 1e3:5.0f} us: S = {st['S_ns'] / 1e9:+.7f} s, "
              f"k = {st['k'] * 1e6:+.3f} ppm, core sigma {st['core_sigma_ns'] / 1e3:.1f} us, "
              f"n {st['n_core']}")
    df = bunch_pairs(td, psb, cand, c2['S_ns'], c2['k'])
    c3 = fit_kappa(df)
    print(f"3 kappa    {c3['kappa'] * 1e6:.1f} ppm from {c3['n_bunches']} two-trigger "
          f"bunches; pair-difference sigma {c3['sigma_pair_ns']:.1f} ns; MAD by "
          f"separation {c3['mad_by_sep']}")

    df = per_bunch(df, c3['kappa'])
    W = a.window_ns
    val = df[np.isfinite(df.res)]
    acc = per_bunch(bunch_pairs(td, psb, cand, c2['S_ns'], c2['k'], shift=5e6),
                    c3['kappa'])
    acc = acc[np.isfinite(acc.res)]
    # delta_b continuity: is the jitter random bunch to bunch?
    db = df.groupby('bi').delta_b.first()
    cons = db[db.index.to_series().diff() == 1]
    jit = float(np.median(np.abs(db - db.median())))
    # the slow part (oscillator curvature the straight drift line misses) is a
    # running median over SMOOTH_S of bunch time; the fast part is what is left
    tb = psb[db.index.to_numpy()]
    slow = np.array([np.median(db.to_numpy()[np.abs(tb - t) < SMOOTH_S * 5e8])
                     for t in tb])
    fast = float(np.median(np.abs(db.to_numpy() - slow)))
    slow_range = float(slow.max() - slow.min())
    step = float(np.median(np.abs(np.diff(db.to_numpy())[np.diff(db.index) == 1])))

    # efficiency, with an honest denominator: EVERY DREAM trigger that falls
    # inside a window whose delta_b is known from OTHER triggers -- not just
    # the ones that already found a candidate within 300 us
    eff, n_den = efficiency(td, psb, df, c2, c3['kappa'], W)
    summary = dict(
        run=a.run, subrun=a.subrun, ntof=a.ntof, log_start=t0,
        n_dream=int(td.size), n_bunches=int(good.sum()),
        n_bunches_no_pstime=int((~good).sum()), n_singles=int(cand['t'].size),
        pss_shift_ns=pss_shift, coarse=c1, drift=c2, kappa=c3,
        delta_b_mad_ns=jit, delta_b_step_mad_ns=step,
        delta_b_fast_mad_ns=fast, delta_b_slow_range_ns=slow_range,
        smooth_s=SMOOTH_S,
        n_pairs_300us=int(len(df)), n_loo=int(len(val)),
        window_ns=W,
        matched_loo=int((np.abs(val.res) < W).sum()),
        efficiency_loo=eff, efficiency_denominator=n_den,
        accidental_loo=int((np.abs(acc.res) < W).sum()),
        n_control=int(len(acc)),
        # sideband: the nearest-candidate residual 1-300 us from the peak is
        # accidental by construction; scale its density to the +-W window
        sideband=int(val.res.abs().between(1e3, 3e5).sum()),
        accidental_expected=float(val.res.abs().between(1e3, 3e5).sum()
                                  * (2 * W) / (2 * (3e5 - 1e3))),
        res_core_mad_ns=float(np.median(np.abs(val.res[np.abs(val.res) < 200]))),
        singletons=int((df.n_b == 1).sum()),
    )
    print(f"4 bunch    delta_b MAD {jit / 1e3:.1f} us = slow part spanning "
          f"{slow_range / 1e3:.0f} us + fast scatter MAD {fast / 1e3:.1f} us "
          f"(step MAD {step / 1e3:.1f} us); {len(df)} triggers within 300 us, {len(val)} "
          f"with a leave-one-out partner, {summary['singletons']} alone in their bunch")
    print(f"           |res| < {W:g} ns: {summary['matched_loo']} matched, efficiency "
          f"{eff * 100:.1f} % of {n_den} in-window triggers; core MAD "
          f"{summary['res_core_mad_ns']:.1f} ns")
    print(f"  control  sideband 1-300 us: {summary['sideband']} triggers -> "
          f"{summary['accidental_expected']:.2f} expected accidentals in +-{W:g} ns "
          f"({summary['accidental_expected'] / max(summary['matched_loo'], 1) * 100:.3f} % "
          f"of matches); DREAM +5 ms: {summary['accidental_loo']} of "
          f"{summary['n_control']} testable pass")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f'{a.run}_{a.subrun}_{a.ntof}'
    (OUT / f'summary_{stem}.json').write_text(json.dumps(summary, indent=1,
                                                         default=float))
    import pandas as pd
    # stage-2 raw view: every pair within +-5 ms of S, before the rate is fitted
    lo = np.searchsorted(t_n, td + c1['S_ns'] - 5e6)
    hi = np.searchsorted(t_n, td + c1['S_ns'] + 5e6)
    n = hi - lo
    i_d = np.repeat(np.arange(td.size), n)
    i_n = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    pd.DataFrame(dict(t_dream_s=td[i_d] / 1e9,
                      r_us=(t_n[i_n] - td[i_d] - c1['S_ns']) / 1e3)).to_csv(
        OUT / f'drift_pairs_{stem}.csv', index=False)
    acc[['res']].to_csv(OUT / f'control_{stem}.csv', index=False)
    pd.DataFrame(dict(bunch_t_s=psb / 1e9, bi=np.arange(psb.size))).merge(
        df.groupby('bi').agg(delta_b_ns=('delta_b', 'first'), n=('v', 'size')
                             ).reset_index(), on='bi').to_csv(
        OUT / f'delta_b_{stem}.csv', index=False)
    df = df.assign(eventId=dr['eventId'][df.d], t_dream_ns=td[df.d],
                   bunch=bunches[df.bi])
    df.drop(columns=['d', 'c', 'bi']).to_csv(OUT / f'pairs_{stem}.csv', index=False)
    print(f'wrote {OUT}/summary_{stem}.json, pairs_{stem}.csv')


if __name__ == '__main__':
    main()
