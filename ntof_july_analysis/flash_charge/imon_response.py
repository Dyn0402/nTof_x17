#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imon_response.py -- measure the CAEN imon readback's impulse response directly,
and thereby close (or price) the one systematic that bounds every charge number
in HANDOFF_FLASH_CHARGE_2026-08-09.md sec 4.

THE QUESTION
------------
`Q_per_pulse = (mean(imon) - median(imon)) / f_pulse` is only the charge if the
readback preserves the TIME-AVERAGE of a current burst far shorter than the
sample spacing.  The supply is read at ~1 Hz; the beam pulse delivers its charge
in milliseconds.  If the readback instead reports something close to an
instantaneous current (or a short boxcar) it mostly misses the bursts and every
Q is a LOWER BOUND by an unmeasured factor.

THE MEASUREMENT
---------------
We have timestamps for the imon samples and for the individual beam pulses, so
we can phase-fold imon against time-since-beam-pulse and read the monitor's
response to a known impulse of charge straight off.

  * a smoothing filter  -> a rise and a ~1 s decay, peak amplitude ~ Q / width
  * missed bursts       -> a few samples at ~ Q / t_burst (tens of uA), the rest
                           at baseline, and an elevated fraction of ~1 %

THE TRAP, AND WHY IT IS REAL HERE
---------------------------------
Any clock offset or timestamp-granularity mismatch smears a narrow spike into a
wide one -- i.e. it FABRICATES the reassuring answer.  In this data set the trap
is not hypothetical:

  * `hv_monitor.csv` timestamps are whole seconds ('%Y-%m-%d %H:%M:%S').
  * The monitor loop period is ~1.0120 s, not 1.000 s: it slips a whole labelled
    second every ~83 samples.  So the sub-second phase of the true CAEN read
    inside its labelled second DRIFTS uniformly over [0, 1).

Folding on the raw labels therefore convolves the true response with a 1 s wide
box (sigma = 289 ms) -- enough on its own to turn a delta into a ~1 s feature.

Two independent defences, both implemented here:

  1. TIMESTAMP-FREE TESTS (sec 3).  The elevated-sample fraction, the run-length
     distribution of consecutive elevated samples, and above all the largest
     single-sample excursion use no timestamps at all.  A sample that contains a
     whole ms-scale burst and reports an excess dI_max can only be averaging over
     w >= Q / dI_max.  That is a hard lower bound on the averaging window.
  2. TIME-BASE RECONSTRUCTION (sec 2).  The drift is the cure as well as the
     disease: `label[k] <= t0 + k*P < label[k]+1` for every sample is 2N linear
     inequalities in two unknowns, so the wrap pattern pins (t0, P).  Solved as
     an LP-feasibility box on greedy-maximal segments, this recovers the true
     sample times to a few tens of ms, and the fold sharpens.

Both are reported.  If they disagree, believe (1).

RUN USED
--------
run_79 sub-runs stat090_0000/0001 (2026-07-26, resist A540/B540/C525/D520,
drift 700 V) -- the SAME setpoint as run_158, and the only production-point
hv_monitor.csv in the local July mirror.  Its det C charge (90-98 nC/pulse)
reproduces run_158's det C (97-101 nC/pulse) to 5 %, so it is the same
measurement.  In July det A carried ~2 uA of leakage (it was clean by August),
so the clean chamber here is C; A and D are leaky cross-checks only.

    .venv/bin/python ntof_july_analysis/flash_charge/imon_response.py \
        --src /media/dylan/data/x17/beam_july
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import charge_lib as C  # noqa: E402

# The clean chamber at the production point in July.  A/D are leaky, B very.
DEFAULT_DETS = ('C', 'D', 'A')
# A drift-cathode channel: carries the field-cage divider current only, so it
# must show NO beam-correlated excursion.  Same crate, same host, same 1 Hz
# logger -- the in-situ control for "is this excursion just crate pickup?".
NULL_CH = '9:2 imon'


# ---------------------------------------------------------------------------- #
# 1. loading
# ---------------------------------------------------------------------------- #

def load_imon(run_dir: str, chans: list[str]) -> dict:
    """Concatenate every sub-run's hv_monitor.csv that exists under run_dir.

    Returns dict with 't' (labelled unix seconds), 'sub' (sub-run index, so the
    time-base reconstruction never spans a logger restart) and one array per
    requested channel.
    """
    out: dict[str, list] = {'t': [], 'sub': []}
    for ch in chans:
        out[ch] = []
    subs = []
    for k, sd in enumerate(sorted(glob.glob(os.path.join(run_dir, '*/')))):
        p = os.path.join(sd, 'hv_monitor.csv')
        if not os.path.exists(p):
            continue
        d = C.read_hv_monitor(p)
        if d['t'].size < 100 or any(ch not in d for ch in chans):
            continue
        subs.append(os.path.basename(sd.rstrip('/')))
        out['t'].append(d['t'])
        out['sub'].append(np.full(d['t'].size, len(subs) - 1))
        for ch in chans:
            out[ch].append(d[ch])
    if not subs:
        raise SystemExit(f'no hv_monitor.csv under {run_dir}')
    res = {k: np.concatenate(v) for k, v in out.items()}
    res['subruns'] = subs
    return res


def load_pulses(intensity_dir: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Beam-pulse times [unix s, ms precision] and intensities [1e10 p] in a window."""
    import datetime as _dt
    days = sorted({_dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d')
                   for x in (t0 - 86400, t0, t1, t1 + 86400)})
    ts, iv = [], []
    for day in days:
        p = os.path.join(intensity_dir, f'beam_intensity_{day}.csv')
        if not os.path.exists(p):
            continue
        for row in csv.DictReader(open(p)):
            try:
                t = float(row['unix_ts'])
                v = float(row['intensity_e10'])
            except (KeyError, ValueError):
                continue
            if t0 <= t <= t1 and v > C.PULSE_E10_MIN:
                ts.append(t)
                iv.append(v)
    o = np.argsort(ts)
    return np.asarray(ts)[o], np.asarray(iv)[o]


# ---------------------------------------------------------------------------- #
# 2. time-base reconstruction:  label[k] <= t0 + k*P < label[k]+1
# ---------------------------------------------------------------------------- #

def _a_window(labels: np.ndarray, k: np.ndarray, b: float) -> tuple[float, float]:
    """Admissible t0 interval at a given period b."""
    return (float((labels - b * k).max()), float((labels + 1.0 - b * k).min()))


def _infeas(labels: np.ndarray, k: np.ndarray, b: float) -> float:
    """F(b) = a_lo(b) - a_hi(b).  Convex in b (max of linears + max of linears),
    so feasibility F(b) <= 0 holds on an interval that a ternary search finds
    exactly -- no grid, hence no false negatives when the feasible period range
    is narrower than a grid step (which it is: ~10 us for a 3 500-sample run)."""
    lo, hi = _a_window(labels, k, b)
    return lo - hi


def _lp_box(labels: np.ndarray, iters: int = 90):
    """Feasible (P, t0) region for a run of consecutively-logged samples, from
    `label[k] <= t0 + k*P < label[k]+1`.

    Returns (P_lo, P_hi, t0_lo, t0_hi) evaluated at the interval centre, or None
    if no single linear time base fits (-> the caller splits the segment).
    """
    n = labels.size
    if n < 2:
        return None
    k = np.arange(n, dtype=float)
    span = float(labels[-1] - labels[0])
    p0 = span / max(k[-1], 1.0)
    lo, hi = p0 - 0.5, p0 + 0.5
    for _ in range(iters):                        # ternary search on convex F
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _infeas(labels, k, m1) < _infeas(labels, k, m2):
            hi = m2
        else:
            lo = m1
    b_star = 0.5 * (lo + hi)
    if _infeas(labels, k, b_star) > 0:
        return None
    # widen to the full feasible interval: F is convex, so bisect on each side
    def edge(sign):
        a, c = b_star, b_star + sign * 0.5
        if _infeas(labels, k, c) <= 0:
            return c
        for _ in range(60):
            m = 0.5 * (a + c)
            if _infeas(labels, k, m) <= 0:
                a = m
            else:
                c = m
        return a
    b_lo, b_hi = edge(-1), edge(+1)
    a_lo, a_hi = _a_window(labels, k, b_star)
    return float(b_lo), float(b_hi), float(a_lo), float(a_hi)


def reconstruct_times(labels: np.ndarray, sub: np.ndarray,
                      min_len: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """True sample times and their 1-sided uncertainty, from the drift pattern.

    Greedy-maximal segments inside each sub-run: extend a segment while a single
    linear time base still satisfies every truncation constraint, then start a
    new one.  Segments too short to pin the period get uncertainty 0.5 s (i.e.
    'labels only'), and callers should cut on it.
    """
    t_hat = labels.astype(float) + 0.5
    unc = np.full(labels.size, 0.5)
    for s in np.unique(sub):
        idx = np.where(sub == s)[0]
        lab = labels[idx].astype(float)
        n = lab.size
        i = 0
        while i < n:
            # exponential search for the longest segment with one linear time
            # base, then bisect back
            L = min(2, n - i)
            while i + 2 * L <= n and _lp_box(lab[i:i + 2 * L]) is not None:
                L *= 2
            lo_len, hi_len = L, min(2 * L, n - i)
            while hi_len - lo_len > 1:
                mid = (lo_len + hi_len) // 2
                if _lp_box(lab[i:i + mid]) is not None:
                    lo_len = mid
                else:
                    hi_len = mid
            L = max(lo_len, 1)
            box = _lp_box(lab[i:i + L])
            j = idx[i:i + L]
            if box is not None and L >= min_len:
                b_lo, b_hi, _, _ = box
                kk = np.arange(L, dtype=float)
                # envelope of t_k = a + k*b over the feasible region, sampled at
                # its boundary in b (convex, so this is a tight outer bound)
                tmin = np.full(L, np.inf)
                tmax = np.full(L, -np.inf)
                for bb in (b_lo, 0.5 * (b_lo + b_hi), b_hi):
                    al, ah = _a_window(lab[i:i + L], kk, bb)
                    if al > ah:
                        continue
                    for aa in (al, ah):
                        tt = aa + kk * bb
                        tmin = np.minimum(tmin, tt)
                        tmax = np.maximum(tmax, tt)
                if np.isfinite(tmin).all():
                    t_hat[j] = 0.5 * (tmin + tmax)
                    unc[j] = 0.5 * (tmax - tmin)
            i += L
    return t_hat, unc


# ---------------------------------------------------------------------------- #
# 3. timestamp-free tests
# ---------------------------------------------------------------------------- #

def rolling_baseline(i: np.ndarray, win: int = 241, q: float = 20.0) -> np.ndarray:
    """Leakage baseline that tracks drift: a rolling low percentile.

    The global median is the right baseline only if the leakage is constant over
    the sub-run.  It is not on the leaky channels (det A drifts 2.11 -> 1.97 uA
    across run_79), and a drifting baseline leaks straight into mean - median.
    A low percentile over a ~4 min window sits in the between-pulse population
    (only ~40 % of samples are elevated) while following the drift.
    """
    n = i.size
    out = np.empty(n)
    h = win // 2
    for k in range(0, n, 16):                  # 16-sample stride, then interpolate
        a, b = max(0, k - h), min(n, k + h + 1)
        out[k] = np.percentile(i[a:b], q)
    idx = np.arange(0, n, 16)
    return np.interp(np.arange(n), idx, out[idx])


def noise_sigma(excess: np.ndarray, tau: np.ndarray, tau_min: float = 3.0) -> float:
    """Readback noise from samples far from any pulse (the fold's flat floor)."""
    far = excess[tau > tau_min]
    if far.size < 50:
        far = excess[excess < np.median(excess)]
    return float(1.4826 * np.median(np.abs(far - np.median(far))))


def run_lengths(mask: np.ndarray) -> dict[int, int]:
    """Histogram of run lengths of consecutive True in mask."""
    out: dict[int, int] = {}
    n = 0
    for m in mask:
        if m:
            n += 1
        elif n:
            out[n] = out.get(n, 0) + 1
            n = 0
    if n:
        out[n] = out.get(n, 0) + 1
    return out


def timestamp_free(excess: np.ndarray, thr: float, q_nc: float,
                   rate_hz: float, dt_s: float = 1.012) -> dict:
    """Counting-only discriminants.  No timestamp enters any of these."""
    elev = excess > thr
    frac = float(elev.mean())
    dmax = float(excess.max())
    # A sample whose averaging window fully contains one burst reads Q/w, so
    # w >= Q/dI_max.  Burst is ms, window is >= 0.4 s, so 'fully contains' holds
    # for all but ~1e-2 of elevated samples.
    w_min = (q_nc * 1e-9) / (dmax * 1e-6) if dmax > 0 else np.nan
    return dict(
        frac_elevated=frac,
        thr_ua=thr,
        n_elevated=int(elev.sum()),
        elevated_seconds_per_pulse=frac * dt_s / rate_hz if rate_hz > 0 else np.nan,
        elevated_per_pulse_samples=frac / (rate_hz * dt_s) if rate_hz > 0 else np.nan,
        di_max_ua=dmax,
        w_min_s=float(w_min),
        # what an instantaneous reader of a 10 ms burst would have given
        instant_frac_expected=float(rate_hz * 0.010),
        instant_dimax_expected_ua=float(q_nc * 1e-9 / 0.010 * 1e6),
        run_lengths=run_lengths(elev),
    )


# ---------------------------------------------------------------------------- #
# 4. clock offset: lag scan
# ---------------------------------------------------------------------------- #

def lag_scan(t_lab: np.ndarray, excess: np.ndarray, t_pulse: np.ndarray,
             lag_max: int = 3600) -> dict:
    """Correlate the 1 s excess series against the 1 s binned pulse train.

    Bounds any clock offset between the HV-monitor host and the beam-intensity
    log.  The PS supercycle makes the correlation quasi-periodic, so alias peaks
    at multiples of the supercycle are expected: report them, and report the
    contrast of the true peak against them.
    """
    t0 = int(t_lab.min()) - lag_max - 10
    t1 = int(t_lab.max()) + lag_max + 10
    n = t1 - t0
    sel = (t_pulse >= t0) & (t_pulse < t1)
    pc = np.bincount((t_pulse[sel] - t0).astype(int), minlength=n)[:n].astype(float)
    idx = (t_lab - t0).astype(int)
    lags = np.arange(-lag_max, lag_max + 1)
    corr = np.empty(lags.size)
    for m, lag in enumerate(lags):
        j = idx - lag
        ok = (j >= 0) & (j < n)
        corr[m] = np.corrcoef(excess[ok], pc[j[ok]])[0, 1]
    best = int(lags[np.argmax(corr)])
    # alias peaks: local maxima at least 30 s away from the best lag
    away = np.abs(lags - best) > 30
    return dict(best_lag_s=best, best_corr=float(corr.max()),
                alias_corr=float(corr[away].max()),
                alias_lag_s=int(lags[away][np.argmax(corr[away])]),
                lags=lags, corr=corr)


# ---------------------------------------------------------------------------- #
# 5. phase fold
# ---------------------------------------------------------------------------- #

def phase_fold(t: np.ndarray, excess: np.ndarray, t_pulse: np.ndarray,
               bins: np.ndarray, offset: float = 0.0) -> tuple:
    """Mean excess vs time since the most recent beam pulse.

    Note this is the fold on the PRECEDING pulse only, so at short tau it also
    carries the tails of earlier pulses (n_TOF pulses come 1.2 s apart in
    trains).  Use unfold_response() for the deconvolved single-pulse kernel.
    """
    tt = t + offset
    j = np.searchsorted(t_pulse, tt, side='right') - 1
    ok = j >= 0
    tau = tt[ok] - t_pulse[j[ok]]
    e = excess[ok]
    nb = bins.size - 1
    k = np.digitize(tau, bins) - 1
    mean = np.full(nb, np.nan)
    err = np.full(nb, np.nan)
    cnt = np.zeros(nb, dtype=int)
    for b in range(nb):
        s = k == b
        cnt[b] = int(s.sum())
        if cnt[b]:
            mean[b] = e[s].mean()
            err[b] = e[s].std(ddof=1) / np.sqrt(cnt[b]) if cnt[b] > 1 else np.nan
    return mean, err, cnt, tau, e


# ---------------------------------------------------------------------------- #
# 6. response unfolding (handles overlapping pulses exactly)
# ---------------------------------------------------------------------------- #

def unfold_response(t: np.ndarray, excess: np.ndarray,
                    t_pulse: np.ndarray, i_pulse: np.ndarray,
                    bins: np.ndarray, per_1e10: bool = False,
                    ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares kernel h(tau) with excess(t) = sum_k w_k h(t - t_k) + c.

    w_k = 1 (response per pulse) or i_k/1e10-protons (response per proton).
    Linear, so overlapping pulses are handled exactly rather than by cutting to
    isolated ones (of which there are only ~30 in two hours).
    """
    nb = bins.size - 1
    w = (i_pulse if per_1e10 else np.ones_like(i_pulse)).astype(float)
    M = np.zeros((t.size, nb + 1))
    M[:, nb] = 1.0
    for b in range(nb):
        lo, hi = bins[b], bins[b + 1]
        # pulses contributing to bin b of sample s:  lo <= t_s - t_k < hi
        a = np.searchsorted(t_pulse, t - hi, side='left')
        z = np.searchsorted(t_pulse, t - lo, side='right')
        cw = np.concatenate([[0.0], np.cumsum(w)])
        M[:, b] = cw[z] - cw[a]
    A = M.T @ M + ridge * np.trace(M.T @ M) / (nb + 1) * np.eye(nb + 1)
    coef = np.linalg.solve(A, M.T @ excess)
    # 1 sigma from the residual scatter
    res = excess - M @ coef
    s2 = float(res @ res) / max(t.size - nb - 1, 1)
    cov = s2 * np.linalg.inv(A)
    return coef[:nb], np.sqrt(np.diag(cov))[:nb], float(coef[nb])


def isolated_fold(t: np.ndarray, excess: np.ndarray,
                  t_pulse: np.ndarray, i_pulse: np.ndarray,
                  bins: np.ndarray, gap_before: float = 3.0,
                  gap_after: float = 2.4,
                  sel: np.ndarray | None = None) -> tuple:
    """Model-free single-pulse response: fold only on pulses with a clear gap.

    n_TOF's PS supercycle is strictly periodic (36 s, 11 pulses, spacings all
    multiples of 1.2 s), which makes a deconvolution that tries to separate the
    two intensity bands ill-conditioned.  This does not deconvolve anything: it
    keeps only pulses with no predecessor within `gap_before` seconds AND no
    successor within `gap_after` -- about a third of them -- so the measured
    excess belongs to one pulse over the whole plotted range.

    Both gaps are required, not just the first.  Cutting only on the predecessor
    and then truncating each sample at its successor makes the long-tau bins a
    DIFFERENT, longer-gap subset of pulses, and since the two n_TOF intensity
    bands sit at fixed places in the supercycle that subset has a different mean
    intensity -- which shows up as a spurious dip in the middle of the response.
    """
    dtp = np.diff(t_pulse)
    keep = np.r_[True, dtp > gap_before] & np.r_[dtp > gap_after, True]
    if sel is not None:
        keep &= sel
    tk = t_pulse[keep]
    nxt = np.r_[t_pulse[1:], np.inf]           # next pulse after each pulse
    nk = nxt[keep]
    # tau >= 0: samples after an isolated pulse, truncated at its successor
    j = np.searchsorted(tk, t, side='right') - 1
    ok = j >= 0
    tau = t[ok] - tk[j[ok]]
    e = excess[ok]
    live = t[ok] < nk[j[ok]]
    tau, e = tau[live], e[live]
    # tau < 0: samples in the `pre` seconds BEFORE an isolated pulse.  The pulse
    # has no predecessor within gap_before, so at tau > -(gap_before - 2.2) the
    # previous pulse's response has decayed -- these bins are the causality
    # check, and they must come out at zero.
    pre = max(gap_before - 2.2, 0.0)
    if pre > 0:
        j2 = np.searchsorted(tk, t, side='left')
        ok2 = j2 < tk.size
        tau2 = t[ok2] - tk[j2[ok2]]
        e2 = excess[ok2]
        keep2 = tau2 >= -pre
        tau = np.r_[tau, tau2[keep2]]
        e = np.r_[e, e2[keep2]]
    nb = bins.size - 1
    k = np.digitize(tau, bins) - 1
    mean = np.full(nb, np.nan)
    err = np.full(nb, np.nan)
    cnt = np.zeros(nb, dtype=int)
    for b in range(nb):
        s = k == b
        cnt[b] = int(s.sum())
        if cnt[b]:
            mean[b] = e[s].mean()
            err[b] = e[s].std(ddof=1) / np.sqrt(cnt[b]) if cnt[b] > 1 else np.nan
    return mean, err, cnt, int(keep.sum()), float(i_pulse[keep].mean()) if keep.any() else np.nan


def flatness_chi2(mean: np.ndarray, err: np.ndarray) -> float:
    """chi2/ndf of a fold against a flat line at its own mean.

    The right null statistic.  max(fold) is not: the maximum of ~40 noisy bin
    means is biased upward, so a randomised-pulse-time control 'sees' a peak.
    """
    ok = np.isfinite(mean) & np.isfinite(err) & (err > 0)
    if ok.sum() < 5:
        return float('nan')
    m, e = mean[ok], err[ok]
    mu = float(np.average(m, weights=1.0 / e ** 2))
    return float(np.sum(((m - mu) / e) ** 2) / (m.size - 1))


def kernel_metrics(h: np.ndarray, bins: np.ndarray) -> dict:
    ctr = 0.5 * (bins[:-1] + bins[1:])
    wid = np.diff(bins)
    area = float(np.sum(h * wid))                       # uA*s = uC
    pk = int(np.argmax(h))
    half = h.max() / 2.0
    above = np.where(h > half)[0]
    fwhm = float(ctr[above[-1]] - ctr[above[0]] + wid[above[0]]) if above.size else np.nan
    mean_t = float(np.sum(h * ctr * wid) / area) if area else np.nan
    rms = float(np.sqrt(max(np.sum(h * (ctr - mean_t) ** 2 * wid) / area, 0.0))) if area else np.nan
    return dict(area_uAs=area, q_nc=area * 1e3, peak_ua=float(h.max()),
                peak_tau_s=float(ctr[pk]), fwhm_s=fwhm,
                centroid_s=mean_t, rms_width_s=rms)


# ---------------------------------------------------------------------------- #
# driver
# ---------------------------------------------------------------------------- #

def analyse(src: str, run: str, dets, out_dir: str, fine: float = 0.20) -> dict:
    run_dir = os.path.join(src, 'runs', run)
    if not os.path.isdir(run_dir):
        run_dir = os.path.join(src, run)
    chans = [C.RESIST_CH[d] + ' imon' for d in dets] + [NULL_CH]
    d = load_imon(run_dir, chans)
    t_lab = d['t']
    tp, ip = load_pulses(os.path.join(src, 'slow_control', 'beam_intensity')
                         if os.path.isdir(os.path.join(src, 'slow_control'))
                         else os.path.join(src, 'beam_intensity'),
                         t_lab.min() - 30, t_lab.max() + 30)
    rate = tp.size / (t_lab.max() - t_lab.min())

    print(f'{run}: {t_lab.size} imon samples over '
          f'{(t_lab.max() - t_lab.min()) / 3600:.2f} h, sub-runs {d["subruns"]}')
    print(f'  {tp.size} beam pulses, rate {rate:.3f} Hz, '
          f'intensity {np.median(ip):.0f}e10 median')

    # -- sampling cadence -----------------------------------------------------
    gap = np.diff(t_lab)
    gaps = np.where(gap > 1.5)[0]
    cadence = dict(
        n=int(t_lab.size), label_granularity_s=1.0,
        n_gaps=int(gaps.size),
        gap_spacing_median=float(np.median(np.diff(gaps))) if gaps.size > 2 else np.nan,
        loop_period_s=float(1.0 + gaps.size / max(t_lab.size, 1)),
    )
    print(f'  cadence: {gaps.size} slipped seconds, every '
          f'{cadence["gap_spacing_median"]:.0f} samples -> loop period '
          f'{cadence["loop_period_s"]:.4f} s (labels are truncated, so the '
          f'sub-second phase DRIFTS)')

    # -- time base ------------------------------------------------------------
    t_hat, unc = reconstruct_times(t_lab, d['sub'])
    good = unc < 0.15
    print(f'  time base: {good.mean() * 100:.1f}% of samples reconstructed to '
          f'<150 ms (median {np.median(unc[good]) * 1e3:.0f} ms, '
          f'p95 {np.percentile(unc[good], 95) * 1e3:.0f} ms)')
    timebase = dict(frac_good=float(good.mean()),
                    median_unc_ms=float(np.median(unc[good]) * 1e3),
                    p95_unc_ms=float(np.percentile(unc[good], 95) * 1e3))

    res: dict = dict(run=run, subruns=d['subruns'], n_samples=int(t_lab.size),
                     n_pulses=int(tp.size), pulse_rate_hz=float(rate),
                     intensity_median_e10=float(np.median(ip)),
                     cadence=cadence, timebase=timebase, dets={})

    # -- clock offset (once, on the cleanest channel) -------------------------
    ch0 = C.RESIST_CH[dets[0]] + ' imon'
    ex0 = d[ch0] - np.median(d[ch0])
    ls = lag_scan(t_lab, ex0, tp)
    res['clock'] = {k: v for k, v in ls.items() if k not in ('lags', 'corr')}
    print(f'  clock: best lag {ls["best_lag_s"]:+d} s (r={ls["best_corr"]:.3f}); '
          f'nearest alias {ls["alias_lag_s"]:+d} s (r={ls["alias_corr"]:.3f})')

    bins_lab = np.arange(-1.0, 6.001, 0.25)
    bins_fin = np.arange(-0.8, 2.401, fine)
    bins_crs = np.arange(-0.8, 2.401, 0.4)
    os.makedirs(out_dir, exist_ok=True)

    # -- beam structure -------------------------------------------------------
    dtp = np.diff(tp)
    res['beam'] = dict(
        dt_unique=[float(x) for x in np.unique(np.round(dtp, 1))[:12]],
        n_dedicated=int((ip >= 600).sum()), n_parasitic=int((ip < 600).sum()),
        mean_e10_dedicated=float(ip[ip >= 600].mean()),
        mean_e10_parasitic=float(ip[ip < 600].mean()),
        n_isolated_gap3=int(np.r_[True, dtp > 3.0].sum()),
    )
    print(f'  beam: {res["beam"]["n_dedicated"]} dedicated '
          f'({res["beam"]["mean_e10_dedicated"]:.0f}e10) + '
          f'{res["beam"]["n_parasitic"]} parasitic '
          f'({res["beam"]["mean_e10_parasitic"]:.0f}e10) on a strict PS '
          f'supercycle; {res["beam"]["n_isolated_gap3"]} pulses have a >3 s '
          f'preceding gap')

    for det in list(dets) + ['NULL']:
        ch = NULL_CH if det == 'NULL' else C.RESIST_CH[det] + ' imon'
        i = d[ch]
        med = float(np.median(i))
        base_roll = rolling_baseline(i)
        q_vs_pct = {int(qq): float((np.mean(i - rolling_baseline(i, q=qq)) * 1e-6
                                    / rate) * 1e9) for qq in (5, 10, 20, 30)}
        ex = i - med                       # the published estimator's excess
        ex_r = i - base_roll               # leakage-detrended excess
        q_nc = (ex.mean() * 1e-6 / rate) * 1e9 if rate else np.nan
        q_nc_r = (ex_r.mean() * 1e-6 / rate) * 1e9 if rate else np.nan

        # folds.  All fine-binned work uses the detrended excess and the
        # reconstructed time base; the label fold is kept to show the smearing.
        m_lab, e_lab, c_lab, tau_lab, _ = phase_fold(t_lab + 0.5, ex_r, tp, bins_lab)
        m_fin, e_fin, c_fin, tau_fin, _ = phase_fold(t_hat[good], ex_r[good], tp, bins_fin)

        sig = noise_sigma(ex_r, tau_lab, 3.0)
        thr = max(5.0 * sig, 0.002)
        tf = timestamp_free(ex_r, thr, q_nc_r, rate, cadence['loop_period_s'])
        # the same count at the threshold HANDOFF sec 4 quoted (27.8 % on run_158)
        tf['frac_elevated_at_20nA'] = float((ex_r > 0.02).mean())

        # model-free single-pulse response, and the same split by intensity band
        m_iso, e_iso, c_iso, n_iso, ibar = isolated_fold(
            t_hat[good], ex_r[good], tp, ip, bins_fin)
        # the SAME fold on the raw labels.  Binned COARSELY on purpose: with
        # whole-second labels tau is aliased onto the beam's own 1.2 s grid, so a
        # 0.2 s binning of the label fold is spiky nonsense, not a measurement --
        # which is itself the point about what the reconstruction buys.
        m_isl, e_isl, c_isl, _, _ = isolated_fold(
            t_lab + 0.5, ex_r, tp, ip, bins_crs)
        iso_lab = kernel_metrics(np.nan_to_num(m_isl), bins_crs)
        iso = kernel_metrics(np.nan_to_num(m_iso), bins_fin)
        iso['n_pulses'] = n_iso
        iso['mean_e10'] = ibar
        iso['chi2_flat'] = flatness_chi2(m_iso, e_iso)
        bands = {}
        for nm, s in (('parasitic', ip < 600.0), ('dedicated', ip >= 600.0)):
            mb, eb, cb, nb_, ib = isolated_fold(t_hat[good], ex_r[good], tp, ip,
                                                bins_fin, sel=s)
            if nb_ < 30:
                continue
            kb = kernel_metrics(np.nan_to_num(mb), bins_fin)
            bands[nm] = dict(n_pulses=nb_, mean_e10=ib, q_nc=kb['q_nc'],
                             peak_ua=kb['peak_ua'], fwhm_s=kb['fwhm_s'],
                             pc_per_1e10=kb['q_nc'] * 1e3 / ib if ib else np.nan)

        # deconvolved kernel (uses every pulse; exact for overlaps, but the
        # supercycle's fixed pattern means DO NOT split it by band)
        h, dh, bfit = unfold_response(t_hat[good], ex_r[good], tp, ip, bins_fin)
        km = kernel_metrics(h, bins_fin)

        # clock stability: same kernel on the two halves of the run
        halves = []
        mid = t_hat[good].size // 2
        tg, eg = t_hat[good], ex_r[good]
        for sl in (slice(0, mid), slice(mid, None)):
            hh, _, _ = unfold_response(tg[sl], eg[sl], tp, ip, bins_fin)
            halves.append(kernel_metrics(hh, bins_fin))

        res['dets'][det] = dict(
            channel=ch, i_median_ua=med, i_mean_ua=float(i.mean()),
            di_ua=float(ex.mean()), di_detrended_ua=float(ex_r.mean()),
            q_nc_mean_median=q_nc, q_nc_detrended=q_nc_r,
            noise_sigma_ua=sig, q_nc_vs_baseline_pct=q_vs_pct, **tf,
            kernel=km, kernel_baseline_ua=bfit, isolated=iso,
            isolated_on_labels=iso_lab, bands=bands,
            halves=[dict(centroid_s=x['centroid_s'], rms_width_s=x['rms_width_s'],
                         fwhm_s=x['fwhm_s'], q_nc=x['q_nc']) for x in halves],
            closure=dict(q_isolated_nc=iso['q_nc'], q_kernel_nc=km['q_nc'],
                         q_mean_median_nc=q_nc, q_detrended_nc=q_nc_r,
                         ratio_kernel_over_meanmed=km['q_nc'] / q_nc if q_nc else np.nan),
        )

        # fold CSVs (all three, so nothing is hidden)
        for tag, (bb, mm, ee, cc) in (
                ('labels', (bins_lab, m_lab, e_lab, c_lab)),
                ('recon', (bins_fin, m_fin, e_fin, c_fin)),
                ('isolated', (bins_fin, m_iso, e_iso, c_iso)),
                ('isolated_labels', (bins_crs, m_isl, e_isl, c_isl))):
            p = os.path.join(out_dir, f'imon_fold_{run}_{det}_{tag}.csv')
            with open(p, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['tau_lo_s', 'tau_hi_s', 'n', 'mean_excess_ua', 'err_ua'])
                for b in range(bb.size - 1):
                    w.writerow([f'{bb[b]:.3f}', f'{bb[b + 1]:.3f}', cc[b],
                                f'{mm[b]:.6f}', f'{ee[b]:.6f}'])
        p = os.path.join(out_dir, f'imon_kernel_{run}_{det}.csv')
        with open(p, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['tau_lo_s', 'tau_hi_s', 'h_ua', 'h_err_ua'])
            for b in range(bins_fin.size - 1):
                w.writerow([f'{bins_fin[b]:.3f}', f'{bins_fin[b + 1]:.3f}',
                            f'{h[b]:.6f}', f'{dh[b]:.6f}'])

        print(f'  det {det}: Q(mean-med) {q_nc:7.1f} / detrended {q_nc_r:7.1f} nC '
              f'| noise {sig * 1e3:5.2f} nA | elevated {tf["frac_elevated"] * 100:4.1f}% '
              f'@{thr * 1e3:.0f}nA, {tf["frac_elevated_at_20nA"] * 100:4.1f}% @20nA '
              f'(instant read: {tf["instant_frac_expected"] * 100:.1f}%)')
        print(f'         dI_max {tf["di_max_ua"]:.3f} uA (instant would be '
              f'{tf["instant_dimax_expected_ua"]:.1f}) -> window >= '
              f'{tf["w_min_s"]:.2f} s | isolated-pulse response: peak '
              f'{iso["peak_ua"]:.3f} uA @ {iso["peak_tau_s"]:.2f} s, FWHM '
              f'{iso["fwhm_s"]:.2f} s, Q {iso["q_nc"]:.1f} nC ({n_iso} pulses)')
        print(f'         deconvolved: peak {km["peak_ua"]:.3f} uA @ '
              f'{km["peak_tau_s"]:.2f} s, FWHM {km["fwhm_s"]:.2f} s, Q '
              f'{km["q_nc"]:.1f} nC | halves centroid '
              f'{halves[0]["centroid_s"]:.3f}/{halves[1]["centroid_s"]:.3f} s, '
              f'rms {halves[0]["rms_width_s"]:.3f}/{halves[1]["rms_width_s"]:.3f} s')
        for nm, b in bands.items():
            print(f'           {nm:>10}: {b["n_pulses"]:4d} pulses, '
                  f'{b["mean_e10"]:.0f}e10 -> Q {b["q_nc"]:6.1f} nC, '
                  f'{b["pc_per_1e10"]:6.1f} pC per 1e10 p')

    # -- nulls ----------------------------------------------------------------
    ch = C.RESIST_CH[dets[0]] + ' imon'
    ex = d[ch] - rolling_baseline(d[ch])
    rng = np.random.default_rng(17)
    real_chi2 = res['dets'][dets[0]]['isolated']['chi2_flat']
    fake = []
    for _ in range(20):
        # uniform-random pulse times: same count, structure destroyed.  A pure
        # time SHIFT is not a null here -- the 36 s supercycle aliases onto
        # itself, so a shifted fold still correlates.
        tf_ = np.sort(rng.uniform(t_hat.min(), t_hat.max(), tp.size))
        m, e, _, _, _ = isolated_fold(t_hat[good], ex[good], tf_,
                                      np.full(tp.size, 500.0), bins_fin)
        fake.append(flatness_chi2(m, e))
    res['null_random_times_chi2_flat'] = float(np.mean(fake))
    res['null_random_times_chi2_flat_std'] = float(np.std(fake))
    res['real_chi2_flat'] = real_chi2
    print(f'  null (pulse times randomised, 20 draws): fold chi2/ndf vs flat '
          f'{np.mean(fake):.2f} +- {np.std(fake):.2f}, real {real_chi2:.1f}')

    with open(os.path.join(out_dir, f'imon_response_{run}.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f'  wrote {out_dir}/imon_response_{run}.json')
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/media/dylan/data/x17/beam_july')
    ap.add_argument('--run', default='run_79')
    ap.add_argument('--dets', default=','.join(DEFAULT_DETS))
    ap.add_argument('--out', default=os.path.join(HERE, 'results'))
    ap.add_argument('--fine', type=float, default=0.2,
                    help='fold bin width [s] on the reconstructed time base')
    a = ap.parse_args()
    analyse(a.src, a.run, tuple(a.dets.split(',')), a.out, fine=a.fine)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
