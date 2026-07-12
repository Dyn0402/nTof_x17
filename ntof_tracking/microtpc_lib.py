#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
microtpc_lib.py — standalone micro-TPC reconstruction library for MX17
strip-Micromegas detectors, distilled from the June 2026 cosmic-bench
characterization (mx_june_cosmic_qa scripts 21/26/33/34/35/36).

Design goals:
  * numpy/pandas only — importable on the analysis laptop AND on the DAQ
    machine (daq_lxplus) without the cosmic-bench package tree.
  * data-source agnostic — every function takes plain arrays for ONE
    plane of ONE track/cluster candidate (position mm, amplitude ADC,
    time ns, time_over_threshold ns). Loading/mapping stays in drivers.
  * algorithm-faithful — each function reproduces the bench script it is
    named after (provenance in each docstring); bench numbers quoted in
    bench_constants.py are the acceptance targets.

The waveform-free ("hits6") chain implemented here needs ONLY the
combined_hits columns (eventId, feu, channel, amplitude, time,
time_over_threshold) and gives:
  - per-plane cluster + anchored time fit  (production S, position anchor)
  - per-plane signature features           (6-feature set of script 35)
  - |tan theta| regression + sign          (script 34 construction)
  - extent-slope / T_sat drift velocity    (script 21/35 estimator)
  - X/Y pairing scores (time IoU + charge balance, PLAN_38 result)

Waveform-level upgrades (unsharing, early-charge centroid, n_u) live on
the DAQ side — see ntof_tracking/TRACK_PLAN_06_daq_waveforms.md.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from . import bench_constants as bc

# re-export the most-used constants at module level for driver convenience
PITCH_MM = bc.PITCH_MM
THR_HIT = bc.THR_HIT
A_LEAD_MIN = bc.A_LEAD_MIN
FEATS_HITS6 = bc.FEATS_HITS6


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
def sigma68(a):
    """Half the 16-84 percentile spread (robust sigma)."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return np.nan
    q = np.percentile(a, [16, 84])
    return 0.5 * (q[1] - q[0])


def robust_line(x, y, band, iters=4, clip=3.0):
    """Iterative deg-1 polyfit with MAD clipping inside x-band (34's helper).
    Returns slope or nan. Used to fit v from S vs tan over a band."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > band[0]) & (x < band[1])
    if m.sum() < 10:
        return np.nan
    xs, ys = x[m], y[m]
    keep = np.ones(len(xs), bool)
    coef = None
    for _ in range(iters):
        if keep.sum() < 10:
            return np.nan
        coef = np.polyfit(xs[keep], ys[keep], 1)
        r = ys - np.polyval(coef, xs)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-12
        keep = np.abs(r - np.median(r[keep])) < clip * 1.4826 * mad
    return float(coef[0]) if coef is not None else np.nan


# ---------------------------------------------------------------------------
# clustering + anchored time fit  (cosmic_micro_tpc_analysis._fit_single_axis)
# ---------------------------------------------------------------------------
def gap_cluster_largest(pos, gap_mm=bc.GAP_THRESHOLD_MM):
    """Split position-sorted hits on gaps > gap_mm; return boolean mask of the
    largest cluster (ties -> first). `pos` need not be sorted."""
    pos = np.asarray(pos, float)
    order = np.argsort(pos)
    ps = pos[order]
    new = np.zeros(len(ps), bool)
    new[1:] = np.diff(ps) > gap_mm
    labels_sorted = np.cumsum(new)
    # count per label, pick largest
    best = np.argmax(np.bincount(labels_sorted))
    mask_sorted = labels_sorted == best
    mask = np.zeros(len(ps), bool)
    mask[order] = mask_sorted
    return mask


def anchored_time_fit(pos, time, amp, gap_mm=bc.GAP_THRESHOLD_MM,
                      min_strips=bc.MIN_STRIPS):
    """Production per-plane micro-TPC fit (_fit_single_axis, bit-faithful):
      1. largest gap-cluster (gap > 12 mm on sorted positions)
      2. anchor at the EARLIEST-time strip (pos_anchor, t_anchor)
      3. amplitude-weighted line t = t_anchor + slope*(pos - pos_anchor),
         weights amp (curve_fit sigma=1/sqrt(amp)); slope only.
    Returns dict or None. S_um_ns = 1000/slope = apparent drift speed x tan.
    NB: this raw time fit carries the known ~10-20 % charge-sharing bias --
    use it for pattern/anchoring, use the regression/unsharing for angles.
    """
    pos = np.asarray(pos, float); time = np.asarray(time, float)
    amp = np.asarray(amp, float)
    ok = np.isfinite(pos) & np.isfinite(time) & np.isfinite(amp)
    pos, time, amp = pos[ok], time[ok], amp[ok]
    if len(pos) < min_strips:
        return None
    m = gap_cluster_largest(pos, gap_mm)
    n_dropped = int((~m).sum())
    pos, time, amp = pos[m], time[m], amp[m]
    if len(pos) < min_strips:
        return None
    i0 = int(np.argmin(time))
    p0, t0 = pos[i0], time[i0]
    dp = pos - p0
    dt = time - t0
    w = amp + 1e-9                       # curve_fit sigma=1/sqrt(amp) => w=amp
    den = np.sum(w * dp * dp)
    slope = np.sum(w * dp * dt) / den if den > 0 else np.nan  # ns/mm
    resid = dt - slope * dp
    red_chi2 = float(np.sum(w * resid**2) / max(len(pos) - 1, 1))
    ext = float(np.ptp(pos))
    return dict(
        slope_ns_per_mm=float(slope),
        S_um_ns=(1000.0 / slope) if slope not in (0.0, np.nan) and np.isfinite(slope) and slope != 0 else np.nan,
        mesh_position_mm=float(p0),      # production position anchor
        earliest_time_ns=float(t0),
        latest_time_ns=float(time.max()),
        duration_ns=float(time.max() - t0),
        n_strips=int(len(pos)),
        n_dropped=n_dropped,
        extent_mm=ext,
        q_sum=float(amp.sum()),
        red_chi2=red_chi2,
    )


# ---------------------------------------------------------------------------
# hit-level signature features (script 35 build_features_hits, exact)
# ---------------------------------------------------------------------------
def hit_features(pos, amp, time, tot,
                 a_lead_min=bc.A_LEAD_MIN, thr_hit=bc.THR_HIT,
                 pitch=bc.PITCH_MM):
    """Signature features for ONE plane of ONE event/track candidate.
    Inputs: per-strip arrays (any order). Returns dict or None (no valid lead).
    Feature definitions are bit-faithful to 35_hybrid_drift_scan.py; the two
    signed features (a_asym_sgn, t_asym_sgn) needed by the sign model are
    added on top (same quantities 34 builds from headon_features.csv).
    """
    pos = np.asarray(pos, float); amp = np.asarray(amp, float)
    time = np.asarray(time, float); tot = np.asarray(tot, float)
    ok = np.isfinite(pos)
    if ok.sum() < 1:
        return None
    pos, amp, time, tot = pos[ok], amp[ok], time[ok], tot[ok]
    o = np.argsort(pos)
    p, a, t, q = pos[o], amp[o], time[o], tot[o]
    k = int(np.argmax(a))
    if a[k] < a_lead_min:
        return None
    l0 = k
    while l0 - 1 >= 0 and a[l0 - 1] >= thr_hit \
            and abs(p[l0] - p[l0 - 1] - pitch) < 0.5 * pitch:
        l0 -= 1
    r0 = k
    while r0 + 1 < len(p) and a[r0 + 1] >= thr_hit \
            and abs(p[r0 + 1] - p[r0] - pitch) < 0.5 * pitch:
        r0 += 1
    n_raw = r0 - l0 + 1
    q_clu = float(a[l0:r0 + 1].sum())

    def nb(side):
        j = k + side
        if 0 <= j < len(p) and abs(p[j] - p[k] - side * pitch) < 0.5 * pitch:
            return float(a[j]), float(t[j])
        return 0.0, np.nan

    aL, tL = nb(-1)
    aR, tR = nb(+1)
    t_nb = [x for x in (tL, tR) if np.isfinite(x)]
    t_delay = (min(t_nb) - t[k]) if t_nb else np.nan
    s = aR + aL
    return dict(
        a_lead=float(a[k]), a_l=aL, a_r=aR, t_l=tL, t_r=tR,
        tot_lead=float(q[k]), n_raw=int(n_raw),
        q_frac=float(a[k] / q_clu) if q_clu > 0 else np.nan,
        a_asym=abs(aR - aL) / s if s > 0 else np.nan,
        a_asym_sgn=(aR - aL) / s if s > 0 else np.nan,
        t_asym_sgn=(tR - tL) if (np.isfinite(tR) and np.isfinite(tL)) else np.nan,
        t_delay=float(t_delay) if t_delay is not None else np.nan,
        pos_lead_mm=float(p[k]),
        q_cluster=q_clu,
    )


# ---------------------------------------------------------------------------
# |tan theta| regression (34/35 construction: standardize -> lstsq ->
# monotonic binned-median calibration) + serialization
# ---------------------------------------------------------------------------
def train_tan_regression(F, y_abs, feats=FEATS_HITS6,
                         min_train=300, n_quant=25, min_bin=30, min_bins=5):
    """Train the |tan| regressor. F: (n, len(feats)) array; y_abs: |tan_ref|.
    Returns model dict {feats, mu, sd, w, calib_s, calib_t} or None."""
    F = np.asarray(F, float); y = np.asarray(y_abs, float)
    ok = np.isfinite(F).all(axis=1) & np.isfinite(y)
    if ok.sum() < min_train:
        return None
    mu, sd = F[ok].mean(axis=0), F[ok].std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (F[ok] - mu) / sd
    A = np.c_[Z, np.ones(ok.sum())]
    w, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    s_lin = Z @ w[:-1] + w[-1]
    qs = np.nanquantile(s_lin, np.linspace(0.02, 0.98, n_quant))
    ctr_s, med_t = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (s_lin >= lo) & (s_lin < hi)
        if m.sum() > min_bin:
            ctr_s.append(float(np.median(s_lin[m])))
            med_t.append(float(np.median(y[ok][m])))
    if len(ctr_s) < min_bins:
        return None
    med_t = np.maximum.accumulate(np.array(med_t)).tolist()
    return dict(feats=list(feats), mu=mu.tolist(), sd=sd.tolist(),
                w=w.tolist(), calib_s=ctr_s, calib_t=med_t)


def apply_tan_regression(model, F, restandardize=False):
    """Apply a trained |tan| model to feature rows F (n, len(feats)).
    restandardize=True recomputes mu/sd on THIS dataset's finite rows
    (weights + calibration stay frozen) — the cheap in-situ adaptation for
    a new gas/HV/window where feature scales shift (tot_lead, t_delay
    rescale with drift time; see 35's per-point-training finding).
    Returns (tan_abs, ok_mask)."""
    F = np.asarray(F, float)
    ok = np.isfinite(F).all(axis=1)
    mu = np.array(model['mu'], float); sd = np.array(model['sd'], float)
    if restandardize and ok.sum() > 100:
        mu = F[ok].mean(axis=0)
        sd = F[ok].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
    w = np.array(model['w'], float)
    Z = (F - mu) / sd
    s_lin = np.full(len(F), np.nan)
    s_lin[ok] = Z[ok] @ w[:-1] + w[-1]
    tan_abs = np.full(len(F), np.nan)
    tan_abs[ok] = np.interp(s_lin[ok],
                            np.array(model['calib_s'], float),
                            np.array(model['calib_t'], float))
    return tan_abs, ok


# ---------------------------------------------------------------------------
# sign model (34's Fisher discriminant on signed asymmetries)
# ---------------------------------------------------------------------------
def train_sign_fisher(a_asym_sgn, t_asym_sgn, sign_true, tan_ref_abs,
                      band=(np.tan(np.radians(2.0)), np.tan(np.radians(10.0)))):
    """Fisher LDA on G=[a_asym_sgn, t_asym_sgn], trained on tracks with
    2 deg < |tan_ref| < 10 deg. Returns wg (len 2) or None."""
    G = np.c_[np.asarray(a_asym_sgn, float), np.asarray(t_asym_sgn, float)]
    y = np.asarray(sign_true, float)
    ta = np.asarray(tan_ref_abs, float)
    m = np.isfinite(G).all(axis=1) & np.isfinite(y) & (ta > band[0]) & (ta < band[1])
    if (m & (y > 0)).sum() < 50 or (m & (y < 0)).sum() < 50:
        return None
    mu1 = G[m & (y > 0)].mean(axis=0)
    mu0 = G[m & (y < 0)].mean(axis=0)
    Sw = np.cov(G[m & (y > 0)].T) + np.cov(G[m & (y < 0)].T)
    try:
        wg = np.linalg.solve(Sw, mu1 - mu0)
    except np.linalg.LinAlgError:
        return None
    return wg.tolist()


def apply_sign(wg, a_asym_sgn, t_asym_sgn, fallback_sign=None):
    """sign = sign(G @ wg); NaN rows fall back to fallback_sign (or +1)."""
    G = np.c_[np.asarray(a_asym_sgn, float), np.asarray(t_asym_sgn, float)]
    g = G @ np.asarray(wg, float)
    s = np.sign(g)
    bad = ~np.isfinite(g) | (s == 0)
    if fallback_sign is not None:
        fb = np.sign(np.asarray(fallback_sign, float))
        fb[~np.isfinite(fb) | (fb == 0)] = 1.0
        s[bad] = fb[bad]
    else:
        s[bad] = 1.0
    return s


def hybrid_tan(tan_seg, tan_reg_abs, sign_hat,
               tan_switch=bc.TAN_SWITCH, seg_sane=1.5):
    """34's hybrid rule: the REGRESSOR decides the regime.
    Use the (unshared/calibrated) segment time-fit angle when it exists,
    is sane, and tan_reg_abs > tan_switch (~5 deg); else signed regression."""
    tan_seg = np.asarray(tan_seg, float)
    tan_reg = np.asarray(sign_hat, float) * np.asarray(tan_reg_abs, float)
    use_seg = np.isfinite(tan_seg) & (np.abs(tan_seg) < seg_sane) \
        & (np.asarray(tan_reg_abs, float) > tan_switch)
    return np.where(use_seg, tan_seg, tan_reg), use_seg


# ---------------------------------------------------------------------------
# extent-slope / T_sat drift velocity (21/35's v_extent, exact)
# ---------------------------------------------------------------------------
def v_extent(tan_abs, n_strips, duration_ns, min_bin=25,
             tan_lo=bc.TAN_LO, tan_hi=bc.TAN_HI, tan_step=bc.TAN_STEP,
             sat_deg=bc.SAT_DEG, pitch=bc.PITCH_MM, min_sat=30):
    """v = (cluster-extent slope vs |tan|) / T_sat. Telescope-free once the
    abscissa is a calibrated angle estimate (regressed or reference).
    Returns dict(v, v_err, slope, tsat, n_bins, n_sat) or None."""
    tan_abs = np.asarray(tan_abs, float)
    ns = np.asarray(n_strips, float)
    dur = np.asarray(duration_ns, float)
    ext = (ns - 1) * pitch
    bins = np.arange(tan_lo, tan_hi + tan_step, tan_step)
    ctr, med, mer = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (tan_abs >= b0) & (tan_abs < b1) & np.isfinite(ext) & np.isfinite(dur)
        if m.sum() >= min_bin:
            ctr.append(0.5 * (b0 + b1))
            med.append(np.median(ext[m]))
            mer.append(1.253 * np.std(ext[m], ddof=1) / np.sqrt(m.sum()))
    if len(ctr) < 4:
        return None
    ctr, med, mer = map(np.array, (ctr, med, mer))
    w = 1.0 / mer**2
    W, Wx, Wy = np.sum(w), np.sum(w * ctr), np.sum(w * med)
    Wxx, Wxy = np.sum(w * ctr**2), np.sum(w * ctr * med)
    den = W * Wxx - Wx**2
    slope = (W * Wxy - Wx * Wy) / den
    slope_err = np.sqrt(W / den)
    m_sat = tan_abs > np.tan(np.radians(sat_deg))
    m_sat &= np.isfinite(dur)
    if m_sat.sum() < min_sat:
        return None
    tsat = float(np.median(dur[m_sat]))
    tsat_err = float(1.253 * np.std(dur[m_sat], ddof=1) / np.sqrt(m_sat.sum()))
    v = slope * 1000.0 / tsat
    v_err = v * np.hypot(slope_err / slope, tsat_err / tsat)
    return dict(v=float(v), v_err=float(v_err), slope=float(slope),
                tsat=tsat, n_bins=len(ctr), n_sat=int(m_sat.sum()))


# ---------------------------------------------------------------------------
# X/Y pairing scores (track_explorer time-IoU + PLAN_38 charge balance)
# ---------------------------------------------------------------------------
def time_iou(t0a, t1a, t0b, t1b):
    """Interval intersection-over-union of two [t0,t1] time spans."""
    lo = max(t0a, t0b); hi = min(t1a, t1b)
    inter = max(0.0, hi - lo)
    union = max(t1a, t1b) - min(t0a, t0b)
    return inter / union if union > 0 else 0.0


def charge_balance_score(qx, qy, f_med, f_s68):
    """PLAN_38: f = qx/(qx+qy) is narrow (sigma68 ~ 0.07) and flat in
    position/angle. Return |f - f_med| / f_s68 (a pull; <~2 = compatible).
    Use per-detector f_med from bench_constants.DETECTORS (or re-measure
    in situ — one histogram)."""
    s = qx + qy
    if not np.isfinite(s) or s <= 0:
        return np.nan
    f = qx / s
    return abs(f - f_med) / f_s68


def pair_planes(x_cands, y_cands, f_med=0.5, f_s68=0.07,
                min_iou=0.20, w_iou=1.0, w_bal=0.35, max_bal_pull=4.0):
    """Greedy X<->Y candidate pairing by combined score.
    x_cands/y_cands: lists of dicts with keys t0, t1 (time span, ns) and
    q (cluster charge). Score = w_iou*IoU - w_bal*min(pull, max)/max.
    Returns list of (ix, iy, iou, pull) accepted pairs (each cand used once).
    """
    scored = []
    for i, cx in enumerate(x_cands):
        for j, cy in enumerate(y_cands):
            iou = time_iou(cx['t0'], cx['t1'], cy['t0'], cy['t1'])
            if iou < min_iou:
                continue
            pull = charge_balance_score(cx['q'], cy['q'], f_med, f_s68)
            pn = min(pull, max_bal_pull) / max_bal_pull if np.isfinite(pull) else 1.0
            scored.append((w_iou * iou - w_bal * pn, i, j, iou, pull))
    scored.sort(reverse=True)
    used_x, used_y, out = set(), set(), []
    for s, i, j, iou, pull in scored:
        if i in used_x or j in used_y:
            continue
        used_x.add(i); used_y.add(j)
        out.append((i, j, iou, pull))
    return out


# ---------------------------------------------------------------------------
# model file I/O
# ---------------------------------------------------------------------------
def save_model(path, det_name, provenance, planes):
    """planes: {'x': {...regression model..., 'wg': [...], 'holdout': {...},
    'v_sig':..., 't_sat_ns':...}, 'y': {...}}"""
    obj = dict(version=1, kind='hits6', det=det_name, **provenance,
               planes=planes)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
    return path


def load_model(path):
    with open(path) as f:
        m = json.load(f)
    assert m.get('kind') == 'hits6', f'unexpected model kind in {path}'
    for p in m['planes'].values():
        assert p['feats'] == list(FEATS_HITS6), \
            f'feature-set mismatch in {path}: {p["feats"]}'
    return m
