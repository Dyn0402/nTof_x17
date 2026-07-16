#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sota_reco.py — state-of-the-art micro-TPC hit reconstruction (reusable).

The production combined-hits path uses RAW per-strip times/positions: it does
NOT undo the resistive-strip charge sharing, which biases the time-position
ladder (raw ladder ~4 deg too steep; see the depth-resolved residual). The
charge-sharing "unsharing" that fixes this has lived only in the prototype
study scripts (26/27/28/36). This module lifts that hit-level correction into a
single reusable entry point so displays and analyses can consume corrected
hits instead of re-implementing the deconvolution.

Pipeline (per event, from decoded_root waveforms):
  1. pedestal (per strip) + common-mode (per chip, per sample) subtraction
  2. UNSHARE: solve the mixed prompt/delayed banded system per time sample,
     over position-ordered contiguous strip blocks   [scripts 26/27]
  3. re-extract per-strip CFD time + amplitude, recluster (largest cluster/plane)
  4. position estimator: early-charge centroid for short/vertical footprints,
     earliest-strip anchor otherwise (the script-36 'combo' winner)

`sota_hits(...)` returns a DataFrame with the SAME schema as the raw
make_event_displays.load_hits output (eventId, feu, amplitude, time,
x_position_mm, y_position_mm) so it is a drop-in for the display/fit code, plus
an `unshared` flag column and the per-plane cluster id.

Kernel note: the charge-sharing coefficients (c1, c2) are per detector/run and
are MEASURED by script 26. det3 (mx17_3, Saturday run) values are baked in as
SOTA_KERNEL_DET3 (from MICROTPC_RUNBOOK.md); pass `kernel=` for other detectors.
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

# ---- constants (match scripts 26/27) ----
SAMPLE_NS = 60.0
THR_HIT = 100.0          # per-strip amplitude threshold for a hit [ADC]
THR_WF = 150.0           # CFD minimum peak [ADC]
CORE_FRAC = 0.30         # core-strip fraction of the peak for ladder fits
PITCH_MM = 0.78
N_PED_EVENTS = 300
ALPHA = 0.5              # prompt fraction of the sharing kernel (script 27 best)
EARLY_K = 2             # early-charge window [samples] for the centroid estimator
N_SWITCH = 9            # combo: early-charge centroid for n_strips <= N_SWITCH

# det3 (mx17_3, sat_det3 run), measured by 26_unsharing_analysis.py (runbook).
# FEU 7 = X, FEU 8 = Y.  c1 = A(+-1)/A(0), c2 = A(+-2)/A(0).
SOTA_KERNEL_DET3 = {7: (0.449, 0.052), 8: (0.516, 0.151)}


def cfd_time(w):
    """50% constant-fraction leading-edge time of a single-strip waveform [ns]."""
    ipk = int(np.argmax(w))
    a = w[ipk]
    if a < THR_WF or ipk == 0:
        return np.nan
    lvl = 0.5 * a
    for i in range(1, ipk + 1):
        if w[i] >= lvl > w[i - 1]:
            return SAMPLE_NS * (i - 1 + (lvl - w[i - 1]) / (w[i] - w[i - 1]))
    return np.nan


def _nsum(x, k):
    """x[i-k] + x[i+k] along the strip axis."""
    out = np.zeros_like(x)
    out[k:] += x[:-k]
    out[:-k] += x[k:]
    return out


def unshare(wb, c1, c2, alpha=ALPHA):
    """Deconvolve charge sharing over a contiguous strip block.

    Mixed prompt/delayed kernel, causal in time samples (script 27):
      W[:,s] = X[:,s] + a*c1*E1 X[:,s] + (1-a)*c1*E1 X[:,s-1]
                      + a*c2*E2 X[:,s] + (1-a)*c2*E2 X[:,s-2]
    Solved sample-by-sample: a banded solve for the prompt part with an explicit
    RHS from already-unshared earlier samples. wb is (n_strips, n_samples).
    """
    n, ns = wb.shape
    if n < 3:
        return wb
    ab = np.zeros((5, n))
    ab[0, 2:] = alpha * c2
    ab[1, 1:] = alpha * c1
    ab[2, :] = 1.0
    ab[3, :-1] = alpha * c1
    ab[4, :-2] = alpha * c2
    X = np.zeros_like(wb)
    for s in range(ns):
        rhs = wb[:, s].copy()
        if alpha < 1.0:
            if s >= 1:
                rhs -= (1 - alpha) * c1 * _nsum(X[:, s - 1], 1)
            if s >= 2:
                rhs -= (1 - alpha) * c2 * _nsum(X[:, s - 2], 2)
        X[:, s] = solve_banded((2, 2), ab, rhs)
    return X


def _plane_blocks_positions(det, cfg):
    """Per FEU: strip-position array (512) and position-ordered contiguous
    strip blocks (as in script 27 load_events)."""
    plane_of_feu = {cfg.MX17_FEU_X: 'x', cfg.MX17_FEU_Y: 'y'}
    blocks, pos_of = {}, {}
    for feu in cfg.MX17_FEUS:
        pi = 0 if plane_of_feu[feu] == 'x' else 1
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))[pi]
                        for ch in range(512)], dtype=float)
        pos_of[feu] = pos
        ok = np.where(np.isfinite(pos))[0]
        o = ok[np.argsort(pos[ok])]
        brk = np.where(np.diff(pos[o]) > 1.5 * PITCH_MM)[0]
        blocks[feu] = np.split(o, brk + 1)
    return blocks, pos_of, plane_of_feu


def load_waveforms(eids, cfg, det, dec_dir=None):
    """Load pedestal+common-mode-subtracted waveforms for the requested eids.

    Returns (blocks, pos_of, plane_of_feu, wf) where wf[(feu, eid)] is a
    (512, 32) float32 array (strips x samples). Only the requested events are
    kept, so this is cheap for a handful of display events.
    """
    import uproot
    if dec_dir is None:
        dec_dir = os.path.join(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, 'decoded_root')
    want = set(int(e) for e in eids)
    blocks, pos_of, plane_of_feu = _plane_blocks_positions(det, cfg)
    wf = {}
    for feu in cfg.MX17_FEUS:
        for fn in sorted(glob.glob(os.path.join(dec_dir, f'*_{feu:02d}.root'))):
            t = uproot.open(fn)['nt']
            eids_all = t.arrays(['eventId'], library='np')['eventId']
            idx = [i for i in range(len(eids_all)) if int(eids_all[i]) in want]
            if not idx:
                continue
            a0 = t.arrays(['amplitude'], entry_stop=N_PED_EVENTS, library='np')['amplitude']
            ped = np.median(np.stack([a.reshape(32, 512) for a in a0
                                      if a.size == 32 * 512]), axis=(0, 1))
            arr = t.arrays(['eventId', 'amplitude'], library='np')
            for i in idx:
                if arr['amplitude'][i].size != 32 * 512:
                    continue
                w = arr['amplitude'][i].reshape(32, 512).astype(np.float32) - ped
                cms = np.median(w.reshape(32, 8, 64), axis=2)
                w -= np.repeat(cms, 64, axis=1)
                wf[(feu, int(arr['eventId'][i]))] = w.T.astype(np.float32)
    return blocks, pos_of, plane_of_feu, wf


def _largest_cluster(amax, min_strips=3):
    hit = np.where(amax >= THR_HIT)[0]
    if len(hit) < min_strips:
        return None
    brk = np.where(np.diff(hit) > 2)[0]
    groups = [g for g in np.split(hit, brk + 1) if len(g) >= min_strips]
    return max(groups, key=len) if groups else None


def _early_charge_centroid(wu, pos, t0):
    """Early-charge centroid position: amplitude-weighted strip position using
    only the first EARLY_K samples after the cluster-start time (script 36).
    Sub-pitch, anchored near the impact point; best for short footprints."""
    if not np.isfinite(t0):
        return np.nan
    s0 = max(0, int(t0 // SAMPLE_NS))
    qe = np.clip(wu[:, s0:s0 + EARLY_K], 0, None).sum(axis=1)
    return float(np.sum(qe * pos) / qe.sum()) if qe.sum() > THR_HIT else np.nan


def sota_hits(eids, cfg, det, kernel=None, alpha=ALPHA, do_unshare=True,
              dec_dir=None, wf_cache=None):
    """Reconstruct per-strip hits for the given events.

    do_unshare=True  -> full SOTA hits (charge sharing removed)
    do_unshare=False -> raw hits extracted identically from the SAME waveforms
                        (apples-to-apples baseline for comparison)

    Returns a DataFrame with columns:
      eventId, feu, plane, position, x_position_mm, y_position_mm, time,
      amplitude, unshared        (one row per strip in the largest cluster/plane)
    plus a per-event attribute frame is not needed — the position estimator
    (early-charge centroid / earliest-strip anchor) is returned separately by
    sota_track_points().
    """
    if kernel is None:
        kernel = SOTA_KERNEL_DET3
    if wf_cache is None:
        blocks, pos_of, plane_of_feu, wf = load_waveforms(eids, cfg, det, dec_dir)
    else:
        blocks, pos_of, plane_of_feu, wf = wf_cache
    rows = []
    for (feu, eid), w in wf.items():
        c1, c2 = kernel[feu]
        pi_col = 'x_position_mm' if plane_of_feu[feu] == 'x' else 'y_position_mm'
        best = None
        for blk in blocks[feu]:
            wb = w[blk]
            if do_unshare:
                wb = unshare(wb, c1, c2, alpha)
            grp = _largest_cluster(wb.max(axis=1))
            if grp is None:
                continue
            if best is None or len(grp) > len(best[0]):
                best = (grp, blk, wb)
        if best is None:
            continue
        grp, blk, wb = best
        pos = pos_of[feu][blk[grp]]
        amp = wb[grp].max(axis=1)
        tt = np.array([cfd_time(wb[g]) for g in grp])
        for p, a, t in zip(pos, amp, tt):
            if not np.isfinite(t):
                continue
            rows.append(dict(eventId=int(eid), feu=int(feu),
                             plane=plane_of_feu[feu], position=float(p),
                             x_position_mm=float(p) if plane_of_feu[feu] == 'x' else np.nan,
                             y_position_mm=float(p) if plane_of_feu[feu] == 'y' else np.nan,
                             time=float(t), amplitude=float(a),
                             unshared=bool(do_unshare)))
    return pd.DataFrame(rows)


def sota_track_points(eids, cfg, det, kernel=None, alpha=ALPHA, dec_dir=None,
                      wf_cache=None):
    """Per event/plane SOTA position anchor (early-charge centroid with the
    earliest-strip fallback, i.e. the script-36 'combo' estimator) and the
    unshared ladder slope. Returns a DataFrame:
      eventId, plane, pos_combo, pos_earliest, n_strips, slope_ns_per_mm, t_start
    """
    if kernel is None:
        kernel = SOTA_KERNEL_DET3
    if wf_cache is None:
        blocks, pos_of, plane_of_feu, wf = load_waveforms(eids, cfg, det, dec_dir)
    else:
        blocks, pos_of, plane_of_feu, wf = wf_cache
    rows = []
    for (feu, eid), w in wf.items():
        c1, c2 = kernel[feu]
        best = None
        for blk in blocks[feu]:
            wb = unshare(w[blk], c1, c2, alpha)
            grp = _largest_cluster(wb.max(axis=1))
            if grp is None:
                continue
            if best is None or len(grp) > len(best[0]):
                best = (grp, blk, wb)
        if best is None:
            continue
        grp, blk, wb = best
        pos = pos_of[feu][blk[grp]]
        amp = wb[grp].max(axis=1)
        tt = np.array([cfd_time(wb[g]) for g in grp])
        ok = np.isfinite(tt)
        if ok.sum() < 2:
            continue
        t0 = float(np.nanmin(tt[ok]))
        i_lead = int(np.nanargmin(np.where(ok, tt, np.inf)))
        pos_early = _early_charge_centroid(wb[grp], pos, t0)
        n = int(ok.sum())
        pos_combo = pos_early if (n <= N_SWITCH and np.isfinite(pos_early)) else float(pos[i_lead])
        mcore = (amp >= CORE_FRAC * amp.max()) & ok
        slope = np.nan
        if mcore.sum() >= 3 and np.ptp(pos[mcore]) > 0:
            slope = float(np.polyfit(pos[mcore], tt[mcore], 1)[0])
        rows.append(dict(eventId=int(eid), plane=plane_of_feu[feu],
                         pos_combo=float(pos_combo), pos_earliest=float(pos[i_lead]),
                         n_strips=n, slope_ns_per_mm=slope, t_start=t0))
    return pd.DataFrame(rows)
