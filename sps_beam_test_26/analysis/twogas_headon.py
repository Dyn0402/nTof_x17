#!/usr/bin/env python3
"""Head-on two-gas A/B on det4 SPS waveforms. See TWOGAS_HEADON_2026-08-11.md.

Same detector, same live band, same electronics, same drift field (243 V/cm):
  CO2 arm : run56 wf_m70V.npz         (Ar/CO2/iso 95/3/2, resist 590/625 V, ZS 5s)
  CF4 arm : run63 wf_run63_flat.npz   (Ar/CF4/iso 88/10/2, resist 770 V, ZS 4s)
  CF4 RAW : run71 wf_run71_raw_det4only.npz (ZS off; drift 700/450/275 V)

ZS-aware clean builder (the repo's robust_waveforms.build_clean assumes RAW):
  * ZS mode ships only samples above threshold in an amplitude-dependent
    window; amplitudes are already pedestal-subtracted by the decoder.
    -> baseline subtract 0; pile-up gate = "no shipped samples in the 540 ns
       pre-window" (a shipped pre-sample IS above ZS, i.e. pile-up).
    -> tails/undershoot are NOT measurable (negative lobes never ship).
  * RAW mode: per-trace pre-window mean subtraction, |pre| < 15 ADC gate
    (identical to robust_waveforms recipe; reproduces its documented numbers).

Y view only (X view tilt-contaminated). Fiducial: uRWELL ev_pX in the live
bands [145, 220] mm when finite, else X-view leading-trace position.

Usage: python3 twogas_lxplus.py <data_dir> <out_json>
"""
import json
import sys

import numpy as np

from det4_sps_map import POSITION_MM, VIEW, PITCH_MM

SNS = 60.0
NSMP = 64
SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)
BAD_CH = (510, 372)
PRE_SMP = 9
FID_X = (145.0, 220.0)
Q0 = (400.0, 3000.0)
NREL = 12


def build(wf_path, raw_mode):
    Z = np.load(wf_path)
    ev, ch, samp, amp = Z["ev"], Z["ch"], Z["samp"], Z["amp"]
    plateau, pX = Z["ev_plateau"], Z["ev_pX"]
    n_ev = len(plateau)

    key = ev.astype(np.int64) * 512 + ch
    ukey, kinv = np.unique(key, return_inverse=True)
    trace = np.full((len(ukey), NSMP), np.nan, np.float32)
    trace[kinv, samp] = amp
    t_ev = (ukey // 512).astype(np.int64)
    t_ch = (ukey % 512).astype(np.int16)
    t_view = VIEW[t_ch]

    n_pre = np.isfinite(trace[:, :PRE_SMP]).sum(axis=1)
    with np.errstate(invalid="ignore"):
        pre = np.nanmean(trace[:, :PRE_SMP], axis=1)
    if raw_mode:
        trace = trace - np.where(np.isfinite(pre), pre, 0.0)[:, None]
        pre_ok = np.isfinite(pre) & (np.abs(pre) < 15.0)
    else:
        pre_ok = n_pre == 0          # any shipped pre-sample = pile-up
    peak_amp = np.nanmax(np.where(np.isfinite(trace), trace, -np.inf), axis=1)
    peak_smp = np.nanargmax(np.nan_to_num(trace, nan=-1e9), axis=1)

    # leading strip per view per event; then strip offsets about it
    cmap, t_d = {}, np.full(len(ukey), 999, np.int16)
    xlead_pos = np.full(n_ev, np.nan)
    for v in ("x", "y"):
        isv = np.flatnonzero(t_view == v)
        pk = peak_amp[isv].copy()
        pk[~np.isfinite(pk)] = -np.inf
        o = np.lexsort((-pk, t_ev[isv]))
        e_s = t_ev[isv][o]
        f = np.r_[True, e_s[1:] != e_s[:-1]]
        lead = isv[o[f]]
        sidx_of_ev = np.full(n_ev, -999, np.int32)
        sidx_of_ev[t_ev[lead]] = SIDX[t_ch[lead]]
        t_d[isv] = SIDX[t_ch[isv]] - sidx_of_ev[t_ev[isv]]
        if v == "x":
            xlead_pos[t_ev[lead]] = POSITION_MM[t_ch[lead]]
        good = (~np.isin(t_ch[lead], BAD_CH) & pre_ok[lead]
                & (peak_amp[lead] >= Q0[0]) & (peak_amp[lead] <= Q0[1]))
        m = np.full(n_ev, -1, np.int64)
        m[t_ev[lead][good]] = lead[good]
        cmap[v] = m

    fid = np.where(np.isfinite(pX), pX, xlead_pos)
    return dict(trace=trace, t_ev=t_ev, t_ch=t_ch, t_d=t_d, t_view=t_view,
                peak_amp=peak_amp, peak_smp=peak_smp, cmap=cmap,
                plateau=plateau, fid=fid, n_ev=n_ev)


def trim20(vals):
    v = np.sort(vals)
    k = int(len(v) * 0.2)
    core = v[k:len(v) - k] if len(v) > 2 * k else v
    return core.mean() if len(core) else np.nan


def cross(w, level, side, ipk, dt=SNS):
    a = w[ipk]
    y = a * level
    t = np.arange(len(w)) * dt
    fin = np.isfinite(w)
    if side < 0:
        idx = np.where(fin[:ipk + 1] & (w[:ipk + 1] < y))[0]
        if len(idx) == 0:
            return np.nan
        i = idx[-1]
        j = i + 1
        while j <= ipk and not np.isfinite(w[j]):
            j += 1
        if j > ipk or w[j] == w[i]:
            return t[i]
        return t[i] + (t[j] - t[i]) * (y - w[i]) / (w[j] - w[i])
    idx = np.where(fin[ipk:] & (w[ipk:] < y))[0]
    if len(idx) == 0:
        return np.nan
    i = ipk + idx[0]
    j = i - 1
    while j >= ipk and not np.isfinite(w[j]):
        j -= 1
    if j < ipk or w[j] == w[i]:
        return t[i]
    return t[j] + (t[i] - t[j]) * (w[j] - y) / (w[j] - w[i])


def stack_metrics(w, dt=SNS):
    ipk = int(np.nanargmax(np.nan_to_num(w, nan=-1e9)))
    t = np.arange(len(w)) * dt
    r10, r90 = cross(w, .1, -1, ipk), cross(w, .9, -1, ipk)
    return dict(peak_ns=float(t[ipk]),
                on50_ns=float(cross(w, .5, -1, ipk)),
                off50_ns=float(cross(w, .5, +1, ipk)),
                rise1090_ns=float(r90 - r10),
                on50_to_pk_ns=float(t[ipk] - cross(w, .5, -1, ipk)),
                under_frac=float(np.nanmean(w[-4:]) / w[ipk])
                if np.isfinite(w[-4:]).any() else None)


def analyse_arm(C, arm, raw_mode):
    res = {}
    rows = C["cmap"]["y"]
    evs = np.flatnonzero((rows >= 0) & (C["plateau"] == arm))
    f = C["fid"][evs]
    evs = evs[np.isfinite(f) & (f >= FID_X[0]) & (f <= FID_X[1])]
    crows = rows[evs]
    res["n_clean"] = int(len(evs))
    if len(evs) < 50:
        return res

    pk, amp0 = C["peak_smp"][crows], C["peak_amp"][crows]
    tr = C["trace"]
    rel = np.full((len(crows), 2 * NREL + 1), np.nan, np.float32)
    for j, (r, p) in enumerate(zip(crows, pk)):
        lo, hi = p - NREL, p + NREL + 1
        s0, s1 = max(0, lo), min(NSMP, hi)
        rel[j, s0 - lo:s1 - lo] = tr[r, s0:s1] / amp0[j]
    stack = np.array([trim20(rel[:, k][np.isfinite(rel[:, k])])
                      for k in range(rel.shape[1])])
    nshare = np.isfinite(rel).sum(axis=0)
    res["stack_trim20"] = [None if not np.isfinite(v) else round(float(v), 4)
                           for v in stack]
    res["stack_n_per_sample"] = nshare.tolist()
    res["stack_metrics"] = stack_metrics(stack)
    res["amp0_q"] = {q: float(np.percentile(amp0, q)) for q in (10, 50, 90)}

    if raw_mode:
        med_abs = np.nanmedian(tr[crows] / amp0[:, None], axis=0)
        res["absolute_metrics"] = stack_metrics(med_abs)
        res["abs_median_trace"] = [round(float(v), 4) for v in med_abs]

    # per-event rise quantiles (on50->peak, shipped samples only)
    r_ev = []
    for r, p in zip(crows, pk):
        c = cross(tr[r], .5, -1, int(p))
        if np.isfinite(c):
            r_ev.append(p * SNS - c)
    if len(r_ev) > 30:
        res["on50pk_ev_q"] = {q: float(np.percentile(r_ev, q))
                              for q in (5, 25, 50, 75, 95)}

    # neighbours: event-wise peak shift, detection fraction, ratios
    ev_index = {}
    isy = np.flatnonzero(C["t_view"] == "y")
    for d in (-2, -1, 1, 2):
        sel = isy[C["t_d"][isy] == d]
        m = np.full(C["n_ev"], -1, np.int64)
        m[C["t_ev"][sel]] = sel
        ev_index[d] = m
    shifts, detfrac, pk_r, ar_r = {}, {}, {}, {}
    for d in (-2, -1, 1, 2):
        rr = ev_index[d][evs]
        ok = rr >= 0
        det = ok & np.isfinite(C["peak_amp"][rr]) & (C["peak_amp"][rr] > 40)
        detfrac[d] = round(float(det.sum()) / len(evs), 3)
        sh, npr, apr = [], [], []
        for rc, rd, p in zip(crows[det], rr[det], pk[det]):
            idd = int(np.nanargmax(np.nan_to_num(tr[rd], nan=-1e9)))
            sh.append((idd - int(p)) * SNS)
            lo, hi = max(0, int(p) - 6), min(NSMP, int(p) + 7)
            wdw, wcw = tr[rd, lo:hi], tr[rc, lo:hi]
            if np.isfinite(wdw).sum() >= 5:
                npr.append(np.nanmax(wdw) / np.nanmax(wcw))
                apr.append(np.nansum(np.where(np.isfinite(wdw), wdw, 0))
                           / np.nansum(np.where(np.isfinite(wcw), wcw, 0)))
        if len(sh) >= 30:
            shifts[d] = dict(median=float(np.median(sh)), n=len(sh))
            pk_r[d] = float(np.median(npr))
            ar_r[d] = float(np.median(apr))
    res["pm_shift_ns"] = shifts
    res["neigh_detfrac"] = detfrac
    res["peak_ratio_matchedwin"] = pk_r
    res["area_ratio_matchedwin"] = ar_r
    return res


def main():
    data, out = sys.argv[1], sys.argv[2]
    R = {"fiducial_x_mm": FID_X, "q0_adc": Q0, "view": "y",
         "note": "ZS arms: no tails/undershoot (unmeasurable); RAW only"}
    jobs = [
        ("co2_590V", data + "/wf_m70V.npz", False, "590V"),
        ("co2_625V", data + "/wf_m70V.npz", False, "625V"),
        ("cf4_zs_770V", data + "/wf_run63_flat.npz", False, "flat700"),
        ("cf4_raw_d700", data + "/wf_run71_raw_det4only.npz", True, "raw700"),
        ("cf4_raw_d450", data + "/wf_run71_raw_det4only.npz", True, "raw450"),
        ("cf4_raw_d275", data + "/wf_run71_raw_det4only.npz", True, "raw275"),
    ]
    built = {}
    for tag, wf, raw, arm in jobs:
        if wf not in built:
            built[wf] = build(wf, raw)
        R[tag] = analyse_arm(built[wf], arm, raw)
        print(tag, "n =", R[tag].get("n_clean"))
    with open(out, "w") as f:
        json.dump(R, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
