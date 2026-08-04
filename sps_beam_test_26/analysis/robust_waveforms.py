#!/usr/bin/env python3
"""Clean per-sample response library for run_71 RAW — the artefact-free stack.

Recreates (in the repo this time) the reanalysis-2026-08-04 selection whose
scripts previously lived only on the campaign machine's data disk.  See
RAW_RUN71_REANALYSIS_2026-08-04.md §1: peak-aligned means with no per-event
baseline inspection sat at 22–31 % of peak *before the pulse existed*, from
three stacked artefacts.  This applies the fixes and emits the clean median /
trimmed-mean W_d(t) library in ABSOLUTE window time:

  1. drop events whose central strip is an oscillating channel (ch 510 / 372
     swing ±400–900 ADC with quiet neighbours; ch 510 alone was the "central
     strip" of 22 % of selected events),
  2. require |pre-window mean| < 15 ADC on the central strip (rejects beam
     pile-up sitting in the 540 ns before the trigger pulse),
  3. subtract each strip's own pre-window level per event,
  4. aggregate with the per-sample median and 20 %-trimmed mean over the
     samples that were actually shipped.  The FEU packet loss is uniform and
     amplitude-independent, so order statistics over present samples are
     unbiased and need NO acceptance division (unlike sums normalised by the
     event count, which is where the ~24 % bias came from).

`build_clean()` is importable — tilt_clean.py runs on the identical selection.

  python robust_waveforms.py run71_raw [--q0 400,3000] [--out DIR]

Writes <out>/robust_library_<dataset>.npz with, per (plateau, view, d):
  med_<p>_<v>_<d>, trim_<p>_<v>_<d>, n_<p>_<v>_<d>  (nsmp arrays / counts)
plus per-event central peak stats and the event-wise ±1 peak-time shifts.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "det4_sps_assessment"))
import datasets                                          # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM     # noqa: E402

SNS = 60.0
SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)
BAD_CH = (510, 372)          # oscillating channels, both Y view
PRE_SMP = 9                  # samples 0..8 = the 540 ns pre-window
PRE_GATE_ADC = 15.0          # |pre-window mean| gate on the central strip
TRIM = 0.20                  # 20 %-trimmed mean


class CleanTraces:
    """Per-(event,strip) baseline-subtracted traces with the clean gates.

    trace   (n_traces, nsmp) float32, NaN = sample never shipped by the FEU
    t_ev/t_ch/t_d/t_view    event index / channel / strip offset / view
    pre     the subtracted pre-window level per trace
    peak_amp/peak_smp       per-trace peak (post-subtraction) and its sample
    cmap[v] event -> central-trace row (or -1 when the event fails a gate)
    """


def build_clean(wf_path, D, q0lo=400.0, q0hi=3000.0, verbose=True):
    nsmp = int(D["n_samples"])
    Z = np.load(wf_path)
    ev, ch, samp, amp = Z["ev"], Z["ch"], Z["samp"], Z["amp"]
    C = CleanTraces()
    C.plateau, pX, pY = Z["ev_plateau"], Z["ev_pX"], Z["ev_pY"]
    C.n_ev = len(C.plateau)
    C.nsmp = nsmp
    if verbose:
        print(f"{wf_path}: {len(ev)} sample records, "
              f"{len(np.unique(ev))} events")

    # Central channel per event per view, from the leading-cluster position.
    cen_ch = {}
    for v, pos in (("x", pX), ("y", pY)):
        s0 = np.where(np.isfinite(pos), np.round(pos / PITCH_MM), -999).astype(int)
        lut = np.full(SIDX.max() + 2, -1, np.int32)
        for c in range(len(SIDX)):
            if VIEW[c] == v:
                lut[SIDX[c]] = c
        cen_ch[v] = np.where((s0 >= 0) & (s0 <= SIDX.max()),
                             lut[np.clip(s0, 0, SIDX.max())], -1)

    d_of = np.full(len(ev), 999, np.int16)
    for v in ("x", "y"):
        m = VIEW[ch] == v
        cc = cen_ch[v][ev[m]]
        ok = cc >= 0
        d_of[np.flatnonzero(m)[ok]] = SIDX[ch[m][ok]] - SIDX[cc[ok]]

    # scatter the records into one (event,strip) x sample matrix
    key = ev.astype(np.int64) * 512 + ch
    ukey, kinv = np.unique(key, return_inverse=True)
    trace = np.full((len(ukey), nsmp), np.nan, np.float32)
    trace[kinv, samp] = amp
    C.t_ev = (ukey // 512).astype(np.int64)
    C.t_ch = (ukey % 512).astype(np.int16)
    first = np.zeros(len(ukey), np.int64)
    first[kinv] = np.arange(len(key))
    C.t_d = d_of[first]
    C.t_view = VIEW[C.t_ch]

    with np.errstate(invalid="ignore"):
        C.pre = np.nanmean(trace[:, :PRE_SMP], axis=1)
    trace -= C.pre[:, None]
    C.trace = trace
    C.peak_amp = np.nanmax(np.where(np.isfinite(trace), trace, -np.inf), axis=1)
    C.peak_smp = np.nanargmax(np.nan_to_num(trace, nan=-1e9), axis=1)

    # The central strip is the LEADING strip — the view's max-amplitude trace —
    # not the strip under the rounded cluster centroid.  When the charge
    # shares comparably across two strips the centroid rounds to the smaller
    # one, and the true maximum then sits at "d=±1": the first pass of this
    # script did exactly that and inflated the ±1 medians to 0.44 of central
    # (the documented value is 0.16–0.19).  Offsets are re-derived from the
    # leading strip; the extraction stores ±4 strips about the centroid, so
    # re-centring keeps at least ±3 coverage.
    C.cmap = {}
    for v in ("x", "y"):
        isv = np.flatnonzero(C.t_view == v)
        pk = C.peak_amp[isv].copy()
        pk[~np.isfinite(pk)] = -np.inf
        o = np.lexsort((-pk, C.t_ev[isv]))
        e_s = C.t_ev[isv][o]
        f = np.r_[True, e_s[1:] != e_s[:-1]]
        lead_rows = isv[o[f]]                       # leading trace per event
        lead_ev = C.t_ev[lead_rows]
        lead_sidx = SIDX[C.t_ch[lead_rows]]
        # re-centre this view's offsets on the leading strip
        sidx_of_ev = np.full(C.n_ev, -999, np.int32)
        sidx_of_ev[lead_ev] = lead_sidx
        C.t_d[isv] = SIDX[C.t_ch[isv]] - sidx_of_ev[C.t_ev[isv]]

        cen_bad = np.isin(C.t_ch[lead_rows], BAD_CH)
        cen_pre_ok = np.abs(C.pre[lead_rows]) < PRE_GATE_ADC
        cen_q = C.peak_amp[lead_rows]
        cen_qok = (cen_q >= q0lo) & (cen_q <= q0hi)
        good = ~cen_bad & cen_pre_ok & cen_qok & np.isfinite(C.pre[lead_rows])
        cmap = np.full(C.n_ev, -1, np.int64)
        cmap[lead_ev[good]] = lead_rows[good]
        C.cmap[v] = cmap
        if verbose:
            print(f"view {v.upper()}: {len(lead_rows)} leading traces -> "
                  f"{good.sum()} clean "
                  f"(bad-ch {cen_bad.sum()}, pre-gate {(~cen_pre_ok).sum()}, "
                  f"q0 {(~cen_qok).sum()})")
    return C


def trimmed_mean(vals, frac=TRIM):
    if len(vals) == 0:
        return np.nan
    v = np.sort(vals)
    k = int(len(v) * frac)
    core = v[k:len(v) - k] if len(v) > 2 * k else v
    return core.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--wf", default="")
    ap.add_argument("--q0", default="400,3000",
                    help="central-strip peak window, ADC (after baseline)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    D = datasets.get(args.dataset)
    wf = args.wf or D["stage"] + f"wf_{args.dataset}_det4only.npz"
    outdir = args.out or D["stage"] + "reanalysis_clean/"
    os.makedirs(outdir, exist_ok=True)
    q0lo, q0hi = (float(x) for x in args.q0.split(","))

    C = build_clean(wf, D, q0lo, q0hi)
    nsmp = C.nsmp

    # Two aggregations, two uses.  In ABSOLUTE window time (med_/trim_) the
    # central peak is washed out — at low drift field each event peaks
    # wherever a large ionisation cluster lands on the ladder — so every
    # per-event trace is first normalised by its event's central peak, which
    # is what makes "returns to < 5 % of peak" and "last-4 % of peak"
    # meaningful.  PEAK-ALIGNED (alm_/altr_) traces, same normalisation, are
    # the basis for the kernel fit and the charge budget, where the shape
    # about the central maximum is what matters.
    NREL = 30
    results, meta = {}, []
    for v in ("x", "y"):
        isv = C.t_view == v
        cmap = C.cmap[v]
        for lab, *_ in D["plateaus"]:
            evsel = np.flatnonzero((C.plateau == lab) & (cmap >= 0))
            if len(evsel) < 50:
                continue
            keep = np.zeros(C.n_ev, bool)
            keep[evsel] = True
            meta.append((lab, v, len(evsel)))
            for dd in (0, 1, -1, 2, -2, 3, -3):
                sel = isv & (C.t_d == dd) & keep[C.t_ev]
                if sel.sum() < 50:
                    continue
                cen_rows = cmap[C.t_ev[sel]]
                q0_ev = C.peak_amp[cen_rows]
                pk_ev = C.peak_smp[cen_rows]
                tr = C.trace[sel] / q0_ev[:, None]
                med = np.full(nsmp, np.nan)
                trm = np.full(nsmp, np.nan)
                nn = np.zeros(nsmp, np.int32)
                for s in range(nsmp):
                    col = tr[:, s]
                    col = col[np.isfinite(col)]
                    nn[s] = len(col)
                    if len(col):
                        med[s] = np.median(col)
                        trm[s] = trimmed_mean(col)
                tag = f"{lab}_{v}_{dd:+d}"
                results[f"med_{tag}"] = med
                results[f"trim_{tag}"] = trm
                results[f"n_{tag}"] = nn
                # peak-aligned: row j's rel window is samples pk_ev[j]-NREL..+NREL
                nal = 2 * NREL + 1
                cols = pk_ev[:, None] + (np.arange(nal) - NREL)[None, :]
                ok = (cols >= 0) & (cols < nsmp)
                al = np.full((len(tr), nal), np.nan, np.float32)
                rows = np.broadcast_to(np.arange(len(tr))[:, None], cols.shape)
                al[ok] = tr[rows[ok], np.clip(cols, 0, nsmp - 1)[ok]]
                alm = np.full(nal, np.nan)
                alt = np.full(nal, np.nan)
                for s in range(nal):
                    col = al[:, s]
                    col = col[np.isfinite(col)]
                    if len(col):
                        alm[s] = np.median(col)
                        alt[s] = trimmed_mean(col)
                results[f"alm_{tag}"] = alm
                results[f"altr_{tag}"] = alt
            # event-wise +-1 peak-time shift on clean events
            for dd in (1, -1):
                sel = isv & (C.t_d == dd) & keep[C.t_ev] & (cmap[C.t_ev] >= 0)
                nb_ev = C.t_ev[sel]
                shift = (C.peak_smp[sel].astype(float)
                         - C.peak_smp[cmap[nb_ev]]) * SNS
                results[f"dtpk_{lab}_{v}_{dd:+d}"] = shift
            c_idx = cmap[evsel]
            results[f"cpk_{lab}_{v}"] = C.peak_amp[c_idx]
            results[f"cpksmp_{lab}_{v}"] = C.peak_smp[c_idx] * SNS

    out = os.path.join(outdir, f"robust_library_{args.dataset}.npz")
    np.savez_compressed(out, t=np.arange(nsmp) * SNS,
                        t_rel=(np.arange(2 * NREL + 1) - NREL) * SNS,
                        **{k: np.asarray(vv) for k, vv in results.items()})
    print(f"\nwrote {out}")

    print(f"\n{'plateau':>8} {'view':>4} {'events':>7}   "
          f"{'peak[ns]':>8} {'ret<5%[ns]':>10} {'last4[%pk]':>10}")
    for lab, v, n in meta:
        med = results.get(f"med_{lab}_{v}_+0")
        if med is None:
            continue
        pk = np.nanargmax(med)
        pkv = med[pk]
        below = np.flatnonzero((np.arange(nsmp) > pk) & (med < 0.05 * pkv))
        ret = below[0] * SNS if len(below) else np.nan
        last4 = 100 * np.nanmean(med[-4:]) / pkv
        print(f"{lab:>8} {v:>4} {n:>7}   {pk * SNS:8.0f} {ret:10.0f} "
              f"{last4:10.1f}")
    for lab, v, n in meta:
        if v != "y":
            continue
        sh = np.concatenate([results.get(f"dtpk_{lab}_{v}_{dd:+d}",
                                         np.array([])) for dd in (1, -1)])
        if len(sh):
            print(f"  event-wise +-1 peak shift, {lab} Y: "
                  f"median {np.median(sh):+.0f} ns (n={len(sh)})")


if __name__ == "__main__":
    main()
