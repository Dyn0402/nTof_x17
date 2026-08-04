#!/usr/bin/env python3
"""Waveform extraction for the kernel fit, using det4 alone.

The sharing kernel is measured from the strip-to-strip pattern WITHIN one
event: the central strip's waveform, its neighbours' waveforms, and the time
structure that separates prompt diffusion from the resistive copy.  None of
that needs an external position reference.  The uRWELL is used elsewhere to
select a band and to prove the mount is flat, but it is not part of the kernel
measurement itself, so dropping it removes a dependency rather than an
ingredient.

Two reasons that matters here:

  * run_71's uRWELL sits on FEU1 and banco's `combined_hits` for it were made
    with the pre-fix decoder AND the ZS analyzer flags, so they are unusable
    until FEU1 is re-decoded.  This path does not wait for that.
  * A det4-only selection and a uRWELL-referenced one have completely
    different failure modes, so agreement between them is a real check.

Selection, all from det4:
  exactly one cluster in each view (a clean single track)
  the X cluster inside a June live band, >= 2 mm from either edge
  not discharge-flagged, central strip below saturation

Emits the same field names as `flat_align_eff.py`, so `kernel_fit_m70V.py`
and `tilt_m70V.py` run on it unchanged.

  python extract_det4_only.py run71_raw [--max-events 60000]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import uproot

import datasets

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM      # noqa: E402

ZS_BASELINE = 256.0
BAND_MARGIN_MM = 2.0
STRIPES = ("/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
           "det4_sps_assessment/stripes_g_det4.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--gate", default="600,3600")
    ap.add_argument("--max-events", type=int, default=60000)
    ap.add_argument("--out", default="")
    ap.add_argument("--cm", default="block", choices=("block", "masked", "none"),
                    help="common-mode handling for RAW: 'block' subtracts the "
                         "per-sample median of each 64-ch connector block "
                         "(biased late in the window, where the dispersed "
                         "charge spreads across the block and the median "
                         "rises with it -- the ZS-study CM signal bias); "
                         "'masked' is the same median but with the strips "
                         "within +-10 of either view's leading cluster (and "
                         "the oscillating channels) excluded, so the signal "
                         "cannot bias it; 'none' leaves CM in entirely -- "
                         "unusable in beam, the CM wanders WITHIN the window "
                         "(10-20x the beam-off sigma, ZS study) so no "
                         "pre-level can absorb it")
    args = ap.parse_args()
    D = datasets.get(args.dataset)
    stage = D["stage"]
    raw = bool(D.get("raw"))
    suff = {"block": "", "masked": "_cmmasked", "none": "_nocm"}[args.cm]
    out = args.out or stage + f"wf_{args.dataset}_det4only{suff}.npz"

    # ---------------------------------------------------- read det4 hits
    ev_off = 0
    H = {k: [] for k in ("ev", "ch", "amp", "t")}
    meta = []          # (subrun, fgroup, ev_id, t_wall) per event
    for sub, stem, t0, idxs in D["subruns"]:
        d = os.path.join(stage, sub) if os.path.isdir(os.path.join(stage, sub)) \
            else stage
        for i in idxs:
            hf = os.path.join(d, f"hits_{sub}_{i}_03.root")
            if not os.path.exists(hf):
                continue
            b = uproot.open(hf + ":hits").arrays(
                ["eventId", "channel", "amplitude", "time",
                 "trigger_timestamp_ns"], library="np")
            ids = np.unique(b["eventId"])
            idx = np.searchsorted(ids, b["eventId"])
            ts = np.zeros(len(ids), np.int64)
            ts[idx] = b["trigger_timestamp_ns"]
            H["ev"].append(idx + ev_off)
            H["ch"].append(b["channel"].astype(np.int16))
            H["amp"].append(np.abs(b["amplitude"]).astype(np.float32))
            H["t"].append(b["time"].astype(np.float32))
            meta.append((np.full(len(ids), sub, dtype="<U24"),
                         np.full(len(ids), i, dtype="<U4"),
                         ids.astype(np.int64),
                         t0 + ts.astype(float) / 1e9))
            ev_off += len(ids)
            print(f"  {sub}/{i}: {len(ids)} events, {len(b['eventId'])} hits")

    h_ev = np.concatenate(H["ev"]); h_ch = np.concatenate(H["ch"])
    h_amp = np.concatenate(H["amp"]); h_t = np.concatenate(H["t"])
    sub_a = np.concatenate([m[0] for m in meta])
    fg_a = np.concatenate([m[1] for m in meta])
    evid = np.concatenate([m[2] for m in meta])
    t_wall = np.concatenate([m[3] for m in meta])
    n_ev = ev_off
    plateau = datasets.plateau_of(args.dataset, t_wall)
    print(f"\n{n_ev} events, {len(h_ev)} det4 hits")

    # ------------------------------------------------ det4-only clustering
    g0, g1 = (float(v) for v in args.gate.split(","))
    sel = (h_t > g0) & (h_t < g1)
    lead, ncl, nst, qmax = {}, {}, {}, {}
    for v in ("x", "y"):
        k = sel & (VIEW[h_ch] == v)
        ev, pos, amp = h_ev[k], POSITION_MM[h_ch[k]], h_amp[k]
        L = np.full(n_ev, np.nan); N = np.zeros(n_ev, np.int32)
        S = np.zeros(n_ev, np.int32); Q = np.zeros(n_ev)
        o = np.lexsort((pos, ev))
        ev, pos, amp = ev[o], pos[o], amp[o]
        new = np.empty(len(ev), bool); new[0] = True
        new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > 3.0)
        cid = np.cumsum(new) - 1
        nc = cid[-1] + 1
        cq = np.bincount(cid, weights=amp, minlength=nc)
        cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
        cn = np.bincount(cid, minlength=nc)
        cmx = np.zeros(nc); np.maximum.at(cmx, cid, amp)
        cev = np.zeros(nc, np.int64); cev[cid] = ev
        np.add.at(N, cev, 1)
        o2 = np.argsort(cq, kind="stable")
        L[cev[o2]] = cp[o2]; S[cev[o2]] = cn[o2]; Q[cev[o2]] = cmx[o2]
        lead[v], ncl[v], nst[v], qmax[v] = L, N, S, Q

    J = np.load(STRIPES)
    inband = np.zeros(n_ev, bool)
    for lo, hi in J["bands"]:
        inband |= (lead["x"] > lo + BAND_MARGIN_MM) & (lead["x"] < hi - BAND_MARGIN_MM)

    single = (ncl["x"] == 1) & (ncl["y"] == 1)
    disch = (nst["x"] > 40) | (nst["y"] > 40)
    unsat = (qmax["x"] < 3400) & (qmax["y"] < 3400)
    good = single & inband & ~disch & unsat & np.isfinite(lead["x"] + lead["y"])
    good &= plateau != ""
    print(f"selection: single-cluster {single.sum()} -> in-band {(single&inband).sum()}"
          f" -> clean {good.sum()}")
    for lab, *_ in D["plateaus"]:
        print(f"   {lab:>8} {int((good & (plateau == lab)).sum()):8d}")

    idx = np.flatnonzero(good)
    per = {}
    labs = [l for l, *_ in D["plateaus"] if (good & (plateau == l)).sum() > 100]
    for lab in labs:
        k = idx[plateau[idx] == lab]
        cap = args.max_events // max(len(labs), 1)
        if len(k) > cap:
            k = np.random.default_rng(0).choice(k, cap, replace=False)
        per[lab] = np.sort(k)
    idx = np.sort(np.concatenate([v for v in per.values() if len(v)]))
    print(f"extracting {len(idx)} events")

    # ------------------------------------------------------- waveforms
    #
    # RAW runs were taken with on-FEU pedestal subtraction AND common-mode
    # subtraction switched OFF, so the decoded `nt` waveforms are raw ADC
    # sitting on per-channel baselines (median 619, spread 344-2947) with the
    # coherent common mode still on top (raw per-channel RMS 297 ADC against
    # 10.4 after CNS).  The analyzer does both corrections internally but only
    # emits hits, so they have to be redone here or every waveform is garbage.
    #
    # Order matters and matches the analyzer: subtract the per-channel pedestal
    # mean first, then the per-sample median across each 64-channel connector
    # block.  Missing samples (the FEU's dropped packets) stay NaN so they are
    # excluded from both the median and the downstream average, rather than
    # being counted as zero.
    NCH, NSMP = 512, int(D["n_samples"])
    ped_mean = np.zeros(NCH, np.float32)
    if raw:
        pth = None
        for sub, _st, _t0, idxs in D["subruns"]:
            dd = os.path.join(stage, sub) if os.path.isdir(os.path.join(stage, sub)) else stage
            for i in idxs:
                cand = os.path.join(dd, f"hits_{sub}_{i}_03.root")
                if os.path.exists(cand):
                    pth = cand
                    break
            if pth:
                break
        P = uproot.open(pth + ":pedestals").arrays(["channel", "mean"], library="np")
        ped_mean[P["channel"].astype(int)] = P["mean"]
        print(f"pedestal means from {os.path.basename(pth)}: "
              f"median {np.median(ped_mean):.1f} ADC, "
              f"range {ped_mean.min():.0f}-{ped_mean.max():.0f}")

    KEEP = 4          # strips either side of the leading strip, per view
    SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)

    CM_MASK_HALF = 10                 # strips either side of a lead to exclude
    CM_BAD_CH = (510, 372)            # oscillating channels, never in the CM

    def correct_chunk(evs, chs, sms, ams):
        """(n_ev,512,nsamp) pedestal- and common-noise-subtracted, NaN = absent."""
        n = len(evs)
        grid = np.full((n, NCH, NSMP), np.nan, np.float32)
        for j in range(n):
            c, sm, am = chs[j], sms[j], ams[j]
            m = (sm >= 0) & (sm < NSMP)
            grid[j, c[m], sm[m]] = am[m] - ped_mean[c[m]]
        if raw and args.cm in ("block", "masked"):
            g = grid.reshape(n, NCH // 64, 64, NSMP)
            if args.cm == "masked":
                gm = grid.copy()
                gm[:, list(CM_BAD_CH), :] = np.nan
                for j, gev in enumerate(evs):
                    for v in ("x", "y"):
                        if not np.isfinite(lead[v][gev]):
                            continue
                        s0 = int(round(lead[v][gev] / PITCH_MM))
                        sig = (VIEW == v) & (np.abs(SIDX - s0) <= CM_MASK_HALF)
                        gm[j, sig, :] = np.nan
                med = np.nanmedian(gm.reshape(n, NCH // 64, 64, NSMP),
                                   axis=2, keepdims=True)
            else:
                med = np.nanmedian(g, axis=2, keepdims=True)
            g -= np.nan_to_num(med)
            grid = g.reshape(n, NCH, NSMP)
        return grid

    recs = {k: [] for k in ("ev", "ch", "samp", "amp")}
    for sub, stem, _t0, idxs in D["subruns"]:
        d = os.path.join(stage, sub) if os.path.isdir(os.path.join(stage, sub)) \
            else stage
        for i in idxs:
            m = idx[(sub_a[idx] == sub) & (fg_a[idx] == i)]
            if not len(m):
                continue
            fn = os.path.join(d, f"dec_{sub}_{i}_03.root")
            if not os.path.exists(fn):
                continue
            T = uproot.open(fn)["nt"]
            nt_ev = T["eventId"].array(library="np").astype(np.int64)
            order = np.argsort(nt_ev)
            jj = np.clip(np.searchsorted(nt_ev, evid[m], sorter=order),
                         0, len(nt_ev) - 1)
            entry = order[jj]
            ok = nt_ev[entry] == evid[m]
            entry, mm = entry[ok], m[ok]
            want = np.zeros(T.num_entries, bool); want[entry] = True
            pos_of = np.full(T.num_entries, -1, np.int64); pos_of[entry] = mm
            print(f"  {sub}/{i}: {len(entry)} events")
            step = 20000
            for s in range(0, T.num_entries, step):
                e = min(s + step, T.num_entries)
                if not want[s:e].any():
                    continue
                a = T.arrays(["sample", "channel", "amplitude"],
                             entry_start=s, entry_stop=e, library="np")
                qs = np.flatnonzero(want[s:e])
                for c0 in range(0, len(qs), 400):
                    blk = qs[c0:c0 + 400]
                    evs = pos_of[s + blk]
                    grid = correct_chunk(evs, [a["channel"][q] for q in blk],
                                         [a["sample"][q] for q in blk],
                                         [a["amplitude"][q] for q in blk])
                    for jj, gev in enumerate(evs):
                        for v in ("x", "y"):
                            if not np.isfinite(lead[v][gev]):
                                continue
                            s0 = int(round(lead[v][gev] / PITCH_MM))
                            want_ch = np.flatnonzero(
                                (VIEW == v) & (np.abs(SIDX - s0) <= KEEP))
                            if not len(want_ch):
                                continue
                            sub_g = grid[jj, want_ch, :]
                            ok = np.isfinite(sub_g)
                            ci, si = np.nonzero(ok)
                            if not len(ci):
                                continue
                            recs["ev"].append(np.full(len(ci), gev, np.int64))
                            recs["ch"].append(want_ch[ci].astype(np.int16))
                            recs["samp"].append(si.astype(np.int16))
                            recs["amp"].append(sub_g[ci, si].astype(np.float32))

    o = {k: np.concatenate(v) for k, v in recs.items()}
    # RAW runs were taken with on-FEU pedestal subtraction OFF, so the decoded
    # amplitudes sit on the raw per-channel baselines and the flat 256 offset
    # used for ZS data does not apply.  The analyzer subtracts the pedestal for
    # the hits tree; here the waveform baseline is removed per strip in
    # per_strip()'s local baseline, so leave the raw values alone.
    if not raw:
        o["amp"] = o["amp"] - ZS_BASELINE
    o["ev_pX"], o["ev_pY"] = lead["x"], lead["y"]
    o["ev_plateau"] = plateau
    o["ev_t_wall"] = t_wall
    o["ev_resid"] = np.zeros(n_ev)
    np.savez_compressed(out, **o)
    print(f"wrote {out}: {len(o['ev'])} sample records over "
          f"{len(np.unique(o['ev']))} events")


if __name__ == "__main__":
    main()
