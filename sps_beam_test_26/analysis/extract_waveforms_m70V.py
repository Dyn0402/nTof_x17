#!/usr/bin/env python3
"""Pull det4 waveform windows for the events that can measure charge spreading.

Why this sub-run.  det4 is **perpendicular** to the beam here, so the whole
ionisation column lands at one transverse position: w = 0 in the wft forward
model.  That removes the track-angle/kernel degeneracy that a cosmic fit has to
carry.  The uRWELL telescope then supplies the transverse position externally
(median |residual| 0.51 mm), so the cluster's own centroid never has to be
trusted either.  What is left in the strip-to-strip pattern is diffusion plus
the resistive kernel -- and those two are separated by TIME, because the
directly-diffused charge is prompt while the shared copies are delayed.

What the zero suppression allows, measured (see zs_diag in the session log):

  * ZS is per channel at 5 sigma (~41 ADC), TPC mode.  A strip below that is
    absent, so the +-2 tail is censored: present 29-47% of the time, and
    beyond |d|=3 what is left is the accidental floor.  Handled downstream by
    the detection-fraction-vs-amplitude curve, not by pretending it is not
    there.
  * The kept sample window grows with amplitude: 5 samples (300 ns) at
    threshold, 17-25 samples (1.0-1.5 us) above ~100 ADC.  So the central
    strip carries a real shape and the weak neighbours carry a peak time.

Selection is deliberately tight, because this is a shape measurement and every
loose event is a background:

  clean single-cluster uRWELL track in all four planes
  predicted position inside a June live band, >= 2 mm from either band edge
  det4 reconstructed and within 5 mm of the track
  not discharge-flagged

  python extract_waveforms_m70V.py [--max-events 60000]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import uproot

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM      # noqa: E402

STAGE = "/media/dylan/data/x17/sps_run53_det4_check/staging/run_56_m70V/"
Z_BACK, Z_DET4 = 1370.0, 1120.0
ZS_BASELINE = 256.0
BAND_MARGIN_MM = 2.0

SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)


def _s(h, m, sec):
    return h * 3600 + m * 60 + sec


PLATEAUS = [("590V", _s(15, 47, 25), _s(15, 52, 50)),
            ("625V", _s(15, 52, 57), _s(15, 59, 34))]


def build_selection(D):
    """Clean tracks, alignment, and the per-event det4 cluster summary."""
    n_ev = len(D["ev_id"])
    clean = np.ones(n_ev, bool)
    for k in ("fx", "fy", "bx", "by"):
        clean &= (D[k + "_n"] == 1) & np.isfinite(D[k + "_p"])
    f = Z_DET4 / Z_BACK
    tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * f
    ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * f
    clean &= np.isfinite(tx + ty)

    h_ev, h_ch, h_amp, h_t = D["h_ev"], D["h_ch"], D["h_amp"], D["h_time"]
    GATE = (h_t > 600) & (h_t < 3600)
    sel = GATE & clean[h_ev]

    lead, ncl, nst = {}, {}, {}
    for v in ("x", "y"):
        k = sel & (VIEW[h_ch] == v)
        ev, pos, amp = h_ev[k], POSITION_MM[h_ch[k]], h_amp[k]
        L = np.full(n_ev, np.nan)
        N = np.zeros(n_ev, np.int32)
        S = np.zeros(n_ev, np.int32)
        o = np.lexsort((pos, ev))
        ev, pos, amp = ev[o], pos[o], amp[o]
        new = np.empty(len(ev), bool)
        new[0] = True
        new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > 3.0)
        cid = np.cumsum(new) - 1
        nc = cid[-1] + 1
        cq = np.bincount(cid, weights=amp, minlength=nc)
        cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
        cn = np.bincount(cid, minlength=nc)
        cev = np.zeros(nc, np.int64)
        cev[cid] = ev
        np.add.at(N, cev, 1)
        o2 = np.argsort(cq, kind="stable")
        L[cev[o2]] = cp[o2]
        S[cev[o2]] = cn[o2]
        lead[v], ncl[v], nst[v] = L, N, S

    reco = np.isfinite(lead["x"] + lead["y"])
    disch = (ncl["x"] + ncl["y"] >= 6) | (nst["x"] > 40) | (nst["y"] > 40)

    keep = clean & reco & ~disch
    U = np.column_stack([tx[keep], ty[keep], np.ones(keep.sum())])
    V = np.column_stack([lead["x"][keep], lead["y"][keep]])
    A, *_ = np.linalg.lstsq(U, V, rcond=None)
    for _ in range(5):
        r = np.hypot(*(V - U @ A).T)
        g = r < np.percentile(r, 80)
        A, *_ = np.linalg.lstsq(U[g], V[g], rcond=None)
    pred = np.column_stack([tx, ty, np.ones(n_ev)]) @ A
    pX, pY = pred[:, 0], pred[:, 1]
    resid = np.hypot(lead["x"] - pX, lead["y"] - pY)

    J = np.load("/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                "det4_sps_assessment/stripes_g_det4.npz")
    inband = np.zeros(n_ev, bool)
    for lo, hi in J["bands"]:
        inband |= (pX > lo + BAND_MARGIN_MM) & (pX < hi - BAND_MARGIN_MM)

    good = clean & reco & ~disch & inband & (resid < 5.0)
    print(f"selection: {clean.sum()} clean -> {(clean & reco).sum()} reco -> "
          f"{good.sum()} in-band, aligned, non-discharge")
    return dict(good=good, pX=pX, pY=pY, lead=lead, A=A, resid=resid, n_ev=n_ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=STAGE + "pair_m70V.npz")
    ap.add_argument("--max-events", type=int, default=60000)
    ap.add_argument("--out", default=STAGE + "wf_m70V.npz")
    args = ap.parse_args()

    D = np.load(args.npz)
    S = build_selection(D)
    good, n_ev = S["good"], S["n_ev"]
    t = D["ev_t_wall"]
    ev_id, file_idx = D["ev_id"], D["file_idx"]

    plateau = np.full(n_ev, "", dtype="<U12")
    for label, t0, t1 in PLATEAUS:
        plateau[(t >= t0) & (t < t1)] = label
    good = good & (plateau != "")

    idx = np.flatnonzero(good)
    if len(idx) > args.max_events:
        idx = np.random.default_rng(0).choice(idx, args.max_events, replace=False)
        idx.sort()
    print(f"extracting waveforms for {len(idx)} events")

    recs = {k: [] for k in ("ev", "ch", "samp", "amp")}
    for fi in np.unique(file_idx[idx]):
        m = idx[file_idx[idx] == fi]
        T = uproot.open(f"{STAGE}dec_{fi:03d}_03.root")["nt"]
        # eventId is a DAQ-global counter that runs on across files, while the
        # nt entry index restarts at 0 in each one -- so map through the tree's
        # own eventId branch rather than assuming entry = eventId - 1.
        nt_ev = T["eventId"].array(library="np").astype(np.int64)
        order = np.argsort(nt_ev)
        j = np.searchsorted(nt_ev, ev_id[m], sorter=order)
        j = np.clip(j, 0, len(nt_ev) - 1)
        entry = order[j]
        ok = nt_ev[entry] == ev_id[m]
        if not ok.all():
            print(f"    {(~ok).sum()} of {len(ok)} events not in this nt tree, dropped")
        entry, m = entry[ok], m[ok]
        print(f"  file {fi:03d}: {len(entry)} events, "
              f"nt entries {entry.min()}-{entry.max()}")
        want = np.zeros(T.num_entries, bool)
        want[entry] = True
        pos_of = np.full(T.num_entries, -1, np.int64)
        pos_of[entry] = m                        # back-reference into pair npz
        step = 50000
        for lo in range(0, T.num_entries, step):
            hi = min(lo + step, T.num_entries)
            if not want[lo:hi].any():
                continue
            a = T.arrays(["sample", "channel", "amplitude"],
                         entry_start=lo, entry_stop=hi, library="np")
            for j in np.flatnonzero(want[lo:hi]):
                ch = a["channel"][j]
                if len(ch) == 0:
                    continue
                recs["ev"].append(np.full(len(ch), pos_of[lo + j], np.int64))
                recs["ch"].append(ch.astype(np.int16))
                recs["samp"].append(a["sample"][j].astype(np.int16))
                recs["amp"].append(a["amplitude"][j].astype(np.float32))
            print(f"    {lo}-{hi}   ", end="\r")
    print()

    out = {k: np.concatenate(v) for k, v in recs.items()}
    out["amp"] = out["amp"] - ZS_BASELINE
    for k in ("pX", "pY", "resid"):
        out["ev_" + k] = S[k]
    out["ev_leadx"] = S["lead"]["x"]
    out["ev_leady"] = S["lead"]["y"]
    out["ev_plateau"] = plateau
    out["ev_t_wall"] = t
    out["A"] = S["A"]
    out["sel_idx"] = idx
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {len(out['ev'])} sample records over "
          f"{len(np.unique(out['ev']))} events")


if __name__ == "__main__":
    main()
