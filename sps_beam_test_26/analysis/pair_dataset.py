#!/usr/bin/env python3
"""Pair det4 (FEU 3) against the uRWELL reference, for a flat-mount dataset.

Generalises `pair_m70V.py` over `datasets.py`, so run_56 and run_63 go through
one code path and the HV-plateau windows are stated in exactly one place.

Wall clock: each sub-run is anchored on its own `dream_daq.log` start time and
the DAQ trigger timestamp is used only for the offset WITHIN that sub-run.
That is more accurate than stitching whole files end to end, and it matters
here because run_63's drift plateaus are 8 minutes long and two of them
straddle a sub-run boundary.

  python pair_dataset.py run63_operating
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import uproot

import datasets

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM      # noqa: E402

URW_MAP = ("/media/dylan/data/x17/sps_run53_det4_check/"
           "flat_ArCO2iso_95-3-2__run53-56/urw_mapping/mapping_urwell.csv")


def urwell_map():
    view = np.full(512, "", dtype="<U1")
    pos = np.full(512, np.nan)
    det = np.full(512, "", dtype="<U1")
    with open(URW_MAP) as f:
        for row in csv.DictReader(f):
            c = int(row["channel"])
            view[c] = row["view"]
            pos[c] = float(row["position_mm"])
            det[c] = "f" if row["detector"] == "EIC_uRWELL_front" else "b"
    return view, pos, det


def clusters(ev, pos, amp, n_ev, gap=3.0):
    lead = np.full(n_ev, np.nan)
    ncl = np.zeros(n_ev, np.int16)
    if not len(ev):
        return lead, ncl
    o = np.lexsort((pos, ev))
    ev, pos, amp = ev[o], pos[o], amp[o]
    new = np.empty(len(ev), bool)
    new[0] = True
    new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > gap)
    cid = np.cumsum(new) - 1
    nc = cid[-1] + 1
    cq = np.bincount(cid, weights=amp, minlength=nc)
    cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
    cev = np.zeros(nc, np.int64)
    cev[cid] = ev
    np.add.at(ncl, cev, 1)
    o2 = np.argsort(cq, kind="stable")
    lead[cev[o2]] = cp[o2]
    return lead, ncl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    D = datasets.get(args.dataset)
    stage = D["stage"]
    out = args.out or stage + f"pair_{args.dataset}.npz"
    uv, up, ud = urwell_map()

    parts, off = [], 0
    for sub, stem, t0, idxs in D["subruns"]:
        d = os.path.join(stage, sub) if os.path.isdir(os.path.join(stage, sub)) \
            else stage
        # collect this sub-run's files first, so the time origin is the
        # sub-run's own minimum timestamp
        acc = []
        for i in idxs:
            # Prefer our own FEU1 decode when it exists.  banco's
            # `combined_hits` for RAW runs were produced with the pre-fix
            # decoder (so their events are merged: run_71 group 023 has 13,126
            # unique eventIds over a span of 17,696, with steps of 2/3/4/5) AND
            # with the ZS analyzer flags, which are wrong for RAW data.  Both
            # uRWELL planes sit on FEU1 -- channels 0-255 front, 256-511 back,
            # exactly the mapping csv -- so a FEU1 hits file needs no combining.
            uf = os.path.join(d, f"hits_{sub}_{i}_01.root")
            if not os.path.exists(uf):
                uf = os.path.join(d, f"{stem}{i}_feu-combined_hits.root")
            hf = os.path.join(d, f"hits_{sub}_{i}_03.root")
            if not os.path.exists(hf):
                hf = os.path.join(d, f"hits_{i}_03.root")
            if not (os.path.exists(uf) and os.path.exists(hf)):
                print(f"  !! {sub}/{i}: missing input, skipped")
                continue
            a = uproot.open(uf + ":hits").arrays(
                ["eventId", "channel", "amplitude", "trigger_timestamp_ns"],
                library="np")
            b = uproot.open(hf + ":hits").arrays(
                ["eventId", "channel", "amplitude", "time", "time_of_max",
                 "integral", "saturated", "significance"], library="np")
            acc.append((i, a, b))
            print(f"  {sub}/{i}: {len(a['eventId'])} uRWELL hits, "
                  f"{len(b['eventId'])} det4 hits")
        if not acc:
            continue
        # trigger_timestamp_ns is ns SINCE THE SUB-RUN START and is continuous
        # across the sub-run's files, so it can be used directly. Subtracting a
        # per-download minimum would shift the whole axis whenever file 000 is
        # not staged -- which is exactly the case for run_71, where banco's
        # pipeline produced the LAST file groups first.
        ts_min = 0.0
        for i, a, b in acc:
            ev_uniq = np.union1d(a["eventId"], b["eventId"])
            n_ev = len(ev_uniq)
            ia = np.searchsorted(ev_uniq, a["eventId"])
            ib = np.searchsorted(ev_uniq, b["eventId"])
            ts = np.zeros(n_ev, np.int64)
            ts[ia] = a["trigger_timestamp_ns"]
            o = {"ev_id": ev_uniq.astype(np.int64), "ev_ts": ts}
            ch, amp = a["channel"], np.abs(a["amplitude"])
            for det in "fb":
                for v in "xy":
                    k = (ud[ch] == det) & (uv[ch] == v)
                    p, n = clusters(ia[k], up[ch[k]], amp[k], n_ev)
                    o[f"{det}{v}_p"], o[f"{det}{v}_n"] = p, n
            o["h_ev"] = ib.astype(np.int64) + off
            o["h_ch"] = b["channel"].astype(np.int16)
            o["h_amp"] = np.abs(b["amplitude"]).astype(np.float32)
            o["h_time"] = b["time"].astype(np.float32)
            o["h_tmax"] = b["time_of_max"].astype(np.float32)
            o["h_sat"] = b["saturated"].astype(np.int8)
            o["h_sig"] = b["significance"].astype(np.float32)
            o["ev_t_wall"] = t0 + (ts.astype(float) - ts_min) / 1e9
            o["ev_t_wall"][ts == 0] = np.nan
            o["subrun"] = np.full(n_ev, sub, dtype="<U40")
            o["fgroup"] = np.full(n_ev, i, dtype="<U4")
            off += n_ev
            parts.append(o)

    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    merged["plateau"] = datasets.plateau_of(args.dataset, merged["ev_t_wall"])
    np.savez_compressed(out, **merged)
    print(f"\nwrote {out}: {len(merged['ev_id'])} events, "
          f"{len(merged['h_ev'])} det4 hits")

    t = merged["ev_t_wall"]
    def hm(x):
        x = x % 86400
        return f"{int(x//3600):02d}:{int(x%3600//60):02d}:{x%60:05.2f}"
    for sub, _, t0, _ in D["subruns"]:
        m = merged["subrun"] == sub
        if m.any():
            tt = t[m][np.isfinite(t[m])]
            print(f"  {sub}: {m.sum():8d} events, {hm(tt.min())} - {hm(tt.max())}"
                  f"   (log start {hm(t0)})")
    print("  plateau occupancy:")
    for lab, lo, hi, dr, re in D["plateaus"]:
        n = (merged["plateau"] == lab).sum()
        print(f"    {lab:>6}  drift {dr:6.1f}  resist {re:6.1f}  {n:9d} events")


if __name__ == "__main__":
    main()
