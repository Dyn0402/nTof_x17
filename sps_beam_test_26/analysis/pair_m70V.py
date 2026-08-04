#!/usr/bin/env python3
"""Pair det4 (FEU 3) against the uRWELL reference for run_56 / meshscan_m70V.

This is the flat-mount, high-gain sub-run: det4 perpendicular to the beam,
Ar/CO2/iso 95/3/2, drift held 700 V, resist stepping 590 V -> 625 V inside the
sub-run.  It is the highest voltage det4 ever ran flat.

Configuration, taken from the machine record rather than assumed:

  64 samples x 60 ns        run_56/run_config.json (NOT the 32 the timeline says)
  ZS on, 5 sigma            ..._03_thr.prg header, 'Threshold value: 5.000000'
  on-FEU pedestal subtr.    -> analyzer run with --zs-baseline 1
  Dream peaking 180 ns      P2B_Beam.cfg, code (0xd023>>4)&0xF = 2

Sub-run boundaries come from run_56/dream_daq.log (which exists on EOS for
runs 54/55/56, contrary to RUN_TIMELINE.md's claim that only run_61 has one):

  meshscan_m70V   15:47:25.058 -> 15:59:33.971

and the det4 HV plateaus inside it, from the recovered hv_monitor trace:

  590.0 V   15:45:20 - 15:52:50     (starts in m60V, ends inside m70V)
  624.7 V   15:52:57 - 16:00:25     (ends in m80V)

Writes one npz per input file pair into staging/, then a merged one.

  python pair_m70V.py [--files 000,001]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM      # noqa: E402

STAGE = ("/media/dylan/data/x17/sps_run53_det4_check/staging/run_56_m70V/")
URW_MAP = ("/media/dylan/data/x17/sps_run53_det4_check/"
           "flat_ArCO2iso_95-3-2__run53-56/urw_mapping/mapping_urwell.csv")
STEM = "EicP2Bt_meshscan_m70V_datrun_260801_15H47_"

Z_FRONT, Z_BACK, Z_DET4 = 0.0, 1370.0, 1120.0
SUBRUN_T0_S = 15 * 3600 + 47 * 60 + 25.058          # m70V start, wall clock

#: det4 global strip index within a view; physically adjacent strips are 127
#: FEU channels apart because every connector is plugged inverted, so nothing
#: may cluster or step in raw channel number.
SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)


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
    """Largest cluster's charge-weighted position, and how many clusters."""
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
    o2 = np.argsort(cq, kind="stable")      # last write wins -> largest cluster
    lead[cev[o2]] = cp[o2]
    return lead, ncl


def one_file(idx: str) -> dict:
    urw_view, urw_pos, urw_det = urwell_map()
    uf = f"{STAGE}{STEM}{idx}_feu-combined_hits.root"
    df = f"{STAGE}hits_{idx}_03.root"
    print(f"  uRWELL {os.path.basename(uf)}")
    print(f"  det4   {os.path.basename(df)}")

    a = uproot.open(uf + ":hits").arrays(
        ["eventId", "channel", "amplitude", "trigger_timestamp_ns"], library="np")
    b = uproot.open(df + ":hits").arrays(
        ["eventId", "channel", "amplitude", "time", "time_of_max",
         "integral", "saturated", "significance", "max_sample"], library="np")

    ev_uniq = np.union1d(a["eventId"], b["eventId"])
    n_ev = len(ev_uniq)
    ia = np.searchsorted(ev_uniq, a["eventId"])
    ib = np.searchsorted(ev_uniq, b["eventId"])

    ts = np.zeros(n_ev, np.int64)
    ts[ia] = a["trigger_timestamp_ns"]

    out = {"ev_id": ev_uniq.astype(np.int64), "ev_ts": ts}
    ch, amp = a["channel"], np.abs(a["amplitude"])
    for det in "fb":
        for v in "xy":
            k = (urw_det[ch] == det) & (urw_view[ch] == v)
            p, n = clusters(ia[k], urw_pos[ch[k]], amp[k], n_ev)
            out[f"{det}{v}_p"], out[f"{det}{v}_n"] = p, n

    out["h_ev"] = ib.astype(np.int64)
    out["h_ch"] = b["channel"].astype(np.int16)
    out["h_amp"] = np.abs(b["amplitude"]).astype(np.float32)
    out["h_time"] = b["time"].astype(np.float32)
    out["h_tmax"] = b["time_of_max"].astype(np.float32)
    out["h_int"] = b["integral"].astype(np.float32)
    out["h_sat"] = b["saturated"].astype(np.int8)
    out["h_sig"] = b["significance"].astype(np.float32)
    out["h_maxsamp"] = b["max_sample"].astype(np.float32)
    print(f"    {n_ev} events, {len(b['eventId'])} det4 hits, "
          f"{len(a['eventId'])} uRWELL hits")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="000,001")
    ap.add_argument("--out", default=STAGE + "pair_m70V.npz")
    args = ap.parse_args()

    parts, off = [], 0
    for idx in args.files.split(","):
        print(f"file {idx}:")
        d = one_file(idx)
        d["h_ev"] = d["h_ev"] + off
        d["file_idx"] = np.full(len(d["ev_id"]), int(idx), np.int16)
        off += len(d["ev_id"])
        parts.append(d)

    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    # wall-clock seconds of each event. trigger_timestamp_ns restarts per file,
    # so rebuild the axis from the per-file span rather than trusting one origin.
    t = np.zeros(len(merged["ev_id"]))
    base = SUBRUN_T0_S
    for p in parts:
        m = merged["file_idx"] == p["file_idx"][0]
        ts = merged["ev_ts"][m].astype(float)
        good = ts > 0
        t0 = ts[good].min() if good.any() else 0.0
        t[m] = base + (ts - t0) / 1e9
        base += (ts[good].max() - t0) / 1e9 if good.any() else 0.0
    merged["ev_t_wall"] = t

    np.savez_compressed(args.out, **merged)
    print(f"\nwrote {args.out}  ({len(merged['ev_id'])} events, "
          f"{len(merged['h_ev'])} det4 hits)")
    span = t[t > 0]
    if len(span):
        def hms(x):
            return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"
        print(f"wall-clock span reconstructed: {hms(span.min())} - {hms(span.max())}")
        print("  (m70V by dream_daq.log: 15:47:25.06 - 15:59:33.97)")


if __name__ == "__main__":
    main()
