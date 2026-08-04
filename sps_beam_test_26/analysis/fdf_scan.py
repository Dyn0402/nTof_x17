#!/usr/bin/env python3
"""Walk a DREAM .fdf at the 16-bit word level and report its frame structure.

Written to answer one question about run_71 (the RAW run, ~24% of sample-groups
lost to FEU bandwidth): **can complete events still be delimited?**  The
decoder currently flushes an event on the FEU end-of-event marker, and when the
packet carrying that marker is dropped, two events merge.  The proposed repair
is to delimit on `eventID` instead -- but that is only valid if `eventID` is
actually present on every surviving frame, which has to be shown, not assumed.

Word format from `decoder/include/dreamdataline.h`:

    NB: words are BIG-endian (read16 does ntohs)
    is_Feu_header    (w & 0x7000)>>12 == 6      8-word FEU header
    is_data_header   (w & 0x6000)>>13 == 1      4-word per-Dream block header
    is_data          (w & 0x6000)>>13 == 0      channel payload
    is_final_trailer (w & 0x7000)>>12 == 7      trailer; get_EoE = (w & 0x800)>>11

FEU header word roles (from DreamDecoder.cpp):
    0: FeuID = w & 0xFF, and sampleID |= (w & 0x800)>>3   (bit 11 -> bit 8)
    1: eventID  |= w & 0xFFF
    3: sampleID += (w & 0xFF8)>>3 ; fine timestamp = w & 0x7
    4: eventID  |= (w & 0xFFF) << 12

  python fdf_scan.py <file.fdf> [--mb 60]
"""
from __future__ import annotations

import argparse
import collections

import numpy as np


def scan(path, max_words):
    w = np.fromfile(path, dtype=">u2", count=max_words)
    n = len(w)
    top3 = (w & 0x7000) >> 12
    top2 = (w & 0x6000) >> 13
    is_feu = top3 == 6
    is_trail = top3 == 7

    frames = []          # (word_index, eventID, sampleID, feuID)
    i = 0
    feu_idx = np.flatnonzero(is_feu)
    # group consecutive FEU-header words into 8-word headers
    if len(feu_idx) == 0:
        return [], [], n
    brk = np.flatnonzero(np.diff(feu_idx) != 1)
    starts = np.r_[feu_idx[0], feu_idx[brk + 1]]
    ends = np.r_[feu_idx[brk], feu_idx[-1]]
    for s, e in zip(starts, ends):
        L = e - s + 1
        if L < 5:
            continue
        h = w[s:s + 8]
        feuID = h[0] & 0xFF
        samp = ((h[0] & 0x800) >> 3) + ((h[3] & 0xFF8) >> 3)
        evt = (h[1] & 0xFFF) | ((h[4] & 0xFFF) << 12)
        frames.append((int(s), int(evt), int(samp), int(feuID), int(L)))

    eoe = [int(k) for k in np.flatnonzero(is_trail & (((w & 0x800) >> 11) == 1))]
    return frames, eoe, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fdf")
    ap.add_argument("--mb", type=float, default=60.0)
    args = ap.parse_args()
    frames, eoe, nwords = scan(args.fdf, int(args.mb * 1e6 / 2))
    print(f"scanned {nwords*2/1e6:.1f} MB, {len(frames)} FEU frames, "
          f"{len(eoe)} EoE markers\n")

    hl = collections.Counter(f[4] for f in frames)
    print("FEU header lengths:", hl.most_common(5))

    ev = np.array([f[1] for f in frames])
    sm = np.array([f[2] for f in frames])
    pos = np.array([f[0] for f in frames])

    print(f"eventID: {ev.min()} .. {ev.max()}, {len(np.unique(ev))} distinct")
    print(f"sampleID: {sm.min()} .. {sm.max()}")

    # --- is eventID present and constant within an event? -----------------
    chg = np.r_[True, ev[1:] != ev[:-1]]
    bounds = np.flatnonzero(chg)
    per = np.diff(np.r_[bounds, len(ev)])
    print(f"\nframes per eventID: {collections.Counter(per.tolist()).most_common(8)}")
    print(f"  (64 = complete event; fewer = groups lost to the FEU)")
    full = (per == 64).sum()
    print(f"  complete events: {full} of {len(per)} = {100*full/len(per):.1f}%")
    print(f"  mean frames/event: {per.mean():.1f}  -> acceptance {per.mean()/64:.3f}")

    # --- does sampleID run 0..63 inside one eventID, without repeats? -----
    bad_dup = bad_order = 0
    for b, c in zip(bounds, np.r_[bounds[1:], len(ev)]):
        s = sm[b:c]
        if len(np.unique(s)) != len(s):
            bad_dup += 1
        if np.any(np.diff(s) <= 0):
            bad_order += 1
    print(f"\nwithin one eventID: {bad_dup} events with a REPEATED sampleID, "
          f"{bad_order} not strictly increasing (of {len(per)})")

    # --- eventID monotonic? ------------------------------------------------
    de = np.diff(ev[bounds])
    print(f"eventID steps between consecutive events: "
          f"{collections.Counter(de.tolist()).most_common(6)}")

    # --- how many events actually carry an EoE marker? --------------------
    eoe = np.array(eoe)
    if len(eoe):
        # an event is 'closed' if an EoE marker falls between its first frame
        # and the next event's first frame
        nb = np.r_[bounds[1:], len(ev)]
        closed = 0
        for b, c in zip(bounds, nb):
            lo = pos[b]
            hi = pos[c] if c < len(pos) else nwords
            if np.any((eoe >= lo) & (eoe < hi)):
                closed += 1
        print(f"\nevents with an EoE marker: {closed} of {len(per)} = "
              f"{100*closed/len(per):.1f}%")
        print(f"  -> {len(per)-closed} events would MERGE into the next one "
              f"with EoE-based delimiting")


if __name__ == "__main__":
    main()
