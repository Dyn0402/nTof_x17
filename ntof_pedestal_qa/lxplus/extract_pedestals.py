#!/usr/bin/env python3
"""extract_pedestals.py -- per-channel pedestal statistics for every DREAM
pedestal run of the n_TOF campaign.  RUNS ON LXPLUS (EOS is POSIX there).

For each pedestal acquisition it reads the decoded ROOT (`nt` tree) and
computes, per channel, straight from the samples:

    mean        raw ADC baseline, averaged over every sample of every event
    raw_sigma   std of the raw ADC about that mean
    cns_sigma   std after subtracting, per sample, the median of the channel's
                own 64-channel DREAM block  (the repo's CNS convention, see
                sps_beam_test_26/det4_sps_assessment/10_pedestals.py)

and per 64-channel DREAM block:

    cm_rms      std of the block-median trace itself -- the coherent
                ("common") noise amplitude of that chip, in ADC

Nothing here trusts the firmware's own numbers, but they are parsed too, from
the `_ped.aux` / `_thr.aux` summary blocks, so the two can be compared and so
the *loaded* zero-suppression threshold (5 sigma) is on the record.

    setup LCG_105, then:
    python3 extract_pedestals.py --out ped_stats.npz

Writes one npz holding every set; ~60 sets x 8 FEUs x 512 channels is small.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import uproot

BASE = "/eos/experiment/ntof/data/x17/july_beam/pedestals"
RUNS = "/eos/experiment/ntof/data/x17/july_beam/runs"
NCH = 512
BLK = 64                       # one DREAM chip = one CNS block = one cable
NBLK = NCH // BLK

# FEU -> chamber and view.  ntof_active_area/clusters.py is the authority.
FEU_DET = {3: ("A", "x"), 4: ("A", "y"),
           5: ("B", "x"), 6: ("B", "y"),
           7: ("C", "x"), 8: ("C", "y"),
           1: ("D", "x"), 2: ("D", "y")}

STAMP_RE = re.compile(r"_pedthr_(\d{6}_\d{2}H\d{2})_\d{3}_(\d{2})\.root$")
AUX_RE = re.compile(r"^\s*(\d{3})\s+D(\d)\s+C\s*(\d+)\s+"
                    r"(?:ped|thr)=\s*(\d+)\s+0x[0-9a-f]+\s+Histo_Stat:.*?"
                    r"Avr\s*=\s*([-\d.]+)\s+Std\s*=\s*([-\d.]+)")


def parse_aux(path):
    """First 512 data lines of a *_ped.aux / *_thr.aux: (loaded, avr, std).

    The file repeats every channel a second time with its full histogram
    appended; the summary block at the top carries the same statistics, so we
    stop as soon as we have 512 channels.
    """
    loaded = np.full(NCH, np.nan, np.float64)
    avr = np.full(NCH, np.nan, np.float64)
    std = np.full(NCH, np.nan, np.float64)
    seen = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = AUX_RE.match(line)
            if not m:
                continue
            ch = int(m.group(1))
            if not np.isnan(avr[ch]):        # into the histogram block
                break
            loaded[ch], avr[ch], std[ch] = (float(m.group(4)),
                                            float(m.group(5)),
                                            float(m.group(6)))
            seen += 1
            if seen == NCH:
                break
    return loaded, avr, std


def read_frames(path, max_samples):
    """(events, samples, channels) float32, capped at ~max_samples per channel.

    Events whose payload is not a complete NCH x nsamp frame are dropped --
    the same guard 10_pedestals.py uses.
    """
    t = uproot.open(path)["nt"]
    n_ent = t.num_entries
    if n_ent == 0:
        return None
    probe = t.arrays(["sample"], entry_stop=1, library="np")
    nsamp = int(probe["sample"][0].max()) + 1
    want = max(1, min(n_ent, int(np.ceil(max_samples / max(nsamp, 1)))))
    a = t.arrays(["channel", "sample", "amplitude"],
                 entry_stop=want, library="np")
    out = []
    for ch, sa, am in zip(a["channel"], a["sample"], a["amplitude"]):
        if len(am) != nsamp * NCH:
            continue
        w = np.full((nsamp, NCH), np.nan, np.float32)
        w[sa, ch] = am
        if np.isnan(w).any():
            continue
        out.append(w)
    if not out:
        return None
    return np.asarray(out, np.float32)


def stats_for_file(path, max_samples):
    """Per-channel noise decomposition for one FEU.

    Order matters and follows the firmware's: subtract the channel's own
    baseline first, *then* take the block median as the common mode.  Taking
    the median of raw ADC instead lets the spread of per-channel DC offsets
    (~100 ADC across a chip) leak into the common-mode estimate, which
    inflates cns_sigma on the quiet channels by ~70 %.
    """
    w = read_frames(path, max_samples)
    if w is None:
        return None
    flat = w.reshape(-1, NCH)                       # (n_samples, 512)
    mean = flat.mean(axis=0)
    raw_sigma = flat.std(axis=0)

    # Is the raw spread random, or a fixed per-sample waveform shape repeated
    # every event?  shape_rms answers it; on this data it is <1 % of raw_sigma,
    # so raw_sigma really is noise.
    shape_rms = w.mean(axis=0).std(axis=0)

    ped = flat - mean[None, :]                      # baseline-subtracted
    cm = np.empty((flat.shape[0], NBLK), np.float32)
    res = ped.copy()
    for b in range(NBLK):
        s = slice(b * BLK, (b + 1) * BLK)
        med = np.median(ped[:, s], axis=1)
        cm[:, b] = med
        res[:, s] -= med[:, None]
    cns_sigma = res.std(axis=0)
    cm_rms = cm.std(axis=0)                         # coherent swing per chip

    return dict(mean=mean.astype(np.float32),
                raw_sigma=raw_sigma.astype(np.float32),
                cns_sigma=cns_sigma.astype(np.float32),
                shape_rms=shape_rms.astype(np.float32),
                cm_rms=cm_rms.astype(np.float32),
                n_samples=np.int32(flat.shape[0]),
                n_events=np.int32(w.shape[0]),
                n_samp_per_event=np.int32(w.shape[1]))


def discover():
    """{(dirname, stamp): {feu: rootpath}} over every pedestals_* directory."""
    sets = defaultdict(dict)
    for d in sorted(os.listdir(BASE)):
        if not d.startswith("pedestals_"):
            continue
        for p in sorted(glob.glob(os.path.join(BASE, d, "pedestals", "*.root"))):
            m = STAMP_RE.search(p)
            if not m:
                continue
            sets[(d, m.group(1))][int(m.group(2))] = p
    return sets


def stamp_to_iso(stamp):
    """'260701_19H52' -> '2026-07-01T19:52'."""
    d, t = stamp.split("_")
    return f"20{d[0:2]}-{d[2:4]}-{d[4:6]}T{t[0:2]}:{t[3:5]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ped_stats.npz")
    ap.add_argument("--max-samples", type=int, default=16000,
                    help="samples per channel to use (32 per event)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sets = discover()
    keys = sorted(sets, key=lambda k: k[1])
    if args.limit:
        keys = keys[:args.limit]
    print(f"{len(keys)} pedestal acquisitions in {BASE}", flush=True)

    store, meta = {}, []
    for i, key in enumerate(keys):
        dirname, stamp = key
        t0 = time.time()
        feus = sets[key]
        got = []
        for feu, path in sorted(feus.items()):
            s = stats_for_file(path, args.max_samples)
            if s is None:
                print(f"  !! {stamp} FEU {feu}: no usable frames", flush=True)
                continue
            pre = f"{stamp}/{feu:02d}"
            for k, v in s.items():
                store[f"{pre}/{k}"] = v
            aux = path[:-5] + "_ped.aux"
            thr = path[:-5] + "_thr.aux"
            if os.path.exists(aux):
                lo, av, sd = parse_aux(aux)
                store[f"{pre}/fw_ped_word"] = lo.astype(np.float32)
                store[f"{pre}/fw_raw_avr"] = av.astype(np.float32)
                store[f"{pre}/fw_raw_std"] = sd.astype(np.float32)
            if os.path.exists(thr):
                lo, av, sd = parse_aux(thr)
                store[f"{pre}/fw_thr"] = lo.astype(np.float32)
                store[f"{pre}/fw_zs_avr"] = av.astype(np.float32)
                store[f"{pre}/fw_zs_std"] = sd.astype(np.float32)
            got.append(feu)
        if not got:
            continue
        meta.append((stamp, stamp_to_iso(stamp), dirname,
                     ",".join(f"{f:02d}" for f in got)))
        print(f"[{i+1}/{len(keys)}] {stamp}  {len(got)} FEUs  "
              f"{time.time()-t0:.1f}s", flush=True)

    store["meta"] = np.array(meta, dtype=object)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB), "
          f"{len(meta)} sets", flush=True)


if __name__ == "__main__":
    sys.exit(main())
