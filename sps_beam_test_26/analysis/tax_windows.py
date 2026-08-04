#!/usr/bin/env python3
"""H4 beam-stopper (TAX) open/blocked windows, from the backfilled NXCALS log.

Why this file exists: `HV_AND_BEAM_RECORD.md` §1 said the beam record "cannot
date accesses", because the H4 beam stopper is not among the variables banco's
SPS spill monitor logs — its H4 story is built entirely from the XBH4.BEND
currents, and those stay energised through an access. That was true of banco's
record. It is not true of NXCALS: `XTAX_022_023:POSITION_MEAS` is the T2 TAX
position, and the mx17-daq fork of the monitor has always logged it.

Backfilled for the whole test-beam period on 2026-08-02 into

    records/beam/backfill_nxcals/h4_tax_<date>.csv

The classifier is the daq controller's: position <= -100 mm is parked out (beam
can reach H4), >= +100 mm is parked in (H4 blocked), anything between is
mid-stroke. Mid-stroke samples are NOT state changes — the barrier takes ~20 s
to travel, and treating each of those samples as a transition is what makes a
naive diff of the `state` column produce hundreds of spurious windows.

    python tax_windows.py                 # the det4 era, 07-30 onward
    python tax_windows.py --all           # the whole period
    python tax_windows.py --min-s 120     # only windows longer than 2 min
"""

import argparse
import csv
import glob
import os
from datetime import datetime

ARCHIVE = ("/media/dylan/data/x17/sps_run53_det4_check/records/beam/"
           "backfill_nxcals")

# Below this the barrier is parked out; above it, parked in. Same thresholds as
# mx17-daq's sps_spill_controller, so the states here match its live log.
OPEN_MAX = -100.0
BLOCK_MIN = 100.0


def classify(pos):
    if pos is None:
        return None
    if pos <= OPEN_MAX:
        return "open"
    if pos >= BLOCK_MIN:
        return "blocked"
    return "moving"


def windows(path):
    """Contiguous blocked windows in one day-file.

    A window opens on the first `blocked` sample and closes on the next `open`
    one. `moving` is deliberately transparent: it is the barrier in transit, so
    it neither opens nor closes a window, and it must not reset the state we are
    tracking.
    """
    out = []
    state = None
    start = None
    for r in csv.DictReader(open(path)):
        try:
            s = classify(float(r["position_mm"]))
        except (TypeError, ValueError, KeyError):
            continue
        if s == "moving" or s is None:
            continue
        if s == "blocked" and state != "blocked":
            start = r["timestamp"]
        elif s == "open" and state == "blocked" and start:
            out.append((start, r["timestamp"]))
            start = None
        state = s
    if state == "blocked" and start:          # still shut at end of file
        out.append((start, None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=ARCHIVE)
    ap.add_argument("--all", action="store_true",
                    help="whole period (default: 07-30 onward, the det4 era)")
    ap.add_argument("--min-s", type=float, default=0.0,
                    help="hide windows shorter than this many seconds")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "h4_tax_2026-*.csv")))
    if not args.all:
        files = [f for f in files
                 if os.path.basename(f)[len("h4_tax_"):] >= "2026-07-30"]
    if not files:
        raise SystemExit(f"no h4_tax_*.csv under {args.dir}")

    total = 0
    for f in files:
        w = windows(f)
        rows = []
        for a, b in w:
            ta = datetime.fromisoformat(a)
            if b is None:
                rows.append((ta, None, None))
                continue
            tb = datetime.fromisoformat(b)
            dur = (tb - ta).total_seconds()
            if dur >= args.min_s:
                rows.append((ta, tb, dur))
        if not rows:
            continue
        print(f"\n{os.path.basename(f)[len('h4_tax_'):-4]}   H4 blocked windows")
        for ta, tb, dur in rows:
            if tb is None:
                print(f"   {ta:%H:%M:%S} -> (still blocked at end of day)")
            else:
                print(f"   {ta:%H:%M:%S} -> {tb:%H:%M:%S}   {dur/60:6.1f} min")
            total += 1
    print(f"\n{total} blocked window(s) over {len(files)} day(s)")


if __name__ == "__main__":
    main()
