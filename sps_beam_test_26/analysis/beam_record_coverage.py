#!/usr/bin/env python3
"""Coverage of the SPS beam record, per day, over the test-beam period.

Answers the one question every beam-corrected number depends on: for a given
wall-clock window, do we actually have a beam record, and how continuous is it?

Run against the backfilled archive (the default) or against any directory of
`sps_spill_<date>.csv` / `beam_intensity_<date>.csv` / `h4_tax_<date>.csv`, so
the same command reports on banco's live logs and on the archive and the two can
be compared directly.

    python beam_record_coverage.py                       # the archive
    python beam_record_coverage.py --dir <other>         # e.g. a banco pull
    python beam_record_coverage.py --gaps                # list intra-day holes

A "gap" is any interval between consecutive cycles longer than --gap-s. The SPS
supercycle is a few seconds, so anything past a minute is the record being down,
the machine being down, or an access — the TAX log (`tax_windows.py`) is what
separates those three.
"""

import argparse
import csv
import glob
import os
from datetime import datetime

ARCHIVE = ("/media/dylan/data/x17/sps_run53_det4_check/records/beam/"
           "backfill_nxcals")

KINDS = [
    ("sps_spill_", "spill cycles"),
    ("beam_intensity_", "intensity pts"),
    ("h4_tax_", "TAX samples"),
]


def read_times(path):
    """Sorted unix timestamps in a day-file. These files are written sorted, but
    a merged file is only sorted if every writer agreed on the key, so sort."""
    ts = []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    ts.append(float(r["unix_ts"]))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    ts.sort()
    return ts


def gaps(ts, gap_s):
    return [(a, b) for a, b in zip(ts, ts[1:]) if b - a > gap_s]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=ARCHIVE)
    ap.add_argument("--gaps", action="store_true", help="list intra-day gaps")
    ap.add_argument("--gap-s", type=float, default=60.0)
    args = ap.parse_args()

    days = sorted({os.path.basename(p).rsplit("_", 1)[-1][:-4]
                   for pre, _ in KINDS
                   for p in glob.glob(os.path.join(args.dir, f"{pre}2026-*.csv"))})
    if not days:
        raise SystemExit(f"no day-files under {args.dir}")

    print(f"{args.dir}\n")
    hdr = f"{'day':<12}" + "".join(f"{lbl:>16}" for _, lbl in KINDS) + "   span (first -> last, spill)"
    print(hdr)
    print("-" * len(hdr))

    for day in days:
        cells, spill_ts = [], []
        for pre, _ in KINDS:
            ts = read_times(os.path.join(args.dir, f"{pre}{day}.csv"))
            if pre == "sps_spill_":
                spill_ts = ts
            cells.append(f"{len(ts):>16,}" if ts else f"{'-':>16}")
        span = ""
        if spill_ts:
            a = datetime.fromtimestamp(spill_ts[0])
            b = datetime.fromtimestamp(spill_ts[-1])
            span = f"   {a:%H:%M:%S} -> {b:%H:%M:%S}"
        print(f"{day:<12}" + "".join(cells) + span)

        if args.gaps and spill_ts:
            for a, b in gaps(spill_ts, args.gap_s):
                print(f"{'':<12}   gap {datetime.fromtimestamp(a):%H:%M:%S}"
                      f" -> {datetime.fromtimestamp(b):%H:%M:%S}"
                      f"   ({(b - a) / 60:.1f} min)")


if __name__ == "__main__":
    main()
