#!/usr/bin/env python3
"""extract_usage.py -- which pedestal each n_TOF sub-run actually ran with.

RUNS ON LXPLUS.  Every sub-run's `raw_daq_data/pedestal_run.txt` names the
pedestal directory whose `.prg` memory files the DAQ loaded into the FEUs for
that sub-run, and the sub-run's own data file names carry its wall-clock start.
Together they turn the list of pedestal acquisitions into a timeline of what
was in force over the beam.

    python3 extract_usage.py --out ped_usage.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

RUNS = "/eos/experiment/ntof/data/x17/july_beam/runs"
DAT_RE = re.compile(r"_datrun_(\d{6}_\d{2}H\d{2})_")
PED_DIR_RE = re.compile(r"^pedestals_(\d{2})-(\d{2})-(\d{2})_"
                        r"(\d{2})-(\d{2})-(\d{2})$")


def subrun_start(raw_dir):
    """Wall-clock stamp of the sub-run, from its own datrun file names."""
    try:
        names = os.listdir(raw_dir)
    except OSError:
        return ""
    stamps = {m.group(1) for n in names if (m := DAT_RE.search(n))}
    return min(stamps) if stamps else ""


def ped_dir_iso(name):
    """'pedestals_07-24-26_22-14-00' -> '2026-07-24T22:14:00'."""
    m = PED_DIR_RE.match(name.strip())
    if not m:
        return ""
    mo, dy, yr, hh, mm, ss = m.groups()
    return f"20{yr}-{mo}-{dy}T{hh}:{mm}:{ss}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ped_usage.csv")
    args = ap.parse_args()

    rows = []
    runs = sorted((d for d in os.listdir(RUNS) if d.startswith("run_")),
                  key=lambda s: int(s.split("_")[1]))
    for run in runs:
        rdir = os.path.join(RUNS, run)
        try:
            subs = sorted(d for d in os.listdir(rdir)
                          if os.path.isdir(os.path.join(rdir, d)))
        except OSError:
            continue
        for sub in subs:
            raw = os.path.join(rdir, sub, "raw_daq_data")
            f = os.path.join(raw, "pedestal_run.txt")
            if not os.path.exists(f):
                continue
            with open(f, errors="replace") as fh:
                ped = fh.read().strip()
            rows.append(dict(run=run, subrun=sub,
                             subrun_start=subrun_start(raw),
                             pedestal_run=ped,
                             pedestal_time=ped_dir_iso(ped)))
        print(f"{run}: {len(rows)} rows so far", flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "subrun", "subrun_start",
                                           "pedestal_run", "pedestal_time"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} sub-runs, "
          f"{len({r['pedestal_run'] for r in rows})} distinct pedestals")


if __name__ == "__main__":
    sys.exit(main())
