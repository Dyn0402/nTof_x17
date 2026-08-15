#!/usr/bin/env python3
"""extract_context.py -- the conditions each pedestal run was taken under.

RUNS ON LXPLUS.  A pedestal is only interpretable next to the configuration it
was taken with, so for every pedestal acquisition this pulls:

  * from `run_config.json`   sample period, samples per waveform, latency,
                             zero-suppression flag, gas, trigger, HV setpoints
  * from `Mx17_init_*.cfg_cpy` the DREAM clock dividers and per-DREAM registers
                             actually loaded into the FEUs
  * from `hv_monitor.csv`    whether the chambers were live while it was taken,
                             and at what voltage

Without this a clock change and a cable change look identical in the noise.

    python3 extract_context.py --out ped_context.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

BASE = "/eos/experiment/ntof/data/x17/july_beam/pedestals"

CFG_KEYS = ["RdClk_Div", "WrClk_Div", "WrClk_Phase", "AdcClk_Phase",
            "Main_Conf_Samples", "Main_Conf_SparseRd", "Main_Conf_DreamPol"]
STAMP_RE = re.compile(r"_pedthr_(\d{6}_\d{2}H\d{2})_")


def cfg_values(path):
    """First value seen for each key of interest, plus the Dream * registers."""
    out = {}
    dream = {}
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                for k in CFG_KEYS:
                    if k in line and k not in out:
                        out[k] = line.split()[-1]
                m = re.match(r"^Feu \* Dream \*\s+(\d+)\s+(.*)$", line.strip())
                if m and m.group(1) not in dream:
                    dream[m.group(1)] = " ".join(m.group(2).split())
    except OSError:
        return {}
    out["dream_regs"] = dream
    return out


def hv_state(path):
    """Median monitored voltage and current per channel over the pedestal run."""
    try:
        with open(path, errors="replace") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {}
    if not rows:
        return {}
    cols = [c for c in rows[0] if c and c.lower() not in ("time", "timestamp")]
    out = {}
    for c in cols:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[c]))
            except (TypeError, ValueError):
                pass
        if vals:
            vals.sort()
            out[c] = round(vals[len(vals) // 2], 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ped_context.json")
    args = ap.parse_args()

    out = {}
    for d in sorted(os.listdir(BASE)):
        if not d.startswith("pedestals_"):
            continue
        pdir = os.path.join(BASE, d, "pedestals")
        roots = sorted(glob.glob(os.path.join(pdir, "*_pedthr_*.root")))
        if not roots:
            continue
        stamps = sorted({m.group(1) for r in roots
                         if (m := STAMP_RE.search(os.path.basename(r)))})

        rc = {}
        rcp = os.path.join(BASE, d, "run_config.json")
        if os.path.exists(rcp):
            try:
                rc = json.load(open(rcp))
            except (OSError, json.JSONDecodeError):
                rc = {}
        dd = rc.get("dream_daq_info", {})
        subs = rc.get("sub_runs") or [{}]

        inits = sorted(glob.glob(os.path.join(pdir, "Mx17_init_*.cfg_cpy")))
        hv = hv_state(os.path.join(pdir, "hv_monitor.csv"))

        for st in stamps:
            # the cfg copy written closest to this acquisition
            cand = [p for p in inits if st[:6] in os.path.basename(p)] or inits
            cfg = cfg_values(cand[0]) if cand else {}
            out[st] = dict(
                directory=d,
                sample_period_ns=dd.get("sample_period"),
                n_samples=dd.get("n_samples_per_waveform"),
                latency=dd.get("latency"),
                zero_suppress=dd.get("zero_suppress"),
                included_feus=dd.get("included_feus"),
                gas=rc.get("gas"),
                trigger=rc.get("trigger"),
                beam_type=rc.get("beam_type"),
                start_time=rc.get("start_time"),
                hv_setpoints=subs[0].get("hvs"),
                hv_monitored=hv,
                cfg=cfg,
                n_acquisitions_in_dir=len(stamps),
            )
        print(f"{d}: {len(stamps)} acquisition(s)", flush=True)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"wrote {args.out}: {len(out)} acquisitions")


if __name__ == "__main__":
    sys.exit(main())
