#!/usr/bin/env python3
"""Build the det4 HV-point -> P2 run/sub-run map from the scan logs.

We never controlled the DAQ: banco started runs and sub-runs on their own
schedule and we moved our HV underneath. So "which P2 sub-run was live while
det4 sat at V" is the join that every det4 beam analysis needs, and it is only
recoverable from the scan logs plus the sub-run boundaries.

Two shapes of log, because the scan driver was rewritten between the days:

- `detE_scan.py` / `detE_resist_scan.py` (2026-08-01) were window-gated: they
  waited for a proved-open data window and stamped the sub-run into each line,
  `point 535.0 V done  [set during run_53/cfg_gain4.5_peaktime200_deflt]`.
  The join is in the log already.
- `det4_drift_scan.py` / `det4_hv_scan.py` (2026-08-02) dropped the gating and
  just stepped, `point 1 done: drift 700.5 V / 0.133 uA, resist 750.0 V ...`.
  Those have to be joined against the sub-run boundaries by wall clock.

Sub-run boundaries come from `dream_daq.log` where it exists (exact, and the
only place run_61's four restarts are visible) and otherwise from the harvested
file-time inventory.

Inputs live on the data disk, not in the repo:
    /media/dylan/data/x17/sps_run53_det4_check/records/scan_logs/

Usage:
    ../../.venv/bin/python build_run_map.py          # writes run_map.csv
    ../../.venv/bin/python build_run_map.py --table  # + markdown to stdout
"""
import csv
import datetime as dt
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = "/media/dylan/data/x17/sps_run53_det4_check/records/scan_logs"
INVENTORY = os.path.join(HERE, "run_inventory.json")
OUT = os.path.join(HERE, "run_map.csv")

TS = r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
# 08-01 driver: sub-run stamped in the line
TAGGED = re.compile(TS + r".*point\s+([\d.]+)\s*V done\s*\[set during ([^/]+)/([^\]]+)\]")
# 08-02 driver: value pairs, no sub-run
PAIRED = re.compile(TS + r"\s+point\s+\d+ done: (drift|resist) ([\d.]+) V / ([\d.]+) uA,"
                         r"\s*(drift|resist) ([\d.]+) V / ([\d.]+) uA")
DAQ = re.compile(TS + r"[,\d]*\s+INFO: Subrun (started|finished): (\S+)")


def parse_ts(s):
    return dt.datetime.strptime(s.replace("T", " "), "%Y-%m-%d %H:%M:%S")


def subrun_windows():
    """[(t0, t1, run, subrun)] — dream_daq.log first, inventory as the fallback."""
    wins = []
    daq = os.path.join(LOGS, "dream_daq.log")
    if os.path.exists(daq):
        open_sub = {}
        for line in open(daq):
            m = DAQ.search(line)
            if not m:
                continue
            t, what, name = parse_ts(m.group(1)), m.group(2), m.group(3)
            if what == "started":
                open_sub[name] = t
            elif name in open_sub:
                wins.append((open_sub.pop(name), t, "run_61", name))
    covered = {(r, s) for _, _, r, s in wins}
    if os.path.exists(INVENTORY):
        inv = json.load(open(INVENTORY))
        for run, R in inv["runs"].items():
            for sr, S in R["subruns"].items():
                if (run, sr) in covered or not S["mtime_first"]:
                    continue
                wins.append((parse_ts(S["mtime_first"]), parse_ts(S["mtime_last"]),
                             run, sr))
        for run, subs in inv["analysis"].items():
            for sr, S in subs.items():
                if (run, sr) in covered or not S["qa_first"]:
                    continue
                # QA mtimes trail the sub-run; widen backwards so a point that
                # landed inside the sub-run still matches. Marked lower trust.
                t1 = parse_ts(S["qa_last"])
                wins.append((t1 - dt.timedelta(minutes=13), t1, run, sr + " (qa-est)"))
    return sorted(wins)


def locate(t, wins):
    for t0, t1, run, sr in wins:
        if t0 <= t <= t1:
            return run, sr
    return "", ""


def rows():
    wins = subrun_windows()
    out = []
    for fn in sorted(os.listdir(LOGS)):
        if not fn.endswith(".log"):
            continue
        path = os.path.join(LOGS, fn)
        for line in open(path, errors="replace"):
            m = TAGGED.search(line)
            if m:
                t = parse_ts(m.group(1))
                scan = "resist" if "resist" in fn else "drift"
                out.append({"time": t.isoformat(sep=" "), "scan": scan,
                            "value_v": float(m.group(2)), "other_v": "",
                            "i_ua": "", "run": m.group(3), "subrun": m.group(4),
                            "source": fn, "join": "logged"})
                continue
            m = PAIRED.search(line)
            if m:
                t = parse_ts(m.group(1))
                vals = {m.group(2): (float(m.group(3)), float(m.group(4))),
                        m.group(5): (float(m.group(6)), float(m.group(7)))}
                scan = "drift" if "drift_scan" in fn else "resist"
                other = "resist" if scan == "drift" else "drift"
                run, sr = locate(t, wins)
                out.append({"time": t.isoformat(sep=" "), "scan": scan,
                            "value_v": vals[scan][0], "other_v": vals[other][0],
                            "i_ua": vals[scan][1], "run": run, "subrun": sr,
                            "source": fn, "join": "by-time"})
    out.sort(key=lambda r: r["time"])
    return out


if __name__ == "__main__":
    data = rows()
    cols = ["time", "scan", "value_v", "other_v", "i_ua", "run", "subrun",
            "source", "join"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(data)
    print("wrote %s (%d points)" % (OUT, len(data)), file=sys.stderr)
    unmatched = [r for r in data if not r["run"]]
    if unmatched:
        print("  %d points fell outside every sub-run window (beam-off or "
              "between runs)" % len(unmatched), file=sys.stderr)
    if "--table" in sys.argv:
        print("| time | scan | V | run | sub-run | join |")
        print("|---|---|---:|---|---|---|")
        for r in data:
            print("| %s | %s | %.1f | %s | %s | %s |"
                  % (r["time"], r["scan"], r["value_v"], r["run"] or "—",
                     r["subrun"] or "—", r["join"]))
