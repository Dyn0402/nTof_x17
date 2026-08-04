#!/usr/bin/env python3
"""Harvest the objective run/sub-run inventory for the SPS H4 campaign from
banco and write it to `run_inventory.json` next to this file.

This is deliberately *only* what the filesystem knows -- file names, FEU ids,
byte counts, mtimes. No interpretation, no HV, no physics. It exists so the
written timeline in `RUN_TIMELINE.md` has something independent to be checked
against, because on this campaign the DAQ's own `run_config.json` is known to
lie (run_61's still says the pre-9AM gas and tilt, and its `start_time` is the
16:06 restart, not when the run began).

Two passes, because the raw run directories get pruned as disk fills:

- `dream_run/` + `runs/` -- raw `.fdf` files. Filenames carry the DAQ wall
  clock (`..._datrun_260802_12H45_000_03.fdf`) and the FEU id in the last
  field (`_01` = FEU1 uRWELL front, `_03` = FEU3 = det4/mx17_E, `_05` = FEU5
  uRWELL back). Present for runs 22-54 and 59-61.
- `analysis/` -- banco's auto-QA PNGs. Survives pruning, so it is the only
  timing left for runs 55-58 and 60. Its per-sub-run detector list is *not*
  a reliable det4-presence flag: the auto-pipeline stops emitting `mx17_E`
  after run_56/meshscan_m40V even though FEU3 keeps writing data all the way
  through run_61. det4 has to be decoded by hand.

Usage:
    ../../.venv/bin/python harvest_inventory.py          # refresh the JSON
    ../../.venv/bin/python harvest_inventory.py --table  # + print the timeline
"""
import datetime as dt
import json
import os
import subprocess
import sys

HOST = "banco_cern"   # the only one of the ~/.ssh/config banco aliases that
                      # resolves from off-site; banco_int/_ext/_alice time out
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "run_inventory.json")

REMOTE = r'''
import os, json, glob, re, datetime as dt
D = "/local/home/banco/P2_data/TB_July2026_H4"
FDF = re.compile(r"^(?P<pfx>.+?)_(?P<kind>datrun|pedestals_pedthr)_(?P<d>\d{6})_"
                 r"(?P<h>\d{2})H(?P<m>\d{2})_(?P<seq>\d{3})_(?P<feu>\d{2})\.fdf$")

runs = {}
for base in ("dream_run", "runs"):
    for rd in sorted(glob.glob(D + "/" + base + "/run_*")):
        run = os.path.basename(rd)
        R = runs.setdefault(run, {"where": [], "subruns": {}, "loose_files": []})
        R["where"].append(base)
        for sr in sorted(os.listdir(rd)):
            sp = os.path.join(rd, sr)
            if not os.path.isdir(sp):
                continue
            S = R["subruns"].setdefault(sr, {
                "feus": {}, "datrun_stamps": [], "ped_stamps": [], "nfdf": 0,
                "bytes": 0, "hv_monitor": None, "logs": [], "scan_logs": [],
                "mtime_first": None, "mtime_last": None})
            ts = []
            for root, _, fs in os.walk(sp):
                for f in fs:
                    p = os.path.join(root, f)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    ts.append(st.st_mtime)
                    m = FDF.match(f)
                    if m:
                        S["nfdf"] += 1
                        S["bytes"] += st.st_size
                        feu = m.group("feu")
                        S["feus"][feu] = S["feus"].get(feu, 0) + 1
                        dd = m.group("d")
                        stamp = ("20%s-%s-%sT%s:%s" % (dd[:2], dd[2:4], dd[4:],
                                                       m.group("h"), m.group("m")))
                        key = ("datrun_stamps" if m.group("kind") == "datrun"
                               else "ped_stamps")
                        S[key].append(stamp)
                    if f == "hv_monitor.csv":
                        S["hv_monitor"] = p
                    if f.startswith("RunCtrl_") and f.endswith(".log"):
                        S["logs"].append(f)
                    if f.endswith((".log", ".csv")) and ("scan" in f or "hold" in f):
                        S["scan_logs"].append(f)
            if ts:
                S["mtime_first"] = dt.datetime.fromtimestamp(min(ts)).isoformat(timespec="seconds")
                S["mtime_last"] = dt.datetime.fromtimestamp(max(ts)).isoformat(timespec="seconds")
            for k in ("datrun_stamps", "ped_stamps", "logs", "scan_logs"):
                S[k] = sorted(set(S[k]))
        for f in sorted(os.listdir(rd)):
            p = os.path.join(rd, f)
            if os.path.isfile(p):
                R["loose_files"].append({
                    "name": f, "bytes": os.path.getsize(p),
                    "mtime": dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")})

ana = {}
for rd in sorted(glob.glob(D + "/analysis/run_*")):
    subs = {}
    for sr in sorted(os.listdir(rd)):
        sp = os.path.join(rd, sr)
        if not os.path.isdir(sp):
            continue
        dets = [d for d in sorted(os.listdir(sp)) if os.path.isdir(os.path.join(sp, d))]
        ts = []
        for root, _, fs in os.walk(sp):
            for f in fs:
                try:
                    ts.append(os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
        subs[sr] = {"detectors": dets,
                    "qa_first": dt.datetime.fromtimestamp(min(ts)).isoformat(timespec="seconds") if ts else None,
                    "qa_last": dt.datetime.fromtimestamp(max(ts)).isoformat(timespec="seconds") if ts else None}
    ana[os.path.basename(rd)] = subs

cfgs = {}
for p in sorted(glob.glob(D + "/runs/run_*/run_config.json")):
    cfgs[os.path.basename(os.path.dirname(p))] = json.load(open(p))

print(json.dumps({"runs": runs, "analysis": ana, "run_configs": cfgs},
                 indent=1, default=str))
'''


def harvest():
    r = subprocess.run(["ssh", HOST, "python3 -"], input=REMOTE,
                       capture_output=True, text=True, timeout=600)
    if r.returncode:
        sys.exit("ssh/harvest failed:\n" + r.stderr[-2000:])
    d = json.loads(r.stdout)
    d["_harvested"] = dt.datetime.now().isoformat(timespec="seconds")
    d["_host"] = HOST
    with open(OUT, "w") as f:
        json.dump(d, f, indent=1)
    return d


def table(d):
    """Chronological sub-run table: raw where we still have it, QA mtimes as
    the fallback for pruned runs (marked 'qa')."""
    rows = []
    for run, R in d["runs"].items():
        for sr, S in R["subruns"].items():
            if S["mtime_first"]:
                rows.append((S["mtime_first"], S["mtime_last"], run, sr, "raw",
                             ",".join(sorted(S["feus"])), S["nfdf"],
                             S["bytes"] / 1e9, S["scan_logs"]))
    seen = {(r[2], r[3]) for r in rows}
    for run, subs in d["analysis"].items():
        for sr, S in subs.items():
            if (run, sr) in seen or not S["qa_first"]:
                continue
            rows.append((S["qa_first"], S["qa_last"], run, sr, "qa ",
                         "/".join(S["detectors"]), 0, 0.0, []))
    for t0, t1, run, sr, src, feus, n, gb, logs in sorted(rows):
        print("%s -> %s  %-7s %-24s %s [%-14s] fdf=%3d %6.1f GB"
              % (t0, (t1 or "")[11:], run, sr, src, feus, n, gb))
        if logs:
            print("%53s scan logs: %s" % ("", ", ".join(logs)))


if __name__ == "__main__":
    data = harvest()
    print("wrote %s (%d runs raw, %d runs QA)"
          % (OUT, len(data["runs"]), len(data["analysis"])), file=sys.stderr)
    if "--table" in sys.argv:
        table(data)
