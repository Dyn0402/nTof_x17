"""Count DREAM events per sub-run, campaign-wide, from EOS.

One entry of the `nt` tree in a decoded_root file is one triggered event on
that FEU.  FEU 1 is read as the proxy for the sub-run: every FEU sees the same
global triggers, so this is an event count, not a per-FEU count.  Only the ROOT
header is touched.  The sub-run's start time comes out of the file name, which
is the same stamp the DAQ's own ledger uses, so this does not depend on the
n_TOF side at all.
"""
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import uproot

ROOT = "/eos/experiment/ntof/data/x17/july_beam/runs"
files = sorted(glob.glob(os.path.join(ROOT, "run_*", "*", "decoded_root", "*_01.root")))
sys.stderr.write("%d files\n" % len(files))

# beam vs cosmics is per run, from the DAQ's own config, not from sub-run names
beam_type = {}
for cfg in sorted(glob.glob(os.path.join(ROOT, "run_*", "run_config.json"))):
    run = cfg.split(os.sep)[-2]
    try:
        with open(cfg) as fh:
            beam_type[run] = json.load(fh).get("beam_type", "?")
    except Exception:
        beam_type[run] = "?"

PAT = re.compile(r"datrun_(\d{2})(\d{2})(\d{2})_(\d{2})H(\d{2})_(\d{3})_")


def one(path):
    parts = path.split(os.sep)
    run, sub = parts[-4], parts[-3]
    m = PAT.search(os.path.basename(path))
    if m:
        yy, mm, dd, hh, mi, tag = m.groups()
        stamp = "20%s-%s-%sT%s:%s" % (yy, mm, dd, hh, mi)
    else:
        stamp, tag = "", ""
    try:
        with uproot.open(path) as fh:
            n = fh["nt"].num_entries
    except Exception as exc:
        return "%s,%s,%s,%s,,%s," % (run, sub, tag, stamp, type(exc).__name__)
    return "%s,%s,%s,%s,%d,," % (run, sub, tag, stamp, n)


out = open(os.path.expanduser("~/x17count/events_per_subrun.csv"), "w")
out.write("run,subrun,tag,stamp,events,error,beam_type\n")
with ThreadPoolExecutor(max_workers=16) as ex:
    for i, line in enumerate(ex.map(one, files)):
        run = line.split(",")[0]
        out.write(line + beam_type.get(run, "?") + "\n")
        if i % 500 == 0:
            out.flush()
            sys.stderr.write("%d/%d\n" % (i, len(files)))
out.close()
sys.stderr.write("done\n")
