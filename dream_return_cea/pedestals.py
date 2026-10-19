"""Proof-of-life pedestals of the DREAM DAQ after its return to CEA.

On 5 October 2026 one pedestal run was taken on each FEU, one at a time
(`Sys Topo` had a single FEU enabled per run).  This module turns those FDFs
into per-channel statistics.

  1. decode every `*_pedthr_*.fdf` with the C++ decoder (cached as ROOT)
  2. tie each file to its FEU:  filename `_feuN_`  ->  `Feu N` in the cfg
     ->  `Feu_RunCtrl_Id`,  and check that against the FEU ID the decoder
     reads from the data stream itself
  3. per channel: baseline, raw sigma, and sigma after common-noise
     subtraction -- the same decomposition `ntof_pedestal_qa` used for the
     n_TOF campaign (baseline first, then per-chip per-sample median)

    ../.venv/bin/python -m dream_return_cea.pedestals
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import subprocess
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
SRC = os.path.expanduser("~/Desktop/ped_validation_5-10-26")
CFG = os.path.join(SRC, "ped_val.cfg")
DECODE = os.path.expanduser(
    "~/CLionProjects/mm_strip_reconstruction/cmake-build-release/decoder/decode")

NCH, BLK = 512, 64
FDF_RE = re.compile(r"_feu(\d+)_pedthr_(\d{6}_\d{2}H\d{2})_\d{3}_(\d{2})\.fdf$")

# One noise decomposition for the whole repo: reuse the campaign's.
_spec = importlib.util.spec_from_file_location(
    "extract_pedestals",
    os.path.join(REPO, "ntof_pedestal_qa", "lxplus", "extract_pedestals.py"))
EP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EP)


def cfg_feu_ids(path=CFG):
    """{cfg FEU slot: Feu_RunCtrl_Id} from the run-control cfg."""
    out = {}
    for line in open(path):
        m = re.match(r"^\s*Feu\s+(\d+)\s+Feu_RunCtrl_Id\s+(\d+)", line)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def cfg_value(key, path=CFG):
    for line in open(path):
        m = re.match(rf"^\s*Feu\s+\*\s+(?:DrmClk\s+)?{key}\s+(\S+)", line)
        if m:
            return m.group(1)
    return None


def decode(fdf):
    """FDF -> ROOT (cached in data/); returns (root path, decoder log)."""
    os.makedirs(DATA, exist_ok=True)
    stem = os.path.basename(fdf)[:-4]
    root = os.path.join(DATA, stem + ".root")
    log = os.path.join(DATA, stem + ".decode.log")
    if not (os.path.exists(root) and os.path.exists(log)):
        r = subprocess.run([DECODE, fdf, root], capture_output=True, text=True,
                           check=True)
        open(log, "w").write(r.stdout + r.stderr)
    return root, open(log).read()


def parse_decode_log(text):
    feu = sorted({int(x) for x in re.findall(r"reading FEU (\d+)", text)})
    num = lambda k: re.search(rf"{k}\s*:\s*([\d.]+)", text)
    return dict(stream_ids=feu,
                events=int(num("events written").group(1)),
                missing=int(num("events MISSING").group(1)),
                completeness=float(num("sample completeness").group(1)))


def acquisition_window(fdf, stamp):
    """(start, end): start from the file-name stamp (minute resolution, set
    when the run opened), end from the FDF's mtime (last byte written)."""
    d, t = stamp.split("_")
    start = datetime(2000 + int(d[:2]), int(d[2:4]), int(d[4:6]),
                     int(t[:2]), int(t[3:5]))
    end = datetime.fromtimestamp(os.path.getmtime(fdf))
    return start, end


def failed_attempts():
    """RunCtrl logs whose run never opened an FEU -- part of the record."""
    out = []
    for log in sorted(glob.glob(os.path.join(SRC, "RunCtrl_*.log"))):
        text = open(log).read()
        if "FeuCtrl_Open failed" in text:
            stamp = re.search(r"RunCtrl_(\d{6}_\d{2}H\d{2})", log).group(1)
            topo = os.path.join(SRC, f"Mx17_init_{stamp}.cfg_cpy")
            slot = None
            if os.path.exists(topo):
                m = re.search(r"^\s*Sys Topo Feu\s+(\d+)", open(topo).read(), re.M)
                slot = int(m.group(1)) if m else None
            out.append(dict(stamp=stamp, slot=slot,
                            reason="FeuCtrl_Open failed"))
    return out


def build():
    ids = cfg_feu_ids()
    rows = []
    for fdf in sorted(glob.glob(os.path.join(SRC, "*_pedthr_*.fdf")),
                      key=lambda p: int(FDF_RE.search(p).group(1))):
        m = FDF_RE.search(fdf)
        slot, stamp = int(m.group(1)), m.group(2)
        root, log = decode(fdf)
        dl = parse_decode_log(log)
        s = EP.stats_for_file(root, max_samples=10 ** 9)   # every event
        _, fw_avr, fw_std = EP.parse_aux(fdf[:-4] + "_ped.aux")
        start, end = acquisition_window(fdf, stamp)
        rows.append(dict(slot=slot, feu_id=ids[slot], stamp=stamp,
                         start=start, end=end, fdf=os.path.basename(fdf),
                         id_match=dl["stream_ids"] == [ids[slot]],
                         fw_avr=fw_avr, fw_std=fw_std, **dl, **s))
    return rows


def save(rows):
    arrays = {}
    meta = []
    for r in rows:
        k = f"{r['slot']:02d}"
        for name in ("mean", "raw_sigma", "cns_sigma", "cm_rms",
                     "fw_avr", "fw_std"):
            arrays[f"{k}/{name}"] = r[name]
        meta.append({k2: (v.isoformat() if isinstance(v, datetime) else
                          v.item() if isinstance(v, np.generic) else v)
                     for k2, v in r.items() if not isinstance(v, np.ndarray)})
    np.savez_compressed(os.path.join(DATA, "ped_stats.npz"), **arrays)
    ctx = dict(meta=meta, failed=failed_attempts(),
               RdClk_Div=cfg_value("RdClk_Div"), WrClk_Div=cfg_value("WrClk_Div"),
               samples=cfg_value("Main_Conf_Samples"))
    json.dump(ctx, open(os.path.join(DATA, "ped_meta.json"), "w"), indent=1)


def load():
    """[row dicts] with arrays re-attached, ordered by FEU slot."""
    z = np.load(os.path.join(DATA, "ped_stats.npz"))
    ctx = json.load(open(os.path.join(DATA, "ped_meta.json")))
    rows = []
    for m in ctx["meta"]:
        r = dict(m)
        r["start"] = datetime.fromisoformat(m["start"])
        r["end"] = datetime.fromisoformat(m["end"])
        k = f"{m['slot']:02d}"
        for name in ("mean", "raw_sigma", "cns_sigma", "cm_rms",
                     "fw_avr", "fw_std"):
            r[name] = z[f"{k}/{name}"]
        rows.append(r)
    return rows, ctx


if __name__ == "__main__":
    rows = build()
    save(rows)
    for r in rows:
        print(f"FEU {r['slot']}  ID {r['feu_id']:>3}  stream {r['stream_ids']}  "
              f"{r['start']:%H:%M}-{r['end']:%H:%M:%S}  ev {r['events']}  "
              f"base {np.median(r['mean']):6.1f}  raw {np.median(r['raw_sigma']):5.2f}  "
              f"cmn {np.median(r['cns_sigma']):5.2f}  cm {np.median(r['cm_rms']):5.2f}")
