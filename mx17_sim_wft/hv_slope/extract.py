#!/usr/bin/env python3
"""
HV-slope extraction — data leg.

Pulls a *threshold-free* peak-strip amplitude per event per view out of the
det3 saturday-scan mesh-voltage ladder, so the measured d ln A / dV can be
compared against the Garfield gain ladder.

Why this observable and this selection
--------------------------------------
* **Same observable as T14.** `t14_compare.py` compares `wfm.max(axis=1)` on
  the raw 32 x 60 ns samples read through `wft.io.FeuReader` (pedestal + CNS).
  This script reads exactly the same way, so the slope is measured on the same
  quantity whose *normalization* is x0.55-0.63 low in the sim.
* **No amplitude threshold.** The peak strip is the plane's strongest strip,
  full stop — no 5 sigma cut is applied before taking the maximum. A threshold
  would truncate the low tail and bias the MPV *up* at low voltage, faking a
  shallower slope. (A 5 sigma count is still recorded per event as a
  diagnostic, never as a selection.)
* **A voltage-independent event population.** The bench trigger is external to
  det3, and the M3 telescope runs at fixed HV, so requiring a good M3 track
  (chi2 < 1, NClus = 4) pointing into det3 selects the *same physical
  population* at every mesh voltage. Detector inefficiency at low V then shows
  up honestly as low-amplitude entries instead of as missing events.
* M3 is used for pointing only — reference side, no det3 geometry is
  reconstructed here (see RECONSTRUCTION_BASIS.md).

Output: one parquet row per (subrun, view, event) with the ref-frame track
position, so the fiducial region can be chosen downstream.

    python3 mx17_sim_wft/hv_slope/extract.py --out <dir>/peaks.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "mx_june_cosmic_qa"))
sys.path.insert(0, os.path.join(REPO, "cosmic_bench_analysis"))

RUN = "mx17_det3_saturday_scan_6-27-26"
BASE = "/home/dylan/x17/cosmic_bench/det3/"
SUBRUN_RE = re.compile(r"^hv_scan(2?)_resist_(\d+)V_drift_1000V$")
# the drift ladder at fixed mesh voltage — the control that says how much of
# the mesh-voltage slope could be drift-field contamination (see report)
DRIFT_RE = re.compile(r"^drift_scan_resist_490V_drift_(\d+)V$")
SIGMA = 5.0            # diagnostic only — never a selection in this script
DET_Z = 702.0


def subruns(base, run, drift=False):
    out = []
    for d in sorted(os.listdir(os.path.join(base, run))):
        m = SUBRUN_RE.match(d)
        if m and not drift:
            out.append((d, int(m.group(2)), "scan2" if m.group(1) else "scan1"))
        m = DRIFT_RE.match(d)
        if m and drift:
            out.append((d, int(m.group(1)), "drift"))
    return sorted(out, key=lambda t: (t[2], t[1]))


def ray_positions(cfg):
    """event id -> (x, y) of the M3 reference track at the det3 plane."""
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    from M3RefTracking import M3RefTracking
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    x, y, evn = rays.get_xy_positions(DET_Z)
    x, y, evn = np.asarray(x), np.asarray(y), np.asarray(evn)
    ok = np.isfinite(x) & np.isfinite(y)
    return {int(e): (float(a), float(b))
            for e, a, b in zip(evn[ok], x[ok], y[ok])}


def view_rows(decoded_dir, feu, pos, refs, subrun, volt, scan, view):
    """Threshold-free peak-strip amplitude for every reference-matched event."""
    from wft.io import FeuReader

    files = sorted(glob.glob(os.path.join(decoded_dir, f"*_{feu:02d}.root")))
    valid = ~np.isnan(pos)
    # position order of the valid strips, so 'neighbour' means physically
    # adjacent rather than adjacent in channel number
    order = np.argsort(pos[valid])
    chs = np.flatnonzero(valid)[order]
    rank = np.full(512, -1)
    rank[chs] = np.arange(len(chs))

    rows = []
    for path in files:
        rdr = FeuReader(path)
        noise = np.where(rdr.noise > 0, rdr.noise, np.inf)
        med_noise = float(np.median(rdr.noise[valid]))
        want = set(int(e) for e in rdr.event_ids) & set(refs)
        if not want:
            continue
        for eid, _ftst, wfm in rdr.iter_events(want):
            amp = wfm.max(axis=1)
            amp_v = np.where(valid, amp, -np.inf)
            pk = int(np.argmax(amp_v))            # NO threshold here
            a = float(amp[pk])
            r = rank[pk]
            nb = 0.0
            for dr in (-1, +1):
                if 0 <= r + dr < len(chs):
                    nb += float(amp[chs[r + dr]])
                    # neighbours sit at ~30 % of the peak, so this stays off
                    # the rail for ~3x longer than the peak strip does
            sig = amp / noise
            over = valid & (sig >= SIGMA)
            rx, ry = refs[eid]
            rows.append(dict(
                subrun=subrun, volt=volt, scan=scan, view=view,
                event_id=int(eid), peak_ch=pk, peak_amp=a, nb_amp=nb,
                n_over=int(over.sum()),
                q_over=float(amp[over].sum()) if over.any() else 0.0,
                ref_x=rx, ref_y=ry, noise=med_noise,
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="restrict to these subrun names")
    ap.add_argument("--subrun", nargs=2, metavar=("NAME", "VOLT"),
                    help="extract one arbitrary subrun (e.g. the 490 V long "
                         "run, whose name matches neither ladder pattern)")
    ap.add_argument("--drift", action="store_true",
                    help="run the drift ladder instead of the mesh ladder; "
                         "`volt` then means the DRIFT voltage at a fixed 490 V "
                         "mesh")
    a = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.io import strip_position_map

    base_cfg = get_config("sat_det3")
    base_cfg.BASE_PATH = a.base
    base_cfg.RUN = a.run
    pos_maps = strip_position_map(base_cfg)

    allrows = []
    todo = ([(a.subrun[0], int(a.subrun[1]), "single")] if a.subrun
            else subruns(a.base, a.run, drift=a.drift))
    for sub, volt, scan in todo:
        if a.only and sub not in a.only:
            continue
        cfg = get_config("sat_det3")
        cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = a.base, a.run, sub
        decoded = os.path.join(a.base, a.run, sub, "decoded_root")
        if not os.path.isdir(decoded):
            print(f"[skip] {sub}: no decoded_root")
            continue
        refs = ray_positions(cfg)
        print(f"[{sub}] {volt} V  {len(refs):,} M3-good rays", flush=True)
        for view, feu in (("x", cfg.MX17_FEU_X), ("y", cfg.MX17_FEU_Y)):
            r = view_rows(decoded, feu, pos_maps[feu], refs, sub, volt, scan, view)
            print(f"   {view}: {len(r):,} events", flush=True)
            allrows += r

    df = pd.DataFrame(allrows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out)
    print(f"wrote {a.out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
