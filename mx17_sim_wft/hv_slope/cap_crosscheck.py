#!/usr/bin/env python3
"""
Does the undershoot cap follow the FEU (electronics) or the layer (detector)?

det3 shows an undershoot that saturates at about -290 ADC on its Y view and
stays proportional to signal on X. CNS is excluded (`cns_undershoot.py`) and
neither the mesh nor the drift ladder can separate an amplitude ceiling from a
charge ceiling (`cap_scan.py`) — on det3 the two are locked together, and the
view that caps is also the view on FEU 8, so "resistive-side layer" and
"FEU 8 electronics" are confounded.

The 6-26 overnight run breaks that confound, because it wires the same two
layer types onto different FEUs:

              X layer      Y layer (resistive side)
    det3        FEU 7            FEU 8
    det6        FEU 3            FEU 4
    det7        FEU 6            FEU 8

If the cap follows the LAYER it appears on FEU 4 and FEU 8 and never on 3, 6 or
7 — three detectors, three different FEUs on the Y side. If it follows the
ELECTRONICS it appears on FEU 8 only, and det6's FEU 4 looks like an X plane.

Selection note: this is a SHAPE comparison at fixed amplitude, so it does not
use M3 — every event with a >= 5 sigma strip enters and the comparison is made
inside absolute amplitude bins. That is deliberately looser than the det3 work
(where the population mattered because amplitude itself was the observable);
the price is that spark and junk events are not vetoed, which the report says.

    python3 mx17_sim_wft/hv_slope/cap_crosscheck.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path[:0] = [_HERE, _REPO, os.path.join(_REPO, "mx_june_cosmic_qa"),
                os.path.join(_REPO, "cosmic_bench_analysis")]

from cns_undershoot import _Reader                          # noqa: E402

SIGMA = 5.0
RAIL = 3500.0
BINS = [(400, 800), (800, 1200), (1200, 1700), (1700, 2300), (2300, 3000),
        (3000, 3500)]

D67 = ("/home/dylan/x17/cosmic_bench/det6_det7/"
       "mx17_det6_det7_overnight_6-26-26/longer_run")
D3 = ("/home/dylan/x17/cosmic_bench/det3/mx17_det3_saturday_scan_6-27-26/"
      "long_run_resist_490V_drift_1000V")

# (label, directory, FEU, detector, layer)
LEGS = [("det3 X", D3, 7, "det3", "X"),
        ("det3 Y", D3, 8, "det3", "Y"),
        ("det6 X", D67, 3, "det6", "X"),
        ("det6 Y", D67, 4, "det6", "Y"),
        ("det7 X", D67, 6, "det7", "X"),
        ("det7 Y", D67, 8, "det7", "Y")]
MAX_FILES = 3


def leg(path, feu, max_files=MAX_FILES):
    rows = []
    files = sorted(glob.glob(os.path.join(path, "decoded_root",
                                          f"*_{feu:02d}.root")))[:max_files]
    for f in files:
        rdr = _Reader(f, cns=True)
        noise = np.where(rdr.noise > 0, rdr.noise, np.inf)
        for eid, wfm, _cm in rdr.iter_events():
            amp = wfm.max(axis=1)
            if not (amp / noise >= SIGMA).any():
                continue
            pk = int(np.argmax(amp))
            w = wfm[pk]
            ipk = int(np.argmax(w))
            a = float(w[ipk])
            if ipk > 20 or a < BINS[0][0]:
                continue
            rows.append((a, float(w[ipk + 1:].min())))
    return pd.DataFrame(rows, columns=["peak_amp", "undershoot_adc"])


def profile(g):
    out = []
    for lo, hi in BINS:
        m = (g.peak_amp >= lo) & (g.peak_amp < hi)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                        adc=float(g.undershoot_adc[m].median()) if m.sum() >= 20 else None,
                        frac=float((g.undershoot_adc[m] / g.peak_amp[m]).median())
                        if m.sum() >= 20 else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/x17/response_sim/hv_slope/cns"))
    a = ap.parse_args()
    cache = os.path.join(a.out_dir, "crosscheck.parquet")

    if os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        frames = []
        for label, path, feu, det, layer in LEGS:
            g = leg(path, feu)
            g["label"], g["feu"], g["det"], g["layer"] = label, feu, det, layer
            print(f"{label:8s} FEU {feu}: {len(g):,} events", flush=True)
            frames.append(g)
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(cache)

    un = df[df.peak_amp < RAIL]
    out = {}
    for label, _p, feu, det, layer in LEGS:
        g = un[un.label == label]
        out[label] = dict(feu=feu, det=det, layer=layer, n=int(len(g)),
                          profile=profile(g))
    json.dump(out, open(os.path.join(a.out_dir, "crosscheck.json"), "w"),
              indent=1)

    print(f"\n{'leg':10s}{'FEU':>4}  " +
          "".join(f"{lo}-{hi}".rjust(12) for lo, hi in BINS))
    print(" " * 14 + "median undershoot [ADC] in fixed absolute bins")
    for label, _p, feu, det, layer in LEGS:
        cells = "".join((f"{p['adc']:12.0f}" if p["adc"] is not None else
                         f"{'-':>12}") for p in out[label]["profile"])
        print(f"{label:10s}{feu:>4}  {cells}")
    print(f"\n{'leg':10s}{'FEU':>4}  ratio of the top measurable bin to the "
          f"800-1200 bin  (a CAP gives a small ratio, proportional gives ~3)")
    for label, _p, feu, det, layer in LEGS:
        pr = out[label]["profile"]
        base = pr[1]["adc"]
        top = next((p["adc"] for p in reversed(pr) if p["adc"] is not None), None)
        topbin = next((f"{p['lo']}-{p['hi']}" for p in reversed(pr)
                       if p["adc"] is not None), "-")
        if base and top:
            print(f"{label:10s}{feu:>4}  {top / base:5.2f}   "
                  f"({top:.0f} / {base:.0f}, top bin {topbin})")
    print("\nwrote", os.path.join(a.out_dir, "crosscheck.json"))


if __name__ == "__main__":
    main()
