#!/usr/bin/env python3
"""Drift-ladder span per HV plateau: the estimator-route v(E) curve.

For each plateau of a dataset, the t10-t90 span of the in-time hit-time
distribution (amp >= 150 ADC, threshold-robust) tracks the drift-time ladder
length: span ~ gap / v_drift + const(shaping).  Against the known 28.8 mm
gap this gives v(E) up to the additive shaping constant, which cancels in
ratios and can be removed with the run_71 end-lobe anchor
(v = 14 um/ns at 233 V/cm, wet CF4).

Works straight off the det4 hits trees + dream_daq t0s -- no pairing needed.

  ../../.venv/bin/python ladder_span.py run63_rot25 run55_flatdrift
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets

TIME_LO, TIME_HI = 200.0, 3800.0
AMP_MIN = 150.0
GAP_MM = 28.8


def spans(name):
    D = datasets.get(name)
    per = {}
    for sub, stem, t0, idxs in D["subruns"]:
        d = os.path.join(D["stage"], sub)
        if not os.path.isdir(d):
            d = D["stage"]
        for fn in sorted(glob.glob(os.path.join(d, f"hits_{sub}_*_03.root"))):
            try:
                with uproot.open(fn) as f:
                    if "hits" not in f:
                        continue
                    a = f["hits"].arrays(
                        ["eventId", "trigger_timestamp_ns", "time",
                         "amplitude"], library="np")
            except Exception:
                continue
            tw = t0 + a["trigger_timestamp_ns"] / 1e9
            lab = datasets.plateau_of(name, tw)
            ok = (a["amplitude"] >= AMP_MIN) & (a["time"] >= TIME_LO) & \
                 (a["time"] <= TIME_HI)
            for l in np.unique(lab[ok]):
                if not l:
                    continue
                per.setdefault(str(l), []).append(a["time"][ok & (lab == l)])
    out = {}
    hv = {p[0]: (p[3], p[4]) for p in D["plateaus"]}
    for l, chunks in per.items():
        tt = np.concatenate(chunks)
        if len(tt) < 500:
            continue
        q10, q50, q90 = np.percentile(tt, (10, 50, 90))
        dv, rv = hv.get(l, (np.nan, np.nan))
        out[l] = dict(drift_V=float(dv), field_Vcm=float(dv / (GAP_MM / 10)),
                      n=int(len(tt)), t10=float(q10), t50=float(q50),
                      t90=float(q90), span=float(q90 - q10))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(datasets.DATASETS))
    ap.add_argument("--anchor", default="233:14.0",
                    help="field:velocity anchor to convert span->v "
                         "(default run_71 end-lobe, 233 V/cm : 14 um/ns)")
    args = ap.parse_args()
    af, av = (float(x) for x in args.anchor.split(":"))

    for name in args.names:
        D = datasets.get(name)
        r = spans(name)
        print(f"\n=== {name} ({D['mount']}, {D['gas']})")
        # shaping constant from the anchor: span = gap/v + c0, anchored where
        # the field matches the anchor field (within 10 V/cm)
        c0 = None
        for l, x in r.items():
            if abs(x["field_Vcm"] - af) < 10:
                c0 = x["span"] - GAP_MM * 1e3 / av
        tag = f" (c0 from {af:.0f} V/cm anchor)" if c0 is not None else \
            "  [no anchor plateau: v quoted only as gap/(span-c0) once c0 known]"
        print(f"  {'plat':>6} {'V/cm':>6} {'hits':>9} {'t10':>6} {'t90':>6} "
              f"{'span':>6}" + (f" {'v [um/ns]':>10}" if c0 is not None else "")
              + tag)
        for l, x in sorted(r.items(), key=lambda kv: -kv[1]["drift_V"]):
            line = (f"  {l:>6} {x['field_Vcm']:6.0f} {x['n']:9d} "
                    f"{x['t10']:6.0f} {x['t90']:6.0f} {x['span']:6.0f}")
            if c0 is not None and x["span"] > c0:
                line += f" {GAP_MM * 1e3 / (x['span'] - c0):10.2f}"
            print(line)
        out = D["stage"] + f"ladder_span_{name}.json"
        with open(out, "w") as f:
            json.dump(r, f, indent=1)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
