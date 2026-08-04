#!/usr/bin/env python3
"""det4 tilt, redone on the CLEAN selection — the §6 open item.

The June/RAW tilt numbers (~0.2–0.4° in X) came from tilt_m70V.py running on
contaminated mean waveforms: no pre-window gate, no bad-channel mask, and the
tilt signal lives in the tail exactly where the pile-up and the oscillating
channels dumped charge.  The SIGN was robust (three runs agree); the MAGNITUDE
was not to be carried as a constant until redone clean.  This redoes it.

Same two estimators as tilt_m70V.py, on robust_waveforms.build_clean() traces:

  1. centroid walk    d<x>/dt = v_drift * tan(theta)
  2. arrival-time antisymmetry across ±1 strips

v_drift is taken from the DATA, not from tables (the gas carried ~1.3–1.7 %
water; run-by-run table values are unusable): the run_71 end-lobe measurement
gives v(233 V/cm) ≈ 13–15 µm/ns, and measured v ∝ E across 92–233 V/cm, so
each plateau's v scales with its drift voltage.

  python tilt_clean.py run71_raw [--v233 14.0]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "det4_sps_assessment"))
import datasets                                          # noqa: E402
from det4_sps_map import PITCH_MM                        # noqa: E402
from robust_waveforms import build_clean, SNS            # noqa: E402

V233_REF_V = 700.5           # drift voltage of the 233 V/cm end-lobe point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--wf", default="")
    ap.add_argument("--q0", default="400,3000")
    ap.add_argument("--v233", type=float, default=14.0,
                    help="um/ns at the 700 V operating field, from the "
                         "end-of-ladder lobe (13-15); v scales ∝ V below it")
    ap.add_argument("--nrel", type=int, default=12,
                    help="samples about the central peak used for the walk")
    args = ap.parse_args()
    D = datasets.get(args.dataset)
    wf = args.wf or D["stage"] + f"wf_{args.dataset}_det4only.npz"
    q0lo, q0hi = (float(x) for x in args.q0.split(","))

    C = build_clean(wf, D, q0lo, q0hi)
    NREL = args.nrel

    drift_of = {lab: dr for lab, _lo, _hi, dr, _re in D["plateaus"]}

    for v in ("x", "y"):
        isv = C.t_view == v
        cmap = C.cmap[v]
        print(f"\n================ view {v.upper()} ================")
        for lab, *_ in D["plateaus"]:
            vd = args.v233 * drift_of[lab] / V233_REF_V / 1000.0   # mm/ns
            evsel = (C.plateau == lab) & (cmap >= 0)
            if evsel.sum() < 100:
                continue
            keep = np.zeros(C.n_ev, bool)
            keep[np.flatnonzero(evsel)] = True

            sel = isv & (np.abs(C.t_d) <= 4) & keep[C.t_ev]
            tr = C.trace[sel]
            td = C.t_d[sel].astype(float)
            tev = C.t_ev[sel]
            pk_c = C.peak_smp[cmap[tev]]           # central peak sample/event

            # ---- 1. centroid walk about the central peak, clean traces.
            # Positive charge only: the undershoot would otherwise flip the
            # weights sign-side and the centroid with them.
            n = 2 * NREL + 1
            num = np.zeros(n)
            den = np.zeros(n)
            smp = np.arange(C.nsmp)
            for j in range(len(tr)):
                rel = smp - pk_c[j]
                m = (np.abs(rel) <= NREL) & np.isfinite(tr[j]) & (tr[j] > 0)
                np.add.at(num, rel[m] + NREL, tr[j][m] * td[j])
                np.add.at(den, rel[m] + NREL, tr[j][m])
            cen = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
            t = (np.arange(n) - NREL) * SNS
            good = np.isfinite(cen) & (den > 0.02 * np.nanmax(den))
            sl, ic = np.polyfit(t[good], cen[good] * PITCH_MM, 1)
            theta = np.degrees(np.arctan(abs(sl) / vd))
            print(f"\n  {lab} ({int(evsel.sum())} ev, v_d {vd*1000:.1f} um/ns):")
            print(f"    centroid walk  dx/dt = {sl*1000:+8.4f} um/ns "
                  f"-> tan = {sl/vd:+.4f} -> theta = {theta:5.2f} deg")

            # ---- 2. arrival-time antisymmetry of the ±1 peak times
            dtm = {}
            for dd in (1, -1):
                s2 = isv & (C.t_d == dd) & keep[C.t_ev]
                dt = (C.peak_smp[s2].astype(float)
                      - C.peak_smp[cmap[C.t_ev[s2]]]) * SNS
                core = np.abs(dt) < 600
                if core.sum() > 30:
                    dtm[dd] = float(np.median(dt[core]))
            if 1 in dtm and -1 in dtm:
                sym1 = 0.5 * (dtm[1] + dtm[-1])
                asym1 = 0.5 * (dtm[1] - dtm[-1])
                line = (f"    dt(+1) {dtm[1]:+6.1f}  dt(-1) {dtm[-1]:+6.1f} ns"
                        f"   sym {sym1:+6.1f} (sharing)  "
                        f"antisym {asym1:+6.1f} (tilt)")
                if abs(asym1) > 1e-6:
                    th2 = np.degrees(np.arctan(PITCH_MM / (vd * abs(asym1))))
                    line += f" -> theta {th2:5.2f} deg"
                print(line)


if __name__ == "__main__":
    main()
