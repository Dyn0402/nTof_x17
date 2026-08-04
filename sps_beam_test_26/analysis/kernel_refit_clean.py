#!/usr/bin/env python3
"""Kernel refit + charge budget on the CLEAN run_71 library.

Consumes robust_waveforms.py's library (absolute window time, artefact-free)
and redoes two things from RAW_RUN71_REANALYSIS_2026-08-04.md §4:

 1. the cascade-model refit  W_d = alpha_d W_0 + beta_d (W_0 ⊛ K_tau^{|d|}),
    global tau per plateau — this is the drift-invariance TEST (tau, c1, c2
    must not move between 450 and 275 V), not a charge accounting: the betas
    are not physical fractions because the basis W_0 itself loses charge
    sideways and undershoots,
 2. the model-independent charge budget: window-integral area and peak
    amplitude per offset, relative to the central strip.

  python kernel_refit_clean.py run71_raw [--view y]

Quote the Y view (the X view carries the ~0.2–0.4° tilt contamination).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets                                          # noqa: E402
from kernel_fit_m70V import rc_cascade                   # noqa: E402

SNS = 60.0


def fit_plateau(t, W, fit_lo_ns=0.0, fit_hi_ns=None):
    """Global tau + per-offset (alpha, beta) on absolute-time traces."""
    w0 = W[0]
    offs = [dd for dd in (1, -1, 2, -2) if dd in W]
    msk = np.isfinite(w0)
    for dd in offs:
        msk &= np.isfinite(W[dd])
    msk &= t >= fit_lo_ns
    if fit_hi_ns:
        msk &= t <= fit_hi_ns

    def resid(p):
        tau = p[0]
        r = []
        for i, dd in enumerate(offs):
            al, be = p[1 + 2 * i], p[2 + 2 * i]
            mdl = al * w0 + be * rc_cascade(np.nan_to_num(w0), tau, abs(dd))
            r.append((mdl - W[dd])[msk])
        return np.concatenate(r)

    p0 = [500.0] + [0.2, 0.3] * len(offs)
    lo = [5.0] + [0.0, 0.0] * len(offs)
    hi = [5000.0] + [3.0, 3.0] * len(offs)
    res = least_squares(resid, p0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
    tau = res.x[0]
    par = {dd: (res.x[1 + 2 * i], res.x[2 + 2 * i]) for i, dd in enumerate(offs)}
    return tau, par, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--lib", default="")
    ap.add_argument("--view", default="y", choices=("x", "y"))
    ap.add_argument("--basis", default="altr",
                    choices=("altr", "alm", "trim", "med"),
                    help="altr/alm = peak-aligned trim20/median (the kernel "
                         "and budget basis — in absolute time the central "
                         "peak is washed out by ladder jitter); trim/med = "
                         "absolute-window aggregates (containment only)")
    args = ap.parse_args()
    D = datasets.get(args.dataset)
    lib = args.lib or (D["stage"] + "reanalysis_clean/"
                       f"robust_library_{args.dataset}.npz")
    Z = np.load(lib)
    t = Z["t_rel"] if args.basis in ("altr", "alm") else Z["t"]
    v = args.view

    plateaus = sorted({k.split("_")[1] for k in Z.files if k.startswith("med_")})
    print(f"library {os.path.basename(lib)}, view {v.upper()}, "
          f"basis {args.basis}, plateaus {plateaus}")

    fits = {}
    for lab in plateaus:
        W = {}
        for dd in (0, 1, -1, 2, -2, 3, -3):
            k = f"{args.basis}_{lab}_{v}_{dd:+d}"
            if k in Z.files:
                W[dd] = Z[k]
        if 0 not in W or 1 not in W:
            continue
        tau, par, res = fit_plateau(t, W)
        c1 = np.mean([par[d][1] for d in (1, -1) if d in par])
        c2 = np.mean([par[d][1] for d in (2, -2) if d in par])
        a1 = np.mean([par[d][0] for d in (1, -1) if d in par])
        fits[lab] = dict(tau=tau, c1=c1, c2=c2, a1=a1)
        print(f"\n=== {lab}:  tau_s = {tau:6.0f} ns   c1 = {c1:.3f}   "
              f"c2 = {c2:.3f}   alpha(+-1) = {a1:.3f}   (cost {res.cost:.2e})")

        # ---------------- model-independent charge budget (doc §4 table)
        w0 = np.nan_to_num(W[0])
        area0, pk0 = w0.sum(), w0.max()
        print(f"    {'d':>3} {'area/central':>13} {'peak/central':>13}")
        for add in (1, 2, 3):
            pair = [dd for dd in (add, -add) if dd in W]
            if not pair:
                continue
            ar = np.mean([np.nan_to_num(W[dd]).sum() for dd in pair]) / area0
            pk = np.mean([np.nanmax(W[dd]) for dd in pair]) / pk0
            print(f"    ±{add} {ar:13.3f} {pk:13.3f}")
        seven = w0.sum() + sum(np.nan_to_num(W[dd]).sum()
                               for dd in (1, -1, 2, -2, 3, -3) if dd in W)
        sevenpk = pk0 + sum(np.nanmax(W[dd])
                            for dd in (1, -1, 2, -2, 3, -3) if dd in W)
        print(f"    central holds {100 * area0 / seven:.0f} % of the 7-strip "
              f"window integral, {100 * pk0 / sevenpk:.0f} % of the peak sum")

    if len(fits) >= 2:
        labs = list(fits)
        taus = [fits[l]["tau"] for l in labs]
        c1s = [fits[l]["c1"] for l in labs]
        c2s = [fits[l]["c2"] for l in labs]
        print(f"\nDRIFT INVARIANCE across {labs}: "
              f"tau ±{100 * np.ptp(taus) / 2 / np.mean(taus):.1f} %  "
              f"c1 ±{100 * np.ptp(c1s) / 2 / np.mean(c1s):.1f} %  "
              f"c2 ±{100 * np.ptp(c2s) / 2 / np.mean(c2s):.1f} %")


if __name__ == "__main__":
    main()
