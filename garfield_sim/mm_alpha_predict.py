#!/usr/bin/env python3
"""
mm_alpha_predict.py — predict the useful voltage window of a new gas table
===========================================================================
Avalanche jobs get exponentially more expensive with gain: a 200-event batch
at G ~ 4e4 costs ~30 min, at G ~ 1e6 it is hopeless. Before launching a scan
on a new mixture we want to know *which* voltages put it in the gain range we
care about — without running any avalanches.

Method
------
The .gas file stores the Magboltz Townsend (alpha) and attachment (eta)
coefficients. Naively G = exp((alpha - eta) * d), but the tracked gain is
larger because Penning transfer converts Ar excitations into extra ionisation,
and that is applied during microscopic tracking, not in the table.

So we calibrate the shortfall on gases we have already simulated:

    K(gas) = ln(G_simulated) / ((alpha - eta) * d)

and apply the calibration to the new mixture. K is empirical but stable across
Ar-based mixtures at the same Penning setting, which is all we need to pick a
voltage window.

Usage (on lxplus, LCG view sourced):
    # calibrate on already-simulated gases and predict for a new one
    python3 mm_alpha_predict.py --calibrate Ar_iC4H10_95_5 Ar_iC4H10_98_2 \
                                            Ar_iC4H10_90_10 Ar_CO2_70_30 \
                                --predict Ar_CO2_iC4H10_93_5_2 \
                                --pressure CERN_450m
"""

import os
import sys
import json
import argparse
import ctypes

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_config as cfg


def gas_file(label, pressure):
    return os.path.join(cfg.GAS_DIR, f"{label}_{pressure}.gas")


def result_file(label, pressure):
    return os.path.join(cfg.RESULTS_DIR, f"{label}_{pressure}.json")


def load_medium(path):
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError
    import Garfield  # noqa: F401
    gas = ROOT.Garfield.MediumMagboltz()
    if not gas.LoadGasFile(path):
        sys.exit(f"[predict] ERROR: could not load {path}")
    return gas


def alpha_eta(gas, voltages, gap_cm):
    """Townsend and attachment coefficients (1/cm) at each mesh voltage."""
    a_out, e_out = [], []
    for v in voltages:
        e = v / gap_cm
        alpha = ctypes.c_double(0.)
        eta   = ctypes.c_double(0.)
        gas.ElectronTownsend(0., 0., e, 0., 0., 0., alpha)
        gas.ElectronAttachment(0., 0., e, 0., 0., 0., eta)
        a_out.append(alpha.value)
        e_out.append(eta.value)
    return np.array(a_out), np.array(e_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", nargs="+", default=[],
                    help="Gas labels that already have results/ JSONs")
    ap.add_argument("--predict",   required=True, help="New gas label")
    ap.add_argument("--pressure",  default="CERN_450m")
    ap.add_argument("--gap-cm",    type=float, default=cfg.GAP_CM)
    ap.add_argument("--v-min",     type=float, default=340.)
    ap.add_argument("--v-max",     type=float, default=620.)
    ap.add_argument("--v-step",    type=float, default=10.)
    args = ap.parse_args()

    d = args.gap_cm
    ks = []

    print(f"{'gas':<24} {'V':>6} {'alpha':>9} {'eta':>7} {'exp(a-e)d':>12} "
          f"{'G_sim':>12} {'K':>6}")
    print("-" * 82)

    for label in args.calibrate:
        gpath, rpath = gas_file(label, args.pressure), result_file(label, args.pressure)
        if not (os.path.exists(gpath) and os.path.exists(rpath)):
            print(f"[predict] skip {label}: missing gas table or results")
            continue
        res = json.load(open(rpath))
        v   = np.array(res["voltages"], float)
        gsim = np.array(res["gain_mean"], float)
        gas = load_medium(gpath)
        a, e = alpha_eta(gas, v, d)
        eff = (a - e) * d
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.log(gsim) / eff
        for i in range(0, len(v), max(1, len(v) // 4)):
            print(f"{label:<24} {v[i]:6.0f} {a[i]:9.1f} {e[i]:7.2f} "
                  f"{np.exp(eff[i]):12.3g} {gsim[i]:12.3g} {k[i]:6.3f}")
        good = np.isfinite(k) & (gsim > 10)
        ks.extend(k[good].tolist())
        print(f"{label:<24} -> K = {np.mean(k[good]):.3f} +- {np.std(k[good]):.3f}")
        print("-" * 82)

    if ks:
        K = float(np.mean(ks))
        print(f"\nCalibration over {len(ks)} points: K = {K:.3f} "
              f"+- {np.std(ks):.3f}\n")
    else:
        K = 1.0
        print("\nNo calibration points — falling back to K = 1 (pure Townsend)\n")

    gpath = gas_file(args.predict, args.pressure)
    if not os.path.exists(gpath):
        sys.exit(f"[predict] gas table not ready yet: {gpath}")

    volts = np.arange(args.v_min, args.v_max + 0.5 * args.v_step, args.v_step)
    gas = load_medium(gpath)
    a, e = alpha_eta(gas, volts, d)
    eff = (a - e) * d
    g_pred = np.exp(K * eff)

    print(f"PREDICTION — {args.predict} @ {args.pressure}  (K = {K:.3f})")
    print(f"{'V':>6} {'alpha':>9} {'eta':>7} {'G_pred':>12}")
    for vv, aa, ee, gg in zip(volts, a, e, g_pred):
        print(f"{vv:6.0f} {aa:9.1f} {ee:7.2f} {gg:12.3g}")

    # Which voltages cover the reference gain span?
    ref = result_file("Ar_iC4H10_95_5", args.pressure)
    if os.path.exists(ref):
        r = json.load(open(ref))
        glo, ghi = min(r["gain_mean"]), max(r["gain_mean"])
        inside = volts[(g_pred >= glo) & (g_pred <= ghi)]
        print(f"\nAr/iC4H10 95/5 reference gain span: {glo:.0f} - {ghi:.0f}")
        if len(inside):
            print(f"Predicted matching voltage window for {args.predict}: "
                  f"{inside.min():.0f} - {inside.max():.0f} V")
        else:
            print("Predicted gains do not overlap the reference span in this range!")


if __name__ == "__main__":
    main()
