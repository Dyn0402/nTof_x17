#!/usr/bin/env python3
"""
mm_gap_scaling.py — how far does an HV-equivalence map travel between detectors?
===============================================================================
The gain simulation in mm_sim_core is a uniform field across a slab
(ComponentConstant, gap = mm_config.GAP_CM). It is not a model of any particular
detector: no mesh, no hole field lines, no electron transparency, no resistive
layer, no primary ionisation. The only detector-specific numbers in it are the
amplification gap and the site pressure.

So a gain *number* from this simulation is not transferable. The question is
whether the equal-gain *map* is, and that has a clean answer:

    G  =  exp( K · (alpha(E) - eta(E)) · d )

Two gases reach the same gain in the same gap when

    K_A·(alpha_A - eta_A)(E_A)  =  K_B·(alpha_B - eta_B)(E_B)          (*)

Both d and the target gain drop out of (*). The equal-gain relation is therefore
a map between *fields*, fixed by the gas physics alone — and the equal-gain
*voltage* map is that field map read at the other detector's gap:

    E_A' = V_A'/d'   ->   E_B = F(E_A')   ->   V_B' = E_B · d'

which is why quoting the map in volts silently attaches it to a 150 µm gap.

K absorbs the fact that the tabulated Townsend coefficient is computed without
Penning transfer while the avalanche is run with it. This script calibrates K
against the simulated gain curve, checks whether it is actually constant in E
(if it drifts, (*) is only approximate and the map carries that error), and then
answers the practical question: what gap would reproduce a claimed gain at a
claimed voltage?

    python3 mm_gap_scaling.py --gas Ar_CO2_70_30 --pressure CERN_450m \\
                              --claim-gain 1e4 --claim-voltage 530

Must run where Garfield++ is available (lxplus, LCG_108 + setupGarfield.sh).
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="Gap scaling of an equal-gain map")
    p.add_argument("--gas",           default="Ar_CO2_70_30")
    p.add_argument("--pressure",      default="CERN_450m")
    p.add_argument("--gas-table",     default=None,
                   help="Override the .gas path (default gas_tables/<gas>_<pressure>.gas)")
    p.add_argument("--claim-gain",    type=float, default=1e4,
                   help="Gain another detector is reported to reach")
    p.add_argument("--claim-voltage", type=float, default=530.0,
                   help="Mesh voltage at which it reaches --claim-gain")
    p.add_argument("--gaps-um",       default="64,96,128,150,192,220",
                   help="Gaps to tabulate (µm)")
    p.add_argument("--pair-with",     default=None,
                   help="Second gas: rebuild the equal-gain voltage map at each "
                        "gap and report how much it moves. This is the actual "
                        "test of whether a map made at 150 µm is usable on a "
                        "detector with a different gap.")
    p.add_argument("--results-dir",   default=None)
    return p.parse_args()


def calibrate_K(gas_file, res_path, d0):
    """K and the simulated (V, G) curve for one gas."""
    with open(res_path) as f:
        res = json.load(f)
    V = np.array(res["voltages"], float)
    G = np.array(res["gain_mean"], float)
    m = G > 1.0
    V, G = V[m], G[m]
    P = res["pressure_torr"]
    T = res.get("temp_k", cfg.TEMP_K)
    a, e = load_alpha_eta(gas_file, P, T, V / d0)
    K = np.log(G) / ((a - e) * d0)
    return {"V": V, "G": G, "P": P, "T": T, "K": K,
            "Kbar": float(K.mean()),
            # K(E) as a linear trend, so it can be evaluated at other fields
            "Kfit": np.polyfit(V / d0, K, 1),
            "penning": res.get("penning"), "file": gas_file}


def solve_field_for_lnG(gas, target_lnG, d, use_ktrend, lo=5e3, hi=2e5):
    """Field at which this gas reaches target_lnG in gap d. Bisection."""
    def f(E):
        a, e = load_alpha_eta(gas["file"], gas["P"], gas["T"], [E])
        K = np.polyval(gas["Kfit"], E) if use_ktrend else gas["Kbar"]
        return K * (a[0] - e[0]) * d - target_lnG
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return float("nan")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def pair_gap_report(A, B, name_a, name_b, gaps, vgrid, use_ktrend):
    """Equal-gain voltage map V_B(V_A) rebuilt at each gap."""
    print(f"\n=== equal-gain map {name_a} -> {name_b}, rebuilt at each gap ===")
    print(f"    (K {'as a linear trend in E' if use_ktrend else 'held at its mean'})")
    hdr = f"{'V_A':>6} " + " ".join(f"{d*1e4:>8.0f}µm" for d in gaps)
    print(hdr)
    table = []
    for VA in vgrid:
        row = []
        for d in gaps:
            EA = VA / d
            a, e = load_alpha_eta(A["file"], A["P"], A["T"], [EA])
            KA = np.polyval(A["Kfit"], EA) if use_ktrend else A["Kbar"]
            lnG = KA * (a[0] - e[0]) * d
            EB = solve_field_for_lnG(B, lnG, d, use_ktrend)
            row.append(EB * d if np.isfinite(EB) else float("nan"))
        table.append(row)
        cells = " ".join(f"{v:9.0f}" if np.isfinite(v) else f"{'--':>9}"
                         for v in row)
        print(f"{VA:6.0f} {cells}")

    print(f"\n  same thing as DeltaV = V_B - V_A:")
    print(hdr)
    for VA, row in zip(vgrid, table):
        cells = " ".join(f"{v-VA:+9.0f}" if np.isfinite(v) else f"{'--':>9}"
                         for v in row)
        print(f"{VA:6.0f} {cells}")

    print(f"\n  and as the ratio V_B/V_A, which is what actually transfers:")
    print(hdr)
    for VA, row in zip(vgrid, table):
        cells = " ".join(f"{v/VA:9.4f}" if np.isfinite(v) else f"{'--':>9}"
                         for v in row)
        print(f"{VA:6.0f} {cells}")


def load_alpha_eta(gas_file, pressure_torr, temp_k, fields):
    """alpha and eta (1/cm) at each field, from the Magboltz table."""
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    import Garfield  # noqa: F401
    import ctypes

    gas = ROOT.Garfield.MediumMagboltz()
    gas.LoadGasFile(gas_file)
    gas.SetTemperature(temp_k)
    gas.SetPressure(pressure_torr)

    a_out, e_out = [], []
    for E in fields:
        a = ctypes.c_double(0.)
        e = ctypes.c_double(0.)
        gas.ElectronTownsend(0., 0., float(E), 0., 0., 0., a)
        gas.ElectronAttachment(0., 0., float(E), 0., 0., 0., e)
        a_out.append(a.value)
        e_out.append(e.value)
    return np.array(a_out), np.array(e_out)


def main():
    args = parse_args()
    results_dir = args.results_dir or cfg.RESULTS_DIR
    here = os.path.dirname(os.path.abspath(__file__))

    gas_file = args.gas_table or os.path.join(
        here, "gas_tables", f"{args.gas}_{args.pressure}.gas")
    res_path = os.path.join(results_dir, f"{args.gas}_{args.pressure}.json")
    if not os.path.exists(gas_file):
        sys.exit(f"[gap] no gas table: {gas_file}")
    if not os.path.exists(res_path):
        sys.exit(f"[gap] no gain results: {res_path}")

    with open(res_path) as f:
        res = json.load(f)
    V = np.array(res["voltages"], float)
    G = np.array(res["gain_mean"], float)
    m = G > 1.0
    V, G = V[m], G[m]

    d0 = cfg.GAP_CM
    P  = res["pressure_torr"]
    T  = res.get("temp_k", cfg.TEMP_K)

    # ── 1. calibrate K on the simulated curve ─────────────────────────────────
    a0, e0 = load_alpha_eta(gas_file, P, T, V / d0)
    K = np.log(G) / ((a0 - e0) * d0)

    print(f"\n=== {args.gas} @ {args.pressure} ({P:.1f} Torr, {T:.2f} K) ===")
    print(f"simulated gap: {d0*1e4:.0f} µm   penning: {res.get('penning')}")
    print(f"\n K = ln G / ((alpha-eta)·d), from the simulated gain curve:")
    print(f"{'V':>6} {'E (kV/cm)':>10} {'alpha':>9} {'eta':>7} {'G_sim':>11} {'K':>7}")
    for v, e, a, et, g, k in zip(V, V / d0 / 1e3, a0, e0, G, K):
        print(f"{v:6.0f} {e:10.2f} {a:9.1f} {et:7.2f} {g:11.1f} {k:7.3f}")
    print(f"\n K = {K.mean():.3f} ± {K.std():.3f}   "
          f"(spread {100*K.std()/K.mean():.1f} % — if this is small, the "
          f"equal-gain map is gap- and gain-independent to about that accuracy)")

    # trend of K with field: the thing that breaks (*) if it is not flat
    slope = np.polyfit(V / d0 / 1e3, K, 1)[0]
    print(f" dK/dE = {slope:+.4f} per kV/cm "
          f"({100*slope*(V.max()-V.min())/d0/1e3/K.mean():+.1f} % across the "
          f"simulated range)")

    Kbar = float(K.mean())

    # ── 2. what gap reproduces the claim? ─────────────────────────────────────
    Vc, Gc = args.claim_voltage, args.claim_gain
    print(f"\n=== claim: G = {Gc:,.0f} at {Vc:.0f} V in {args.gas} ===")
    g_here = float(np.exp(np.interp(Vc, V, np.log(G))))
    print(f" this simulation at {d0*1e4:.0f} µm and {Vc:.0f} V gives "
          f"G = {g_here:,.0f}  ->  claim is {Gc/g_here:,.0f}x higher")

    gaps = np.array([float(x) for x in args.gaps_um.split(",")]) * 1e-4
    print(f"\n same voltage {Vc:.0f} V, other gaps (K = {Kbar:.3f} held fixed):")
    print(f"{'gap (µm)':>9} {'E (kV/cm)':>10} {'alpha-eta':>10} {'G':>14}")
    fine = np.linspace(20e-4, 400e-4, 400)
    aF, eF = load_alpha_eta(gas_file, P, T, Vc / fine)
    lnG_fine = Kbar * (aF - eF) * fine
    for d in gaps:
        aa, ee = load_alpha_eta(gas_file, P, T, [Vc / d])
        lnG = Kbar * (aa[0] - ee[0]) * d
        print(f"{d*1e4:9.0f} {Vc/d/1e3:10.2f} {aa[0]-ee[0]:10.1f} "
              f"{np.exp(lnG):14,.0f}")

    target = np.log(Gc)
    hit = [(fine[i], fine[i + 1]) for i in range(len(fine) - 1)
           if (lnG_fine[i] - target) * (lnG_fine[i + 1] - target) < 0]
    print(f"\n gap that would give G = {Gc:,.0f} at {Vc:.0f} V:")
    if not hit:
        best = fine[int(np.argmax(lnG_fine))]
        print(f"   NONE in 20-400 µm. Best possible at this voltage is "
              f"G = {np.exp(lnG_fine.max()):,.0f} at {best*1e4:.0f} µm.")
        print(f"   -> the claim cannot be reached by changing the gap alone; "
              f"something outside this model is doing the work.")
    else:
        for lo, hi in hit:
            print(f"   d ≈ {0.5*(lo+hi)*1e4:.0f} µm")

    # ── 3. does the equal-gain map itself survive a change of gap? ────────────
    if args.pair_with:
        gas_b = os.path.join(here, "gas_tables",
                             f"{args.pair_with}_{args.pressure}.gas")
        res_b = os.path.join(results_dir, f"{args.pair_with}_{args.pressure}.json")
        if not (os.path.exists(gas_b) and os.path.exists(res_b)):
            print(f"\n[gap] skipping pair: need {gas_b} and {res_b}")
        else:
            A = {"V": V, "G": G, "P": P, "T": T, "K": K, "Kbar": Kbar,
                 "Kfit": np.polyfit(V / d0, K, 1), "file": gas_file}
            B = calibrate_K(gas_b, res_b, d0)
            print(f"\n {args.pair_with}: K = {B['Kbar']:.3f} ± {B['K'].std():.3f}"
                  f"   penning {B['penning']}")
            vgrid = np.linspace(max(V.min(), 420), min(V.max(), 700), 6)
            gaps = np.array([float(x) for x in args.gaps_um.split(",")]) * 1e-4
            pair_gap_report(A, B, args.gas, args.pair_with, gaps, vgrid, False)
            pair_gap_report(A, B, args.gas, args.pair_with, gaps, vgrid, True)

    print(f"\n note: 'gain' here is electrons per seed electron. A measured gain "
          f"folds in mesh transparency and electron collection efficiency, both "
          f"< 1, so a real detector reads LOWER than this at the same field — "
          f"a measured value ABOVE simulation is not explained by those.")


if __name__ == "__main__":
    main()
