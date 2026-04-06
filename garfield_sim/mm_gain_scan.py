#!/usr/bin/env python3
"""
mm_gain_scan.py — Micromegas gain vs voltage simulation
=========================================================
Loads pre-computed gas tables and runs AvalancheMicroscopic to measure
gain as a function of mesh voltage for each (gas, pressure) combination.

Usage:
    python3 mm_gain_scan.py [--gas LABEL] [--pressure LABEL] [--events N]
                            [--vstep V] [--dry-run]

Options:
    --gas LABEL       Only run for this gas label (default: all)
    --pressure LABEL  Only run for this pressure label (default: all)
    --events N        Override N_EVENTS from config
    --vstep V         Override V_STEP (e.g. 10 for a quick pass)
    --dry-run         Print the run plan and estimated time, then exit

Results are written to:
    results/<gas_label>_<pressure_label>.json

Each JSON file contains:
  {
    "gas":        "...",
    "pressure":   "...",
    "pressure_torr": ...,
    "gap_cm":     ...,
    "n_events":   ...,
    "voltages":   [...],           # V
    "fields":     [...],           # V/cm
    "gain_mean":  [...],           # mean gain per voltage step
    "gain_std":   [...],           # std dev of gain distribution
    "gain_rms":   [...],           # RMS gain fluctuation / mean (relative)
    "gain_raw":   [[...], ...],    # full list of per-event gains
    "survival":   [...],           # fraction of events with ne > 0
    "runtime_s":  [...],           # wall time per voltage step (s)
  }

After all runs, a summary CSV is also written to results/summary.csv.
"""

import sys
import os
import time
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mm_config as cfg

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)
import Garfield

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def result_filename(gas_label, pressure_label):
    return os.path.join(cfg.RESULTS_DIR, f"{gas_label}_{pressure_label}.json")


def load_gas(gas_cfg, pressure_torr):
    """Load a previously generated .gas file and set up Penning transfer."""
    from mm_generate_gas import gas_filename
    pressure_label = {v: k for k, v in cfg.PRESSURES.items()}[pressure_torr]
    fname = gas_filename(gas_cfg["label"], pressure_label)

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Gas file not found: {fname}\n"
            f"Run mm_generate_gas.py first."
        )

    gas = ROOT.Garfield.MediumMagboltz()
    gas.LoadGasFile(fname)

    # Apply Penning transfer AFTER loading the gas file.
    # The gas file stores excitation/ionisation frequencies; EnablePenningTransfer
    # then scales alpha accordingly (see Garfield++ Penning docs).
    penning = gas_cfg["penning"]
    if penning["mode"] == "auto":
        # Built-in parameterisation (works for Ar/iC4H10, Ar/CO2, etc.)
        gas.EnablePenningTransfer()
        print(f"    Penning: auto (built-in rP parameterisation)")
    elif penning["mode"] == "manual":
        rP  = penning["rP"]
        noble = penning["gas"]
        gas.EnablePenningTransfer(rP, 0., noble)
        print(f"    Penning: manual, rP = {rP:.2f} for '{noble}' excitations")
        if gas_cfg["label"].startswith("He"):
            print(f"    NOTE: rP={rP:.2f} for He/C2H6 is an estimate.")
            print(f"          Validate against measured gain data and adjust")
            print(f"          cfg.GAS_MIXTURES[0]['penning']['rP'] accordingly.")

    return gas


def build_geometry(gas, e_field_vcm):
    """
    Create a ComponentConstant + Sensor for a uniform amplification field.

    Coordinate system:
      z = 0         → resistive layer (anode, absorbing boundary)
      z = GAP_CM    → mesh (electron injection point)

    Field points in +z (from anode toward mesh), electrons drift toward z=0.
    """
    cmp = ROOT.Garfield.ComponentConstant()
    cmp.SetArea(-1., -1., 0.,
                 1.,  1., cfg.GAP_CM)
    cmp.SetElectricField(0., 0., e_field_vcm)
    cmp.SetMedium(gas)

    sensor = ROOT.Garfield.Sensor()
    sensor.AddComponent(cmp)
    sensor.SetArea(-1., -1., 0.,
                    1.,  1., cfg.GAP_CM)

    return cmp, sensor


def run_voltage_step(sensor, e_field_vcm, n_events, voltage_v):
    """
    Run n_events single-electron avalanches at a given field.
    Returns (gains, n_attached, wall_time_s).
    """
    aval = ROOT.Garfield.AvalancheMicroscopic()
    aval.SetSensor(sensor)

    gains      = []
    n_attached = 0
    t0         = time.time()

    # ROOT.Long was removed in ROOT 6.22+; use ctypes instead.
    import ctypes
    ne_out = ctypes.c_int(0)
    ni_out = ctypes.c_int(0)

    checkpoint = cfg.TIMING_CHECKPOINT
    t_checkpoint = time.time()

    for i in range(n_events):
        # Seed electron at the mesh with zero kinetic energy
        aval.AvalancheElectron(0., 0., cfg.GAP_CM, 0., 0.)
        aval.GetAvalancheSize(ne_out, ni_out)
        ne = int(ne_out.value)

        if ne == 0:
            n_attached += 1
        else:
            gains.append(ne)

        # Print ETA after first checkpoint
        if (i + 1) == checkpoint:
            dt = time.time() - t_checkpoint
            rate = checkpoint / dt  # events/s
            remaining = n_events - (i + 1)
            eta_s = remaining / rate
            print(f"      Event {i+1}/{n_events}  "
                  f"rate={rate:.1f} ev/s  "
                  f"ETA {eta_s:.0f}s  "
                  f"current mean gain={np.mean(gains):.0f}" if gains
                  else f"      Event {i+1}/{n_events}  rate={rate:.1f} ev/s  ETA {eta_s:.0f}s",
                  flush=True)

    wall_time = time.time() - t0
    return gains, n_attached, wall_time


def run_combination(gas_cfg, pressure_label, pressure_torr, voltages, n_events):
    """Run the full voltage scan for one (gas, pressure) combination."""
    label    = gas_cfg["label"]
    out_file = result_filename(label, pressure_label)

    print(f"\n{'='*60}")
    print(f"Gas:      {label}")
    print(f"Pressure: {pressure_torr:.2f} Torr  ({pressure_label})")
    print(f"Voltages: {voltages[0]:.0f}–{voltages[-1]:.0f} V  "
          f"({len(voltages)} points, step={voltages[1]-voltages[0]:.0f} V)")
    print(f"Events:   {n_events} per voltage step")
    print(f"Gap:      {cfg.GAP_CM*1e4:.0f} µm")

    # Load gas
    print(f"\n  Loading gas table...")
    gas = load_gas(gas_cfg, pressure_torr)

    # Storage
    result = {
        "gas":            label,
        "pressure_label": pressure_label,
        "pressure_torr":  pressure_torr,
        "gap_cm":         cfg.GAP_CM,
        "temp_k":         cfg.TEMP_K,
        "n_events":       n_events,
        "penning":        gas_cfg["penning"],
        "voltages":       voltages.tolist(),
        "fields":         (voltages / cfg.GAP_CM).tolist(),
        "gain_mean":      [],
        "gain_median":    [],
        "gain_std":       [],
        "gain_rms_rel":   [],   # sigma/mean (relative gain spread)
        "gain_raw":       [],   # list of lists
        "survival":       [],   # fraction of events with ne > 0
        "n_attached":     [],
        "runtime_s":      [],
    }

    t_total = time.time()

    for iv, voltage in enumerate(voltages):
        e_field = voltage / cfg.GAP_CM   # V/cm

        print(f"\n  [{iv+1}/{len(voltages)}] V = {voltage:.0f} V  "
              f"  E = {e_field:.0f} V/cm  ({e_field/1e3:.1f} kV/cm)")

        # Rebuild geometry at this field (cheap)
        cmp, sensor = build_geometry(gas, e_field)

        gains, n_att, wall_time = run_voltage_step(
            sensor, e_field, n_events, voltage
        )

        # Statistics
        if len(gains) == 0:
            # All electrons attached — unphysical at these fields, flag it
            print(f"    WARNING: all {n_events} events had zero gain! "
                  f"Check gas table coverage.")
            mean_g   = 0.
            med_g    = 0.
            std_g    = 0.
            rms_rel  = float("nan")
            survival = 0.
        else:
            arr      = np.array(gains, dtype=float)
            mean_g   = float(np.mean(arr))
            med_g    = float(np.median(arr))
            std_g    = float(np.std(arr))
            rms_rel  = std_g / mean_g if mean_g > 0 else float("nan")
            survival = len(gains) / n_events

        result["gain_mean"].append(mean_g)
        result["gain_median"].append(med_g)
        result["gain_std"].append(std_g)
        result["gain_rms_rel"].append(rms_rel)
        result["gain_raw"].append(gains)
        result["survival"].append(survival)
        result["n_attached"].append(n_att)
        result["runtime_s"].append(wall_time)

        print(f"    Gain:     {mean_g:.0f} ± {std_g:.0f}  "
              f"(median={med_g:.0f}, σ/μ={rms_rel:.2f})")
        print(f"    Survival: {100*survival:.1f}%  "
              f"({n_att}/{n_events} attached)")
        print(f"    Time:     {wall_time:.1f} s  "
              f"({wall_time/n_events*1000:.0f} ms/event)")

        # Save incrementally after each voltage step so you don't lose data
        # if the run is interrupted
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

    total_time = time.time() - t_total
    result["total_runtime_s"] = total_time
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ✓ Saved → {out_file}")
    print(f"  Total time: {total_time/60:.1f} min")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dry-run estimate
# ──────────────────────────────────────────────────────────────────────────────

def estimate_runtime(n_events, voltages, gas_label):
    """
    Very rough estimate of runtime. Microscopic tracking time scales roughly
    with gain (more electrons = more steps). We use a heuristic based on
    typical Ar/CO2 timing of ~10 ms/event at gain ~1000.
    """
    ms_per_event = {
        "He_C2H6": 5,     # He mixtures tend to have lower gain, faster
        "Ar_iC4H10": 20,  # Ar + hydrocarbon can reach high gains
    }
    key = next((k for k in ms_per_event if k in gas_label), "Ar_iC4H10")
    ms = ms_per_event[key]
    total_s = len(voltages) * n_events * ms / 1000
    return total_s


# ──────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ──────────────────────────────────────────────────────────────────────────────

def write_summary_csv(all_results):
    import csv
    csv_path = os.path.join(cfg.RESULTS_DIR, "summary.csv")
    rows = []
    for res in all_results:
        for i, v in enumerate(res["voltages"]):
            rows.append({
                "gas":            res["gas"],
                "pressure_label": res["pressure_label"],
                "pressure_torr":  f"{res['pressure_torr']:.2f}",
                "voltage_V":      v,
                "field_Vcm":      res["fields"][i],
                "gain_mean":      f"{res['gain_mean'][i]:.2f}",
                "gain_median":    f"{res['gain_median'][i]:.2f}",
                "gain_std":       f"{res['gain_std'][i]:.2f}",
                "gain_rms_rel":   f"{res['gain_rms_rel'][i]:.4f}",
                "survival":       f"{res['survival'][i]:.4f}",
                "n_events":       res["n_events"],
                "runtime_s":      f"{res['runtime_s'][i]:.1f}",
            })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Summary CSV → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Micromegas gain scan via AvalancheMicroscopic"
    )
    parser.add_argument("--gas",      default=None,
                        help="Run only this gas label (substring match)")
    parser.add_argument("--pressure", default=None,
                        help="Run only this pressure label (substring match)")
    parser.add_argument("--events",   type=int, default=cfg.N_EVENTS,
                        help=f"Events per voltage step (default {cfg.N_EVENTS})")
    parser.add_argument("--vstep",    type=int, default=cfg.V_STEP,
                        help=f"Voltage step in V (default {cfg.V_STEP})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print run plan and estimated time, then exit")
    args = parser.parse_args()

    # Build voltage array (possibly overriding step)
    voltages = np.arange(cfg.V_MIN, cfg.V_MAX + args.vstep, args.vstep)

    # Filter gas mixtures
    gases = cfg.GAS_MIXTURES
    if args.gas:
        gases = [g for g in gases if args.gas in g["label"]]
        if not gases:
            print(f"ERROR: no gas matches '{args.gas}'")
            sys.exit(1)

    # Filter pressures
    pressures = dict(cfg.PRESSURES)
    if args.pressure:
        pressures = {k: v for k, v in pressures.items() if args.pressure in k}
        if not pressures:
            print(f"ERROR: no pressure matches '{args.pressure}'")
            sys.exit(1)

    # Print run plan
    print("Micromegas Gain Scan — Run Plan")
    print("================================")
    print(f"Gas mixtures : {[g['label'] for g in gases]}")
    print(f"Pressures    : {list(pressures.keys())}")
    print(f"Voltages     : {voltages[0]:.0f}–{voltages[-1]:.0f} V  "
          f"(step={args.vstep} V, {len(voltages)} pts)")
    print(f"E-field range: {voltages[0]/cfg.GAP_CM:.0f}–"
          f"{voltages[-1]/cfg.GAP_CM:.0f} V/cm")
    print(f"Events/step  : {args.events}")
    print()
    print("Pressure conditions:")
    for label, torr in pressures.items():
        print(f"  {label:20s}  {torr:.2f} Torr  ({torr/760*1013.25:.1f} hPa)")
    print()
    print("Estimated runtimes (rough):")
    total_est = 0.
    for g in gases:
        for plabel, ptorr in pressures.items():
            est = estimate_runtime(args.events, voltages, g["label"])
            total_est += est
            print(f"  {g['label']:30s} × {plabel:15s}  ~{est/60:.0f} min")
    print(f"  {'TOTAL':>48s}  ~{total_est/60:.0f} min")
    print()

    if args.dry_run:
        print("(dry-run mode — exiting without running)")
        return

    # Run
    all_results = []
    t_wall = time.time()
    n_combos = len(gases) * len(pressures)
    combo = 0

    for gas_cfg in gases:
        for pressure_label, pressure_torr in pressures.items():
            combo += 1
            print(f"\n[Combination {combo}/{n_combos}]")
            res = run_combination(
                gas_cfg, pressure_label, pressure_torr,
                voltages, args.events
            )
            all_results.append(res)

    # Write summary CSV
    if all_results:
        write_summary_csv(all_results)

    elapsed = time.time() - t_wall
    print(f"\n{'='*60}")
    print(f"All done in {elapsed/60:.1f} min")
    print(f"Results in: {cfg.RESULTS_DIR}/")
    print(f"Next step:  python3 mm_plot.py")


if __name__ == "__main__":
    main()
