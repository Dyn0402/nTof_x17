#!/usr/bin/env python3
"""
mm_condor_worker.py — Single HTCondor job worker
=================================================
Runs on an lxplus worker node. Executes N avalanche simulations for one
(gas, pressure, voltage) point and writes a fragment JSON to EOS.

Called by mm_condor_job.sh, which sources the LCG view first.
Physics delegated to mm_sim_core.run_avalanche_batch (shared with the
local parallel scanner mm_gain_scan_parallel.py).

Usage (for testing locally):
    python3 mm_condor_worker.py \\
        --gas-file /eos/user/d/dneff/garfield_sim/gas_tables/He_C2H6_96p5_3p5_Saclay_160m.gas \\
        --gas-label He_C2H6_96p5_3p5 \\
        --pressure-label Saclay_160m \\
        --pressure-torr 745.8 \\
        --penning-mode manual --penning-rP 0.40 --penning-gas he \\
        --voltage 475 \\
        --events 5 \\
        --gap-cm 0.015 \\
        --output /tmp/test_fragment.json \\
        --batch-id 000
"""

import sys
import os
import json
import argparse

import numpy as np

# Make mm_sim_core importable from the same directory (works both on lxplus
# after HTCondor file transfer and locally).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Garfield++ avalanche worker (one HTCondor job)")
    p.add_argument("--gas-file",        required=True,  help="Full path to .gas file")
    p.add_argument("--gas-label",       required=True,  help="Gas label string, e.g. He_C2H6_96p5_3p5")
    p.add_argument("--pressure-label",  required=True,  help="Pressure label, e.g. Saclay_160m")
    p.add_argument("--pressure-torr",   required=True,  type=float, help="Pressure in Torr")
    p.add_argument("--penning-mode",    required=True,  choices=["auto", "manual"])
    p.add_argument("--penning-rP",      default=0.0,    type=float, help="Penning rP (manual mode only)")
    p.add_argument("--penning-gas",     default="",     help="Noble gas for Penning (manual mode only)")
    p.add_argument("--voltage",         required=True,  type=float, help="Mesh voltage (V)")
    p.add_argument("--events",          required=True,  type=int,   help="Number of avalanches to simulate")
    p.add_argument("--gap-cm",          default=0.015,  type=float, help="Amplification gap (cm)")
    p.add_argument("--temp-k",          default=293.15, type=float, help="Temperature (K)")
    p.add_argument("--output",          required=True,  help="Output fragment JSON path")
    p.add_argument("--batch-id",        required=True,  help="Batch identifier string (e.g. 003)")
    return p.parse_args()


def run(args):
    from mm_sim_core import run_avalanche_batch

    print(f"[worker] gas_label={args.gas_label}  pressure={args.pressure_label}"
          f"  V={args.voltage:.0f}V  events={args.events}  batch={args.batch_id}",
          flush=True)
    print(f"[worker] gas_file={args.gas_file}", flush=True)

    if not os.path.exists(args.gas_file):
        print(f"[worker] ERROR: gas file not found: {args.gas_file}", file=sys.stderr)
        sys.exit(1)

    # Build penning config dict for mm_sim_core
    penning_cfg = {"mode": args.penning_mode}
    if args.penning_mode == "manual":
        penning_cfg["rP"]  = args.penning_rP
        penning_cfg["gas"] = args.penning_gas

    e_field = args.voltage / args.gap_cm
    print(f"[worker] E-field = {e_field:.0f} V/cm ({e_field/1e3:.2f} kV/cm)", flush=True)
    print(f"[worker] Penning: {args.penning_mode}"
          + (f"  rP={args.penning_rP}  gas={args.penning_gas}"
             if args.penning_mode == "manual" else ""),
          flush=True)

    # Run simulation — progress printed every 20 events for job log readability
    gains, n_attached, wall = run_avalanche_batch(
        args.gas_file, penning_cfg, args.voltage, args.events,
        gap_cm=args.gap_cm,
        pressure_torr=args.pressure_torr,
        temp_k=args.temp_k,
        progress_interval=20,
    )

    print(f"[worker] Done in {wall:.1f}s  "
          f"({len(gains)} survived, {n_attached} attached)", flush=True)

    # ── Write fragment JSON ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    fragment = {
        "gas_label":          args.gas_label,
        "pressure_label":     args.pressure_label,
        "pressure_torr":      args.pressure_torr,
        "voltage":            args.voltage,
        "field":              e_field,
        "batch_id":           args.batch_id,
        "n_events_attempted": args.events,
        "gain_raw":           gains,
        "n_attached":         n_attached,
        "runtime_s":          wall,
        "gap_cm":             args.gap_cm,
        "temp_k":             args.temp_k,
        "penning_mode":       args.penning_mode,
        "penning_rP":         args.penning_rP,
        "penning_gas":        args.penning_gas,
    }

    with open(args.output, "w") as f:
        json.dump(fragment, f, indent=2)

    print(f"[worker] Fragment written → {args.output}", flush=True)

    # Quick stats for log readability
    if gains:
        arr = np.array(gains, dtype=float)
        print(f"[worker] gain: mean={np.mean(arr):.0f}  "
              f"median={np.median(arr):.0f}  "
              f"std={np.std(arr):.0f}  "
              f"survival={len(gains)/args.events:.2f}", flush=True)
    else:
        print(f"[worker] WARNING: all {args.events} electrons attached — zero gain", flush=True)


if __name__ == "__main__":
    args = parse_args()
    run(args)
