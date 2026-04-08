#!/usr/bin/env python3
"""
mm_condor_collect.py — Collect and merge HTCondor job fragment results
=======================================================================
Run this interactively on lxplus (or locally if EOS is mounted) after
HTCondor jobs finish. Scans the jobs/ directory for fragment JSONs,
merges them into per-(gas, pressure) result files in the standard schema
that mm_plot.py expects.

Supports append mode (default): if a result JSON already exists, new
events are concatenated onto existing gain_raw before recomputing stats.
Stats and I/O delegated to mm_sim_core (shared with mm_gain_scan_parallel.py).

Usage:
    python3 mm_condor_collect.py [--no-append] [--dry-run] [--gas STR] [--pressure STR]

Options:
    --no-append     Discard existing results, start fresh from fragments
    --dry-run       Show what would be merged without writing anything
    --gas STR       Filter by gas label substring
    --pressure STR  Filter by pressure label substring
    --jobs-dir PATH Override jobs directory (default: EOS_BASE/jobs)
    --results-dir   Override results directory
"""

import os
import sys
import glob
import argparse
import math
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_sim_core import recompute_stats, load_existing, build_result_dict, save_result

# ── Paths ──────────────────────────────────────────────────────────────────────
EOS_BASE    = "/afs/cern.ch/user/d/dneff/work/git/nTof_x17/garfield_sim"
JOBS_DIR    = f"{EOS_BASE}/jobs"
RESULTS_DIR = f"{EOS_BASE}/results"

# Gap and temperature (must match worker settings)
GAP_CM = 0.015
TEMP_K = 293.15


def altitude_to_torr(h_m):
    return 101325.0 * math.exp(-h_m / 8500.0) * 0.00750062


PRESSURES = {
    "Saclay_160m": altitude_to_torr(160),
    "CERN_450m":   altitude_to_torr(450),
}

GAS_PENNING = {
    "He_C2H6_96p5_3p5":     {"mode": "manual", "rP": 0.40, "gas": "he"},
    "Ar_iC4H10_95_5":       {"mode": "auto"},
    "Ar_CO2_70_30":          {"mode": "auto"},   # no Penning: Ar metastables < CO2 IP
    "Ne_iC4H10_95_5_rP040": {"mode": "manual", "rP": 0.40, "gas": "ne"},
    "Ne_iC4H10_95_5_rP050": {"mode": "manual", "rP": 0.50, "gas": "ne"},
    "Ne_iC4H10_95_5_rP060": {"mode": "manual", "rP": 0.60, "gas": "ne"},
}


# ── Main collection logic ──────────────────────────────────────────────────────

def collect(jobs_dir, results_dir, gas_filter, pressure_filter,
            append, dry_run):

    # ── Scan fragment files ────────────────────────────────────────────────────
    pattern = os.path.join(jobs_dir, "*.json")
    all_fragments = sorted(glob.glob(pattern))

    if not all_fragments:
        print(f"No fragment files found in {jobs_dir}")
        print("Check that HTCondor jobs have completed.")
        return

    print(f"Found {len(all_fragments)} fragment files in {jobs_dir}")

    # Group fragments by (gas_label, pressure_label) → {voltage: [frag, ...]}
    grouped = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for fpath in all_fragments:
        try:
            import json
            with open(fpath) as f:
                frag = json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not read {os.path.basename(fpath)}: {e}")
            skipped += 1
            continue

        gas  = frag.get("gas_label", "")
        pres = frag.get("pressure_label", "")
        volt = float(frag.get("voltage", 0))

        if gas_filter    and gas_filter    not in gas:  continue
        if pressure_filter and pressure_filter not in pres: continue

        grouped[(gas, pres)][volt].append(frag)

    if skipped:
        print(f"  ({skipped} files skipped due to read errors)")

    if not grouped:
        print("No fragments matched the filter criteria.")
        return

    print()

    # ── Process each (gas, pressure) combination ───────────────────────────────
    all_results = []

    for (gas_label, pressure_label), volt_dict in sorted(grouped.items()):
        print(f"{'='*60}")
        print(f"Merging: {gas_label} × {pressure_label}")

        # Load prior results if appending
        if append:
            prior = load_existing(results_dir, gas_label, pressure_label)
            if prior:
                sample_v = next(iter(prior))
                n_sample = (len(prior[sample_v]["gain_raw"]) +
                            prior[sample_v]["n_attached"])
                print(f"  [existing] {len(prior)} voltage points, "
                      f"~{n_sample} events/point")
        else:
            prior = {}

        pressure_torr = PRESSURES.get(pressure_label, 0.)
        penning       = GAS_PENNING.get(gas_label, {"mode": "unknown"})

        volt_data  = []
        max_events = 0

        for voltage in sorted(volt_dict.keys()):
            fragments = volt_dict[voltage]
            e_field   = voltage / GAP_CM

            # Merge all fragment gain_raw + n_attached for this voltage
            new_gains    = []
            new_attached = 0
            new_runtime  = 0.
            n_attempted  = 0

            for frag in fragments:
                new_gains    += frag.get("gain_raw", [])
                new_attached += frag.get("n_attached", 0)
                new_runtime  += frag.get("runtime_s", 0.)
                n_attempted  += frag.get("n_events_attempted", 0)

            # Merge with prior
            p = prior.get(voltage, {})
            merged_gains    = p.get("gain_raw", []) + new_gains
            merged_attached = p.get("n_attached", 0) + new_attached
            merged_runtime  = p.get("runtime_s", 0.) + new_runtime

            n_total    = len(merged_gains) + merged_attached
            max_events = max(max_events, n_total)

            mean_g, med_g, std_g, rms_rel, surv = recompute_stats(
                merged_gains, merged_attached
            )

            prior_n = len(p.get("gain_raw", [])) + p.get("n_attached", 0)
            rms_str = f"{rms_rel:.2f}" if not (rms_rel != rms_rel) else "  —"
            print(f"  V={voltage:>5.0f}V  "
                  f"frags={len(fragments):>3d}  "
                  f"new={n_attempted:>5d}  "
                  f"prior={prior_n:>5d}  "
                  f"total={n_total:>5d}  "
                  f"gain={mean_g:>8.0f}±{std_g:.0f}  "
                  f"surv={100*surv:.0f}%")

            volt_data.append({
                "voltage":    voltage,
                "field":      e_field,
                "gain_raw":   merged_gains,
                "n_attached": merged_attached,
                "runtime_s":  merged_runtime,
            })

        result = build_result_dict(
            gas_label, pressure_label, pressure_torr,
            GAP_CM, TEMP_K, penning,
            max_events, volt_data,
        )

        print(f"  → {len(volt_data)} voltage points, "
              f"up to {max_events} events/point")

        all_results.append((gas_label, pressure_label, result))

    print()

    # ── Write results ──────────────────────────────────────────────────────────
    if dry_run:
        print("(dry-run — not writing any files)")
        return

    for gas_label, pressure_label, result in all_results:
        out_path = save_result(result, results_dir)
        print(f"✓ Saved → {out_path}")

    # ── Summary CSV ────────────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(results_dir, "summary.csv")
    rows = []
    for gas_label, pressure_label, res in all_results:
        for i, v in enumerate(res["voltages"]):
            rms   = res["gain_rms_rel"][i]
            n_tot = len(res["gain_raw"][i]) + res["n_attached"][i]
            rows.append({
                "gas":            gas_label,
                "pressure_label": pressure_label,
                "pressure_torr":  f"{res['pressure_torr']:.2f}",
                "voltage_V":      v,
                "field_Vcm":      res["fields"][i],
                "n_events_total": n_tot,
                "gain_mean":      f"{res['gain_mean'][i]:.2f}",
                "gain_median":    f"{res['gain_median'][i]:.2f}",
                "gain_std":       f"{res['gain_std'][i]:.2f}",
                "gain_rms_rel":   f"{rms:.4f}" if not (rms != rms) else "nan",
                "survival":       f"{res['survival'][i]:.4f}",
            })
    if rows:
        os.makedirs(results_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Summary CSV → {csv_path}")

    print()
    print("Done. Run mm_plot.py to visualise results.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect HTCondor gain scan fragment results"
    )
    parser.add_argument("--no-append",    action="store_true",
                        help="Discard existing results and rebuild from fragments only")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Show merge plan without writing files")
    parser.add_argument("--gas",          default=None,
                        help="Filter by gas label substring")
    parser.add_argument("--pressure",     default=None,
                        help="Filter by pressure label substring")
    parser.add_argument("--jobs-dir",     default=JOBS_DIR,
                        help=f"Fragment JSON directory (default: {JOBS_DIR})")
    parser.add_argument("--results-dir",  default=RESULTS_DIR,
                        help=f"Results directory (default: {RESULTS_DIR})")
    args = parser.parse_args()

    print("HTCondor Result Collector")
    print("=" * 55)
    print(f"Jobs dir    : {args.jobs_dir}")
    print(f"Results dir : {args.results_dir}")
    print(f"Append mode : {'no (overwrite)' if args.no_append else 'yes (merge with existing)'}")
    if args.gas:      print(f"Gas filter  : {args.gas}")
    if args.pressure: print(f"Pres filter : {args.pressure}")
    print()

    collect(
        jobs_dir=args.jobs_dir,
        results_dir=args.results_dir,
        gas_filter=args.gas,
        pressure_filter=args.pressure,
        append=not args.no_append,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
