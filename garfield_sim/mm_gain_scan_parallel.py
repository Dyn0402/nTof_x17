#!/usr/bin/env python3
"""
mm_gain_scan_parallel.py — Parallel Micromegas gain scan
=========================================================
Drop-in replacement for mm_gain_scan.py. Uses a single flat pool where
every task is one (gas, pressure, voltage) triple. No nested pools, no
daemon-children constraint.

Architecture
------------
  Main process (coordinator)
    └─ Pool of N workers  (spawn, not fork — ROOT safety)
         Each worker handles ONE voltage step for ONE combo:
           load gas file → build geometry → run N avalanches → return stats

  The main process collects results via imap_unordered as they finish,
  groups them by combo, writes incremental JSON after each completed voltage
  step, and assembles final files at the end.

Parallelism
-----------
With 8 cores and 4 combos × 31 voltage steps = 124 tasks, all 8 cores
stay busy for the full run. The --workers flag controls pool size directly.

Physics shared with mm_condor_worker.py via mm_sim_core.run_avalanche_batch.
Stats and result I/O shared via mm_sim_core.recompute_stats / build_result_dict.

Usage
-----
    python3 mm_gain_scan_parallel.py                      # auto workers
    python3 mm_gain_scan_parallel.py --workers 8          # explicit
    python3 mm_gain_scan_parallel.py --workers 4          # conservative
    python3 mm_gain_scan_parallel.py --vstep 10 --events 50 --dry-run
    python3 mm_gain_scan_parallel.py --gas He --pressure Saclay
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mm_config as cfg
from mm_generate_gas import gas_filename

# ROOT/Garfield imported ONLY inside worker — never in the main process.
# mm_sim_core is imported inside the worker too (no ROOT at its module level).


# ──────────────────────────────────────────────────────────────────────────────
# Worker: one (gas_file, penning, voltage) task
# ──────────────────────────────────────────────────────────────────────────────

def _worker(task):
    """
    Run n_events avalanches for one (combo, voltage) pair.
    Each call is a fully independent spawned process with its own ROOT instance.

    task = (combo_key, gas_file, penning_cfg, voltage, n_events, gap_cm)
    Returns (combo_key, result_dict).
    """
    combo_key, gas_file, penning_cfg, voltage, n_events, gap_cm = task

    # Import inside the worker — safe for spawn multiprocessing
    from mm_sim_core import run_avalanche_batch, recompute_stats

    gains, n_attached, wall = run_avalanche_batch(
        gas_file, penning_cfg, voltage, n_events, gap_cm
    )

    e_field = voltage / gap_cm
    mean_g, med_g, std_g, rms_rel, surv = recompute_stats(gains, n_attached)

    return combo_key, {
        "voltage":      voltage,
        "field":        e_field,
        "gain_mean":    mean_g,
        "gain_median":  med_g,
        "gain_std":     std_g,
        "gain_rms_rel": rms_rel,
        "gain_raw":     gains,
        "survival":     surv,
        "n_attached":   n_attached,
        "runtime_s":    wall,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Result assembly & saving  (delegates to mm_sim_core)
# ──────────────────────────────────────────────────────────────────────────────

def _assemble_and_save(combo_meta, step_results,
                       n_events, total_time, partial=False):
    from mm_sim_core import build_result_dict, save_result

    gas_cfg, pressure_label, pressure_torr = combo_meta

    # Only include voltage steps we have results for
    have = {sr["voltage"] for sr in step_results}
    volt_data = [
        {
            "voltage":    sr["voltage"],
            "field":      sr["field"],
            "gain_raw":   sr["gain_raw"],
            "n_attached": sr["n_attached"],
            "runtime_s":  sr["runtime_s"],
        }
        for sr in step_results
        if sr["voltage"] in have
    ]

    result = build_result_dict(
        gas_cfg["label"], pressure_label, pressure_torr,
        cfg.GAP_CM, cfg.TEMP_K, gas_cfg["penning"],
        n_events, volt_data, total_time, partial
    )

    out = save_result(result, cfg.RESULTS_DIR)
    return result, out


def write_summary_csv(all_results):
    import csv
    csv_path = os.path.join(cfg.RESULTS_DIR, "summary.csv")
    rows = []
    for res in all_results:
        for i, v in enumerate(res["voltages"]):
            rms = res["gain_rms_rel"][i]
            rows.append({
                "gas":            res["gas"],
                "pressure_label": res["pressure_label"],
                "pressure_torr":  f"{res['pressure_torr']:.2f}",
                "voltage_V":      v,
                "field_Vcm":      res["fields"][i],
                "gain_mean":      f"{res['gain_mean'][i]:.2f}",
                "gain_median":    f"{res['gain_median'][i]:.2f}",
                "gain_std":       f"{res['gain_std'][i]:.2f}",
                "gain_rms_rel":   f"{rms:.4f}" if not np.isnan(rms) else "nan",
                "survival":       f"{res['survival'][i]:.4f}",
                "n_events":       res["n_events"],
                "runtime_s":      f"{res['runtime_s'][i]:.1f}",
            })
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    n_cores = mp.cpu_count()

    parser = argparse.ArgumentParser(
        description="Parallel Micromegas gain scan (flat pool)"
    )
    parser.add_argument("--gas",      default=None,
                        help="Run only this gas label (substring match)")
    parser.add_argument("--pressure", default=None,
                        help="Run only this pressure label (substring match)")
    parser.add_argument("--events",   type=int, default=cfg.N_EVENTS,
                        help=f"Events per voltage step (default {cfg.N_EVENTS})")
    parser.add_argument("--vstep",    type=int, default=cfg.V_STEP,
                        help=f"Voltage step in V (default {cfg.V_STEP})")
    parser.add_argument("--workers",  type=int, default=None,
                        help=f"Worker processes (default: cpu_count={n_cores})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print plan and exit")
    args = parser.parse_args()

    voltages = np.arange(cfg.V_MIN, cfg.V_MAX + args.vstep, args.vstep)

    # Filter combos
    gases = cfg.GAS_MIXTURES
    if args.gas:
        gases = [g for g in gases if args.gas in g["label"]]
        if not gases:
            print(f"ERROR: no gas matches '{args.gas}'"); sys.exit(1)
    pressures = dict(cfg.PRESSURES)
    if args.pressure:
        pressures = {k: v for k, v in pressures.items() if args.pressure in k}
        if not pressures:
            print(f"ERROR: no pressure matches '{args.pressure}'"); sys.exit(1)

    combos = [
        (gas_cfg, plabel, ptorr)
        for gas_cfg in gases
        for plabel, ptorr in pressures.items()
    ]

    workers = min(args.workers or n_cores, n_cores)
    n_tasks = len(combos) * len(voltages)

    # ── Print plan ─────────────────────────────────────────────────────────────
    print("Parallel Micromegas Gain Scan")
    print("=" * 55)
    print(f"Available cores : {n_cores}")
    print(f"Workers         : {workers}")
    print(f"Combinations    : {len(combos)}")
    print(f"Voltages/combo  : {len(voltages)}  "
          f"({voltages[0]:.0f}–{voltages[-1]:.0f} V, step={args.vstep} V)")
    print(f"Total tasks     : {n_tasks}  "
          f"({len(combos)} combos × {len(voltages)} V steps)")
    print(f"Events/task     : {args.events}")
    print()
    print("Combinations:")
    missing = []
    for gas_cfg, plabel, ptorr in combos:
        gfile = gas_filename(gas_cfg["label"], plabel)
        ok = os.path.exists(gfile)
        print(f"  {'✓' if ok else '✗'}  {gas_cfg['label']} × {plabel}"
              f"  ({ptorr:.1f} Torr)")
        if not ok:
            missing.append(gfile)
    print()

    if missing:
        print("ERROR: Missing gas files — run mm_generate_gas.py first:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    if args.dry_run:
        print("(dry-run — exiting)")
        return

    # ── Build flat task list ───────────────────────────────────────────────────
    tasks = []
    for ci, (gas_cfg, plabel, ptorr) in enumerate(combos):
        gfile = gas_filename(gas_cfg["label"], plabel)
        for v in voltages:
            tasks.append((
                ci,                  # combo index (picklable int)
                gfile,
                gas_cfg["penning"],
                float(v),
                args.events,
                cfg.GAP_CM,
            ))

    combo_meta  = {ci: combos[ci] for ci in range(len(combos))}
    step_store  = defaultdict(list)
    combo_start = {}
    n_done      = defaultdict(int)
    n_expected  = len(voltages)
    t_wall      = time.time()

    print(f"{'Combo':<38} {'V(V)':>6} {'Gain':>8} {'Surv%':>6} "
          f"{'t(s)':>6} {'Done':>10}")
    print("-" * 80)

    # ── Run flat pool ──────────────────────────────────────────────────────────
    ctx = mp.get_context("spawn")
    all_results = []

    with ctx.Pool(processes=workers) as pool:
        for combo_key, sr in pool.imap_unordered(_worker, tasks):
            gas_cfg, plabel, ptorr = combo_meta[combo_key]
            tag = f"{gas_cfg['label']}|{plabel}"

            if combo_key not in combo_start:
                combo_start[combo_key] = time.time()

            step_store[combo_key].append(sr)
            n_done[combo_key] += 1
            done = n_done[combo_key]

            print(f"  {tag:<36} {sr['voltage']:>6.0f} "
                  f"{sr['gain_mean']:>8.0f} "
                  f"{100*sr['survival']:>5.0f}% "
                  f"{sr['runtime_s']:>6.0f}  "
                  f"[{done}/{n_expected}]",
                  flush=True)

            # Incremental save after every result
            elapsed_combo = time.time() - combo_start[combo_key]
            _assemble_and_save(
                combo_meta[combo_key], step_store[combo_key],
                args.events, elapsed_combo, partial=True
            )

            # Combo fully complete
            if done == n_expected:
                total_combo = time.time() - combo_start[combo_key]
                result, out = _assemble_and_save(
                    combo_meta[combo_key], step_store[combo_key],
                    args.events, total_combo, partial=False
                )
                all_results.append(result)
                print(f"\n  ✓ {tag}  complete in {total_combo/60:.1f} min"
                      f"  →  {os.path.basename(out)}\n", flush=True)

    elapsed = time.time() - t_wall
    write_summary_csv(all_results)

    print(f"\n{'='*55}")
    print(f"All done in {elapsed/60:.1f} min  ({workers} workers)")
    print(f"Results in: {cfg.RESULTS_DIR}/")
    print(f"Next step:  python3 mm_plot.py")


if __name__ == "__main__":
    mp.freeze_support()
    main()
