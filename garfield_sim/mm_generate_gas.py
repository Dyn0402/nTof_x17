#!/usr/bin/env python3
"""
mm_generate_gas.py — Pre-compute Magboltz gas tables (parallel)
================================================================
Run this ONCE (or whenever you change gas compositions or pressure).
Gas tables are cached as .gas files; the gain scanner loads them directly,
skipping the expensive Magboltz calculation.

Each (gas, pressure) combination is independent and runs in its own process,
so all 4 tables can be generated simultaneously if you have ≥4 cores.

Usage:
    python3 mm_generate_gas.py                  # auto-detect cores
    python3 mm_generate_gas.py --workers 4      # explicit worker count
    python3 mm_generate_gas.py --workers 1      # sequential (old behaviour)
    python3 mm_generate_gas.py --ncoll 5        # faster, less accurate
    python3 mm_generate_gas.py --force          # regenerate existing files

Each (gas, pressure) combination produces one .gas file, e.g.:
    gas_tables/He_C2H6_96p5_3p5_Saclay_160m.gas
    gas_tables/He_C2H6_96p5_3p5_CERN_450m.gas
    gas_tables/Ar_iC4H10_95_5_Saclay_160m.gas
    gas_tables/Ar_iC4H10_95_5_CERN_450m.gas

Typical runtime with --workers 4:
    ~same as the slowest single file (~20–80 min depending on ncoll & CPU)
"""

import sys
import os
import time
import argparse
import multiprocessing as mp

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))
import mm_config as cfg

# NOTE: ROOT and Garfield are intentionally NOT imported at module level.
# They are imported inside _worker() so that each spawned process gets a
# clean, isolated ROOT instance. Importing ROOT in the parent process and
# then forking causes subtle corruption in Magboltz's Fortran RNG state.


def gas_filename(gas_label, pressure_label):
    return os.path.join(cfg.GAS_DIR, f"{gas_label}_{pressure_label}.gas")


# ──────────────────────────────────────────────────────────────────────────────
# Worker — runs in its own process
# ──────────────────────────────────────────────────────────────────────────────

def _worker(args):
    """
    Generate one gas table. Runs in an isolated process so each Magboltz
    instance has its own RNG state and ROOT global state.
    Returns (fname, elapsed_minutes, alpha_at_mid_field) on success,
    or raises on failure.
    """
    gas_cfg, pressure_torr, pressure_label, ncoll, force = args

    # Import ROOT/Garfield fresh in this process
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    import Garfield
    import ctypes

    label = gas_cfg["label"]
    fname = gas_filename(label, pressure_label)
    pid   = os.getpid()
    tag   = f"[pid={pid} | {label} | {pressure_label}]"

    if os.path.exists(fname) and not force:
        print(f"{tag} SKIP — file exists", flush=True)
        return fname, 0., None

    print(f"{tag} Starting Magboltz", flush=True)
    print(f"{tag}   Pressure: {pressure_torr:.2f} Torr", flush=True)
    print(f"{tag}   nColl:    {ncoll} × 10^7", flush=True)
    print(f"{tag}   E grid:   {cfg.E_GRID_MIN_VCM:.0f}–"
          f"{cfg.E_GRID_MAX_VCM:.0f} V/cm, {cfg.E_GRID_NPTS} pts", flush=True)

    gas = ROOT.Garfield.MediumMagboltz()

    # Set composition
    components = gas_cfg["components"]
    if len(components) == 1:
        gas.SetComposition(components[0][0], components[0][1])
    elif len(components) == 2:
        gas.SetComposition(
            components[0][0], components[0][1],
            components[1][0], components[1][1],
        )
    elif len(components) == 3:
        gas.SetComposition(
            components[0][0], components[0][1],
            components[1][0], components[1][1],
            components[2][0], components[2][1],
        )
    else:
        raise ValueError(f"Too many gas components: {components}")

    gas.SetTemperature(cfg.TEMP_K)
    gas.SetPressure(pressure_torr)
    gas.SetFieldGrid(
        cfg.E_GRID_MIN_VCM,
        cfg.E_GRID_MAX_VCM,
        cfg.E_GRID_NPTS,
        True,   # logarithmic spacing
    )

    t0 = time.time()
    gas.GenerateGasTable(ncoll)
    elapsed = time.time() - t0

    print(f"{tag} Magboltz done in {elapsed/60:.1f} min", flush=True)

    gas.WriteGasFile(fname)
    print(f"{tag} Saved → {fname}", flush=True)

    # Sanity check: Townsend coefficient at the middle of our voltage scan
    e_mid = (cfg.V_MIN + cfg.V_MAX) / 2.0 / cfg.GAP_CM
    alpha = ctypes.c_double(0.)
    # Signature: ElectronTownsend(ex, ey, ez, bx, by, bz, alpha)
    gas.ElectronTownsend(0., 0., e_mid, 0., 0., 0., alpha)
    alpha_val = alpha.value
    print(f"{tag} Sanity: α({e_mid:.0f} V/cm) = {alpha_val:.1f} cm⁻¹", flush=True)

    return fname, elapsed / 60.0, alpha_val


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    n_cores = mp.cpu_count()

    parser = argparse.ArgumentParser(
        description="Generate Magboltz gas tables (parallel)"
    )
    parser.add_argument("--ncoll", type=int, default=cfg.MAGBOLTZ_NCOLL,
                        help=f"Magboltz collision sets ×10^7 "
                             f"(default: {cfg.MAGBOLTZ_NCOLL})")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate gas files even if they exist")
    parser.add_argument("--workers", type=int,
                        default=None,
                        help=f"Parallel workers (default: min(n_combos, "
                             f"cpu_count={n_cores}))")
    args = parser.parse_args()

    # Build full list of (gas, pressure) combinations to generate
    combos = [
        (gas_cfg, pressure_torr, pressure_label)
        for gas_cfg in cfg.GAS_MIXTURES
        for pressure_label, pressure_torr in cfg.PRESSURES.items()
    ]
    n_total = len(combos)

    workers = args.workers if args.workers else min(n_total, n_cores)
    workers = min(workers, n_total)   # no point having more workers than jobs

    print("Magboltz Gas Table Generation")
    print("=" * 50)
    print(f"Combinations : {n_total}")
    print(f"Workers      : {workers}  (cpu_count={n_cores})")
    print(f"nColl        : {args.ncoll} × 10^7")
    print(f"Output dir   : {cfg.GAS_DIR}")
    print()
    print("Files to generate:")
    for gas_cfg, ptorr, plabel in combos:
        fname   = gas_filename(gas_cfg["label"], plabel)
        status  = "exists" if os.path.exists(fname) else "will generate"
        if os.path.exists(fname) and not args.force:
            status = "SKIP (exists)"
        print(f"  {gas_cfg['label']} × {plabel:15s}  {ptorr:.1f} Torr  "
              f"→ {os.path.basename(fname)}  [{status}]")
    print()

    # Pack args for the worker
    worker_args = [
        (gas_cfg, ptorr, plabel, args.ncoll, args.force)
        for gas_cfg, ptorr, plabel in combos
    ]

    t_wall = time.time()

    if workers == 1:
        # Sequential — simpler, easier to debug
        results = [_worker(a) for a in worker_args]
    else:
        # Parallel — use spawn to avoid ROOT fork-safety issues
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_worker, worker_args)

    elapsed_total = time.time() - t_wall

    # Summary
    print(f"\n{'='*50}")
    print(f"All done in {elapsed_total/60:.1f} min  "
          f"(wall time with {workers} worker(s))")
    print()
    print(f"{'File':<45} {'Time(min)':>10} {'α_mid(cm⁻¹)':>13}")
    print("-" * 70)
    for (fname, elapsed_min, alpha), (gas_cfg, ptorr, plabel) in \
            zip(results, combos):
        exists   = "✓" if os.path.exists(fname) else "✗"
        t_str    = f"{elapsed_min:.1f}" if elapsed_min > 0 else "skipped"
        a_str    = f"{alpha:.1f}"        if alpha is not None else "—"
        print(f"  {exists} {os.path.basename(fname):<43} "
              f"{t_str:>10} {a_str:>13}")
    print()
    print("Next step: python3 mm_gain_scan_parallel.py")


if __name__ == "__main__":
    mp.freeze_support()
    main()
