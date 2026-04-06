#!/usr/bin/env python3
"""
mm_condor_submit.py — Submit Micromegas gain scan jobs to HTCondor
===================================================================
Run this interactively on lxplus. Submits one HTCondor job per
(gas, pressure, voltage, batch) combination.

Usage:
    python3 mm_condor_submit.py [--dry-run] [--batches N] [--events-per-batch N]

Options:
    --dry-run           Print job plan and JDL without submitting
    --batches N         Batches per (gas, pressure, voltage) point (default: 10)
    --events-per-batch  Events per batch (default: 200)
    --gas               Filter by gas label substring
    --pressure          Filter by pressure label substring

The run configuration is hardcoded below in RUN_CONFIG to match the
specific scan Dylan wants:
  - He/C2H6:    450–530 V, step 10 V, both Saclay and CERN
  - Ar/iC4H10:  400–490 V, step 10 V, both Saclay and CERN
  - Target: 2000 events/point = 10 batches × 200 events
"""

import os
import sys
import math
import argparse
import subprocess
import textwrap
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# Adjust REPO_DIR to wherever the garfield_sim repo lives on lxplus/EOS.
# The EOS path is visible from both lxplus interactive nodes and worker nodes.
LXPLUS_BASE    = "/afs/cern.ch/user/d/dneff/work/git/nTof_x17/garfield_sim"
REPO_DIR    = LXPLUS_BASE                          # scripts live here too
GAS_DIR     = f"{LXPLUS_BASE}/gas_tables"
JOBS_DIR    = f"{LXPLUS_BASE}/jobs"
RESULTS_DIR = f"{LXPLUS_BASE}/results"
LOGS_DIR    = f"{LXPLUS_BASE}/logs"


# ── Pressure helper ────────────────────────────────────────────────────────────
def altitude_to_torr(h_m):
    import math
    return 101325.0 * math.exp(-h_m / 8500.0) * 0.00750062


PRESSURES = {
    "Saclay_160m": altitude_to_torr(160),
    "CERN_450m":   altitude_to_torr(450),
}


# ── Run configuration ──────────────────────────────────────────────────────────
# Edit this to change what gets submitted.
# voltages: explicit list in V
# penning:  dict with mode/rP/gas keys matching mm_condor_worker.py args

RUN_CONFIG = [
    {
        "gas_label":       "He_C2H6_96p5_3p5",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(450, 531, 10)),   # 450..530 step 10
        "gap_cm":          0.015,
        "penning_mode":    "manual",
        "penning_rP":      0.40,
        "penning_gas":     "he",
    },
    {
        "gas_label":       "Ar_iC4H10_95_5",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 491, 10)),   # 400..490 step 10
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
]


# ── HTCondor settings ──────────────────────────────────────────────────────────
REQUEST_CPUS   = 1
REQUEST_MEMORY = "2GB"
REQUEST_DISK   = "1GB"
MAX_RUNTIME_S  = 7200    # 2 hours — conservative for high-gain Ar at high V
MAX_RETRIES    = 2


def get_schedd():
    """Get the custom schedd name via 'myschedd show', fall back to default."""
    import re
    try:
        result = subprocess.run(
            ["myschedd", "show"], capture_output=True, text=True, timeout=10
        )
        # myschedd show prints a table; extract the first hostname-like token
        # e.g. "schedd  bigbird09.cern.ch  ..."
        match = re.search(r'([\w.-]+\.cern\.ch)', result.stdout)
        if match:
            schedd = match.group(1)
            print(f"[submit] Using schedd: {schedd}")
            return schedd
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("[submit] myschedd not available — using default schedd")
    return None


def gas_file_path(gas_label, pressure_label):
    return f"{GAS_DIR}/{gas_label}_{pressure_label}.gas"


def fragment_path(gas_label, pressure_label, voltage, batch_id):
    return f"{JOBS_DIR}/{gas_label}_{pressure_label}_{voltage:.0f}V_b{batch_id:03d}.json"


def build_jdl(jobs, batches_per_point, events_per_batch):
    """
    Build a single JDL string for all jobs.
    Each job is one (gas, pressure, voltage, batch) combination.
    """
    lines = []

    # Global settings
    lines += [
        f"executable          = {REPO_DIR}/mm_condor_job.sh",
        f"request_cpus        = {REQUEST_CPUS}",
        f"request_memory      = {REQUEST_MEMORY}",
        f"request_disk        = {REQUEST_DISK}",
        f"+MaxRuntime         = {MAX_RUNTIME_S}",
        f"max_retries         = {MAX_RETRIES}",
        f"should_transfer_files = YES",
        f"when_to_transfer_output = ON_EXIT",
        f"output              = {LOGS_DIR}/$(ClusterId).$(ProcId).out",
        f"error               = {LOGS_DIR}/$(ClusterId).$(ProcId).err",
        f"log                 = {LOGS_DIR}/condor.log",
        "",
    ]

    # One queue entry per job
    for job in jobs:
        out_path = fragment_path(
            job["gas_label"], job["pressure_label"],
            job["voltage"], job["batch_id"]
        )
        gfile = gas_file_path(job["gas_label"], job["pressure_label"])
        ptorr = PRESSURES[job["pressure_label"]]

        # Gas file lands in the job's CWD after transfer — use basename only
        gfile_basename = os.path.basename(gfile)

        # Transfer worker, physics core, and the gas file for this job
        lines.append(
            f"transfer_input_files = {REPO_DIR}/mm_condor_worker.py,"
            f"{REPO_DIR}/mm_sim_core.py,{gfile}"
        )

        # Build arguments string
        penning_args = f"--penning-mode {job['penning_mode']}"
        if job["penning_mode"] == "manual":
            penning_args += f" --penning-rP {job['penning_rP']}"
            penning_args += f" --penning-gas {job['penning_gas']}"

        args = (
            f"--gas-file {gfile_basename} "
            f"--gas-label {job['gas_label']} "
            f"--pressure-label {job['pressure_label']} "
            f"--pressure-torr {ptorr:.4f} "
            f"{penning_args} "
            f"--voltage {job['voltage']:.1f} "
            f"--events {events_per_batch} "
            f"--gap-cm {job['gap_cm']} "
            f"--output {out_path} "
            f"--batch-id {job['batch_id']:03d}"
        )

        lines += [
            f"# {job['gas_label']} | {job['pressure_label']} | "
            f"{job['voltage']:.0f}V | batch {job['batch_id']:03d}",
            f"arguments = {args}",
            "queue",
            "",
        ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Submit Micromegas gain scan to HTCondor"
    )
    parser.add_argument("--dry-run",           action="store_true",
                        help="Print plan and JDL without submitting")
    parser.add_argument("--batches",           type=int, default=10,
                        help="Batches per (gas, pressure, voltage) point (default: 10)")
    parser.add_argument("--events-per-batch",  type=int, default=200,
                        help="Events per batch job (default: 200)")
    parser.add_argument("--gas",               default=None,
                        help="Filter: only submit jobs for this gas label substring")
    parser.add_argument("--pressure",          default=None,
                        help="Filter: only submit jobs for this pressure label substring")
    args = parser.parse_args()

    batches_per_point  = args.batches
    events_per_batch   = args.events_per_batch
    target_per_point   = batches_per_point * events_per_batch

    # ── Create directories ─────────────────────────────────────────────────────
    for d in [JOBS_DIR, RESULTS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Build job list ─────────────────────────────────────────────────────────
    jobs = []
    config = RUN_CONFIG

    if args.gas:
        config = [c for c in config if args.gas in c["gas_label"]]
    if args.pressure:
        config = [c for c in config
                  if any(args.pressure in p for p in c["pressures"])]

    for gas_cfg in config:
        pressures = gas_cfg["pressures"]
        if args.pressure:
            pressures = [p for p in pressures if args.pressure in p]

        for plabel in pressures:
            gfile = gas_file_path(gas_cfg["gas_label"], plabel)
            if not os.path.exists(gfile):
                print(f"[submit] ERROR: gas file missing: {gfile}")
                print(f"         Run mm_generate_gas.py first")
                sys.exit(1)

            for voltage in gas_cfg["voltages"]:
                for batch_id in range(batches_per_point):
                    # Skip if fragment already exists (idempotent resubmission)
                    fpath = fragment_path(
                        gas_cfg["gas_label"], plabel, voltage, batch_id
                    )
                    if os.path.exists(fpath):
                        continue   # already done

                    jobs.append({
                        "gas_label":    gas_cfg["gas_label"],
                        "pressure_label": plabel,
                        "voltage":      float(voltage),
                        "batch_id":     batch_id,
                        "gap_cm":       gas_cfg["gap_cm"],
                        "penning_mode": gas_cfg["penning_mode"],
                        "penning_rP":   gas_cfg["penning_rP"],
                        "penning_gas":  gas_cfg["penning_gas"],
                    })

    if not jobs:
        print("[submit] No jobs to submit — all fragments already exist.")
        print("         Run mm_condor_collect.py to merge results.")
        sys.exit(0)

    # ── Print plan ─────────────────────────────────────────────────────────────
    total_events = len(jobs) * events_per_batch

    print("HTCondor Gain Scan Submission")
    print("=" * 55)
    print(f"Jobs to submit    : {len(jobs)}")
    print(f"Events per job    : {events_per_batch}")
    print(f"Batches per point : {batches_per_point}")
    print(f"Target per point  : {target_per_point} events")
    print(f"Total events      : {total_events:,}")
    print(f"EOS base          : {LXPLUS_BASE}")
    print()

    # Count by gas × pressure
    from collections import Counter
    counts = Counter(
        (j["gas_label"], j["pressure_label"]) for j in jobs
    )
    print("Jobs per combination:")
    for (gas, pres), n in sorted(counts.items()):
        ptorr = PRESSURES.get(pres, 0)
        print(f"  {gas} × {pres:<15s}  {ptorr:.1f} Torr  →  {n} jobs")
    print()

    # Voltage breakdown
    print("Voltage points:")
    for gas_cfg in config:
        pressures_to_show = gas_cfg["pressures"]
        if args.pressure:
            pressures_to_show = [p for p in pressures_to_show if args.pressure in p]
        for plabel in pressures_to_show:
            pending = [j["voltage"] for j in jobs
                       if j["gas_label"] == gas_cfg["gas_label"]
                       and j["pressure_label"] == plabel]
            unique_v = sorted(set(pending))
            print(f"  {gas_cfg['gas_label']} × {plabel}: "
                  f"{[int(v) for v in unique_v]} V")
    print()

    if args.dry_run:
        jdl = build_jdl(jobs, batches_per_point, events_per_batch)
        print("--- JDL preview (first 60 lines) ---")
        for line in jdl.split("\n")[:60]:
            print(line)
        if jdl.count("\n") > 60:
            print(f"... ({jdl.count(chr(10)) - 60} more lines)")
        print("--- end JDL ---")
        print("\n(dry-run — not submitting)")
        return

    # ── Write JDL and submit ───────────────────────────────────────────────────
    jdl_path = os.path.join(LXPLUS_BASE, "gain_scan.jdl")
    jdl = build_jdl(jobs, batches_per_point, events_per_batch)
    with open(jdl_path, "w") as f:
        f.write(jdl)
    print(f"[submit] JDL written → {jdl_path}")

    schedd = get_schedd()
    cmd = ["condor_submit"]
    if schedd:
        cmd += ["-name", schedd]
    cmd.append(jdl_path)

    print(f"[submit] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n[submit] ✓ Submitted {len(jobs)} jobs")
        print()
        print("Monitor with:")
        if schedd:
            print(f"  condor_q -name {schedd}")
        else:
            print(f"  condor_q")
        print()
        print("When jobs finish, collect results with:")
        print(f"  python3 mm_condor_collect.py")
    else:
        print(f"\n[submit] ✗ condor_submit failed (exit code {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
