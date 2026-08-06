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
  - Ne/iC4H10:  400–530 V, step 10 V, both Saclay and CERN × rP=0.40/0.50/0.60
  - He/C2H6 and Ar/iC4H10 commented out (already run)
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

# Voltage grid for the Ar/CO2/iC4H10 93/5/2 scan. Set from the prescan (see
# PRESCAN_VOLTAGES below) so the mixture's gain span brackets the Ar/iC4H10
# 95/5 reference span without extrapolating either curve.
# 2026-07-31: mm_alpha_predict.py on the finished nColl=10 table (CERN) puts the
# 95/5 reference gain span (3.1e3 - 5.4e4) at 450-500 V in this mixture. K is
# only known to +-0.28, i.e. ~+-25 V on that window, so the scan runs 420-530 V
# to keep the match interior to simulated data at both ends.
TERNARY_VOLTAGES = list(range(420, 531, 10))

# Wide, cheap grid used once with --batches 1 --events-per-batch 20 to locate
# the useful window before committing to the production scan.
PRESCAN_VOLTAGES = list(range(340, 601, 20))

# Voltage grid for Ne/CF4/C2H6 80/10/10. NOT set from alpha: the alpha-based
# predictor needs a calibration constant K that is strongly mixture-dependent
# (2.07 for the Ar/iC4H10 family, 1.44 for Ar/CO2 70/30) and there is no value
# for a Ne mixture with a hand-set Penning probability. K = 1.0 vs 1.5 moved the
# predicted window by more than 100 V, so it was measured instead: a 40-job
# prescan at 20 events/point over 180-560 V (cluster 13324573, 2026-08-01,
# r = 0.50), fragments in prescan_jobs/.
#
# Prescan result at CERN pressure: G = 3 at 180 V, 41 at 300 V, 1047 at 420 V,
# 9810 at 500 V, 14068 at 520 V. The Ar/CO2 70/30 reference spans G = 10 (400 V)
# to ~1e4 (740 V), so the matching Ne range is ~245-500 V. Runs 240-540 to keep
# the r = 0.40 and 0.60 brackets — which shift the curve in voltage — interior
# to simulated data at both ends.
NE_TERNARY_VOLTAGES = list(range(240, 541, 10))

# Voltage grids for the 50 µm uRWELL branch, set from a prescan at THIS gap
# (68 jobs, 20 events/point, 150-470 V, 2026-08-01, fragments in
# prescan50_jobs/). They are NOT the 150 µm window rescaled: equal gain means
# equal (alpha-eta)·d and alpha is steeply non-linear in E, so a 3x smaller gap
# is not a 3x smaller voltage.
#
# Prescan at CERN pressure, gain per seed electron:
#   Ar/CO2 70/30   G = 10 at ~212 V,  1e3 at ~400 V,  1e4 at ~490 V
#   Ne/CF4/C2H6    G = 10 at ~145 V,  1e3 at ~340 V,  1e4 at ~440 V
#
# So at uRWELL fields the two gases sit only ~50-60 V apart at operating gain,
# against ~226 V apart at 150 µm. That is the alpha ratio collapsing from ~2.5
# at 27-49 kV/cm to ~1.2 at 60-120 kV/cm — the gases genuinely converge here.
# The two grids differ because the mixtures cover the same GAIN range over
# different voltage ranges; matching them is the whole point.
URWELL_VOLTAGES_ARCO2 = list(range(280, 521, 10))
URWELL_VOLTAGES_NE    = list(range(220, 481, 10))

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
        "voltages":        list(range(400, 601, 10)),   # 400..600 step 10 (extended for quencher scan)
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    # ── Ar/iC4H10 quencher scan: 98/2, 90/10, 85/15, 80/20, 75/25 ──────────────
    # Relative-gain study referenced to Ar/iC4H10 95/5 at 490 V. More isobutane
    # lowers gain at fixed V, so the full 400..600 V range gives the high-quencher
    # mixtures room to reach the reference gain. Penning auto (Sahin Ar/iC4H10).
    {
        "gas_label":       "Ar_iC4H10_98_2",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 601, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    {
        "gas_label":       "Ar_iC4H10_90_10",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 601, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    {
        "gas_label":       "Ar_iC4H10_85_15",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 601, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    {
        "gas_label":       "Ar_iC4H10_80_20",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 601, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    {
        "gas_label":       "Ar_iC4H10_75_25",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 601, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    # Ar/CO2 70/30. mode=auto, and auto is NOT "no Penning" here: Garfield++ has
    # a built-in Ar/CO2 curve and applies rP = 0.547 at 30% CO2 (probed 2026-07-31;
    # the old comment claiming rP = 0 on energetics was wrong). Heavily quenched:
    # 400-530 V only reaches G = 10-120, so the 540-700 V block is what covers
    # operating gain. See ARCO2_NECF4C2H6.md.
    {
        "gas_label":       "Ar_CO2_70_30",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 741, 10)),   # 400..740 step 10
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    # ── Ar/CO2/iC4H10 93/5/2 — the n_TOF operating mixture ────────────────────
    # Garfield++ has NO built-in Penning parameterisation for this ternary
    # (EnablePenningTransfer() returns False → rP = 0), so it must be set by
    # hand or the mixture would be simulated with no Penning at all while the
    # 95/5 reference runs at rP = 0.40. Central value 0.40, bracketed 0.30–0.50;
    # see the note in mm_config.py. All three variants share ONE gas table
    # (Penning is applied after LoadGasFile), hence "gas_table_label".
    {
        "gas_label":       "Ar_CO2_iC4H10_93_5_2_rP030",
        "gas_table_label": "Ar_CO2_iC4H10_93_5_2",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        TERNARY_VOLTAGES,
        "gap_cm":          0.015,
        "penning_mode":    "manual",
        "penning_rP":      0.30,
        "penning_gas":     "ar",
    },
    {
        "gas_label":       "Ar_CO2_iC4H10_93_5_2_rP040",
        "gas_table_label": "Ar_CO2_iC4H10_93_5_2",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        TERNARY_VOLTAGES,
        "gap_cm":          0.015,
        "penning_mode":    "manual",
        "penning_rP":      0.40,
        "penning_gas":     "ar",
    },
    {
        "gas_label":       "Ar_CO2_iC4H10_93_5_2_rP050",
        "gas_table_label": "Ar_CO2_iC4H10_93_5_2",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        TERNARY_VOLTAGES,
        "gap_cm":          0.015,
        "penning_mode":    "manual",
        "penning_rP":      0.50,
        "penning_gas":     "ar",
    },

    # ── Ne/CF4/C2H6 80/10/10 ─────────────────────────────────────────────────
    # Same trap as the Ar ternary, worse: Garfield++ has no Penning curve for
    # Ne/CF4, for Ne/C2H6, or for the ternary, so auto would give rP = 0 while
    # the Ar/CO2 70/30 it is compared against runs on a built-in curve at 0.547.
    # Central 0.50, bracketed 0.40-0.60 — an assumption, not a measurement, and
    # the dominant systematic on this map. All three share one gas table.
    {
        "gas_label":       "Ne_CF4_C2H6_80_10_10_rP040",
        "gas_table_label": "Ne_CF4_C2H6_80_10_10",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        NE_TERNARY_VOLTAGES,
        "gap_cm":          0.015,
        "penning_mode":    "manual",
        "penning_rP":      0.40,
        "penning_gas":     "ne",
    },
    {
        "gas_label":       "Ne_CF4_C2H6_80_10_10_rP050",
        "gas_table_label": "Ne_CF4_C2H6_80_10_10",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        NE_TERNARY_VOLTAGES,
        "gap_cm":          0.015,
        "penning_mode":    "manual",
        "penning_rP":      0.50,
        "penning_gas":     "ne",
    },
    {
        "gas_label":       "Ne_CF4_C2H6_80_10_10_rP060",
        "gas_table_label": "Ne_CF4_C2H6_80_10_10",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        NE_TERNARY_VOLTAGES,
        "gap_cm":          0.015,
        "penning_mode":    "manual",
        "penning_rP":      0.60,
        "penning_gas":     "ne",
    },

    # ── 50 µm uRWELL branch ──────────────────────────────────────────────────
    # Same two gases as the 150 µm Micromegas work above, at the uRWELL foil
    # thickness. Three things had to change and none of them are cosmetic:
    #
    #   1. gap_cm 0.0150 -> 0.0050. Distinct gas_labels (_uRW50) keep these
    #      results from being merged into the 150 µm ones.
    #   2. Different gas tables. At 50 µm the amplification field is 60-120
    #      kV/cm, and EVERY earlier table stops at 60 kV/cm or below — Garfield
    #      would have been reading off the end of the table. The _hiE tables run
    #      to 170 kV/cm (~700 Td, still inside Magboltz's range).
    #   3. K must be re-calibrated here. The 150 µm value drifts -18.5 % across
    #      its own field range, so carrying it up to 3x the field is not safe.
    #
    # And the caveat that outlives all three: a uRWELL amplifies inside conical
    # holes where the field is strongly non-uniform, so a ComponentConstant slab
    # is a much poorer model here than it is for a Micromegas mesh gap. Expect
    # this to over-predict absolute gain. Use it to compare gases, not to
    # predict a working point.
    {
        "gas_label":       "Ar_CO2_70_30_uRW50",
        "gas_table_label": "Ar_CO2_70_30_hiE",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        URWELL_VOLTAGES_ARCO2,
        "gap_cm":          0.0050,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    {
        "gas_label":       "Ne_CF4_C2H6_80_10_10_uRW50_rP040",
        "gas_table_label": "Ne_CF4_C2H6_80_10_10_hiE",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        URWELL_VOLTAGES_NE,
        "gap_cm":          0.0050,
        "penning_mode":    "manual",
        "penning_rP":      0.40,
        "penning_gas":     "ne",
    },
    {
        "gas_label":       "Ne_CF4_C2H6_80_10_10_uRW50_rP050",
        "gas_table_label": "Ne_CF4_C2H6_80_10_10_hiE",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        URWELL_VOLTAGES_NE,
        "gap_cm":          0.0050,
        "penning_mode":    "manual",
        "penning_rP":      0.50,
        "penning_gas":     "ne",
    },
    {
        "gas_label":       "Ne_CF4_C2H6_80_10_10_uRW50_rP060",
        "gas_table_label": "Ne_CF4_C2H6_80_10_10_hiE",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        URWELL_VOLTAGES_NE,
        "gap_cm":          0.0050,
        "penning_mode":    "manual",
        "penning_rP":      0.60,
        "penning_gas":     "ne",
    },

    # ── New gases (uncomment when gas tables have been generated) ──────────────
    {
        "gas_label":       "Ar_CF4_90_10",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 531, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    {
        "gas_label":       "Ar_CF4_iC4H10_88_10_2",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 531, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    # Ne/CF4 — strong Penning; consider scanning rP=0.30/0.40/0.50 like Ne/iC4H10
    {
        "gas_label":       "Ne_CF4_90_10",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 531, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "manual",
        "penning_rP":      0.40,
        "penning_gas":     "ne",
    },
    {
        "gas_label":       "Ar_CF4_CO2_45_40_15",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 531, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    {
        "gas_label":       "CF4_100",
        "pressures":       ["Saclay_160m", "CERN_450m"],
        "voltages":        list(range(400, 531, 10)),
        "gap_cm":          0.015,
        "penning_mode":    "auto",
        "penning_rP":      0.0,
        "penning_gas":     "",
    },
    # Ne/iC4H10 — three rP values to bracket Penning uncertainty (no measured rP in literature)
    # {
    #     "gas_label":       "Ne_iC4H10_95_5_rP040",
    #     "pressures":       ["Saclay_160m", "CERN_450m"],
    #     "voltages":        list(range(400, 531, 10)),   # 400..530 step 10 (range TBD)
    #     "gap_cm":          0.015,
    #     "penning_mode":    "manual",
    #     "penning_rP":      0.40,
    #     "penning_gas":     "ne",
    # },
    # {
    #     "gas_label":       "Ne_iC4H10_95_5_rP050",
    #     "pressures":       ["Saclay_160m", "CERN_450m"],
    #     "voltages":        list(range(400, 531, 10)),   # 400..530 step 10 (range TBD)
    #     "gap_cm":          0.015,
    #     "penning_mode":    "manual",
    #     "penning_rP":      0.50,
    #     "penning_gas":     "ne",
    # },
    # {
    #     "gas_label":       "Ne_iC4H10_95_5_rP060",
    #     "pressures":       ["Saclay_160m", "CERN_450m"],
    #     "voltages":        list(range(400, 531, 10)),   # 400..530 step 10 (range TBD)
    #     "gap_cm":          0.015,
    #     "penning_mode":    "manual",
    #     "penning_rP":      0.60,
    #     "penning_gas":     "ne",
    # },
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


def np_arange_inclusive(lo, hi, step):
    """Inclusive float range, used by the --voltages 'lo:hi:step' form."""
    out, v = [], lo
    while v <= hi + 1e-6:
        out.append(round(v, 3))
        v += step
    return out


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
        gfile = gas_file_path(job.get("gas_table_label", job["gas_label"]),
                              job["pressure_label"])
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
    parser.add_argument("--voltages",          default=None,
                        help="Override the voltage grid: 'lo:hi:step' or a "
                             "comma-separated list (used for prescans)")
    parser.add_argument("--jobs-dir",          default=None,
                        help="Override the fragment output directory (keeps a "
                             "prescan out of the production fragments)")
    args = parser.parse_args()

    batches_per_point  = args.batches
    events_per_batch   = args.events_per_batch
    target_per_point   = batches_per_point * events_per_batch

    if args.jobs_dir:
        global JOBS_DIR
        # MUST be absolute. The worker writes the fragment to this path on the
        # execute node, where it only reaches us because AFS is mounted there;
        # a relative path silently lands in the Condor scratch dir and is thrown
        # away when the job exits. (Cost this a prescan on 2026-08-01.)
        JOBS_DIR = os.path.abspath(args.jobs_dir)
        if JOBS_DIR != args.jobs_dir:
            print(f"[submit] --jobs-dir resolved to absolute: {JOBS_DIR}")
        os.makedirs(JOBS_DIR, exist_ok=True)
        print(f"[submit] fragments → {JOBS_DIR}")

    volt_override = None
    if args.voltages:
        if ":" in args.voltages:
            lo, hi, step = (float(x) for x in args.voltages.split(":"))
            volt_override = list(np_arange_inclusive(lo, hi, step))
        else:
            volt_override = [float(x) for x in args.voltages.split(",")]
        print(f"[submit] voltage override: {[int(v) for v in volt_override]}")

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

        table_label = gas_cfg.get("gas_table_label", gas_cfg["gas_label"])

        for plabel in pressures:
            gfile = gas_file_path(table_label, plabel)
            if not os.path.exists(gfile):
                print(f"[submit] ERROR: gas file missing: {gfile}")
                print(f"         Run mm_generate_gas.py (or mm_gasgen.sub) first")
                sys.exit(1)

            for voltage in (volt_override or gas_cfg["voltages"]):
                for batch_id in range(batches_per_point):
                    # Skip if fragment already exists (idempotent resubmission)
                    fpath = fragment_path(
                        gas_cfg["gas_label"], plabel, voltage, batch_id
                    )
                    if os.path.exists(fpath):
                        continue   # already done

                    jobs.append({
                        "gas_label":    gas_cfg["gas_label"],
                        "gas_table_label": table_label,
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
