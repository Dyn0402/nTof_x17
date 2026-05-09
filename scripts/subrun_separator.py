#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 09 1:29 PM 2026
Created in PyCharm
Created as nTof_x17/subrun_separator.py

@author: Dylan Neff, dylan
"""

#!/usr/bin/env python3
"""
fix_sub_runs.py
===============
Fixes two problems caused by the sub_run_name counter not incrementing:

1. JSON fix  : Renames all "hv_scan_0" sub_run_name entries to hv_scan_0,
               hv_scan_1, hv_scan_2 ... in run_config.json, writing a
               corrected copy alongside the original.

2. File fix  : All data landed in hv_scan_0/raw_daq_data/.  Files are sorted
               chronologically by the timestamp embedded in their names
               (YYMMDD_HHhMM) and moved — in order — into the corresponding
               hv_scan_N directory, mirroring the same inner structure
               (raw_daq_data/, decoded_root/, etc.).

Usage
-----
    python3 fix_sub_runs.py [--run-dir PATH] [--dry-run] [--yes]

Options
-------
    --run-dir PATH   Path to run_1 directory  [default: ~/beam_may/runs/run_1]
    --dry-run        Print every action but do NOT touch the filesystem.
    --yes            Skip the confirmation prompt and proceed immediately.
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"(\d{6}_\d{2}H\d{2})")


def parse_file_timestamp(name: str) -> datetime | None:
    """Extract and parse the YYMMDD_HHhMM timestamp from a filename."""
    m = TIMESTAMP_RE.search(name)
    if not m:
        return None
    return datetime.strptime("20" + m.group(1), "%Y%m%d_%HH%M")


def group_files_by_timestamp(src_dir: Path) -> dict[str, list[Path]]:
    """Return {timestamp_string: [file, ...]} for every file in src_dir."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in src_dir.iterdir():
        if f.is_file():
            m = TIMESTAMP_RE.search(f.name)
            if m:
                groups[m.group(1)].append(f)
            else:
                print(f"  [WARN] Cannot parse timestamp from: {f.name} — skipping")
    return groups


# ---------------------------------------------------------------------------
# Step 1 – Fix JSON
# ---------------------------------------------------------------------------

def fix_json(run_dir: Path, dry_run: bool) -> tuple[list[str], Path]:
    """
    Rename sub_run_name entries hv_scan_0 … hv_scan_N in the config JSON.
    Returns (list_of_new_names, output_path).
    """
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        sys.exit(f"ERROR: run_config.json not found at {config_path}")

    with config_path.open() as fh:
        cfg = json.load(fh)

    sub_runs = cfg.get("sub_runs", [])
    if not sub_runs:
        sys.exit("ERROR: No sub_runs found in run_config.json")

    new_names = []
    for i, sr in enumerate(sub_runs):
        old = sr["sub_run_name"]
        new = re.sub(r"_\d+$", f"_{i}", old)  # replace trailing _N with _i
        sr["sub_run_name"] = new
        new_names.append(new)
        if old != new:
            print(f"  JSON sub_run[{i}]: '{old}' → '{new}'")
        else:
            print(f"  JSON sub_run[{i}]: '{old}' (already correct)")

    out_path = run_dir / "run_config_fixed.json"
    if not dry_run:
        with out_path.open("w") as fh:
            json.dump(cfg, fh, indent=4)
        print(f"\n  Written: {out_path}")
    else:
        print(f"\n  [DRY-RUN] Would write: {out_path}")

    return new_names, out_path


# ---------------------------------------------------------------------------
# Step 2 – Reorganise files
# ---------------------------------------------------------------------------

def fix_files(run_dir: Path, sub_run_names: list[str], dry_run: bool) -> None:
    """
    Move files from hv_scan_0/{inner_dir}/ into hv_scan_N/{inner_dir}/.

    Strategy
    --------
    1. Collect every file in hv_scan_0/ (recursively, preserving inner dirs).
    2. Group files by their embedded timestamp.
    3. Sort timestamp groups chronologically — the i-th group belongs to
       sub_run_names[i].
    4. Create destination directories and move/copy files.
    """
    src_base = run_dir / "hv_scan_0"
    if not src_base.exists():
        sys.exit(f"ERROR: Source directory not found: {src_base}")

    # Collect all files grouped by (inner_subdir, timestamp)
    # e.g. inner_subdir = "raw_daq_data"
    all_groups: dict[str, dict[str, list[Path]]] = {}  # inner_dir -> ts -> files

    for inner_dir in src_base.iterdir():
        if not inner_dir.is_dir():
            continue
        rel = inner_dir.name
        groups = group_files_by_timestamp(inner_dir)
        if groups:
            all_groups[rel] = groups

    if not all_groups:
        sys.exit(f"ERROR: No timestamped files found under {src_base}")

    # Get the sorted timestamp list from the first inner_dir found
    # (they should be identical across inner dirs)
    ref_inner = next(iter(all_groups))
    sorted_timestamps = sorted(
        all_groups[ref_inner].keys(),
        key=lambda ts: datetime.strptime("20" + ts, "%Y%m%d_%HH%M"),
    )

    n_ts = len(sorted_timestamps)
    n_sr = len(sub_run_names)
    if n_ts != n_sr:
        print(
            f"\n  [WARN] Timestamp groups found ({n_ts}) ≠ sub_runs in JSON ({n_sr})."
        )
        print("  Will map as many as possible (min of the two).")

    n = min(n_ts, n_sr)

    print(f"\n  Mapping {n} timestamp groups → sub_run directories:")
    for i in range(n):
        ts = sorted_timestamps[i]
        dst_name = sub_run_names[i]
        print(f"    [{i:2d}] ts={ts}  →  {dst_name}/")

    print()

    # Move files
    moved = 0
    skipped = 0
    for i in range(n):
        ts = sorted_timestamps[i]
        dst_sub = sub_run_names[i]

        for inner_dir, ts_groups in all_groups.items():
            files = ts_groups.get(ts, [])
            for src_file in sorted(files):
                dst_dir = run_dir / dst_sub / inner_dir
                dst_file = dst_dir / src_file.name

                if dst_file == src_file:
                    # Already in the right place (sub_run_0 files)
                    skipped += 1
                    continue

                if not dry_run:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_file), dst_file)

                print(f"  {'[DRY-RUN] MOVE' if dry_run else 'MOVE'}: "
                      f"{src_file.relative_to(run_dir)}  →  "
                      f"{dst_file.relative_to(run_dir)}")
                moved += 1

    print(f"\n  Files moved : {moved}")
    print(f"  Files kept  : {skipped}  (hv_scan_0 files already in correct dir)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        default="~/beam_may/runs/run_1",
        help="Path to the run_1 directory (default: ~/beam_may/runs/run_1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        sys.exit(f"ERROR: run directory not found: {run_dir}")

    print("=" * 65)
    print("  MX17 Sub-Run Fixer")
    print("=" * 65)
    print(f"  Run directory : {run_dir}")
    print(f"  Dry-run mode  : {args.dry_run}")
    print()

    # ── Step 1: Fix JSON ──────────────────────────────────────────────────
    print("[ STEP 1 ] Fixing sub_run_names in run_config.json")
    print("-" * 65)
    sub_run_names, json_out = fix_json(run_dir, dry_run=True)  # always preview first

    # ── Step 2: Plan file moves ───────────────────────────────────────────
    print("\n[ STEP 2 ] Planning file reorganisation")
    print("-" * 65)
    fix_files(run_dir, sub_run_names, dry_run=True)  # always preview first

    # ── Confirmation ──────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY-RUN] No changes made. Re-run without --dry-run to apply.")
        return

    if not args.yes:
        print("\n" + "=" * 65)
        ans = input("  Apply all changes? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("  Aborted — no changes made.")
            return

    print("\n[ APPLYING CHANGES ]")
    print("-" * 65)
    print("\n  Writing corrected JSON …")
    fix_json(run_dir, dry_run=False)
    print("\n  Moving files …")
    fix_files(run_dir, sub_run_names, dry_run=False)
    print("\nDone.")


if __name__ == "__main__":
    main()