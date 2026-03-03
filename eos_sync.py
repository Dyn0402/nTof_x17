#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 03 2:30 PM 2026
Created in PyCharm
Created as nTof_x17/eos_sync.py

@author: Dylan Neff, dylan


eos_sync.py — Synchronize DAQ run data to EOS via rsync over SSH to lxplus.

Runs on the DAQ machine. For each complete run, calls:

    rsync -av --checksum-choice=md5 --size-only \
        /media/dylan/data/x17/feb_beam/runs/run_N/ \
        lxplus.cern.ch:/eos/experiment/ntof/data/x17/feb_beam/runs/run_N/

EOS is POSIX-mounted on lxplus at /eos/..., so no xrdcp or xrdfs needed.
rsync handles skip logic, directory creation, and partial transfers natively.

A run is COMPLETE when no file inside it has been modified for
--idle-seconds (default: 60s).

On failure: logged and skipped, retried on next invocation. Already-synced
files are skipped automatically by rsync's --size-only check.

Usage:
    python eos_sync.py                        # sync all complete runs
    python eos_sync.py --run run_1            # sync one specific run
    python eos_sync.py --dry-run              # preview without copying
    python eos_sync.py --all                  # include in-progress runs
    python eos_sync.py --watch                # continuous mode (future DAQ)
    python eos_sync.py --idle-seconds 120     # custom idle threshold
"""

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_RUNS_DIR = Path("/mnt/data/x17/beam_feb/runs")

# LXPLUS_HOST     = "lxplus.cern.ch"
# LXPLUS_USER     = "dneff"                          # your CERN username
LXPLUS_ALIAS    = "lxplus_kerb"  # From ssh config file
EOS_RUNS_PATH   = "/eos/experiment/ntof/data/x17/feb_beam/runs"

# rsync flags:
#   -a  archive: preserves permissions, timestamps, recursive
#   -v  verbose: one line per file transferred
#   --size-only      skip files where local and remote size match
#   --partial        keep partially transferred files for resume
#   --progress       per-file progress (only useful interactively)
RSYNC_FLAGS = ["-av", "--size-only", "--partial"]

# SSH options: authenticate via Kerberos/GSSAPI (kinit must be valid)
#   GSSAPIAuthentication=yes      use the local Kerberos ticket
#   GSSAPIDelegateCredentials=yes forward the ticket to lxplus (needed for EOS access)
#   BatchMode=yes                 fail immediately if GSSAPI auth fails, no password prompt
SSH_OPTS = (
    "ssh -o GSSAPIAuthentication=yes "
    "-o GSSAPIDelegateCredentials=yes "
    "-o BatchMode=yes "
    "-o ConnectTimeout=15"
)

# Temp/partial extensions written by DAQ — exclude from sync
SKIP_EXTENSIONS = {".tmp", ".part", ".swp"}

# Seconds of no file modification before a run is considered complete
IDLE_SECONDS = 60

# Seconds between watch-mode polling cycles
WATCH_INTERVAL = 30

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("eos_sync.log"),
    ],
)
log = logging.getLogger("eos_sync")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_name: str
    success: bool = False
    files_transferred: int = 0
    bytes_transferred: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Completion detection
# ---------------------------------------------------------------------------

def discover_runs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        log.error(f"Local runs directory does not exist: {runs_dir}")
        sys.exit(1)
    return sorted(p for p in runs_dir.iterdir() if p.is_dir())


def latest_mtime(run_dir: Path) -> float:
    try:
        return max(p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file())
    except ValueError:
        return 0.0


def is_run_complete(run_dir: Path, idle_seconds: int) -> bool:
    mtime = latest_mtime(run_dir)
    return mtime == 0.0 or (time.time() - mtime) >= idle_seconds


def seconds_idle(run_dir: Path) -> float:
    mtime = latest_mtime(run_dir)
    return (time.time() - mtime) if mtime > 0 else float("inf")


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def build_exclude_args() -> list[str]:
    return [arg for ext in SKIP_EXTENSIONS for arg in ("--exclude", f"*{ext}")]


def sync_run(run_dir: Path, dry_run: bool = False) -> RunResult:
    """
    Sync one run directory to EOS via rsync over SSH to lxplus.
    rsync trailing slash on source means: sync contents into the destination dir.
    """
    result = RunResult(run_name=run_dir.name)

    # remote = f"{LXPLUS_USER}@{LXPLUS_HOST}:{EOS_RUNS_PATH}/{run_dir.name}/"
    remote = f"{LXPLUS_ALIAS}:{EOS_RUNS_PATH}/{run_dir.name}/"
    source = str(run_dir) + "/"   # trailing slash = sync contents, not the dir itself

    cmd = (
        ["rsync"]
        + RSYNC_FLAGS
        + (["-n"] if dry_run else [])          # -n = dry run in rsync
        + ["--stats"]                           # summary stats at the end
        + ["-e", SSH_OPTS]
        + build_exclude_args()
        + [source, remote]
    )

    log.debug(f"  $ {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        result.error = "timed out after 1h"
        return result

    # Log rsync's per-file output at debug level, summary at info level
    for line in proc.stdout.splitlines():
        if line.startswith("Number of") or line.startswith("Total"):
            log.info(f"  [{run_dir.name}] {line}")
        elif line and not line.startswith("sending") and not line.startswith("sent"):
            log.debug(f"  [{run_dir.name}] {line}")

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        result.error = f"rsync failed (exit {proc.returncode}): {err}"
        return result

    # Parse stats from rsync --stats output
    for line in proc.stdout.splitlines():
        if "Number of regular files transferred:" in line:
            result.files_transferred = int(line.split(":")[-1].strip().replace(",", ""))
        if "Total transferred file size:" in line:
            # "Total transferred file size: 1,234,567 bytes"
            parts = line.split(":")[-1].strip().split()
            try:
                result.bytes_transferred = int(parts[0].replace(",", ""))
            except (ValueError, IndexError):
                pass

    result.success = True
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_sync(
    specific_run: Optional[str] = None,
    dry_run: bool = False,
    sync_all: bool = False,
    idle_seconds: int = IDLE_SECONDS,
) -> list[RunResult]:
    runs = discover_runs(LOCAL_RUNS_DIR)

    if specific_run:
        runs = [r for r in runs if r.name == specific_run]
        if not runs:
            log.error(f"Run '{specific_run}' not found in {LOCAL_RUNS_DIR}")
            sys.exit(1)

    results = []
    skipped = 0

    for run_dir in runs:
        complete = is_run_complete(run_dir, idle_seconds)

        if not complete and not sync_all:
            idle = seconds_idle(run_dir)
            log.info(f"Skipping: {run_dir.name}  (idle {idle:.0f}s / need {idle_seconds}s)")
            skipped += 1
            continue

        status = "complete" if complete else "in-progress (forced)"
        log.info(f"Syncing: {run_dir.name}  [{status}]")
        result = sync_run(run_dir, dry_run=dry_run)
        results.append(result)

        if result.success:
            log.info(
                f"  → {result.files_transferred} transferred  "
                f"({result.bytes_transferred / 1e6:.1f} MB)"
            )
        else:
            log.warning(f"  → FAILED: {result.error}  (will retry on next sync)")

    if skipped:
        log.info(f"Skipped {skipped} in-progress run(s).  Use --all to force.")

    return results


def print_summary(results: list[RunResult]):
    sep = "─" * 56
    failed = [r for r in results if not r.success]
    log.info(sep)
    log.info(f"  Runs processed : {len(results)}")
    log.info(f"  Runs OK        : {len(results) - len(failed)}")
    log.info(f"  Files synced   : {sum(r.files_transferred for r in results)}")
    log.info(f"  Data sent      : {sum(r.bytes_transferred for r in results) / 1e6:.1f} MB")
    log.info(f"  Runs FAILED    : {len(failed)}")
    if failed:
        log.warning("Failures (will retry on next sync):")
        for r in failed:
            log.warning(f"  {r.run_name}: {r.error}")
    log.info(sep)


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_mode(dry_run: bool = False, idle_seconds: int = IDLE_SECONDS):
    """
    Continuously poll for completed runs and sync them.
    Fully synced runs are remembered for the lifetime of the process.
    Failed runs are retried each cycle.
    """
    fully_synced: set[str] = set()
    log.info(f"Watch mode — polling every {WATCH_INTERVAL}s.  Ctrl-C to stop.")

    while True:
        try:
            for run_dir in discover_runs(LOCAL_RUNS_DIR):
                if run_dir.name in fully_synced:
                    continue
                if not is_run_complete(run_dir, idle_seconds):
                    continue
                log.info(f"New complete run: {run_dir.name}")
                result = sync_run(run_dir, dry_run=dry_run)
                if result.success:
                    fully_synced.add(run_dir.name)
                    log.info(f"Run {run_dir.name} fully synced.")
                else:
                    log.warning(f"Run {run_dir.name} failed — retrying next cycle.")
        except KeyboardInterrupt:
            log.info("Watch mode stopped.")
            break
        except Exception as e:
            log.error(f"Watch loop error: {e}")

        time.sleep(WATCH_INTERVAL)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync DAQ runs to EOS via rsync over SSH to lxplus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python eos_sync.py                     sync all complete runs
  python eos_sync.py --run run_1         sync one run only
  python eos_sync.py --dry-run           preview without transferring
  python eos_sync.py --all               include in-progress runs
  python eos_sync.py --watch             continuous mode for DAQ
  python eos_sync.py --idle-seconds 120  require 2 min of inactivity
        """
    )
    parser.add_argument("--run",          metavar="RUN_NAME")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--all",          action="store_true")
    parser.add_argument("--watch",        action="store_true")
    parser.add_argument("--idle-seconds", type=int, default=IDLE_SECONDS)
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.dry_run:
        log.info("DRY RUN — no files will be transferred.")

    if args.watch:
        watch_mode(dry_run=args.dry_run, idle_seconds=args.idle_seconds)
    else:
        results = run_sync(
            specific_run=args.run,
            dry_run=args.dry_run,
            sync_all=args.all,
            idle_seconds=args.idle_seconds,
        )
        print_summary(results)
        sys.exit(0 if all(r.success for r in results) else 1)


if __name__ == "__main__":
    main()