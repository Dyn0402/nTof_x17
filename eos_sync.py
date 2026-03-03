#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 03 2:30 PM 2026
Created in PyCharm
Created as nTof_x17/eos_sync.py

@author: Dylan Neff, dylan


eos_sync.py — Synchronize DAQ run data to EOS via xrdcp.

Recursively walks each run directory and syncs every file found,
preserving the directory structure on EOS. New file types or subdirectory
layouts are handled automatically.

Run structure (flexible — any layout under runs/ is handled):
    runs/
      run_1/
        run_config.json
        sub_run_1/
          raw_data/
            file_1.fdf
        ...

A run is COMPLETE (safe to sync) when no file inside it has been modified
for --idle-seconds (default: 60s).

On failure: the file is logged and skipped. Already-transferred files
(size-matched on EOS) are skipped on retry, so only failures are re-sent.

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
EOS_REDIRECTOR = "eosntof.cern.ch"
EOS_RUNS_PATH  = "/eos/experiment/ntof/data/x17/feb_beam/runs"
EOS_RUNS_DIR   = f"root://{EOS_REDIRECTOR}/{EOS_RUNS_PATH}"

# A run idle for this many seconds with no file modifications is complete
IDLE_SECONDS = 60

# How long between watch-mode polling cycles
WATCH_INTERVAL = 30  # seconds

# xrdcp flags: -f force overwrite, -N no progress bar, -S parallel streams
XRDCP_FLAGS = ["-f", "-N", "-S", "4"]

# Temp/partial extensions written by DAQ — skip these
SKIP_EXTENSIONS = {".tmp", ".part", ".swp"}

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
    files_transferred: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    bytes_transferred: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.files_failed == 0


# ---------------------------------------------------------------------------
# Completion detection
# ---------------------------------------------------------------------------

def discover_runs(runs_dir: Path) -> list[Path]:
    """Return sorted list of run subdirectories."""
    if not runs_dir.exists():
        log.error(f"Local runs directory does not exist: {runs_dir}")
        sys.exit(1)
    return sorted(p for p in runs_dir.iterdir() if p.is_dir())


def latest_mtime(run_dir: Path) -> float:
    """Most recent modification time of any file in the run tree."""
    try:
        return max(p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file())
    except ValueError:
        return 0.0  # empty directory


def is_run_complete(run_dir: Path, idle_seconds: int) -> bool:
    mtime = latest_mtime(run_dir)
    return mtime == 0.0 or (time.time() - mtime) >= idle_seconds


def seconds_idle(run_dir: Path) -> float:
    mtime = latest_mtime(run_dir)
    return (time.time() - mtime) if mtime > 0 else float("inf")


# ---------------------------------------------------------------------------
# EOS helpers
# ---------------------------------------------------------------------------

def to_eos_url(local_path: Path, local_base: Path, eos_base: str) -> str:
    """Map any local path to its corresponding EOS xroot URL."""
    return f"{eos_base}/{local_path.relative_to(local_base)}"


def split_eos_url(eos_url: str) -> tuple[str, str]:
    """Split 'root://host/path' into (host, '/path')."""
    without_scheme = eos_url[len("root://"):]
    host, _, fpath = without_scheme.partition("/")
    return host, ("/" + fpath.lstrip("/"))


def eos_file_size(host: str, fpath: str) -> Optional[int]:
    """Return size of a file on EOS, or None if it doesn't exist."""
    try:
        result = subprocess.run(
            ["xrdfs", f"root://{host}", "stat", fpath],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if "Size:" in line:
                return int(line.split("Size:")[1].split()[0])
    except Exception:
        pass
    return None


def make_eos_dir(host: str, fpath: str) -> None:
    """Create a directory (and parents) on EOS. Logs a warning on failure."""
    result = subprocess.run(
        ["xrdfs", f"root://{host}", "mkdir", "-p", fpath],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        log.warning(f"mkdir failed for {fpath}: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

def transfer_file(
    local_file: Path,
    eos_url: str,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Copy one file to EOS via xrdcp.
    Skips if the remote already has the same size.
    Returns (success, reason).
    """
    local_size = local_file.stat().st_size
    host, fpath = split_eos_url(eos_url)

    if eos_file_size(host, fpath) == local_size:
        return True, f"skipped ({local_size:,} bytes already on EOS)"

    if dry_run:
        return True, f"dry-run: would copy ({local_size:,} bytes)"

    try:
        result = subprocess.run(
            ["xrdcp"] + XRDCP_FLAGS + [str(local_file), eos_url],
            capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        return False, "timed out after 600s"

    if result.returncode == 0:
        return True, f"transferred ({local_size:,} bytes)"

    err = result.stderr.strip() or result.stdout.strip() or "unknown error"
    return False, f"xrdcp failed: {err}"


# ---------------------------------------------------------------------------
# Run sync — the core logic
# ---------------------------------------------------------------------------

def sync_run(
    run_dir: Path,
    local_base: Path,
    eos_base: str,
    dry_run: bool = False,
) -> RunResult:
    """
    Recursively walk a run directory and sync every file to EOS,
    mirroring the directory structure exactly.
    """
    result = RunResult(run_name=run_dir.name)
    host, _ = split_eos_url(eos_base)

    # Collect all transferable files
    files = sorted(
        p for p in run_dir.rglob("*")
        if p.is_file() and p.suffix not in SKIP_EXTENSIONS
    )

    if not files:
        log.info(f"[{run_dir.name}] No files found.")
        return result

    # Pre-create all required remote directories in one pass
    if not dry_run:
        remote_dirs = sorted({split_eos_url(to_eos_url(f.parent, local_base, eos_base))[1] for f in files})
        for fpath in remote_dirs:
            make_eos_dir(host, fpath)

    # Transfer files
    for local_file in files:
        eos_url = to_eos_url(local_file, local_base, eos_base)
        rel = local_file.relative_to(run_dir)
        success, reason = transfer_file(local_file, eos_url, dry_run=dry_run)

        if "skipped" in reason or "dry-run" in reason:
            result.files_skipped += 1
            log.info(f"[{run_dir.name}] SKIP  {rel}  ({reason})")
        elif success:
            result.files_transferred += 1
            result.bytes_transferred += local_file.stat().st_size
            log.info(f"[{run_dir.name}] OK    {rel}  ({reason})")
        else:
            result.files_failed += 1
            result.errors.append(f"{rel}: {reason}")
            log.error(f"[{run_dir.name}] FAIL  {rel}  ({reason})")

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
        result = sync_run(run_dir, LOCAL_RUNS_DIR, EOS_RUNS_DIR, dry_run=dry_run)
        results.append(result)

        if result.success:
            log.info(f"  → {result.files_transferred} transferred, {result.files_skipped} skipped.")
        else:
            log.warning(f"  → {result.files_failed} failure(s) — will retry on next sync.")

    if skipped:
        log.info(f"Skipped {skipped} in-progress run(s).  Use --all to force.")

    return results


def print_summary(results: list[RunResult]):
    sep = "─" * 56
    log.info(sep)
    log.info(f"  Runs      : {len(results)}")
    log.info(f"  OK        : {sum(r.files_transferred for r in results)} files  "
             f"({sum(r.bytes_transferred for r in results) / 1e6:.1f} MB)")
    log.info(f"  Skipped   : {sum(r.files_skipped for r in results)} files  (already on EOS)")
    log.info(f"  Failed    : {sum(r.files_failed for r in results)} files")
    failed = [r for r in results if not r.success]
    if failed:
        log.warning("Failures (will retry on next sync):")
        for r in failed:
            for err in r.errors:
                log.warning(f"  {r.run_name} / {err}")
    log.info(sep)


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_mode(dry_run: bool = False, idle_seconds: int = IDLE_SECONDS):
    """
    Continuously poll and sync completed runs. Fully synced runs are
    remembered for the lifetime of the process. Failed runs are retried
    each cycle. Intended to run inside a k5reauth + tmux session.
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
                result = sync_run(run_dir, LOCAL_RUNS_DIR, EOS_RUNS_DIR, dry_run=dry_run)
                if result.success:
                    fully_synced.add(run_dir.name)
                else:
                    log.warning(f"Run {run_dir.name} has failures — retrying next cycle.")
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
        description="Sync DAQ runs to EOS via xrdcp (recursive, structure-agnostic).",
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