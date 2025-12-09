#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 27 11:54 2025
Created in PyCharm
Created as nTof_x17/download_new_runs

@author: Dylan Neff, dn277127
"""

import subprocess
import os

# --- CONFIGURATION ---
PI_HOST = "pi@10.42.0.159"
JUMP_HOST = "ntofall@128.141.177.225"

PI_BASE_DIR = "/home/pi/x17/dream_run"
LOCAL_BASE_DIR = "/local/home/dn277127/x17/dream_run"

# rsync options:
# -a : archive mode preserves permissions and timestamps
# -v : verbose
# -u : skip files newer on receiver
# --progress : display file transfer progress
RSYNC_OPTS = "-avu --progress"


def main():
    print("Checking runs on Pi...")
    runs = get_pi_runs()
    print(f'Pi runs found: {runs}')

    min_run = 20  # only sync runs >= this number
    runs = [run for run in runs if run.startswith("run_") and int(run.split("_")[1]) >= min_run]
    print(f'Filtering to runs >= {min_run}: {runs}')

    for run in runs:
        # if not run.startswith("run"):  # optional sanity filter
        #     continue
        run_path = os.path.join(LOCAL_BASE_DIR, run)
        sync_run(run)

    print("Sync complete.")
    print('donzo')


def get_pi_runs():
    """Return a list of run directories on the Raspberry Pi."""
    cmd = [
        "ssh",
        "-J", f"{JUMP_HOST}",
        f"{PI_HOST}",
        f"ls -1 {PI_BASE_DIR}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    runs = result.stdout.split()
    return runs


def ensure_local_dir(path):
    """Create local directory if it does not exist."""
    if not os.path.exists(path):
        print(f"Creating local directory: {path}")
        os.makedirs(path, exist_ok=True)


def sync_run(run_name):
    """Sync .fdf files for a single run using rsync."""
    pi_dir = f"{PI_HOST}:{PI_BASE_DIR}/{run_name}/"
    local_dir = f"{LOCAL_BASE_DIR}/{run_name}/"

    ensure_local_dir(local_dir)

    rsync_patterns = [
        "--include=*/",        # keep directory structure
        "--include=*.fdf",     # include only .fdf files
        "--exclude=*",         # exclude everything else
        "--prune-empty-dirs",
    ]

    cmd = [
        "rsync",
        *RSYNC_OPTS.split(),
        *rsync_patterns,
        "-e", f"ssh -J {JUMP_HOST}",
        pi_dir,
        local_dir,
    ]

    print(f"Syncing {run_name}...")
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
