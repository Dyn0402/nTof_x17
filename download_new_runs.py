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
# PI_HOST = "pi@10.42.0.159"
# JUMP_HOST = "ntofall@128.141.177.225"
# PI_BASE_DIR = "/home/pi/x17/dream_run"
# LOCAL_BASE_DIR = "/local/home/dn277127/x17/dream_run"

# HOST = 'daq'
# # DAQ_BASE_DIR = "/home/mx17/feb_beam/dream_run"
# # LOCAL_BASE_DIR = "/media/dylan/data/x17/feb_beam/dream_run"
# DAQ_BASE_DIR = "/home/mx17/feb_beam/runs"
# LOCAL_BASE_DIR = "/media/dylan/data/x17/feb_beam/runs"

HOST = 'lxplus'
# DAQ_BASE_DIR = "/home/mx17/feb_beam/dream_run"
# LOCAL_BASE_DIR = "/media/dylan/data/x17/feb_beam/dream_run"
DAQ_BASE_DIR = "/eos/experiment/ntof/data/x17/feb_beam/runs"
LOCAL_BASE_DIR = "/media/dylan/data/x17/feb_beam/runs"

# rsync options:
# -a : archive mode preserves permissions and timestamps
# -v : verbose
# -u : skip files newer on receiver
# --progress : display file transfer progress
RSYNC_OPTS = "-avu --progress"
FDF_ONLY = False
EXCLUDE_FDF = True


def main():
    print("Checking runs on DAQ...")
    # runs = get_pi_runs()
    runs = get_daq_runs()
    print(f'DAQ runs found: {runs}')

    # min_run = 140  # only sync runs >= this number
    # max_run = 140
    # runs = [run for run in runs if run.startswith("run_") and max_run >= int(run.split("_")[1]) >= min_run]
    # print(f'Filtering to runs >= {min_run}: {runs}')

    # runs = [17, 18, 19, 25, 26, 27, 29, 30, 31, 32, 33, 35, 38, 39, 49, 50, 52, 55, 57, 59, 63, 64, 67, 71, 74, 76, 80,
    #         84, 85, 86, 88, 94, 97, 107, 114, 118, 126, 128, 130, 131, 136, 138, 139, 141, 142, 143]
    runs = [98]
    runs = [f'run_{run}' for run in runs]

    for run in runs:
        sync_run(run, FDF_ONLY, EXCLUDE_FDF)

    print("Sync complete.")
    print('donzo')


def get_daq_runs():
    """Return a list of run directories on the Raspberry Pi."""
    cmd = [
        "ssh",
        f"{HOST}",
        f"ls -1 {DAQ_BASE_DIR}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    runs = result.stdout.split()
    return runs


# def get_pi_runs():
#     """Return a list of run directories on the Raspberry Pi."""
#     cmd = [
#         "ssh",
#         "-J", f"{JUMP_HOST}",
#         f"{PI_HOST}",
#         f"ls -1 {PI_BASE_DIR}"
#     ]
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     runs = result.stdout.split()
#     return runs


def ensure_local_dir(path):
    """Create local directory if it does not exist."""
    if not os.path.exists(path):
        print(f"Creating local directory: {path}")
        os.makedirs(path, exist_ok=True)


def sync_run(run_name, fdf_only=True, exclude_fdf=False):
    """Sync .fdf files for a single run using rsync."""
    # pi_dir = f"{PI_HOST}:{PI_BASE_DIR}/{run_name}/"
    daq_dir = f"{HOST}:{DAQ_BASE_DIR}/{run_name}/"
    local_dir = f"{LOCAL_BASE_DIR}/{run_name}/"

    ensure_local_dir(local_dir)

    rsync_patterns = [
        "--include=*/",        # keep directory structure
        "--prune-empty-dirs",
    ]
    if fdf_only:
        rsync_patterns.append("--include=*.fdf") # include only .fdf files
        rsync_patterns.append("--exclude=*")  # exclude everything else
    elif exclude_fdf:
        rsync_patterns.append("--exclude=*.fdf")

    cmd = [
        "rsync",
        *RSYNC_OPTS.split(),
        *rsync_patterns,
        # "-e", f"ssh -J {JUMP_HOST}",
        "-e", f"ssh",
        daq_dir,
        local_dir,
    ]

    print(f"Syncing {run_name}...")
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
