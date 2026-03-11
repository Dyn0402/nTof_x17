#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 10 11:38 AM 2026
Created in PyCharm
Created as nTof_x17/run_processor.py

@author: Dylan Neff, dylan
"""

import os
import subprocess
import re
from typing import Optional, Tuple


def main():
    run_dir = '/eos/experiment/ntof/data/x17/feb_beam/runs/'
    raw_dir_name = 'raw_daq_data'
    combined_dir_name = 'combined_hits_root'

    # runs_to_process = get_runs_to_process(run_dir, raw_dir_name, combined_dir_name)
    runs_to_process = get_all_runs(run_dir)
    print(f'Runs to process: {runs_to_process}')
    print(f'Number of runs to process: {len(runs_to_process)}')

    run_list_path = '/eos/experiment/ntof/data/x17/feb_beam/runs_to_process.txt'

    runs_to_process = ['run_88']

    write_run_list_to_file(runs_to_process, run_list_path)

    processor_script_dir = '/afs/cern.ch/work/d/dneff/git/mm_strip_reconstruction/orchestrator/'
    processor_script_name = 'process_run.py'
    log_dir = '/afs/cern.ch/work/d/dneff/condor/run_processor/logs'

    submit_condor_jobs(run_list_path, processor_script_dir, processor_script_name, log_dir)

    print('donzo')


def submit_condor_jobs(
    runs_file: str,
    processor_script_dir: str = "process_run.py",
    processor_script_name: str = "process_run.py",
    log_dir: str = "logs",
    extra_jdl_args: dict = None,
):
    """
    Read run numbers from a file and submit one HTCondor job per run.

    Args:
        runs_file:             Path to a text file with one run name per line
                               (e.g. "run_2", "run_5", …).
        processor_script_dir:  Path to the directory containing the processor script.
        processor_script_name: Name of the Python script to execute.
        log_dir:               Directory where Condor log/out/err files are written.
        extra_jdl_args:        Optional dict of extra JDL key-value pairs to add
                               (e.g. {"request_memory": "2 GB", "request_cpus": "2"}).
    """

    # ------------------------------------------------------------------ #
    # 1. Read run list                                                     #
    # ------------------------------------------------------------------ #
    with open(runs_file) as f:
        runs = [line.strip() for line in f if line.strip()]

    if not runs:
        raise ValueError(f"No runs found in {runs_file}")

    print(f"Found {len(runs)} run(s): {runs}")

    # ------------------------------------------------------------------ #
    # 2. Prepare directories                                               #
    # ------------------------------------------------------------------ #
    os.makedirs(log_dir, exist_ok=True)

    processor_script_path = os.path.join(processor_script_dir, processor_script_name)

    processor_abs = os.path.abspath(processor_script_path)
    if not os.path.isfile(processor_abs):
        raise FileNotFoundError(f"Processor script not found: {processor_abs}")

    # ------------------------------------------------------------------ #
    # 3. Build a single JDL file with one Queue entry per run             #
    # ------------------------------------------------------------------ #
    jdl_path = "submit_jobs.jdl"

    extra = extra_jdl_args or {}

    jdl_lines = [
        # --- Environment ---
        "universe   = vanilla",
        f"executable = /usr/bin/python3",
        "",
        # --- Output / logging ($(run) is substituted per queue item) ---
        f"output     = {log_dir}/$(run).out",
        f"error      = {log_dir}/$(run).err",
        f"log        = {log_dir}/condor.log",
        "",
        # --- Ship the processor script to the worker node ---
        f"transfer_input_files  = {processor_abs}",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "",
        # --- Pass the run name as an argument ---
        f"arguments  = {processor_script_name} $(run)",
        "",
        # --- Resource requests (sensible defaults, override via extra_jdl_args) ---
        f'request_memory = {extra.pop("request_memory", "1 GB")}',
        f'request_cpus   = {extra.pop("request_cpus", "1")}',
        f'+JobFlavour    = "{extra.pop("+JobFlavour", "espresso")}"',
        "",
    ]

    # Any remaining extra args
    for key, value in extra.items():
        jdl_lines.append(f"{key} = {value}")

    # One queue entry per run
    jdl_lines.append("# --- Per-run queue entries ---")
    for run in runs:
        jdl_lines.append(f"run = {run}")
        jdl_lines.append("Queue")
        jdl_lines.append("")

    jdl_content = "\n".join(jdl_lines)

    with open(jdl_path, "w") as f:
        f.write(jdl_content)

    print(f"\nGenerated JDL file: {jdl_path}")
    print("-" * 40)
    print(jdl_content)
    print("-" * 40)

    # ------------------------------------------------------------------ #
    # 4. Submit                                                            #
    # ------------------------------------------------------------------ #
    result = subprocess.run(
        ["condor_submit", jdl_path],
        capture_output=True,
        text=True,
    )

    print("\n[condor_submit stdout]")
    print(result.stdout)

    if result.returncode != 0:
        print("[condor_submit stderr]")
        print(result.stderr)
        raise RuntimeError(f"condor_submit failed with return code {result.returncode}")

    print("Jobs submitted successfully.")
    return result.stdout



def get_all_runs(run_dir):
    """
    Get full list of runs.
    """
    runs_to_process = []
    for run in os.listdir(run_dir):
        run_dir_path = os.path.join(run_dir, run)
        if not os.path.isdir(run_dir_path):
            continue
        runs_to_process.append(run)

    return runs_to_process


def get_runs_to_process(run_dir, raw_dir_name, combined_dir_name):
    """
    Get list of runs to process.
    """

    runs_to_process = []
    for run in os.listdir(run_dir):
        run_dir_path = os.path.join(run_dir, run)
        for subrun in run_dir_path:
            subrun_dir_path = os.path.join(run_dir_path, subrun)
            if not os.path.isdir(subrun_dir_path):
                continue
            subrun_combined_dir_path = os.path.join(subrun_dir_path, combined_dir_name)
            if not os.path.isdir(subrun_combined_dir_path):
                runs_to_process.append(run)  # Combined dir doesn't exist for this subrun, process this run
                break

            # Check if combined_dir has all expected files from raw_dir
            raw_files_dir = os.path.join(subrun_dir_path, raw_dir_name)
            raw_file_nums = get_raw_file_nums(raw_files_dir)
            combined_file_nums = get_combined_file_nums(subrun_combined_dir_path)
            if raw_file_nums != combined_file_nums:  # If some combined files are missing, process this run
                runs_to_process.append(run)
                break

    return runs_to_process


def get_raw_file_nums(raw_files_dir):
    """
    Get file numbers from raw dir. Files look like this Mx17_resist_560V_drift_600V_datrun_260209_22H56_000_04.fdf
    Extract _000_ at end
    """
    file_nums = []
    for file in os.listdir(raw_files_dir):
        if '_datrun_' not in file or not file.endswith('.fdf'):
            continue
        file_num, feu_num = extract_file_numbers_tuple(file)
        file_nums.append(file_num)
    file_nums = list(set(file_nums))
    file_nums.sort()
    return file_nums


def get_combined_file_nums(combined_files_dir):
    """
    Get file numbers from combined dir.
    """
    file_nums = []
    for file in os.listdir(combined_files_dir):
        if '_feu-combined' not in file or not file.endswith('.root'):
            continue
        file_num = extract_combined_file_num(file)
        file_nums.append(file_num)
    file_nums = list(set(file_nums))
    file_nums.sort()
    return file_nums


def extract_file_numbers_tuple(filename: str, end_dot=True) -> Optional[Tuple[int, int]]:
    if end_dot:
        match = re.match(r'.*_(\d{3})_(\d{2})\..*', filename)
    else:
        match = re.match(r'.*_(\d{3})_(\d{2}).*', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def extract_combined_file_num(filename: str, end_dot=True) -> Optional[int]:
    match = re.match(r'.*_(\d{3})_feu-combined*', filename)
    if match:
        return int(match.group(1))
    return None


def write_run_list_to_file(run_list, file_path):
    with open(file_path, 'w') as f:
        for run in run_list:
            f.write(f'{run}\n')


if __name__ == '__main__':
    main()
