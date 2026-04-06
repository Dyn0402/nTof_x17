#!/usr/bin/env python3
"""
mm_sim_core.py — Shared physics core for Micromegas gain simulation
====================================================================
Provides the Garfield++ avalanche simulation, statistics helpers, and
result I/O functions that are shared between:

  mm_gain_scan_parallel.py  — local parallel scan (multiprocessing pool)
  mm_condor_worker.py       — single HTCondor job worker
  mm_condor_collect.py      — HTCondor fragment merger

ROOT/Garfield++ are imported INSIDE run_avalanche_batch() so this module is
safe to import in the main process without loading ROOT.  Spawned worker
processes call run_avalanche_batch() and get their own clean ROOT instance.
"""

import os
import json
import time

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Physics — single-voltage avalanche batch
# ──────────────────────────────────────────────────────────────────────────────

def run_avalanche_batch(gas_file, penning_cfg, voltage, n_events,
                        gap_cm=0.015, pressure_torr=None, temp_k=None,
                        progress_interval=0):
    """
    Run n_events single-electron avalanches at one mesh voltage.

    Imports ROOT/Garfield++ inside the function — safe to call from spawned
    multiprocessing workers (each spawned process gets its own ROOT state).

    Parameters
    ----------
    gas_file          : path to pre-generated .gas file
    penning_cfg       : {"mode": "auto"}  or
                        {"mode": "manual", "rP": float, "gas": str}
    voltage           : mesh voltage in V
    n_events          : number of avalanches to simulate
    gap_cm            : amplification gap in cm (default 0.015 = 150 µm)
    pressure_torr     : override pressure; None = use value in .gas file
    temp_k            : override temperature; None = use value in .gas file
    progress_interval : print a progress line every N events (0 = silent)

    Returns
    -------
    gains       : list[int]  — electron count for each surviving avalanche
    n_attached  : int        — events where all electrons attached (zero gain)
    wall_time_s : float      — elapsed wall time in seconds
    """
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    import Garfield
    import ctypes

    gas = ROOT.Garfield.MediumMagboltz()
    gas.LoadGasFile(gas_file)
    if temp_k is not None:
        gas.SetTemperature(temp_k)
    if pressure_torr is not None:
        gas.SetPressure(pressure_torr)

    # Penning transfer must be applied AFTER LoadGasFile
    if penning_cfg["mode"] == "auto":
        gas.EnablePenningTransfer()
    else:
        gas.EnablePenningTransfer(penning_cfg["rP"], 0., penning_cfg["gas"])

    e_field = voltage / gap_cm

    cmp = ROOT.Garfield.ComponentConstant()
    cmp.SetArea(-1., -1., 0., 1., 1., gap_cm)
    cmp.SetElectricField(0., 0., e_field)
    cmp.SetMedium(gas)

    sensor = ROOT.Garfield.Sensor()
    sensor.AddComponent(cmp)
    sensor.SetArea(-1., -1., 0., 1., 1., gap_cm)

    aval = ROOT.Garfield.AvalancheMicroscopic()
    aval.SetSensor(sensor)

    gains = []
    n_attached = 0
    ne_out = ctypes.c_int(0)
    ni_out = ctypes.c_int(0)

    t0 = time.time()
    for i in range(n_events):
        # Seed electron at z=gap_cm (mesh), drifts toward z=0 (anode)
        aval.AvalancheElectron(0., 0., gap_cm, 0., 0.)
        aval.GetAvalancheSize(ne_out, ni_out)
        ne = int(ne_out.value)
        if ne == 0:
            n_attached += 1
        else:
            gains.append(ne)

        if progress_interval > 0 and (i + 1) % progress_interval == 0:
            mean_str = f"  mean_gain={np.mean(gains):.0f}" if gains else ""
            print(f"  [{i+1}/{n_events}] events done{mean_str}", flush=True)

    wall = time.time() - t0
    return gains, n_attached, wall


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def recompute_stats(gains, n_attached):
    """
    Compute summary statistics from a raw gain list.

    Parameters
    ----------
    gains      : list of ints — per-event electron counts (attached excluded)
    n_attached : int — events where gain was zero

    Returns
    -------
    (mean_g, med_g, std_g, rms_rel, surv)
    """
    n_total = len(gains) + n_attached
    if gains:
        arr     = np.array(gains, dtype=float)
        mean_g  = float(np.mean(arr))
        med_g   = float(np.median(arr))
        std_g   = float(np.std(arr))
        rms_rel = std_g / mean_g if mean_g > 0 else float("nan")
        surv    = len(gains) / n_total if n_total > 0 else 0.
    else:
        mean_g = med_g = std_g = 0.
        rms_rel = float("nan")
        surv    = 0.
    return mean_g, med_g, std_g, rms_rel, surv


# ──────────────────────────────────────────────────────────────────────────────
# Result I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_existing(results_dir, gas_label, pressure_label):
    """
    Load a previously saved result JSON for append-mode accumulation.

    Returns
    -------
    dict keyed by float voltage:
      {voltage: {"gain_raw": list, "n_attached": int, "runtime_s": float}}
    Returns {} if no file exists or the file is unreadable.
    """
    path = os.path.join(results_dir, f"{gas_label}_{pressure_label}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            existing = json.load(f)
        voltages = existing.get("voltages", [])
        runtimes = existing.get("runtime_s", [0.] * len(voltages))
        prior = {}
        for i, v in enumerate(voltages):
            prior[float(v)] = {
                "gain_raw":   list(existing["gain_raw"][i]),
                "n_attached": int(existing["n_attached"][i]),
                "runtime_s":  float(runtimes[i]),
            }
        return prior
    except Exception as e:
        print(f"[sim_core] WARN: could not load {path}: {e}")
        return {}


def build_result_dict(gas_label, pressure_label, pressure_torr, gap_cm, temp_k,
                      penning, n_events, volt_data,
                      total_runtime=0., partial=False):
    """
    Assemble the standard result JSON schema from per-voltage data.

    Parameters
    ----------
    n_events  : int — events per voltage point (for metadata; stored as-is)
    volt_data : list of dicts, each with:
                  {"voltage": float, "gain_raw": list,
                   "n_attached": int, "runtime_s": float}
                (field is computed from voltage/gap_cm if the key is absent)

    Returns
    -------
    dict matching the JSON schema documented in CLAUDE_CODE_BRIEFING.md
    """
    result = {
        "gas":             gas_label,
        "pressure_label":  pressure_label,
        "pressure_torr":   pressure_torr,
        "gap_cm":          gap_cm,
        "temp_k":          temp_k,
        "n_events":        n_events,
        "penning":         penning,
        "voltages":        [],
        "fields":          [],
        "gain_mean":       [],
        "gain_median":     [],
        "gain_std":        [],
        "gain_rms_rel":    [],
        "gain_raw":        [],
        "survival":        [],
        "n_attached":      [],
        "runtime_s":       [],
        "total_runtime_s": total_runtime,
        "partial":         partial,
    }

    for vd in sorted(volt_data, key=lambda x: x["voltage"]):
        gains = vd["gain_raw"]
        n_att = vd["n_attached"]
        e_field = vd.get("field", vd["voltage"] / gap_cm)

        mean_g, med_g, std_g, rms_rel, surv = recompute_stats(gains, n_att)

        result["voltages"].append(float(vd["voltage"]))
        result["fields"].append(float(e_field))
        result["gain_mean"].append(mean_g)
        result["gain_median"].append(med_g)
        result["gain_std"].append(std_g)
        result["gain_rms_rel"].append(rms_rel)
        result["gain_raw"].append(list(gains))
        result["survival"].append(surv)
        result["n_attached"].append(n_att)
        result["runtime_s"].append(float(vd["runtime_s"]))

    return result


def save_result(result, results_dir):
    """
    Write a result dict to the standard JSON path inside results_dir.

    Returns the output path.
    """
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(
        results_dir,
        f"{result['gas']}_{result['pressure_label']}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path
