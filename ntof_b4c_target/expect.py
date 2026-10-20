#!/usr/bin/env python3
"""expect.py -- what the July-2026 B4C target should have given, from Geant4.

    python -m ntof_b4c_target.expect          # -> OUT/expect.json

Reads the reduced per-event arrays (b4c_reduce.py on lxplus, copied to
SIM_PARTS) and turns them into rates per nominal n_TOF pulse (7e12 protons)
and totals for the run_39 + run_40 exposure.  Three sources, two disc radii:

  neutrons  analog EAR2 neutrons, 2.2 meV - 2 eV (TOF 1-30 ms at 19.5 m), with
            the events where the slab made a ¹⁰B(n,γ) capture removed (Geant4
            samples those lines independently, with no energy conservation)
  casc      the measured EGAF ¹⁰B(n,γ)¹¹B thermal cascade, from the slab's
            capture vertices, scaled to captures/pulse
  ipc       Born internal pairs of the seven strongest ¹¹B lines
            (sept26_prelim_analysis.ipc_born), scaled to the same captures

Thresholds follow production: SiPM bar >= 0.5 MIP, plastic >= 0.9 MIP, MIP from
the 2 cm calibration run (CAMPAIGN_STATUS 2026-07-26).  A Micromegas is "lit"
when charged particles leave >= MM_KEV in its drift gas.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

SIM_PARTS = Path("/media/dylan/data/x17/b4c_target/sim_parts")
OUT = Path("/media/dylan/data/x17/b4c_target")

MIP_SIPM, MIP_PLASTIC = 0.475, 3.349         # MeV, analysis/mip_2cm
A_SIPM, B_PLASTIC = 0.5, 0.9                  # production thresholds [MIP]
MM_KEV = 3.0                                  # drift-gas charge to call a chamber lit
N_PULSE = 5.327e6                             # Phase-4 flux, 2.2 meV-2 eV, per 7e12 protons
F_ABS = 0.984                                 # beam neutrons absorbed in the slab (100k-neutron run)
F_NG = 0.29 / (0.29 + 3837.0)                 # ¹⁰B (n,γ) / absorption, thermal (both 1/v)
CAPT_PER_PULSE = N_PULSE * F_ABS * F_NG       # ¹⁰B(n,γ) cascades per nominal pulse
EXPOSURE_PULSES = 2.175e16 / 7e12             # run_39 + run_40 protons / nominal pulse
N_PULSES_REAL = 2635                          # n_TOF pulses during run_39 + run_40

# EGAF line intensity per 100 captures and Born IPC coefficient (ipc_born.alpha_pair)
IPC_LINES = {"7.007_E1": (54.2, 2.952e-3), "4.711_E1": (25.7, 2.330e-3),
             "2.533_E1": (13.8, 1.341e-3), "11.448_E1": (4.64, 3.716e-3),
             "4.444_M1": (63.3, 1.346e-3), "8.917_M1": (13.0, 2.378e-3),
             "6.740_M1": (17.4, 2.077e-3)}   # 6.740 is E2; kinematics run as M1
OPPOSITE = {(0, 1), (2, 3)}                   # arm ids: 0 +X, 1 -X, 2 +Z, 3 -Z


def load(pattern):
    acc = {}; ngen = 0
    for fp in sorted(glob.glob(str(SIM_PARTS / pattern))):
        d = np.load(fp)
        ngen += int(d["ngen"])
        for k in d.files:
            if d[k].ndim:
                acc.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in acc.items()}, ngen


def observables(d, a=A_SIPM, b=B_PLASTIC):
    """Per-event booleans for the trigger / chamber classes."""
    S = d["S"] / MIP_SIPM; P = d["P"] / MIP_PLASTIC; mm = d["MM"] >= MM_KEV
    leg = (S >= a) & (P >= b)
    nleg = leg.sum(1); nmm = mm.sum(1)
    legmm = leg & mm
    pair_arms = d["EP"][:, :, None] & d["EM"][:, None, :] & ~np.eye(4, dtype=bool)[None]
    split_pair = pair_arms.any((1, 2))
    # opposite vs adjacent among two-leg events (first two leg arms)
    opp = np.zeros(len(S), bool)
    two = nleg >= 2
    for i in np.where(two)[0]:
        arms = tuple(np.where(leg[i])[0][:2]); opp[i] = arms in OPPOSITE
    return {
        "sipm_any": (S >= a).any(1),
        "leg1": nleg >= 1,
        "leg2": nleg >= 2,
        "leg2_opposite": two & opp,
        "leg2_adjacent": two & ~opp,
        "mm1": nmm >= 1,
        "mm2": nmm >= 2,
        "leg1_mm2": (nleg >= 1) & (nmm >= 2),
        "leg2_mm2legs": legmm.sum(1) >= 2,
        "pair_split": split_pair,
        "pair_split_leg2mm": split_pair & (legmm.sum(1) >= 2),
    }


def rates(d, ngen, per_event, a=A_SIPM, b=B_PLASTIC, keep=None):
    """Rate per nominal pulse of each class, with the MC statistical error."""
    w = d["w"] * per_event
    obs = observables(d, a, b)
    out = {}
    for k, m in obs.items():
        if keep is not None:
            m = m & keep
        out[k] = (float(w[m].sum()), float(np.sqrt((w[m] ** 2).sum())), int(m.sum()))
    return out


def source_rates(r):
    res = {}
    dn, ngn = load(f"neutrons_r{r}.npz")
    res["neutrons"] = rates(dn, ngn, N_PULSE / ngn, keep=~dn["ng"].astype(bool))
    res["neutrons"]["_n_gen"] = ngn
    dc, ngc = load(f"casc_r{r}__*.npz")
    res["casc"] = rates(dc, ngc, CAPT_PER_PULSE / ngc)
    res["casc"]["_n_gen"] = ngc
    ipc = {}
    for line, (inten, alpha) in IPC_LINES.items():
        e, k = line.split("_")
        di, ngi = load(f"ipc_r{r}__ipc_r{r}_{float(e):.3f}_{k}__*.npz")
        ipc[line] = rates(di, ngi, CAPT_PER_PULSE * inten / 100 * alpha / ngi)
        ipc[line]["_n_gen"] = ngi
    tot = {}
    for line in ipc.values():
        for key, v in line.items():
            if key.startswith("_"):
                continue
            t = tot.setdefault(key, [0.0, 0.0, 0])
            t[0] += v[0]; t[1] = float(np.hypot(t[1], v[1])); t[2] += v[2]
    res["ipc"] = {k: tuple(v) for k, v in tot.items()}
    res["ipc_lines"] = ipc
    return res


def threshold_scan(r, b_grid=np.round(np.arange(0.1, 2.01, 0.1), 2)):
    """Two-leg rate per pulse vs plastic threshold, SiPM at 0.5 MIP, per source."""
    dn, ngn = load(f"neutrons_r{r}.npz"); keep = ~dn["ng"].astype(bool)
    dc, ngc = load(f"casc_r{r}__*.npz")
    out = {"b": b_grid.tolist(), "neutrons": [], "casc": [], "ipc": []}
    ipcd = []
    for line, (inten, alpha) in IPC_LINES.items():
        e, k = line.split("_")
        di, ngi = load(f"ipc_r{r}__ipc_r{r}_{float(e):.3f}_{k}__*.npz")
        ipcd.append((di, CAPT_PER_PULSE * inten / 100 * alpha / ngi))
    for b in b_grid:
        out["neutrons"].append(rates(dn, ngn, N_PULSE / ngn, b=b, keep=keep)["leg2"][0])
        out["casc"].append(rates(dc, ngc, CAPT_PER_PULSE / ngc, b=b)["leg2"][0])
        out["ipc"].append(sum(rates(di, 1.0, pe, b=b)["leg2"][0] for di, pe in ipcd))
    return out


def main():
    res = {"constants": dict(N_PULSE=N_PULSE, F_ABS=F_ABS, F_NG=F_NG,
                             CAPT_PER_PULSE=CAPT_PER_PULSE, EXPOSURE_PULSES=EXPOSURE_PULSES,
                             N_PULSES_REAL=N_PULSES_REAL, MIP_SIPM=MIP_SIPM,
                             MIP_PLASTIC=MIP_PLASTIC, A_SIPM=A_SIPM, B_PLASTIC=B_PLASTIC,
                             MM_KEV=MM_KEV, IPC_LINES=IPC_LINES)}
    for r in (50, 25):
        res[f"r{r}"] = source_rates(r)
        res[f"scan_r{r}"] = threshold_scan(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "expect.json").write_text(json.dumps(res, indent=1))
    for r in (50, 25):
        print(f"--- disc radius {r} mm: per nominal pulse (MC events)")
        for src in ("neutrons", "casc", "ipc"):
            v = res[f"r{r}"][src]
            print(f"  {src:9s}", "  ".join(f"{k}={v[k][0]:.3g}({v[k][2]})" for k in
                  ("sipm_any", "leg1", "leg2", "mm2", "leg1_mm2", "leg2_mm2legs", "pair_split_leg2mm")))


if __name__ == "__main__":
    main()
