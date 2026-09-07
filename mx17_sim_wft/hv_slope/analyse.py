#!/usr/bin/env python3
"""
HV-slope test — does the det3 amplitude-vs-mesh-voltage slope match the
Garfield gain-vs-voltage slope?

Problem 1 of the T14 follow-up. T14 left the simulation's peak amplitude at
x0.55-0.63 of data with an angle-independent floor, and the question is whether
that floor is ONE constant (a Garfield absolute-gain miscalibration, which a
single fitted gain factor could legitimately absorb) or a wrong response to
field (in which case a fitted factor hides broken physics).

The discriminator is the slope. Gain is the only thing the mesh voltage
changes, so d ln A / dV measured on the bench and d ln G / dV from the
avalanche calibration have to agree if the only defect is normalization.

Inputs
------
data : peaks.parquet / peaks_drift.parquet from extract.py (threshold-free
       peak-strip waveform maxima on a voltage-independent, M3-selected,
       100 %-efficient event population)
sim  : MX17_Geant response/avalanche/aval_calib_meshfield_hvscan.json
       (per-voltage T6 field maps, dry Ar/iso 95/5, 150 um gap)
       and aval_calib_diagnosis_grid.json for the wet mini-ladder

    python3 mx17_sim_wft/hv_slope/analyse.py --in-dir ~/x17/response_sim/hv_slope
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

# ── Fiducial region, chosen ONCE from the >=505 V hit map and applied at every
# voltage. Reference-frame mm at the det3 plane (z = 702). One 25 mm bin inset
# from the efficiency plateau edge.
FID_X = (-190.0, 115.0)
FID_Y = (-190.0, 165.0)

# An estimator is used at a voltage only while its quantile sits below this
# fraction of the ADC rail. Declared before looking at the slopes.
SAT_FRAC = 0.70

SIM_DIR = os.path.expanduser("~/CLionProjects/MX17_Geant/response/avalanche")
SIM_HVSCAN = "aval_calib_meshfield_hvscan.json"
SIM_GRID = "aval_calib_diagnosis_grid.json"
DRY = "Ar_iC4H10_95_5_Saclay_160m.gas"
WET = "Ar_iC4H10_H2O_94_5_1_Saclay_160m.gas"
ISO10 = "Ar_iC4H10_90_10_Saclay_160m.gas"

# Head-to-head window: the widest voltage span covered by BOTH ladders in which
# the data median is unsaturated.
HEAD_WINDOW = (460, 490)

ESTIMATORS = [("p02", 0.02), ("p10", 0.10), ("p25", 0.25), ("p50", 0.50)]
NBOOT = 400
RNG = np.random.default_rng(20260809)


def fiducial(df):
    return df[(df.ref_x > FID_X[0]) & (df.ref_x < FID_X[1]) &
              (df.ref_y > FID_Y[0]) & (df.ref_y < FID_Y[1])]


def quantile_table(df, rail):
    """Per (view, voltage) quantiles with bootstrap errors on ln(quantile)."""
    rows = []
    for (view, volt), g in df.groupby(["view", "volt"]):
        a = g.peak_amp.values
        nb = g.nb_amp.values
        row = dict(view=view, volt=int(volt), scan=g.scan.iloc[0], n=len(a),
                   eff=float((g.n_over > 0).mean()),
                   fsat=float((a > 0.88 * rail).mean()),
                   nb_p50=float(np.median(nb)))
        for name, q in ESTIMATORS:
            v = float(np.quantile(a, q))
            bs = np.quantile(a[RNG.integers(0, len(a), (NBOOT, len(a)))], q, axis=1)
            row[name] = v
            row[f"{name}_lnerr"] = float(np.std(np.log(bs)))
            row[f"{name}_ok"] = bool(v < SAT_FRAC * rail)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["view", "volt"])


def loglin(v, lna, err):
    """Weighted straight-line fit of ln A on V. Returns slope per 10 V."""
    w = 1.0 / np.maximum(err, 1e-6) ** 2
    S, Sx = w.sum(), (w * v).sum()
    Sxx, Sy, Sxy = (w * v * v).sum(), (w * lna).sum(), (w * v * lna).sum()
    d = S * Sxx - Sx * Sx
    slope = (S * Sxy - Sx * Sy) / d
    inter = (Sxx * Sy - Sx * Sxy) / d
    return slope * 10.0, np.sqrt(S / d) * 10.0, inter


def sim_ladder(path, gas):
    pts = json.load(open(path))["points"]
    v, g, e = [], [], []
    for k, p in pts.items():
        if not k.startswith(gas + "@"):
            continue
        v.append(float(p["voltage_V"]))
        pol = p["polya"]
        g.append(pol["gain_mean"])
        # Polya MC error on the mean: sqrt(rel_var / n)
        e.append(np.sqrt(pol["rel_var"] / pol["n"]))
    o = np.argsort(v)
    return np.array(v)[o], np.array(g)[o], np.array(e)[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.expanduser("~/x17/response_sim/hv_slope"))
    ap.add_argument("--sim-dir", default=SIM_DIR)
    a = ap.parse_args()

    mesh = fiducial(pd.read_parquet(os.path.join(a.in_dir, "peaks.parquet")))
    drift = fiducial(pd.read_parquet(os.path.join(a.in_dir, "peaks_drift.parquet")))

    # The effective ADC rail: at 525 V >99 % of events clip, so the median peak
    # amplitude there IS the rail (averaged over the per-channel pedestal).
    rail = float(np.median(mesh[mesh.volt == 525].peak_amp))

    tab = quantile_table(mesh, rail)
    dtab = quantile_table(drift, rail)

    out = dict(rail=rail, fiducial=dict(x=FID_X, y=FID_Y), sat_frac=SAT_FRAC,
               head_window=HEAD_WINDOW, n_boot=NBOOT)

    # ── data slopes ──────────────────────────────────────────────────────────
    out["data"] = {}
    for view in ("x", "y"):
        t = tab[tab.view == view]
        per = {}
        for name, _q in ESTIMATORS:
            for tag, win in (("head", HEAD_WINDOW), ("full", (0, 10_000))):
                m = t[f"{name}_ok"] & (t.volt >= win[0]) & (t.volt <= win[1])
                if m.sum() < 3:
                    continue
                s, se, _ = loglin(t.volt[m].values, np.log(t[name][m].values),
                                  t[f"{name}_lnerr"][m].values)
                per[f"{name}_{tag}"] = dict(
                    slope10=s, err10=se, n=int(m.sum()),
                    vmin=int(t.volt[m].min()), vmax=int(t.volt[m].max()))
        # neighbour-strip sum: the saturation-robust cross-check (neighbours sit
        # at ~1/3 of the peak, so they clip ~3x later)
        m = (t.nb_p50 < SAT_FRAC * 2 * rail) & (t.volt >= HEAD_WINDOW[0]) & \
            (t.volt <= HEAD_WINDOW[1])
        s, se, _ = loglin(t.volt[m].values, np.log(t.nb_p50[m].values),
                          np.full(m.sum(), 0.03))
        per["nb_head"] = dict(slope10=s, err10=se, n=int(m.sum()),
                              vmin=int(t.volt[m].min()), vmax=int(t.volt[m].max()))
        out["data"][view] = per

    # ── sim slopes ───────────────────────────────────────────────────────────
    hv = os.path.join(a.sim_dir, SIM_HVSCAN)
    grid = os.path.join(a.sim_dir, SIM_GRID)
    out["sim"] = {}
    for tag, path, gas, win in (
            ("dry_head", hv, DRY, HEAD_WINDOW),
            ("dry_full", hv, DRY, (0, 10_000)),
            ("iso10_full", hv, ISO10, (0, 10_000)),
            ("wet1pct", grid, WET, (0, 10_000))):
        v, g, e = sim_ladder(path, gas)
        m = (v >= win[0]) & (v <= win[1])
        s, se, _ = loglin(v[m], np.log(g[m]), e[m])
        out["sim"][tag] = dict(slope10=s, err10=se, n=int(m.sum()),
                               vmin=float(v[m].min()), vmax=float(v[m].max()),
                               gas=gas)

    # local (finite-difference) slope of the dry sim ladder, for "no voltage
    # offset can rescue this" — the max over the whole simulated range
    v, g, _e = sim_ladder(hv, DRY)
    loc = np.diff(np.log(g)) / np.diff(v) * 10.0
    out["sim"]["dry_local_slope10"] = dict(
        v_mid=((v[1:] + v[:-1]) / 2).tolist(), slope10=loc.tolist(),
        min=float(loc.min()), max=float(loc.max()))

    # ── drift-field control ──────────────────────────────────────────────────
    dt = dtab[dtab.view == "x"].set_index("volt")
    lo, hi = 900, 1100
    out["drift_control"] = dict(
        p50={int(k): float(vv) for k, vv in dt.p50.items()},
        dln_per_10V_at_1000=float((np.log(dt.p50[hi]) - np.log(dt.p50[lo]))
                                  / (hi - lo) * 10.0),
        signal_at_100V=bool(dt.eff[100] > 0.95))

    # ── shape invariance: is the spectrum a pure rescaling? ──────────────────
    t = tab[tab.view == "x"]
    m = t.p50_ok & (t.volt >= 425)
    r1, r2 = (t.p50[m] / t.p10[m]).values, (t.p25[m] / t.p02[m]).values
    out["shape"] = dict(
        volt=t.volt[m].tolist(),
        p50_over_p10=r1.tolist(), p50_over_p10_range=[r1.min(), r1.max()],
        p25_over_p02=r2.tolist(), p25_over_p02_range=[r2.min(), r2.max()],
        n_range=[int(t.n[m].min()), int(t.n[m].max())])

    tab.to_csv(os.path.join(a.in_dir, "mesh_ladder.csv"), index=False)
    dtab.to_csv(os.path.join(a.in_dir, "drift_ladder.csv"), index=False)
    with open(os.path.join(a.in_dir, "slopes.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ── console summary ──────────────────────────────────────────────────────
    print(f"rail = {rail:.0f} ADC   fiducial x{FID_X} y{FID_Y}")
    print(f"\nDATA slope [dln/10V] over {HEAD_WINDOW[0]}-{HEAD_WINDOW[1]} V")
    for view in ("x", "y"):
        for k, r in out["data"][view].items():
            if k.endswith("head"):
                print(f"  {view} {k:10s} {r['slope10']:.4f} +- {r['err10']:.4f}"
                      f"  ({r['n']} pts {r['vmin']}-{r['vmax']} V)")
    print("\nSIM slope [dln/10V]")
    for k, r in out["sim"].items():
        if isinstance(r, dict) and isinstance(r.get("slope10"), float):
            print(f"  {k:12s} {r['slope10']:.4f} +- {r['err10']:.4f}"
                  f"  ({r['n']} pts {r['vmin']:.0f}-{r['vmax']:.0f} V)")
    d = out["data"]["x"]["p50_head"]["slope10"]
    s = out["sim"]["dry_head"]["slope10"]
    print(f"\nratio data/sim = {d/s:.2f}"
          f"   gain doubles every {np.log(2)/d*10:.1f} V (data)"
          f" vs {np.log(2)/s*10:.1f} V (sim)")
    print("wrote mesh_ladder.csv, drift_ladder.csv, slopes.json")


if __name__ == "__main__":
    main()
