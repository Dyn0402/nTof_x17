#!/usr/bin/env python3
"""
Figures + report.html for the HV-slope test (Problem 1 of the T14 follow-up).

Run analyse.py first; this reads its CSV/JSON products, adds the HV-current
check, and writes everything into one directory so the DAQ Analysis tab can
serve it.

    python3 mx17_sim_wft/hv_slope/make_report.py --dir ~/x17/response_sim/hv_slope
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse import (DRY, ESTIMATORS, HEAD_WINDOW, ISO10, SIM_DIR, SIM_GRID,
                     SIM_HVSCAN, WET, sim_ladder)

# Okabe-Ito subset, validated for this use (dataviz validator, light surface):
# lightness band PASS, chroma PASS, normal-vision floor PASS. Only two series
# carry identity by hue (data / sim) and both also carry a distinct marker and
# a direct end-label, so identity never rests on colour alone.
C_DATA = "#0072B2"      # blue
C_SIM = "#D55E00"       # vermillion
C_ALT = "#009E73"       # bluish green — the contaminant/gas variants
INK, MUTED, GRID = "#1b2430", "#6a7583", "#d4d9e0"
# quantile series are ORDINAL, so they get one hue light->dark, not categories
Q_RAMP = ["#9dc9e8", "#5ba3d0", "#2b7fb8", "#0072B2"]

# HV current check — the arithmetic the null rests on, all assumptions explicit
COSMIC_RATE_HZ = 24.0        # ~1 muon cm^-2 min^-1 through the 1440 cm2 active area
PRIMARY_E = 300.0            # ~100 e/cm x 30 mm drift, Ar/iC4H10 95/5 MIP
GAIN_AT_490 = 4.0e4          # data-side gain implied by the sim x0.6 deficit
QE = 1.602e-19
IMON_JUNE = ("/media/dylan/data/x17/cosmic_bench/det3/"
             "mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/"
             "hv_monitor.csv")
IMON_JUNE_CH = "3:4"
IMON_MAY_GLOB = ("/media/dylan/data/x17/cosmic_bench/det_3/"
                 "mx17_det3_HV_Scan_5-5-26/resist_*/hv_monitor.csv")
IMON_MAY_CH = "3:0"

# frozen T14 record — READ ONLY, never regenerated here
T14_SUMMARY = os.path.expanduser(
    "~/x17/response_sim/stageB_w2/t14_compare/t14_summary.json")


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(alpha=0.35, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")


def imon_check(out):
    """The avalanche current a cosmic actually puts into the resist supply,
    against what the supply can see."""
    q_muon = PRIMARY_E * GAIN_AT_490 * QE
    i_aval = COSMIC_RATE_HZ * q_muon
    d = pd.read_csv(IMON_JUNE)
    v, i = d[f"{IMON_JUNE_CH} vmon"], d[f"{IMON_JUNE_CH} imon"]
    st = v > 0.9 * v.max()
    june = dict(vmon=float(v[st].mean()), imon_mean=float(i[st].mean()),
                imon_std=float(i[st].std()), n=int(st.sum()),
                t0=str(d.timestamp.iloc[0]), t1=str(d.timestamp.iloc[-1]),
                series_min=((pd.to_datetime(d.timestamp[st])
                             - pd.to_datetime(d.timestamp.iloc[0]))
                            .dt.total_seconds().values / 60.0),
                series_i=i[st].values)
    rows = []
    for p in sorted(glob.glob(IMON_MAY_GLOB)):
        dd = pd.read_csv(p)
        vm, im = dd[f"{IMON_MAY_CH} vmon"], dd[f"{IMON_MAY_CH} imon"]
        mode = vm.round(0).mode()[0]
        s = vm.round(0) == mode
        rows.append((float(mode), float(im[s].mean()), float(im[s].std()),
                     int(s.sum()),
                     int(re.search(r"resist_(\d+)V", p).group(1))))
    may = pd.DataFrame(rows, columns=["vmon", "imon", "std", "n", "dirV"]
                       ).sort_values("vmon")
    out["imon"] = dict(
        q_per_muon_pC=q_muon * 1e12, rate_hz=COSMIC_RATE_HZ,
        primary_e=PRIMARY_E, gain=GAIN_AT_490,
        i_avalanche_pA=i_aval * 1e12,
        june={k: v for k, v in june.items() if not k.startswith("series")},
        june_R_Mohm=june["vmon"] / (june["imon_mean"] * 1e-6) / 1e6,
        ratio_to_standing=i_aval / (june["imon_mean"] * 1e-6),
        ratio_to_rms=i_aval / (june["imon_std"] * 1e-6),
        may=may.to_dict("list"),
        may_span=float(may.imon.iloc[-1] / may.imon.iloc[0]),
        may_gain_needed=float(may.imon.iloc[-1] * 1e-6
                              / (COSMIC_RATE_HZ * PRIMARY_E * QE)))
    return june, may


def figures(d, tab, dtab, out, june, may):
    figs = []

    def _save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(os.path.join(d, "figures", name), dpi=130,
                    facecolor="white")
        plt.close(fig)
        figs.append((name, caption))

    rail = out["rail"]
    tx = tab[tab.view == "x"].set_index("volt")
    sv, sg, _ = sim_ladder(os.path.join(SIM_DIR, SIM_HVSCAN), DRY)
    wv, wg, _ = sim_ladder(os.path.join(SIM_DIR, SIM_GRID), WET)
    iv, ig, _ = sim_ladder(os.path.join(SIM_DIR, SIM_HVSCAN), ISO10)

    # ── 1: the ladder, both legs indexed to a common base at 490 V ──────────
    ref = np.log(tx.p50[490])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for c, (name, _q) in zip(Q_RAMP, ESTIMATORS):
        ok = tx[f"{name}_ok"]
        ax.plot(tx.index[ok], np.log(tx[name][ok]), "o-", color=c, ms=5, lw=1.6,
                label=f"data {name}")
        ax.plot(tx.index[~ok], np.log(tx[name][~ok]), "o", color=c, ms=4,
                mfc="white", lw=0)
    off = ref - np.interp(490, sv, np.log(sg))
    ax.plot(sv, np.log(sg) + off, "s--", color=C_SIM, ms=5, lw=1.8,
            label="sim dry 95/5")
    ax.plot(wv, np.log(wg) + (ref - np.interp(490, wv, np.log(wg))), "^:",
            color=C_ALT, ms=5, lw=1.5, label="sim wet (1 % H2O)")
    ax.axhline(np.log(rail), color=MUTED, lw=1, ls=(0, (1, 3)))
    ax.text(534, np.log(rail) + 0.06, "ADC rail", color=MUTED, fontsize=8,
            ha="right")
    ax.annotate("data p50", (433, np.log(tx.p50[435]) + 0.45), color=C_DATA,
                fontsize=9.5, fontweight="bold", va="center")
    ax.annotate("sim", (532, np.log(sg[-1]) + off), color=C_SIM, fontsize=9.5,
                fontweight="bold", va="center")
    ax.set_xlim(420, 548)
    _style(ax, "mesh (resist) voltage [V]",
           "ln(response), offset to match at 490 V",
           "Both ladders on one axis, indexed at 490 V")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[1]
    ratio = json.load(open(T14_SUMMARY))["views"]["x"]["peak_amp_med"]["ratio"]
    sd = out["data"]["x"]["p50_head"]["slope10"] / 10
    ss = out["sim"]["dry_head"]["slope10"] / 10
    vv = np.arange(460, 531)
    ax.plot(vv, ratio * np.exp((ss - sd) * (vv - 490)), "-", color=C_SIM, lw=2)
    ax.plot([490], [ratio], "o", color=C_SIM, ms=8, zorder=5)
    ax.annotate(f"T14 measured\n×{ratio:.2f} at 490 V", (490, ratio),
                textcoords="offset points", xytext=(10, -28), fontsize=9,
                color=INK)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.text(462, 1.02, "sim = data", color=MUTED, fontsize=8)
    _style(ax, "mesh (resist) voltage [V]", "sim / data peak amplitude",
           "Where the two slopes put the deficit")
    _save(fig, "ladder.png",
          "LEFT — ln(peak-strip amplitude) vs mesh voltage for four data "
          "quantiles (blue, light→dark = p02→p50; hollow markers are above "
          "the 0.70×rail saturation cut and excluded from every fit) with the "
          "Garfield ln(gain) ladders overlaid. Both legs are shifted "
          "vertically to coincide at 490 V, so only the SHAPE is being "
          "compared — one axis, no second scale. The data rises visibly "
          "faster than either simulated gas. RIGHT — the consequence: the "
          "frozen T14 amplitude ratio measured at 490 V (dot) propagated to "
          "other voltages with the two fitted slopes. A deficit that is one "
          "constant would be a flat line.")

    # ── 2: local slope vs voltage ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    # the two ladders are differenced SEPARATELY: they were taken ~3 h apart
    # and interleave in voltage, so mixing them turns the ~2 % subrun-to-subrun
    # scatter into a spurious zigzag on a 5 V baseline
    for scan, mfc, lab in (("scan1", C_DATA, "data, ladder 1 (425–525 V)"),
                           ("scan2", "white", "data, ladder 2 (460–520 V)")):
        t = tab[(tab.view == "x") & (tab.scan == scan) & tab.p25_ok]
        lv, ls_ = t.volt.values, np.log(t.p25.values)
        ax.plot((lv[1:] + lv[:-1]) / 2, np.diff(ls_) / np.diff(lv) * 10, "o-",
                color=C_DATA, mfc=mfc, ms=5, lw=1.4, label=lab)
    ax.plot((sv[1:] + sv[:-1]) / 2, np.diff(np.log(sg)) / np.diff(sv) * 10,
            "s--", color=C_SIM, ms=5, lw=1.4, label="sim dry 95/5")
    ax.plot((iv[1:] + iv[:-1]) / 2, np.diff(np.log(ig)) / np.diff(iv) * 10,
            "^:", color=C_ALT, ms=5, lw=1.3, label="sim Ar/iso 90/10")
    for r, col in ((out["data"]["x"]["p50_head"], C_DATA),
                   (out["sim"]["dry_head"], C_SIM)):
        ax.axhspan(r["slope10"] - r["err10"], r["slope10"] + r["err10"],
                   color=col, alpha=0.15, lw=0)
    ax.set_xlim(420, 600)
    _style(ax, "mesh voltage [V]", "d ln(response) / dV  [per 10 V]",
           f"Local slope. Bands = the {HEAD_WINDOW[0]}–{HEAD_WINDOW[1]} V fits")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, "slope_local.png",
          "Finite-difference slope between adjacent points of the SAME ladder, "
          "so no fit window is assumed and no scan-to-scan offset leaks in. "
          "The two data ladders were taken ~3 h apart and agree. The simulated "
          "slope sits at 0.26–0.34 per 10 V everywhere in 460–590 V and for "
          "BOTH simulated gases; the data sits near 0.45. No shift of the "
          "voltage axis moves the sim onto the data, because the sim never "
          "gets there anywhere in its range.")

    # ── 3: spectra and their shape invariance ───────────────────────────────
    mesh = pd.read_parquet(os.path.join(d, "peaks.parquet"))
    mesh = mesh[(mesh.view == "x") & (mesh.ref_x > out["fiducial"]["x"][0])
                & (mesh.ref_x < out["fiducial"]["x"][1])
                & (mesh.ref_y > out["fiducial"]["y"][0])
                & (mesh.ref_y < out["fiducial"]["y"][1])]
    # only voltages whose MEDIAN is unsaturated, so the rescaling divisor is
    # itself untouched by the rail
    show = [425, 445, 465, 485]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    XMAX = 1.85       # 485 V reaches the rail at rail/p50 = 2.02
    for c, vlt in zip(Q_RAMP, show):
        a = mesh[mesh.volt == vlt].peak_amp.values
        axes[0].hist(a, bins=np.linspace(0, 4200, 45), histtype="step",
                     density=True, color=c, lw=1.7, label=f"{vlt} V")
        axes[1].hist(a / np.median(a), bins=np.linspace(0, XMAX, 40),
                     histtype="step", density=True, color=c, lw=1.7,
                     label=f"{vlt} V")
    axes[0].axvline(rail, color=MUTED, lw=1, ls=(0, (1, 3)))
    axes[0].text(rail - 80, 3e-3, "rail", color=MUTED, fontsize=8, ha="right")
    _style(axes[0], "peak-strip amplitude [ADC]", "density",
           "Raw spectra (log y)")
    axes[0].set_yscale("log")
    axes[1].set_xlim(0, XMAX)
    _style(axes[1], "peak amplitude / that voltage's own median", "density",
           "The same four spectra, each rescaled by its median")
    for ax in axes:
        ax.legend(fontsize=9, frameon=False)
    _save(fig, "spectra.png",
          "The data spectrum is a pure rescaling with voltage. Divided by its "
          "own median, 425–485 V collapse onto one curve over the whole range "
          "plotted — which stops at 1.85 because the rail cuts into the "
          "485 V spectrum just above it (rail/median = 2.02). All four "
          "medians are themselves "
          "unsaturated, so the divisor is clean. This is the premise the whole "
          "'one gain factor' idea rests on, and on the data side it holds: one "
          "number really does carry the whole voltage dependence. What fails "
          "is that the sim's number is not the same number at every voltage.")

    # ── 4: the controls that make the slope trustworthy ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    for view, mk in (("x", "o"), ("y", "s")):
        t = tab[tab.view == view]
        ax.plot(t.volt, t.eff, mk + "-", color=C_DATA if view == "x" else C_ALT,
                ms=5, lw=1.4, label=f"{view} view")
    ax.set_ylim(0.9, 1.02)
    _style(ax, "mesh voltage [V]", "fraction of events with a ≥5σ strip",
           "Detection efficiency in the fiducial")
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    ax = axes[1]
    dt = dtab[dtab.view == "x"]
    ax.plot(dt.volt, dt.p50, "o-", color=C_DATA, ms=5, lw=1.6)
    ax.axvline(1000, color=C_SIM, lw=1.5, ls="--")
    ax.text(1005, 700, "HV-scan\noperating point", color=C_SIM, fontsize=8)
    _style(ax, "drift voltage [V]  (mesh fixed at 490 V)",
           "median peak amplitude [ADC]", "Drift-field control")

    ax = axes[2]
    hi = mesh[mesh.volt >= 505]
    ax.hexbin(hi.ref_x, hi.ref_y, gridsize=26, cmap="Blues", mincnt=1)
    fx, fy = out["fiducial"]["x"], out["fiducial"]["y"]
    ax.add_patch(plt.Rectangle((fx[0], fy[0]), fx[1] - fx[0], fy[1] - fy[0],
                               fill=False, ec=C_SIM, lw=1.8))
    ax.text(fx[0] + 5, fy[1] + 8, "fiducial", color=C_SIM, fontsize=9)
    _style(ax, "M3 reference x at z = 702 [mm]", "reference y [mm]",
           "Where the selected tracks land")
    _save(fig, "controls.png",
          "The three things that could have faked a slope, closed. LEFT — "
          "detection efficiency is 1.000 at every voltage down to 425 V, so "
          "there is no threshold turn-on truncating the low tail. MIDDLE — the "
          "amplitude is on a plateau in drift field at the 1000 V operating "
          "point (−0.13 % per 10 V), and signal survives all the way down to "
          "100 V drift, which is what says the mesh is grounded and the drift "
          "field does not move when the mesh voltage does. RIGHT — the M3 "
          "fiducial, fixed once from the ≥505 V map and applied unchanged at "
          "every voltage.")

    # ── 5: the HV current monitor ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    t = june["series_min"]
    m, sd_ = june["imon_mean"] * 1e3, june["imon_std"] * 1e3
    ax.plot(t, june["series_i"] * 1e3, "-", color=C_DATA, lw=0.6, alpha=0.8)
    ax.axhspan(m - sd_, m + sd_, color=C_DATA, alpha=0.18, lw=0)
    i_av = out["imon"]["i_avalanche_pA"] / 1000            # nA
    ax.annotate(f"the entire cosmic avalanche current is {i_av:.3f} nA —\n"
                f"{sd_ / i_av:.0f}× thinner than this band, and thinner than\n"
                f"the line drawn above",
                xy=(0.03, 0.06), xycoords="axes fraction", color=C_SIM,
                fontsize=9)
    ax.annotate(f"±1 RMS = ±{sd_:.0f} nA", xy=(0.99, 0.94),
                xycoords="axes fraction", color=C_DATA, fontsize=8.5,
                ha="right")
    _style(ax, "time into the 490 V long run [min]",
           "resist-channel current [nA]",
           "What the supply sees at the bench operating point")

    ax = axes[1]
    ax.errorbar(may.vmon, may.imon, yerr=may["std"], fmt="o-", color=C_DATA,
                ms=5, lw=1.5, capsize=3, label="measured imon")
    ohm = may.imon.iloc[0] * may.vmon / may.vmon.iloc[0]
    ax.plot(may.vmon, ohm, "--", color=MUTED, lw=1.3,
            label="ohmic (I ∝ V), tied at 460 V")
    ax.set_yscale("log")
    _style(ax, "mesh voltage [V]", "resist-channel current [µA]",
           "det3 May HV scan — imon really does move with V")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.text(461, may.imon.max() * 0.30,
            f"×{out['imon']['may_span']:.1f} over 70 V, far above ohmic —\n"
            f"but crediting it to avalanche would need\n"
            f"a gas gain of {out['imon']['may_gain_needed']:.0e}",
            color=C_SIM, fontsize=9)
    _save(fig, "imon.png",
          "LEFT — the resist-supply current through the 490 V long run. The "
          f"standing current is {june['imon_mean']:.3f} µA with an RMS of "
          f"{sd_:.0f} nA (shaded); the whole cosmic-induced avalanche term is "
          f"{i_av:.3f} nA, {sd_ / i_av:.0f}× narrower than that band and "
          "thinner than the plotted line — there is nothing to see, which is "
          "the point. RIGHT — imon on a real det3 HV scan (May, a different "
          "bulk) does rise steeply with voltage and far above the ohmic line, "
          "which is exactly the trap: the rise is leakage / dark current, "
          "since crediting it to avalanche would require a gas gain of "
          f"{out['imon']['may_gain_needed']:.0e}.")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
CSS = """
body{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 color:#1b2430;max-width:1080px;margin:0 auto;padding:28px 22px 80px;
 background:#fff}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:34px 0 10px;
 border-bottom:1px solid #e6e9ee;padding-bottom:5px}
h3{font-size:15px;margin:22px 0 6px;color:#39424f}
.sub{color:#6a7583;margin:0 0 22px;font-size:13px}
.verdict{border-left:5px solid #D55E00;background:#fff6f0;padding:14px 18px;
 margin:18px 0;border-radius:0 6px 6px 0}
.verdict b{color:#a34600}
.ok{border-left-color:#009E73;background:#f1fbf7}.ok b{color:#00694e}
table{border-collapse:collapse;margin:12px 0;font-size:13px;width:100%}
th,td{border:1px solid #e6e9ee;padding:5px 9px;text-align:right}
th{background:#f7f9fb;font-weight:600;color:#39424f}
td:first-child,th:first-child{text-align:left}
figure{margin:22px 0} img{width:100%;border:1px solid #e6e9ee;border-radius:6px}
figcaption{color:#6a7583;font-size:12.5px;margin-top:7px}
code{background:#f2f4f7;padding:1px 5px;border-radius:3px;font-size:12.5px}
.big{font-size:30px;font-weight:600;color:#0072B2}
.grid{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}
.card{flex:1 1 210px;border:1px solid #e6e9ee;border-radius:8px;padding:12px 15px}
.card .lab{font-size:12px;color:#6a7583;text-transform:uppercase;
 letter-spacing:.05em}
ul{margin:8px 0 8px 20px}li{margin:4px 0}
"""



def townsend(o, ratio_x, V=490.0, lng_sim=None):
    """Decompose the slope+gain mismatch in the parallel-plate Townsend form
    alpha = A p exp(-B p / E), E = V/d  =>  lnG = A p d exp(-beta),
    beta = B p d / V, and d lnG/dV = lnG * beta / V.

    Returns the factors on (B p d) and on A needed to reproduce BOTH the
    measured slope and the measured gain. Deliberately crude — a woven-mesh
    gap is not parallel-plate — so this orders suspects, it does not measure
    anything."""
    if lng_sim is None:
        v, g, _ = sim_ladder(os.path.join(SIM_DIR, SIM_HVSCAN), DRY)
        lng_sim = float(np.interp(V, v, np.log(g)))
    s_sim = o["sim"]["dry_head"]["slope10"] / 10.0
    s_dat = o["data"]["x"]["p50_head"]["slope10"] / 10.0
    lng_dat = lng_sim + np.log(1.0 / ratio_x)
    beta_sim = s_sim * V / lng_sim
    beta_dat = s_dat * V / lng_dat
    f_bp = beta_dat / beta_sim
    gain_if_bp_only = f_bp * np.exp(-beta_sim * (f_bp - 1))
    gain_needed = lng_dat / lng_sim
    return dict(beta_sim=beta_sim, beta_data=beta_dat, f_bp=f_bp,
                lng_sim=lng_sim, lng_data=lng_dat,
                gain_if_bp_only=gain_if_bp_only, gain_needed=gain_needed,
                f_A=gain_needed / gain_if_bp_only)


def build_html(d, out, tab, figs):
    o, imon = out, out["imon"]
    dx = o["data"]["x"]
    sd, ss = dx["p50_head"]["slope10"], o["sim"]["dry_head"]["slope10"]
    t14 = json.load(open(T14_SUMMARY))["views"]
    e = html.escape

    def card(lab, val, note=""):
        return (f'<div class="card"><div class="lab">{lab}</div>'
                f'<div class="big">{val}</div><div class="lab">{note}</div></div>')

    rows = []
    tx = tab[tab.view == "x"]
    ty = tab[tab.view == "y"].set_index("volt")
    for _, r in tx.iterrows():
        mark = "" if r.p50_ok else " *"
        rows.append(
            f"<tr><td>{int(r.volt)}</td><td>{int(r.n)}</td>"
            f"<td>{r.eff:.3f}</td><td>{r.p10:.0f}</td><td>{r.p25:.0f}</td>"
            f"<td>{r.p50:.0f}{mark}</td><td>{ty.p50[r.volt]:.0f}</td>"
            f"<td>{r.nb_p50:.0f}</td><td>{r.fsat:.3f}</td></tr>")

    est = []
    for name, _q in ESTIMATORS:
        k = f"{name}_head"
        if k in dx:
            est.append(f"<tr><td>{name} peak amplitude</td>"
                       f"<td>{dx[k]['slope10']:.3f} ± {dx[k]['err10']:.3f}</td>"
                       f"<td>{o['data']['y'][k]['slope10']:.3f} ± "
                       f"{o['data']['y'][k]['err10']:.3f}</td>"
                       f"<td>{dx[k]['n']}</td></tr>")
    est.append(f"<tr><td>neighbour-strip sum (rail-robust)</td>"
               f"<td>{dx['nb_head']['slope10']:.3f} ± {dx['nb_head']['err10']:.3f}</td>"
               f"<td>{o['data']['y']['nb_head']['slope10']:.3f} ± "
               f"{o['data']['y']['nb_head']['err10']:.3f}</td>"
               f"<td>{dx['nb_head']['n']}</td></tr>")

    simrows = []
    for k, lab in (("dry_head", "dry Ar/iC4H10 95/5, 460–490 V (head-to-head)"),
                   ("dry_full", "dry Ar/iC4H10 95/5, 460–530 V (full ladder)"),
                   ("wet1pct", "Ar/iC4H10/H2O 94/5/1, 480–500 V"),
                   ("iso10_full", "Ar/iC4H10 90/10, 530–590 V")):
        r = o["sim"][k]
        simrows.append(f"<tr><td>{lab}</td><td>{r['slope10']:.3f} ± "
                       f"{r['err10']:.3f}</td><td>{r['n']}</td></tr>")

    figblocks = "\n".join(
        f'<figure><img src="figures/{n}" alt="{e(c[:80])}">'
        f"<figcaption><b>{n}</b> — {e(c)}</figcaption></figure>"
        for n, c in figs)

    loc = o["sim"]["dry_local_slope10"]
    ratio_x = t14["x"]["peak_amp_med"]["ratio"]
    tw = townsend(o, ratio_x)
    at460 = ratio_x * np.exp((ss - sd) / 10 * (460 - 490))
    at520 = ratio_x * np.exp((ss - sd) / 10 * (520 - 490))
    v_match = 490 + np.log(1 / ratio_x) / (ss / 10)     # sim V at the data gain
    s_match = float(np.interp(v_match, loc["v_mid"], loc["slope10"]))

    return f"""<!doctype html><meta charset="utf-8">
<title>HV-slope test — det3 gain vs mesh voltage, data vs Garfield</title>
<style>{CSS}</style>
<h1>HV-slope test: is the T14 amplitude deficit one constant?</h1>
<p class="sub">det3 · <code>mx17_det3_saturday_scan_6-27-26</code> mesh ladder
425–525 V vs <code>aval_calib_meshfield_hvscan.json</code> (dry Ar/iC4H10 95/5,
per-voltage T6 field maps, 150 µm gap) · Problem 1 of the T14 follow-up ·
2026-08-09</p>

<div class="verdict"><b>THE SLOPES DISAGREE — a single fitted gain factor is
NOT defensible.</b> Over the 460–490 V window covered by both ladders the bench
amplitude rises at <b>{sd:.3f} ± {dx['p50_head']['err10']:.3f}</b> per 10 V in
ln, the simulated gain at <b>{ss:.3f} ± {o['sim']['dry_head']['err10']:.3f}</b>
— a ratio of <b>{sd / ss:.2f}</b>, {abs(sd - ss) / np.hypot(dx['p50_head']['err10'], o['sim']['dry_head']['err10']):.0f}σ.
The detector doubles its gain every {np.log(2) / sd * 10:.1f} V; the simulation
every {np.log(2) / ss * 10:.1f} V. The deficit is therefore not a normalization
error: it is a wrong response to field, and fitting one constant at 490 V would
bury that rather than absorb it.</div>

<div class="grid">
{card("Data slope", f"{sd:.3f}", "Δln per 10 V, 460–490 V, median peak")}
{card("Sim slope", f"{ss:.3f}", "Δln per 10 V, same window")}
{card("Ratio", f"×{sd / ss:.2f}", "data / sim")}
{card("Efficiency", "1.000", "at every voltage — no turn-on bias")}
</div>

<h2>What was compared</h2>
<p><b>Data leg.</b> The det3 saturday scan's two interleaved mesh ladders
(<code>hv_scan</code> 425–525 V and <code>hv_scan2</code> 460–520 V, 10 V steps,
same day and same gas as the T14 target subrun). The observable is the one T14
itself compares — <code>wfm.max(axis=1)</code> on the raw 32 × 60 ns samples
through <code>wft.io.FeuReader</code>, so pedestal and CNS are bit-identical to
the frozen run. Three properties make the slope meaningful:</p>
<ul>
<li><b>No amplitude threshold anywhere.</b> The peak strip is the plane's
strongest strip, full stop. A threshold would truncate the low tail and bias the
median up at low voltage, faking a shallower slope.</li>
<li><b>A voltage-independent population.</b> The bench trigger and the M3
telescope are both external to det3, so requiring a good M3 track
(χ² &lt; 1, NClus = 4) landing inside a fiducial fixed once from the ≥505 V map
selects the same physical muons at every voltage.</li>
<li><b>Efficiency 1.000 at every point</b>, 425 V included — so nothing is
missing from the low tail even in principle. At 425 V the median peak is still
22σ above noise.</li>
</ul>
<p><b>Sim leg.</b> <code>aval_calib_meshfield_hvscan.json</code>, 8 dry 95/5
points 460–530 V, each with its own T6 FEM field map
(<code>meshfield_vmesh0460.txt</code> … <code>0530</code>) — verified to be the
per-voltage rerun and not the quarantined single-map campaign. Slope on
ln(gain_mean); the Polya MC error is {100 * np.sqrt(0.4556 / 1066):.1f} % per
point at 490 V.</p>

<h2>The numbers</h2>
<h3>Data — every estimator, over {HEAD_WINDOW[0]}–{HEAD_WINDOW[1]} V</h3>
<table><tr><th>estimator</th><th>X view [Δln / 10 V]</th>
<th>Y view [Δln / 10 V]</th><th>points</th></tr>
{''.join(est)}</table>
<p class="sub">Five estimators with different saturation onsets and both views
agree inside ±5 %. The low quantiles are the ones immune to the rail and they
sit if anything <i>below</i> the median, so the steep slope is not a saturation
artefact.</p>

<h3>Simulation</h3>
<table><tr><th>ladder</th><th>slope [Δln / 10 V]</th><th>points</th></tr>
{''.join(simrows)}</table>
<p class="sub">The simulated slope is {min(loc['min'], loc['max']):.2f}–{loc['max']:.2f}
per 10 V <i>everywhere</i> in 460–590 V and for both simulated gases, so no
offset of the voltage axis and no gas choice in hand brings it to 0.45. The wet
mixture is marginally steeper ({o['sim']['wet1pct']['slope10']:.3f}), i.e. the
contaminant axis moves in the right direction but covers about a tenth of the
gap — and, per the diagnosis-grid README, every contaminant also
<i>lowers</i> the gain, so it deepens the amplitude deficit while barely
touching the slope.</p>

<h3>The full data ladder (X view; Y median shown for comparison)</h3>
<table><tr><th>V</th><th>N</th><th>eff</th><th>p10</th><th>p25</th>
<th>p50 (X)</th><th>p50 (Y)</th><th>neighbour sum</th><th>frac at rail</th></tr>
{''.join(rows)}</table>
<p class="sub">* = above the 0.70 × rail cut ({o['sat_frac']:.2f} × {o['rail']:.0f}
ADC) and excluded from every fit. Cross-check against the frozen T14 record: at
490 V this scan's independent <code>hv_scan2</code> subrun gives a median peak
of {tx.set_index('volt').p50[490]:.0f} ADC (X), against T14's long-run
{t14['x']['peak_amp_med']['data']:.0f} ADC — 3 % apart, on different subruns and
with T14 additionally applying a per-view selection. The Y view differs more
({tab[tab.view == 'y'].set_index('volt').p50[490]:.0f} vs
{t14['y']['peak_amp_med']['data']:.0f}); that was first attributed here to
T14's θ window, which the follow-up
(<a href="iso_ve/report.html">iso_ve/report.html</a> §3) shows is <b>not</b> the
cause — peak amplitude is flat against the M3 reference angle across both
windows. The detector's two views are in fact identical (2606.5 vs 2607.0 ADC
on the long run, unselected); the difference is made by T14's per-view
reconstruction-quality cut, which removes saturated events three times more
often in Y.</p>

<h2>What could have faked this, and why it didn't</h2>
<table><tr><th>threat</th><th>test</th><th>outcome</th></tr>
<tr><td>Rail saturation compressing the high end</td>
<td>Five estimators with onsets spanning a factor ~4 in amplitude; every point
above 0.70 × rail dropped</td><td>all agree within ±5 %; the rail-immune p02 and the
neighbour sum give the same slope</td></tr>
<tr><td>Threshold turn-on inflating the low-V median</td>
<td>No threshold applied; efficiency measured in the fiducial</td>
<td>1.000 at every voltage — nothing to truncate</td></tr>
<tr><td>The event population changing with voltage</td>
<td>M3-track selection external to det3, fiducial fixed at high V</td>
<td>N per point stays in {o['shape']['n_range'][0]}–{o['shape']['n_range'][1]}
across the fitted range — Poisson-level scatter only</td></tr>
<tr><td>Gas or pressure drifting during the ~6 h scan</td>
<td>The two ladders were taken ~3 h apart and interleave in voltage</td>
<td>scan2's 460/470/480/490 V sit within 2–4 % of the geometric mean of scan1's
bracketing points</td></tr>
<tr><td>The drift field moving with the mesh voltage</td>
<td>Drift ladder at fixed 490 V mesh; signal at 100 V drift</td>
<td>signal survives at 100 V ⇒ mesh is grounded ⇒ drift field is independent of
the scan; and at the 1000 V operating point the amplitude is on a plateau
({o['drift_control']['dln_per_10V_at_1000'] * 100:+.2f} % per 10 V)</td></tr>
<tr><td>Sparks or discharges inflating the high-V points</td>
<td>Spectrum shape vs voltage</td><td>the spectrum is a pure rescaling
(see <code>spectra.png</code>): p50/p10 stays in
{o['shape']['p50_over_p10_range'][0]:.2f}–{o['shape']['p50_over_p10_range'][1]:.2f}
and p25/p02 in {o['shape']['p25_over_p02_range'][0]:.2f}–{o['shape']['p25_over_p02_range'][1]:.2f}
over 425–490 V, with no widening at the top of the range</td></tr>
<tr><td>Electronics gain/shaper differing between the legs</td>
<td>Done independently in the parallel T14 session: 44 saved
<code>CosmicTb_MX17.cfg</code> copies across the bench disk (Jan–mid-June,
det1/3/4, det3 6-16-26 included) all carry <code>Feu * Dream * 1 = 0x081F
0xD023</code> (peaking code 2 = 180 ns) and <code>Dream 6/7 = 0xAAAA</code>
(200 fC, 10 mV/fC). The saturday scan saved no cfg of its own but its
<code>run_config.json</code> points at the same template.</td>
<td>the electronics-scale hypothesis is dead; the shaper assumptions are
confirmed</td></tr>
<tr><td>A series resistance dropping part of the set voltage</td>
<td>Arithmetic: any drop ∝ I ∝ V shrinks the <i>true</i> ΔV</td>
<td>makes the true data slope <i>steeper</i> — wrong direction to rescue the
sim</td></tr>
</table>

<h2>Where this leaves the deficit</h2>
<p>Propagating the frozen T14 ratio (×{ratio_x:.2f} at 490 V, X view) along the
two fitted slopes: the simulation would sit at
<b>×{at460:.2f}</b> of data at 460 V and <b>×{at520:.2f}</b> at 520 V. A
deficit that were one constant would be flat. Matching on <i>gain</i> instead of
voltage does not help either: the sim reaches the data's 490 V gain at about
{v_match:.0f} V, where its local slope is {s_match:.3f} per 10 V — still
{sd / s_match:.2f}× short.</p>
<p>Read as physics: the simulated avalanche is not just too small, its
<i>logarithmic derivative with respect to field is too small</i>. Take the
textbook parallel-plate form α = A·p·exp(−B·p/E) with E = V/d, so
ln G = A·p·d·e<sup>−β</sup> and d ln G/dV = ln G · β/V with β = B·p·d/V. The
simulation sits at β = {tw['beta_sim']:.2f} at 490 V; reproducing the bench's
slope <i>and</i> the bench's gain (the T14 ×{ratio_x:.2f} put back) needs
β = {tw['beta_data']:.2f}, i.e. <b>B·p·d larger by
{100 * (tw['f_bp'] - 1):.0f} %</b>.</p>
<p>No single knob does that. Raising B·p·d on its own — a longer gap, a denser
or more quenching gas — buys the slope but costs gain, landing at
×{tw['gain_if_bp_only']:.2f} of the model instead of ×{tw['gain_needed']:.2f}.
Raising A on its own moves gain and slope in the <i>same</i> ratio, so it can
never change the slope/gain ratio at all. Getting both at once takes
~{tw['f_bp']:.2f}× in B·p and ~{tw['f_A']:.1f}× in A together — a genuinely
different α(E), not a rescaling of the one in hand. And it is more than the
quencher can supply: doubling the isobutane leaves the simulated slope at
{o['sim']['iso10_full']['slope10']:.3f} per 10 V, no steeper than 95/5.</p>
<p>So the ordering of suspects is the gas table and the Penning treatment
first, geometry second — with the caveat that a woven-mesh gap is not
parallel-plate at all, so a T6 field map whose <i>shape</i> (not just its
scale) is wrong would mimic exactly this and is not excluded by the algebra
above.</p>

<h2>HV current monitor — the null, crossed off</h2>
<div class="verdict ok"><b>NULL, as expected. The absolute gain cannot be read
out of imon.</b> A cosmic puts
{imon['q_per_muon_pC']:.2f} pC into the resist supply
({imon['primary_e']:.0f} primary e × gain {imon['gain']:.0e}); at
{imon['rate_hz']:.0f} Hz through the active area that is
<b>{imon['i_avalanche_pA']:.0f} pA</b>. The June 490 V long run stands at
<b>{imon['june']['imon_mean']:.3f} µA</b> with an RMS of
{imon['june']['imon_std'] * 1e3:.0f} nA — so the avalanche term is
{imon['ratio_to_standing']:.1e} of the standing current and
{imon['ratio_to_rms']:.1e} of its noise.</div>
<p>Empirically the trap is worse than "flat". det3's May HV scan does have a
per-subrun <code>hv_monitor.csv</code>, and imon there rises ×{imon['may_span']:.1f}
over 460→530 V — steeply and reproducibly. It is <i>not</i> avalanche current:
crediting it would require a gas gain of {imon['may_gain_needed']:.0e}. It is
leakage / dark current across the pillars, and it would masquerade as a gain
curve for anyone who fitted it. (The saturday scan itself wrote no per-subrun
monitor; only the 490 V long run has one, covering 22:48–01:10, after the
ladders finished.) For contrast, the n_TOF gamma flash <i>is</i> visible in the
same channel at ~100–150 nC per pulse — five orders of magnitude more charge,
delivered in a burst.</p>

<h2>Figures</h2>
{figblocks}

<h2>What this does not rule out</h2>
<ul>
<li><b>It does not identify which input is wrong.</b> The slope says the
field response is wrong; it does not separate gap, gas density, Penning
transfer, α(E) or the FEM field map. The Townsend-model reading above is an
ordering of suspects, not a measurement of any of them.</li>
<li><b>It does not measure the absolute gain.</b> This is a shape test. The
absolute scale still comes from T14, and still carries the W2 kernel's ~1.3 %
grid systematic and the unknown electronics ADC-per-fC.</li>
<li><b>It does not close the angle-dependent part of the deficit.</b> The
angled-ladder trend (×0.55 → ×0.38 with inclination) is a separate problem and
is untouched here: everything above is at the bench's own cosmic angle mix, at
one drift field.</li>
<li><b>It does not prove the gas was dry.</b> No humidity was ever measured.
The wet mini-ladder shifts the slope by ~10 % of the gap; a much wetter or
otherwise contaminated gas has not been simulated across a long enough voltage
lever arm to be excluded on slope alone.</li>
<li><b>It assumes the mesh voltage set by the supply is the gap voltage.</b>
vmon reads 490.24 V for a 490 V setting, but the drop across any filter
resistance between the supply and the resistive layer was not measured. Any
such drop steepens the true data slope, so it cannot rescue the simulation —
but it does mean the absolute voltage axis carries an unquantified offset.</li>
<li><b>The two ladders are one detector on one day.</b> No second det3 mesh
ladder was checked against this one, and det3's cathode is known to be dished
(GAP_STUDY_2026-07-30), which is itself a candidate for the gap suspect above.</li>
</ul>

<h2>Reproducing</h2>
<pre><code>python3 mx17_sim_wft/hv_slope/extract.py --out ~/x17/response_sim/hv_slope/peaks.parquet
python3 mx17_sim_wft/hv_slope/extract.py --drift --out ~/x17/response_sim/hv_slope/peaks_drift.parquet
python3 mx17_sim_wft/hv_slope/analyse.py
python3 mx17_sim_wft/hv_slope/make_report.py</code></pre>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/x17/response_sim/hv_slope"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.dir, "figures"), exist_ok=True)

    out = json.load(open(os.path.join(a.dir, "slopes.json")))
    tab = pd.read_csv(os.path.join(a.dir, "mesh_ladder.csv"))
    dtab = pd.read_csv(os.path.join(a.dir, "drift_ladder.csv"))
    june, may = imon_check(out)
    figs = figures(a.dir, tab, dtab, out, june, may)
    with open(os.path.join(a.dir, "slopes.json"), "w") as f:
        json.dump({k: v for k, v in out.items()}, f, indent=1, default=float)
    p = os.path.join(a.dir, "report.html")
    with open(p, "w") as f:
        f.write(build_html(a.dir, out, tab, figs))
    print("wrote", p)


if __name__ == "__main__":
    main()
