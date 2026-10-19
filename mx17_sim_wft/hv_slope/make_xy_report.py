#!/usr/bin/env python3
"""
Figures + report.html for the per-view shape-target re-derivation.

Which members of the "Y anomaly family" survive an unbiased selection?

    python3 mx17_sim_wft/hv_slope/make_xy_report.py
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

C_X, C_Y, C_SIM = "#0072B2", "#009E73", "#D55E00"
INK, MUTED, GRID = "#1b2430", "#6a7583", "#d4d9e0"
DIR = os.path.expanduser("~/x17/response_sim/hv_slope/xy_shape")


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


LADDER = ["unselected", "+ θ window", "+ rail depletion", "T14 leg"]


def _ladder_vals(o, view, key, t14key=None):
    r = o["views"][view]
    return [r[0][key], r[3][key], r[4][key],
            o["t14"][view]["data"][t14key or key]]


def figures(o, df):
    figs = []

    def _save(fig, name, cap):
        fig.tight_layout()
        fig.savefig(os.path.join(DIR, "figures", name), dpi=130,
                    facecolor="white")
        plt.close(fig)
        figs.append((name, cap))

    # ── 1: what each selection did, per observable ──────────────────────────
    specs = [("undershoot_p50", "undershoot [% of peak]", 100.0),
             ("fwhm_p50", "FWHM [ns]", 1.0),
             ("rise_p50", "rise 10–90 % [ns]", 1.0),
             ("frac_rise_lt240", "fraction with rise < 240 ns", 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    xpos = np.arange(4)
    for ax, (key, lab, sc) in zip(axes, specs):
        for view, c in (("x", C_X), ("y", C_Y)):
            v = np.array(_ladder_vals(o, view, key)) * sc
            ax.plot(xpos[:3], v[:3], "o-", color=c, ms=6, lw=1.9,
                    label=f"{view.upper()} view")
            ax.plot(xpos[3], v[3], "*", color=c, ms=15)
            ax.plot(xpos[2:], v[2:], ":", color=c, lw=1.4)
        ax.set_xticks(xpos)
        ax.set_xticklabels(LADDER, rotation=28, ha="right", fontsize=8.5)
        _style(ax, None, lab, lab.split(" [")[0])
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].text(0.02, 0.04, "★ = the frozen T14 leg", transform=axes[0].transAxes,
                 fontsize=8.5, color=MUTED)
    _save(fig, "decomposition.png",
          "Every per-view shape target, rebuilt one selection at a time from "
          "the unselected detector (left) to the frozen T14 leg (star). Adding "
          "T14's θ window to an M3-selected sample, then depleting the rail to "
          "T14's own surviving fraction, reproduces all four observables in "
          "both views — so the legs are fully accounted for by those two cuts. "
          "The θ window does almost all the work on the WIDTH observables "
          "(FWHM, rise); the rail depletion does almost all of it on the Y "
          "undershoot. Note how far the unselected detector sits from the "
          "leg: X FWHM 366 vs 544 ns, X rise 191 vs 285 ns.")

    # ── 2: the mechanism ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    prof = o["profile"]
    ctr = [min(p["hi"], 4000) * 0.5 + p["lo"] * 0.5 for p in prof["x"]]
    ax = axes[0]
    for view, c in (("x", C_X), ("y", C_Y)):
        ax.plot(ctr, [100 * p["undershoot"] for p in prof[view]], "o-",
                color=c, ms=6, lw=1.9, label=f"{view.upper()} view")
    ax.axvline(3500, color=MUTED, lw=1.2, ls=(0, (1, 3)))
    ax.text(3450, -12.5, "rail", color=MUTED, fontsize=8, ha="right")
    _style(ax, "peak amplitude [ADC]", "undershoot [% of peak]",
           "Undershoot as a FRACTION falls with amplitude in Y")
    ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    for view, c in (("x", C_X), ("y", C_Y)):
        ax.plot(ctr, [p["undershoot_adc"] for p in prof[view]], "o-",
                color=c, ms=6, lw=1.9, label=f"{view.upper()} view")
    ax.axvline(3500, color=MUTED, lw=1.2, ls=(0, (1, 3)))
    ax.text(3430, -30, "rail — clipped peak,\nnot a valid denominator",
            color=MUTED, fontsize=8, ha="right")
    _style(ax, "peak amplitude [ADC]", "undershoot [ADC]",
           "In ADC both views grow with signal — there is no ceiling")

    ax = axes[2]
    aprof = o["angle_profile"]
    actr = [0.5 * (p["lo"] + min(p["hi"], 20)) for p in aprof["x"]]
    for view, c in (("x", C_X), ("y", C_Y)):
        ax.plot(actr, [p["fwhm"] for p in aprof[view]], "o-", color=c, ms=6,
                lw=1.9, label=f"{view.upper()} view")
        w = o["theta_windows"][view][1]
        ax.axvline(w, color=c, lw=1.1, ls="--", alpha=0.7)
        ax.text(w + 0.2, 505 - 22 * (view == "y"), f"T14 {view.upper()} window",
                color=c, fontsize=8)
    _style(ax, "|M3 in-plane angle| [deg]", "FWHM [ns]",
           "Width falls steeply with inclination")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    _save(fig, "mechanism.png",
          "Why the selections bite. LEFT and MIDDLE — the undershoot: as a "
          "fraction of peak is amplitude-dependent in BOTH views, and in Y it "
          "is non-monotonic: it deepens to about −12.5 % near 2000 ADC and then "
          "falls back to −8.9 % by 3200 ADC, while X falls monotonically from "
          "−3.4 % to −2.7 %. In absolute ADC neither view has a ceiling — over "
          "the unrailed range Y grows x3.01 for a x3.00 peak range and X x2.36. "
          "Removing the high-amplitude events therefore lands on the shallow "
          "side of the curve in both views, which is what T14's selection did, "
          "and twice as much of it in Y. The last point on each curve is the "
          "rail: its peak is clipped, so it is not a valid denominator and it "
          "is excluded from every number quoted. RIGHT — FWHM "
          "against the M3 in-plane angle: an inclined track spreads its charge "
          "over more strips so each strip sees a shorter pulse, and the two "
          "T14 θ windows differ by 2× in width, which is why the legs show an "
          "X/Y width asymmetry the unselected detector mostly does not.")


    # ── 4: is it CNS? ───────────────────────────────────────────────────────
    cd = os.path.expanduser("~/x17/response_sim/hv_slope/cns")
    cns = json.load(open(os.path.join(cd, "cns_undershoot.json")))
    noise = json.load(open(os.path.join(cd, "noise.json")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for view, c in (("x", C_X), ("y", C_Y)):
        for on, ls, mk in ((1, "-", "o"), (0, "--", "s")):
            pr = cns["profiles"][f"{view}_cns{on}"]
            ax.plot([p["peak"] for p in pr], [p["adc"] for p in pr],
                    mk + ls, color=c, ms=6, lw=1.8, mfc=c if on else "white",
                    label=f"{view.upper()}  CNS {'on' if on else 'off'}")
    _style(ax, "peak amplitude [ADC]", "undershoot [ADC]",
           "The Y cap survives with CNS switched off")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")

    ax = axes[1]
    t = (np.arange(len(cns["cm_avg"]["x"]["avg"])) - 12) * 60.0
    for view, c in (("x", C_X), ("y", C_Y)):
        ax.plot(t, cns["cm_avg"][view]["avg"], "-", color=c, lw=2,
                label=f"{view.upper()} view (FEU {cns['cm_avg'][view]['feu']})")
    ax.axhline(cns["profiles"]["y_cns1"][-1]["adc"], color=C_Y, lw=1.3, ls=":")
    ax.text(-600, cns["profiles"]["y_cns1"][-1]["adc"] + 12,
            "the Y undershoot being explained (−292 ADC)", color=C_Y, fontsize=8.5)
    _style(ax, "t − t_peak [ns]", "block common mode [ADC]",
           "What CNS actually removes, averaged and peak-aligned")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, "cns_test.png",
          "The common-mode hypothesis, tested and rejected. LEFT — absolute "
          "undershoot against amplitude with CNS on (filled, solid) and off "
          "(hollow, dashed). The Y curve is untouched: "
          "−292 → −281 ADC in the top bin, and the whole Y curve moves by at "
          "most 4 %. X does not move at all. RIGHT — the per-block common mode "
          "itself, peak-aligned and averaged: it is a few ADC deep, two orders "
          "of magnitude below the feature it was supposed to explain. CNS is "
          "not doing nothing in this run (it takes the median channel noise "
          f"from {noise['8']['noise_raw']:.2f} to {noise['8']['noise_cns']:.2f} "
          "ADC), it is simply far too small to sculpt this.")

    # ── 3: the NaN population ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    ax = axes[0]
    for view, c in (("x", C_X), ("y", C_Y)):
        t = pd.read_parquet(os.path.expanduser(
            f"~/x17/response_sim/stageB_w2/t14_compare/wf_data_{view}.parquet"))
        nan = t.rise_ns.isna()
        bins = np.logspace(1, np.log10(4500), 45)
        ax.hist(t.peak_amp[nan], bins=bins, histtype="step", lw=2, color=c,
                label=f"{view.upper()}: rise = NaN ({nan.mean() * 100:.1f} %)")
        ax.hist(t.peak_amp[~nan], bins=bins, histtype="step", lw=1.2, color=c,
                ls=":", label=f"{view.upper()}: measurable")
    ax.set_xscale("log")
    _style(ax, "peak amplitude [ADC]", "events",
           "The frozen legs' un-measurable events are near-empty")
    ax.legend(fontsize=8.5, frameon=False)

    ax = axes[1]
    lab, vals, cols = [], [], []
    for view, c in (("x", C_X), ("y", C_Y)):
        np_ = o["t14"][view]["data"]["nan_pop"]
        lab += [f"{view.upper()}\nNaN", f"{view.upper()}\nmeasurable"]
        vals += [np_["peak_p50_nan"], np_["peak_p50_ok"]]
        cols += [c, c]
    b = ax.bar(range(4), vals, 0.6, color=cols)
    for i, r in enumerate(b):
        r.set_alpha(1.0 if i % 2 == 0 else 0.4)
        ax.text(i, vals[i] + 60, f"{vals[i]:.0f}", ha="center", fontsize=9,
                color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lab, fontsize=9)
    _style(ax, None, "median peak amplitude [ADC]",
           "Median peak of each population")
    _save(fig, "nan_population.png",
          "The 9.0 % (X) / 3.0 % (Y) of frozen-leg events whose rise time is "
          "un-measurable are not slow events — they are events with almost no "
          "signal in that view: median peak 137 ADC (X) and 65 ADC (Y) against "
          "2685 and 2187 for the rest, and none of them are railed. They "
          "cannot bias rise or FWHM (a NaN is skipped by the median) but they "
          "do enter the peak-amplitude and undershoot statistics, and they are "
          "part of why the frozen X median sits 3 % below the detector's. In "
          "the M3-selected sample used here the same fraction is 0.2 %, "
          "because requiring a reference track through the fiducial "
          "guarantees a muon actually crossed.")

    # ── 5: does the deep undershoot follow the FEU or the layer? ────────────
    cc = json.load(open(os.path.join(cd, "crosscheck.json")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    for label, r in cc.items():
        pr = [q for q in r["profile"] if q["adc"] is not None]
        c = C_Y if r["layer"] == "Y" else C_X
        ls = {"det3": "-", "det6": "--", "det7": ":"}[r["det"]]
        ax.plot([0.5 * (q["lo"] + q["hi"]) for q in pr], [q["adc"] for q in pr],
                "o" + ls, color=c, ms=5, lw=1.8,
                label=f"{label}  (FEU {r['feu']})")
    _style(ax, "peak amplitude [ADC]", "undershoot [ADC]",
           "Three detectors, six planes")
    ax.legend(fontsize=8, frameon=False, ncol=2)

    ax = axes[1]
    labs, vals, cols, feus = [], [], [], []
    for label, r in cc.items():
        pr = [q for q in r["profile"] if q["adc"] is not None]
        labs.append(label.replace(" ", "\n"))
        vals.append(pr[-1]["adc"])
        cols.append(C_Y if r["layer"] == "Y" else C_X)
        feus.append(r["feu"])
    xp = np.arange(len(labs))
    ax.bar(xp, vals, 0.62, color=cols)
    for i, (v, f) in enumerate(zip(vals, feus)):
        ax.text(i, v - 22, f"FEU {f}", ha="center", fontsize=8.5, color=INK)
        ax.text(i, v * 0.45, f"{v:.0f}", ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels(labs, fontsize=9)
    _style(ax, None, "undershoot at 3000–3500 ADC [ADC]",
           "The deep side is the LAYER, not the FEU")
    _save(fig, "feu_vs_layer.png",
          "The confound broken. On det3 the deep-undershoot view is also the "
          "view on FEU 8, so layer and electronics cannot be separated. The "
          "6-26 overnight run wires the same two layer types onto different "
          "FEUs: det6's Y layer sits on FEU 4 while det3's and det7's sit on "
          "FEU 8, and the X layers sit on FEUs 7, 3 and 6. Measured in the same "
          "absolute amplitude bin on all six planes, every Y layer is 3–8x "
          "deeper than every X layer, and FEU 4 (det6 Y, −436 ADC) is the "
          "deepest of all while FEU 6 (det7 X, −39 ADC) is the shallowest. The "
          "effect follows the resistive-side layer across three detectors and "
          "three different FEUs. Cross-detector magnitudes carry gas, gain and "
          "epoch differences, but the X-versus-Y contrast inside each detector "
          "is internally controlled — same events, same run, same gas.")
    return figs


CSS = """
body{font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 color:#1b2430;max-width:1080px;margin:0 auto;padding:28px 22px 80px;background:#fff}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:34px 0 10px;
 border-bottom:1px solid #e6e9ee;padding-bottom:5px}
h3{font-size:15px;margin:22px 0 6px;color:#39424f}
.sub{color:#6a7583;margin:0 0 22px;font-size:13px}
.verdict{border-left:5px solid #D55E00;background:#fff6f0;padding:14px 18px;
 margin:18px 0;border-radius:0 6px 6px 0}
.verdict b{color:#a34600}
.ok{border-left-color:#009E73;background:#f1fbf7}.ok b{color:#00694e}
.info{border-left-color:#0072B2;background:#f2f8fc}.info b{color:#005285}
table{border-collapse:collapse;margin:12px 0;font-size:13px;width:100%}
th,td{border:1px solid #e6e9ee;padding:5px 9px;text-align:right}
th{background:#f7f9fb;font-weight:600;color:#39424f}
td:first-child,th:first-child{text-align:left}
tr.hi td{background:#fff6f0;font-weight:600}
figure{margin:22px 0} img{width:100%;border:1px solid #e6e9ee;border-radius:6px}
figcaption{color:#6a7583;font-size:12.5px;margin-top:7px}
code{background:#f2f4f7;padding:1px 5px;border-radius:3px;font-size:12.5px}
ul{margin:8px 0 8px 20px}li{margin:4px 0} a{color:#0072B2}
"""


def build_html(o, figs):
    e = html.escape
    figblocks = "\n".join(
        f'<figure><img src="figures/{n}" alt="{e(c[:80])}">'
        f"<figcaption><b>{n}</b> — {e(c)}</figcaption></figure>"
        for n, c in figs)

    def rows(view):
        out = []
        for r in o["views"][view]:
            out.append(
                f"<tr><td>{e(r['label'])}</td><td>{r['n']}</td>"
                f"<td>{r['sat_frac']:.3f}</td><td>{r['peak_p50']:.0f}</td>"
                f"<td>{100 * r['undershoot_p50']:.1f} %</td>"
                f"<td>{r['fwhm_p50']:.0f}</td><td>{r['rise_p50']:.0f}</td>"
                f"<td>{r['rise_p5']:.0f}</td>"
                f"<td>{r['frac_rise_lt240']:.3f}</td>"
                f"<td>{r['nan_rise']:.3f}</td></tr>")
        d = o["t14"][view]["data"]
        out.append(
            f"<tr class='hi'><td>T14 data leg (frozen)</td><td>{d['n']}</td>"
            f"<td>{d['sat_frac']:.3f}</td><td>{d['peak_p50']:.0f}</td>"
            f"<td>{100 * d['undershoot_p50']:.1f} %</td>"
            f"<td>{d['fwhm_p50']:.0f}</td><td>{d['rise_p50']:.0f}</td>"
            f"<td>{d['rise_p5']:.0f}</td><td>{d['frac_rise_lt240']:.3f}</td>"
            f"<td>{d['nan_rise']:.3f}</td></tr>")
        return "".join(out)

    ux0 = o["views"]["x"][0]
    uy0 = o["views"]["y"][0]
    uxw = o["views"]["x"][3]
    uyw = o["views"]["y"][3]
    tx = o["t14"]["x"]["data"]
    ty = o["t14"]["y"]["data"]
    npx = tx["nan_pop"]
    npy = ty["nan_pop"]
    cd = os.path.expanduser("~/x17/response_sim/hv_slope/cns")
    cns = json.load(open(os.path.join(cd, "cns_undershoot.json")))
    noise = json.load(open(os.path.join(cd, "noise.json")))
    ycap_on = cns["profiles"]["y_cns1"][-1]["adc"]
    ycap_off = cns["profiles"]["y_cns0"][-1]["adc"]
    nraw, ncns = noise["8"]["noise_raw"], noise["8"]["noise_cns"]
    nratio = noise["8"]["ratio"]
    cc = json.load(open(os.path.join(cd, "crosscheck.json")))
    ccrows = ""
    for det in ("det3", "det6", "det7"):
        rx = cc[f"{det} X"]
        ry = cc[f"{det} Y"]
        vx = [q["adc"] for q in rx["profile"] if q["adc"] is not None][-1]
        vy = [q["adc"] for q in ry["profile"] if q["adc"] is not None][-1]
        ccrows += (f"<tr><td>{det}</td><td>FEU {rx['feu']}</td>"
                   f"<td>{vx:.0f}</td><td>FEU {ry['feu']}</td>"
                   f"<td>{vy:.0f}</td><td>{vy / vx:.1f}x</td></tr>")
    cnsrows = "".join(
        f"<tr><td>{p['lo']:.0f}–{min(p['hi'], 4200):.0f} ADC</td>"
        f"<td>{cns['profiles']['x_cns1'][i]['adc']:.0f}</td>"
        f"<td>{cns['profiles']['x_cns0'][i]['adc']:.0f}</td>"
        f"<td>{cns['profiles']['y_cns1'][i]['adc']:.0f}</td>"
        f"<td>{cns['profiles']['y_cns0'][i]['adc']:.0f}</td>"
        f"<td>{p['cm_tail_min']:.1f}</td></tr>"
        for i, p in enumerate(cns["profiles"]["y_cns1"]))

    return f"""<!doctype html><meta charset="utf-8">
<title>Per-view shape targets, re-derived unbiased — det3</title>
<style>{CSS}</style>
<h1>Which of the Y anomalies survive an unbiased selection?</h1>
<p class="sub">Re-derivation of the per-view DATA shape targets (undershoot,
FWHM, rise) on the 490 V long run · continuation of
<a href="../iso_ve/report.html">iso_ve/report.html</a> §3 · det3 · 2026-08-09</p>

<div class="verdict"><b>Both T14 data legs are fully accounted for by two
selections, and the per-view shape targets change.</b> Adding T14's θ window to
an M3-selected sample and then depleting the rail to T14's own surviving
fraction reproduces all four observables in both views to within a few percent.
The two cuts act on different observables: <b>the θ window drives the width
targets</b> (X FWHM 366 → 524 ns before the rail step even enters) and <b>the
rail depletion drives the Y undershoot</b> (−9.3 % → −11.6 %).</div>

<div class="verdict ok"><b>Recommended unbiased targets: Y undershoot
{100 * uy0['undershoot_p50']:.1f} % (not −12.0 %), X undershoot
{100 * ux0['undershoot_p50']:.1f} % (unchanged).</b> The Y target is
{abs(100 * (uy0['undershoot_p50'] - ty['undershoot_p50']) / ty['undershoot_p50']):.0f} %
shallower than the one in use. That makes the "Y undershoot is unreachable by
any β ≥ 0" conclusion <i>stronger</i>, not weaker: β = 0 gives −18.9 % against
a target that has moved further away, from −12.0 % to −9.3 %.</div>

<h2>The decomposition</h2>
<h3>X view</h3>
<table><tr><th>sample</th><th>n</th><th>rail frac</th><th>peak p50</th>
<th>undershoot</th><th>FWHM</th><th>rise p50</th><th>rise p5</th>
<th>rise &lt; 240</th><th>NaN rise</th></tr>{rows("x")}</table>
<h3>Y view</h3>
<table><tr><th>sample</th><th>n</th><th>rail frac</th><th>peak p50</th>
<th>undershoot</th><th>FWHM</th><th>rise p50</th><th>rise p5</th>
<th>rise &lt; 240</th><th>NaN rise</th></tr>{rows("y")}</table>
<p class="sub">Same subrun, same <code>FeuReader</code> path and the same
observable definitions as <code>t14_compare.extract_view</code>. Selection is
M3 (χ² &lt; 1, NClus = 4) inside the fiducial, with no reconstruction cut. The
θ window is applied on the <i>M3 reference</i> angle, which is sharper than the
reconstructed one T14 used, so that row is a slightly harder cut than T14's.
The undershoot uses events whose peak leaves at least 11 tail samples in the
32-sample window; the other observables use everything, as T14 does.</p>

<h2>Which member survives</h2>
<table><tr><th>claim</th><th>verdict</th><th>corrected number</th></tr>
<tr><td>Y undershoot −12.0 %, unreachable by any β ≥ 0</td>
<td><b>SURVIVES, strengthened</b> — the anomaly is real but the target moves
away from the sim</td>
<td>{100 * uy0['undershoot_p50']:.1f} % unselected,
{100 * uyw['undershoot_p50']:.1f} % with the θ window</td></tr>
<tr><td>X undershoot −3.4 %</td><td><b>SURVIVES unchanged</b></td>
<td>{100 * ux0['undershoot_p50']:.1f} % unselected,
{100 * uxw['undershoot_p50']:.1f} % with the θ window</td></tr>
<tr><td>Data X/Y undershoot asymmetry (Y undershoots ~3.5× deeper)</td>
<td><b>SURVIVES</b>, at ~2.9× rather than 3.5×</td>
<td>{ux0['undershoot_p50'] / uy0['undershoot_p50']:.2f} → ratio
{uy0['undershoot_p50'] / ux0['undershoot_p50']:.1f}×</td></tr>
<tr><td>Y FWHM sim/data = ×1.39 vs X ×1.19</td>
<td><b>DOES NOT SURVIVE as stated</b> — the data denominators come from
angular windows that differ by 2× in width, and FWHM is strongly
angle-dependent</td>
<td>unselected data FWHM: X {ux0['fwhm_p50']:.0f} ns, Y {uy0['fwhm_p50']:.0f} ns
— {100 * abs(ux0['fwhm_p50'] - uy0['fwhm_p50']) / ux0['fwhm_p50']:.0f} % apart,
against {tx['fwhm_p50'] / ty['fwhm_p50']:.2f}× in the legs</td></tr>
<tr><td>Data X/Y peak-amplitude asymmetry</td>
<td><b>DOES NOT SURVIVE</b> (already reported)</td>
<td>{ux0['peak_p50']:.1f} vs {uy0['peak_p50']:.1f} ADC — 0.02 % apart</td></tr>
<tr><td>"The data has a fast rise population" (frac &lt; 240 ns = 0.34 / 0.41)</td>
<td><b>UNDERSTATED</b> — the unselected detector is much faster still</td>
<td>frac &lt; 240 ns = {ux0['frac_rise_lt240']:.3f} (X) /
{uy0['frac_rise_lt240']:.3f} (Y); rise p50
{ux0['rise_p50']:.0f} / {uy0['rise_p50']:.0f} ns</td></tr>
</table>

<h2>The mechanism, measured</h2>
<p><b>Why the rail depletion moves Y and not X.</b> The undershoot
<i>fraction</i> is amplitude-dependent in both views, and in Y it is
non-monotonic: measured on unrailed events in fine bins it runs −8.9 %, −11.6 %,
−12.5 %, −10.4 %, −8.9 % from 1065 to 3196 ADC, peaking near 2000 ADC. X falls
monotonically, −3.4 % to −2.7 %. So removing the high-amplitude events lands on
the shallow side of the curve in both views and deepens the surviving median —
but Y's fraction is ~3.5x larger and Y lost twice as many events (rail fraction
0.35 → 0.11 against X's 0.38 → 0.26), so the shift is visible there and
negligible in X.</p>
<div class="verdict"><b>Correction to an earlier reading of this data.</b> An
earlier version of this report described the Y undershoot as <i>saturating</i>
at about −290 ADC. That was wrong. It came from a coarse top bin that included
railed events — whose clipped peak is not a valid denominator — sitting next to
the 3000–3500 bin at a similar value and looking like a plateau. With the rail
excluded and finer bins there is no ceiling in either view: over a ×3.00 peak
range the absolute undershoot grows ×3.01 in Y and ×2.36 in X. <b>The target
correction is unaffected</b> — −12.0 % → −9.3 % is a direct measurement of the
median on two samples and never depended on the mechanism — but the
"amplitude-limited return current" reading of it is withdrawn.</div>
<p><b>Why the θ window moves the widths.</b> An inclined track spreads its
charge across more strips, so each strip sees a shorter pulse: FWHM falls from
{o['angle_profile']['x'][0]['fwhm']:.0f} ns at |θ| &lt; 1° to
{o['angle_profile']['x'][4]['fwhm']:.0f} ns at 7–12° in X, and
{o['angle_profile']['y'][0]['fwhm']:.0f} → {o['angle_profile']['y'][4]['fwhm']:.0f} ns
in Y. T14's windows are ±{o['theta_windows']['x'][1]:.1f}° in X but
±{o['theta_windows']['y'][1]:.1f}° in Y — twice as wide, because the sim's Y
reco angle spread is wider — so the X leg is measured on near-vertical tracks
and the Y leg on a much broader mix. That is most of the legs' X/Y width
asymmetry. (Which M3 angle belongs to which view was decided by the data, not
assumed: the pairing is the one with the steeper FWHM gradient, and it comes
out crossed — the X view responds to M3's y angle — which is the known 90°
strip-map-to-M3 rotation.)</p>

<h2>The NaN-rise population (S3_rerun's question)</h2>
<div class="verdict info"><b>The un-measurable events are near-empty events,
not slow ones — and the effect is on the same axis as the saturation bias but
in the opposite direction.</b></div>
<p>{100 * npx['frac']:.1f} % of the frozen X leg and {100 * npy['frac']:.1f} %
of the Y leg have <code>rise_ns = NaN</code>. Their median peak amplitude is
<b>{npx['peak_p50_nan']:.0f} ADC (X)</b> and <b>{npy['peak_p50_nan']:.0f} ADC
(Y)</b>, against {npx['peak_p50_ok']:.0f} and {npy['peak_p50_ok']:.0f} for the
measurable ones; {100 * npx['sat_frac_nan']:.1f} % of them are railed. They are
events in which that view saw essentially nothing — the 10 % and 90 % crossings
do not exist because there is no pulse. In the M3-selected sample the same
fraction is {o['views']['x'][0]['nan_rise']:.3f} (X) /
{o['views']['y'][0]['nan_rise']:.3f} (Y), because requiring a reference track
through the fiducial guarantees a muon crossed.</p>
<p>Consequences, and they differ per observable:</p>
<ul>
<li><b>Rise and FWHM are unaffected</b> — a NaN is skipped by the median, so
these events simply do not enter. The 9 % is a lost-statistics issue, not a
bias.</li>
<li><b>Peak amplitude and undershoot ARE affected</b>, because those are
finite for a near-empty event. Nine percent of the X leg sitting at ~137 ADC
pulls the X median peak down, which is part of why the leg reads
{tx['peak_p50']:.0f} against the detector's {ux0['peak_p50']:.0f}.</li>
<li><b>Against the saturation bias it partly cancels in X and barely matters
in Y</b> — the two act in the same direction on the undershoot (both remove
signal-rich events or add signal-poor ones, making the fraction deeper), but
the near-empty population is 3× smaller in Y where the saturation bias is 3×
bigger. Net: the corrections do not cancel; they compound mildly in X (where
the observable is flat anyway) and the rail term dominates Y.</li>
<li>The simulation legs have <b>zero</b> NaN rise, so this is a data-leg-only
defect and it does not cancel in any sim/data ratio.</li>
</ul>


<h2>Is the deep Y undershoot made by CNS? No.</h2>
<div class="verdict ok"><b>Common-mode subtraction is exonerated, and the sim's
missing CNS emulation is not a material bias on the undershoot.</b> Switching
CNS off leaves the Y undershoot where it was — {ycap_on:.0f} → {ycap_off:.0f}
ADC in the top bin, at most 4 % anywhere on the curve — and moves X not at
all. The clipping is therefore upstream of the processing, in the detector or
the electronics return path, and it is a genuine modelling item for the
chain.</div>
<p><b>No reprocessing was needed.</b> The saturday scan was written with
<code>pedestal_subtraction: false</code> and
<code>common_noise_subtraction: false</code>, so <code>decoded_root</code> holds
raw samples and <code>wft.io.FeuReader</code> applies pedestal and CNS itself,
in software, on every read. CNS is a switch on the read. Better still,
FeuReader computes its pedestal from the raw stack <i>before</i> the CNS step,
so the two legs share one pedestal exactly and the A/B is clean. (The
mm_processor gotcha that pedestal RMS is always post-CNS does not apply — that
path is not used here.) The local reader that carries the switch is checked
bit-identical to <code>FeuReader</code> at CNS = on before it is used
(<code>cns_undershoot.py --verify</code>).</p>
<table><tr><th>peak amplitude bin</th><th>X, CNS on</th><th>X, CNS off</th>
<th>Y, CNS on</th><th>Y, CNS off</th><th>block common mode, tail min</th></tr>
{cnsrows}</table>
<p class="sub">Median undershoot in ADC. The last column is the deepest excursion
of the peak strip's own 64-channel block median after the peak — what CNS
subtracts from that strip.</p>
<p>Two numbers settle it. The common mode CNS removes is
<b>−4 to −7.5 ADC</b> in the tail, against the <b>−292 ADC</b> Y undershoot it
was supposed to explain — two orders of magnitude short. And CNS is not idle in
this run: it takes the median channel noise from {nraw:.2f} to {ncns:.2f} ADC
(a factor {nratio:.2f}) on both FEUs. It is working, and it is simply the wrong
size. (Note for the record: this run does <i>not</i> show the large FEU 6/8
common mode seen elsewhere on the bench — here CNS buys 20 %, not a factor
ten.)</p>
<div class="verdict info"><b>Consequence for the β discipline.</b> The concern
that the data legs are post-CNS while the simulation has no CNS emulation, so
the sim-vs-data undershoot comparison could be an apples-to-oranges processing
artifact, is <b>quantitatively closed</b>: CNS moves the data undershoot by
≤ 0.3 percentage points in Y and ≤ 0.05 in X. Undershoot remains a fair
β-fitting observable on that axis. What it is <i>not</i> fair against is the
amplitude DEPENDENCE of the undershoot — the fraction is not one number, it
varies by ~30 % across the amplitude range and non-monotonically in Y, so a β
fitted to a single undershoot median inherits whatever amplitude mix that
sample happened to have. Quote the amplitude distribution alongside any β
fitted this way.</div>


<h2>Does the deep Y undershoot follow the FEU or the layer? The layer.</h2>
<div class="verdict ok"><b>Detector-side, and the electronics are excluded.</b>
On det3 alone the deep view is also the view on FEU 8, so "resistive-side
layer" and "FEU 8 chain" are perfectly confounded. The 6-26 overnight run
breaks that, because it wires the same two layer types onto different FEUs —
and the deep undershoot follows the layer across three detectors and three
different FEUs.</div>
<table><tr><th></th><th>X layer</th><th>undershoot at 3000–3500 ADC</th>
<th>Y layer</th><th>undershoot at 3000–3500 ADC</th><th>Y / X</th></tr>
{ccrows}</table>
<p class="sub">Median undershoot in ADC in one shared absolute amplitude bin,
railed events excluded. These legs use no M3 and no fiducial — this is a shape
comparison at fixed amplitude, so the population matters far less than it did
for the amplitude work, but the price is that spark and junk events are not
vetoed and the cross-detector magnitudes carry gas, gain and epoch differences.
The X-versus-Y contrast <i>inside</i> each detector is internally controlled.</p>
<p>The deepest plane in the set is det6's Y on <b>FEU 4</b> (−436 ADC) and the
shallowest is det7's X on <b>FEU 6</b> (−39 ADC). If this were a property of the
FEU 8 chain, det6's FEU 4 would look like an X plane and it emphatically does
not. So the return current that makes the Y side undershoot several times
deeper than the X side is a property of the detector's resistive-side layer,
and it belongs in the response model rather than in an electronics
calibration.</p>
<p class="sub">Secondary, and weak: within det3 at 490 V and peak ≥ 2300 ADC the
undershoot by 64-channel block runs −259 to −328 ADC in Y and −66 to −100 in X,
with no block standing out in either view — but with only 18–39 events per block
and 6 of 8 blocks populated, this does not exclude connector-level structure at
the ±20 % level.</p>

<h2>What to use downstream</h2>
<div class="verdict info"><b>Take the θ window, drop the rail depletion.</b>
The θ window is a legitimate population match — it exists to make the data's
angular spread resemble the gun's, and both legs need it. The rail depletion is
a one-sided defect: the simulation saturates
{100 * o['t14']['y']['sim']['sat_frac']:.1f} % of the time in Y against the
detector's {100 * o['views']['y'][0]['sat_frac']:.1f} %, so there is nothing on
the sim side for the same cut to remove. So the like-for-like data targets are
the θ-window rows: <b>undershoot X {100 * uxw['undershoot_p50']:.1f} %,
Y {100 * uyw['undershoot_p50']:.1f} %; FWHM X {uxw['fwhm_p50']:.0f} ns,
Y {uyw['fwhm_p50']:.0f} ns; rise p50 X {uxw['rise_p50']:.0f} ns,
Y {uyw['rise_p50']:.0f} ns.</b></div>

<h2>Figures</h2>
{figblocks}

<h2>What this does not rule out</h2>
<ul>
<li><b>It does not re-derive the sim leg.</b> Every sim/data ratio quoted
downstream needs the sim side recomputed under the same treatment, which is the
T14 owner's to do. Nothing in <code>t14_compare/</code> was regenerated here;
its tables were read only.</li>
<li><b>The railed events are still not usable for shape.</b> Their peak is
clipped, so their measured undershoot fraction has a shrunken denominator and
their FWHM has a flat top. "Keep the rail" is right for the population and
wrong for the waveform — which is why the unsaturated-only row is given
alongside, and why the Y undershoot has a genuine range,
{100 * o['views']['y'][1]['undershoot_p50']:.1f} % (unsaturated) to
{100 * uy0['undershoot_p50']:.1f} % (all).</li>
<li><b>The θ window here is on the M3 angle, not the reconstructed one.</b>
T14 cut on reco θ, which is smeared by 1–2.4° and differently per view. My
window is therefore sharper than T14's, which is why the θ-window rows do not
land exactly on the legs.</li>
<li><b>The deep Y undershoot is localised to the detector but not
explained.</b> The cross-check below places it on the resistive-side layer
rather than on any FEU, and CNS is excluded, but what in that layer produces a
return current 3–8x deeper than the X side is not settled here.</li>
<li><b>One subrun, one voltage.</b> 490 V only; the amplitude dependence was
measured across the natural Landau spread, not by varying the gain.</li>
</ul>

<h2>Reproducing</h2>
<pre><code>python3 mx17_sim_wft/hv_slope/xy_shape.py
python3 mx17_sim_wft/hv_slope/make_xy_report.py</code></pre>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.dir, "figures"), exist_ok=True)
    o = json.load(open(os.path.join(a.dir, "xy_shape.json")))
    df = pd.read_parquet(os.path.join(a.dir, "shape.parquet"))
    figs = figures(o, df)
    p = os.path.join(a.dir, "report.html")
    with open(p, "w") as f:
        f.write(build_html(o, figs))
    print("wrote", p)


if __name__ == "__main__":
    main()
