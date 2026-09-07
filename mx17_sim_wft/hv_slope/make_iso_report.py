#!/usr/bin/env python3
"""
Figures + report.html for the isobutane follow-up to the HV-slope test.

Three questions, one document:
  1. Does the measured v(E) exclude a richer isobutane fraction than 95/5?
  2. Does anything on record pin the June mixture?
  3. Why does the frozen T14 data leg show X != Y when the detector does not?

    python3 mx17_sim_wft/hv_slope/make_iso_report.py
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
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path[:0] = [_HERE, _REPO, os.path.join(_REPO, "mx_june_cosmic_qa"),
                os.path.join(_REPO, "cosmic_bench_analysis")]
from analyse import DRY, ISO10, SIM_DIR, SIM_HVSCAN, sim_ladder   # noqa: E402
import iso_ve as IV                                              # noqa: E402

C_DATA, C_SIM, C_ALT = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1b2430", "#6a7583", "#d4d9e0"
RAMP = ["#9dc9e8", "#5ba3d0", "#2b7fb8", "#0072B2", "#004c78"]

HV_DIR = os.path.expanduser("~/x17/response_sim/hv_slope")
T14 = os.path.expanduser("~/x17/response_sim/stageB_w2/t14_compare/t14_summary.json")
FID = dict(x=(-190.0, 115.0), y=(-190.0, 165.0))
V0 = 490.0


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


def fid(d):
    return d[(d.ref_x > FID["x"][0]) & (d.ref_x < FID["x"][1])
             & (d.ref_y > FID["y"][0]) & (d.ref_y < FID["y"][1])]


# ── the gain-side argument ───────────────────────────────────────────────────
def townsend_fit(v, g):
    """ln G = A' exp(-B'/V)  =>  ln(ln G) linear in 1/V. Two parameters, and
    it is the right functional form to carry a gas off its own voltage range
    (a straight line in ln G is not)."""
    y = np.log(np.log(g))
    s, c = np.polyfit(1.0 / v, y, 1)
    return float(np.exp(c)), float(-s)          # A', B'


def gas_axis(ratio_x, slope_data10):
    """Interpolate the two simulated gases in isobutane fraction and ask what
    iso fraction would fix the gain, and what would fix the slope."""
    v5, g5, _ = sim_ladder(os.path.join(SIM_DIR, SIM_HVSCAN), DRY)
    v10, g10, _ = sim_ladder(os.path.join(SIM_DIR, SIM_HVSCAN), ISO10)
    A5, B5 = townsend_fit(v5, g5)
    A10, B10 = townsend_fit(v10, g10)

    def at(iso, V=V0):
        f = (iso - 5.0) / 5.0
        A, B = A5 + f * (A10 - A5), B5 + f * (B10 - B5)
        lng = A * np.exp(-B / V)
        return lng, lng * B / V ** 2 * 10.0     # ln G, slope per 10 V

    lng_sim, sl_sim = at(5.0)
    lng_need = lng_sim + np.log(1.0 / ratio_x)
    iso = np.linspace(1.0, 12.0, 2201)
    lng, sl = np.array([at(i) for i in iso]).T
    i_gain = float(np.interp(-lng_need, -lng, iso))       # lng decreasing in iso
    i_slope = float(np.interp(slope_data10, sl[::-1], iso[::-1])) \
        if sl.min() <= slope_data10 <= sl.max() else float("nan")
    return dict(A5=A5, B5=B5, A10=A10, B10=B10,
                lng_sim=float(lng_sim), slope_sim10=float(sl_sim),
                lng_need=float(lng_need),
                gain_sim=float(np.exp(lng_sim)), gain_need=float(np.exp(lng_need)),
                iso_for_gain=i_gain, slope_at_iso_for_gain=float(at(i_gain)[1]),
                iso_for_slope=i_slope,
                gain_at_iso10=float(np.exp(at(10.0)[0])),
                slope_at_iso10=float(at(10.0)[1]),
                gain_at_iso7=float(np.exp(at(7.0)[0])),
                iso_grid=iso.tolist(), lng_grid=lng.tolist(),
                slope_grid=sl.tolist())


# ── the X/Y question ─────────────────────────────────────────────────────────
def xy_check():
    d = fid(pd.read_parquet(os.path.join(HV_DIR, "peaks_longrun.parquet")))
    t = json.load(open(T14))["views"]
    out = dict(theta_windows=json.load(open(T14))["theta_windows_deg"], views={})
    for v in ("x", "y"):
        g = d[d.view == v]
        out["views"][v] = dict(
            n=int(len(g)), p50=float(np.median(g.peak_amp)),
            sat3500=float((g.peak_amp >= 3500).mean()),
            t14_p50=float(t[v]["peak_amp_med"]["data"]),
            t14_sat3500=float(t[v]["sat_frac_3500"]["data"]),
            t14_sim=float(t[v]["peak_amp_med"]["sim"]),
            t14_ratio=float(t[v]["peak_amp_med"]["ratio"]))
        out["views"][v]["ratio_vs_unselected"] = (
            out["views"][v]["t14_sim"] / out["views"][v]["p50"])
    # first-2500 cap and within-run drift
    for v in ("x", "y"):
        g = d[d.view == v].sort_values("event_id").reset_index(drop=True)
        idx = np.linspace(0, len(g), 9).astype(int)
        oct_ = [float(np.median(g.peak_amp[i:j]))
                for i, j in zip(idx[:-1], idx[1:])]
        out["views"][v]["octiles"] = oct_
        out["views"][v]["cap2500_ratio"] = float(
            np.median(g.peak_amp[:2500]) / np.median(g.peak_amp))
    return d, out


def angle_join(d):
    """Peak amplitude against the M3 REFERENCE angle (reference side, allowed)."""
    from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
    setup_paths()
    from M3RefTracking import M3RefTracking, get_xy_angles
    cfg = get_config("sat_det3")
    cfg.BASE_PATH = "/home/dylan/x17/cosmic_bench/det3/"
    cfg.RUN = "mx17_det3_saturday_scan_6-27-26"
    cfg.SUB_RUN = "long_run_resist_490V_drift_1000V"
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    ax_, ay_, evn = get_xy_angles(rays.ray_data)
    ang = pd.DataFrame(dict(event_id=np.asarray(evn).astype("int64"),
                            thx=np.degrees(np.arctan(np.asarray(ax_, float))),
                            thy=np.degrees(np.arctan(np.asarray(ay_, float)))))
    return d.merge(ang, on="event_id")


def figures(out_dir, ive, gx, xy, dang):
    figs = []

    def _save(fig, name, cap):
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", name), dpi=130,
                    facecolor="white")
        plt.close(fig)
        figs.append((name, cap))

    HV, VM = IV.measured()
    SIG = np.hypot(IV.SIG_REL * VM, IV.SIG_FLOOR)
    curves = IV.load_curves()

    # ── 1: v(E) — the curves, and the (non-)discrimination ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    prof = ive["results"]["ge500_profileB"]
    for c, iso in zip(RAMP, (3.0, 5.0, 7.0, 10.0, 15.0)):
        r = prof.get(f"{iso}") or prof.get(f"{iso:.1f}")
        if r is None:
            continue
        key = (round(r["iso"], 2), round(r["h2o"], 3), round(r["other"], 3))
        cur = curves[key]
        g = r["gap_tied"]
        E = np.linspace(60, 420, 300)
        ax.plot(E * (g / 10.0), r["scale_tied"] * np.interp(E, cur["E"], cur["V"]),
                "-", color=c, lw=1.8,
                label=f"iso {iso:g} % + {r['h2o']:.2f} % H2O  (gap {g:.1f} mm)")
    ax.errorbar(HV, VM, yerr=SIG, fmt="o", color=INK, ms=6, capsize=3, zorder=5,
                label="measured (forward fit)")
    ax.set_xlim(200, 1200)
    _style(ax, "drift voltage [V]", "drift velocity [µm/ns]",
           "Measured v(HV) against Magboltz, each mixture at its own best gap")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    for c, (tag, lab, key) in zip(
            (C_DATA, C_SIM, C_ALT),
            (("A", "gap free 24–32 mm", "_profile"),
             ("B", "gap tied to the measured 758 ns column", "_profileB"),
             ("C", "gap fixed 30 mm, no v-scale (June convention)", "_profileC"))):
        p = ive["results"]["ge500" + key]
        chi_key = {"A": "chi2", "B": "chi2_tied", "C": "chi2_fixed"}[tag]
        iso = sorted(float(k) for k in p)
        ch = np.array([p[f"{k:.1f}"][chi_key] if f"{k:.1f}" in p else p[str(k)][chi_key]
                       for k in iso])
        m = np.array(iso) <= 10.5
        ax.plot(np.array(iso)[m], ch[m] - ch[m].min(), "o-", color=c, lw=1.8,
                ms=5, label=f"{tag}: {lab}")
    ax.axhspan(0, 4, color=MUTED, alpha=0.12, lw=0)
    ax.text(3.2, 4.3, "Δχ² < 4  (not distinguishable)", color=MUTED, fontsize=8.5)
    ax.set_ylim(-0.5, 26)
    _style(ax, "isobutane fraction [%]", "Δχ² from the best mixture",
           "How much v(E) can say about isobutane (≥500 V points)")
    ax.legend(fontsize=8.5, frameon=False)
    _save(fig, "iso_ve.png",
          "LEFT — the measured forward-fit v(HV) ladder with the best Magboltz "
          "mixture at each isobutane fraction, each shown at the drift gap its "
          "own fit prefers. Between 3 % and 10 % isobutane the curves are "
          "interchangeable once the gap is allowed to move; only 15 % is "
          "visibly wrong. RIGHT — Δχ² against isobutane fraction under three "
          "treatments of the drift gap. Under the June convention (C, gap "
          "pinned at exactly 30 mm) there is apparent preference; under either "
          "honest treatment of the gap there is none. Nothing in 3–10 % is "
          "excluded by v(E).")

    # ── 2: the gain-side argument ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    iso = np.array(gx["iso_grid"])
    lng = np.array(gx["lng_grid"])
    sl = np.array(gx["slope_grid"])
    ax = axes[0]
    ax.plot(iso, np.exp(lng), "-", color=C_SIM, lw=2)
    ax.axhline(gx["gain_need"], color=C_DATA, lw=1.6, ls="--")
    ax.text(10.6, gx["gain_need"] * 1.25, "gain the bench actually has",
            color=C_DATA, fontsize=9, ha="right")
    ax.plot([5.0], [gx["gain_sim"]], "s", color=C_SIM, ms=9, zorder=5)
    ax.annotate("the modelled 95/5", (5.0, gx["gain_sim"]),
                textcoords="offset points", xytext=(10, -22), fontsize=9,
                color=INK)
    ax.plot([gx["iso_for_gain"]], [gx["gain_need"]], "o", color=C_DATA, ms=9,
            zorder=5)
    ax.set_yscale("log")
    _style(ax, "isobutane fraction [%]", f"gain at {V0:.0f} V",
           "Richer isobutane moves the gain the WRONG way")
    ax = axes[1]
    ax.plot(iso, sl, "-", color=C_SIM, lw=2, label="sim, interpolated in iso")
    ax.axhline(0.449, color=C_DATA, lw=1.6, ls="--")
    ax.text(11.8, 0.455, "measured slope", color=C_DATA, fontsize=9, ha="right")
    ax.axvline(gx["iso_for_gain"], color=MUTED, lw=1.2, ls=":")
    ax.text(gx["iso_for_gain"] + 0.15, 0.20,
            f"iso that fixes the gain\n({gx['iso_for_gain']:.1f} %) leaves the\n"
            f"slope at {gx['slope_at_iso_for_gain']:.3f}", color=INK, fontsize=8.5)
    ax.set_ylim(0.1, 0.50)
    _style(ax, "isobutane fraction [%]",
           f"d ln G / dV at {V0:.0f} V  [per 10 V]",
           "…and never reaches the measured slope")
    _save(fig, "iso_gain.png",
          "The isobutane axis on the GAIN side, from a two-parameter Townsend "
          "fit (ln G = A' e^(−B'/V)) to each simulated gas, interpolated "
          "linearly in isobutane fraction. LEFT — gain at 490 V falls steeply "
          "with isobutane, so a richer mixture makes the T14 amplitude deficit "
          f"deeper, not shallower; matching the bench's gain wants "
          f"{gx['iso_for_gain']:.1f} % isobutane, i.e. LESS than 5 %. RIGHT — "
          "the slope at 490 V never gets near the measured 0.449 anywhere in "
          "1–12 %, and the fraction that fixes the gain leaves the slope at "
          f"{gx['slope_at_iso_for_gain']:.3f}. The two requirements pull in "
          "opposite directions: no isobutane fraction satisfies both. "
          "(Extrapolation between two simulated gases — the S3 iso ladder is "
          "the proper test.)")

    # ── 3: the X/Y question ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8)]
    ctr = [0.5 * (a + b) for a, b in bins]
    for v, col, c in (("x", "thx", C_DATA), ("y", "thy", C_ALT)):
        g = dang[dang.view == v]
        med = [np.median(g[np.abs(g[col]).between(a, b)].peak_amp) for a, b in bins]
        ax.plot(ctr, med, "o-", color=c, ms=6, lw=1.8, label=f"{v.upper()} view")
    wx = xy["theta_windows"]["x"][1]
    wy = xy["theta_windows"]["y"][1]
    ax.axvline(wx, color=MUTED, lw=1.2, ls="--")
    ax.axvline(wy, color=MUTED, lw=1.2, ls=":")
    ax.text(wx + 0.1, 2300, f"T14 X window\n|θ| < {wx:.1f}°", color=MUTED, fontsize=8)
    ax.text(wy + 0.1, 2300, f"T14 Y window\n|θ| < {wy:.1f}°", color=MUTED, fontsize=8)
    ax.set_ylim(2200, 3100)
    _style(ax, "|M3 reference angle| [deg]", "median peak amplitude [ADC]",
           "Amplitude is flat in angle over both T14 windows")
    ax.legend(fontsize=9, frameon=False, loc="lower left")

    ax = axes[1]
    lbl = ["X view", "Y view"]
    det = [xy["views"][v]["sat3500"] for v in ("x", "y")]
    t14 = [xy["views"][v]["t14_sat3500"] for v in ("x", "y")]
    xpos = np.arange(2)
    ax.bar(xpos - 0.19, det, 0.34, color=C_DATA, label="detector (no reco cut)")
    ax.bar(xpos + 0.19, t14, 0.34, color=C_SIM, label="T14 data leg")
    for i, (a, b) in enumerate(zip(det, t14)):
        ax.text(i - 0.19, a + 0.008, f"{a:.2f}", ha="center", fontsize=9, color=INK)
        ax.text(i + 0.19, b + 0.008, f"{b:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(xpos)
    ax.set_xticklabels(lbl)
    ax.set_ylim(0, 0.42)
    _style(ax, None, "fraction of events with peak ≥ 3500 ADC",
           "What T14's per-view reco selection removes")
    ax.legend(fontsize=9, frameon=False)
    _save(fig, "xy_window.png",
          "LEFT — median peak amplitude against the M3 REFERENCE angle (not a "
          "reconstructed one) in the 490 V long run. It is flat to ±3 % from 0 "
          "to 8°, and both T14 angle windows sit inside that flat region, so "
          "the window cannot be what makes T14's X and Y differ. RIGHT — what "
          "can: T14's per-view reconstruction-quality selection removes "
          "saturated events, and removes three times more of them in Y "
          "(0.33 → 0.11) than in X (0.33 → 0.26). The Y data leg is therefore "
          "a low-amplitude-biased subsample of the detector.")
    return figs


CSS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_report.css")).read() if os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_report.css")) else """
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
figure{margin:22px 0} img{width:100%;border:1px solid #e6e9ee;border-radius:6px}
figcaption{color:#6a7583;font-size:12.5px;margin-top:7px}
code{background:#f2f4f7;padding:1px 5px;border-radius:3px;font-size:12.5px}
ul{margin:8px 0 8px 20px}li{margin:4px 0}
a{color:#0072B2}
"""


def build_html(ive, gx, xy, figs):
    e = html.escape
    figblocks = "\n".join(
        f'<figure><img src="figures/{n}" alt="{e(c[:80])}">'
        f"<figcaption><b>{n}</b> — {e(c)}</figcaption></figure>"
        for n, c in figs)

    def prof_table(setname):
        rows = []
        pa = ive["results"][setname + "_profile"]
        pb = ive["results"][setname + "_profileB"]
        pc = ive["results"][setname + "_profileC"]
        amin = min(r["chi2"] for r in pa.values())
        bmin = min(r["chi2_tied"] for r in pb.values())
        cmin = min(r["chi2_fixed"] for r in pc.values())
        for k in sorted(pa, key=float):
            if float(k) > 10.5:
                continue
            a, b, c = pa[k], pb.get(k), pc.get(k)
            rows.append(
                f"<tr><td>{float(k):.0f} %</td><td>{a['h2o']:.2f}</td>"
                f"<td>{a['chi2'] - amin:.1f}</td><td>{a['gap_mm']:.1f}</td>"
                f"<td>{b['chi2_tied'] - bmin:.1f}</td><td>{b['gap_tied']:.1f}</td>"
                f"<td>{c['chi2_fixed'] - cmin:.1f}</td></tr>")
        return "".join(rows)

    hv = json.load(open(os.path.join(HV_DIR, "slopes.json")))
    _p7 = ive["results"]["ge700_profileC"]
    _c0 = min(r["chi2_fixed"] for r in _p7.values())
    c7 = {float(k): r["chi2_fixed"] - _c0 for k, r in _p7.items()}
    v = xy["views"]
    wx, wy = xy["theta_windows"]["x"], xy["theta_windows"]["y"]
    sd = hv["data"]["x"]["p50_head"]["slope10"]

    return f"""<!doctype html><meta charset="utf-8">
<title>Isobutane follow-up to the HV-slope test — det3</title>
<style>{CSS}</style>
<h1>Is the gain-slope gap an isobutane problem?</h1>
<p class="sub">Follow-up to the HV-slope test (<a href="../report.html">../report.html</a>),
which found det3's gain rising {sd:.3f} per 10 V against the simulation's
{hv['sim']['dry_head']['slope10']:.3f} — a factor {sd / hv['sim']['dry_head']['slope10']:.2f}.
det3 · saturday scan 6-27-26 · 2026-08-09</p>

<div class="verdict"><b>v(E) CANNOT DECIDE IT — the mixture axis stays open on
the drift-velocity side.</b> Once the drift gap is treated as the unknown it
actually is, every isobutane fraction from 3 % to 10 % fits the measured v(E)
equally well (Δχ² ≤ 5 in every configuration tried, with the ordering flipping
between point sets). Only ≥ 15 % is excluded. The June conclusion that "iso 5 %
is clearly preferred" is reproducible <i>only</i> under the convention gap =
30.0 mm exactly with no velocity-scale freedom — and det3's own gap study says
the gap is 27.9 ± 1.0 mm, not 30.</div>

<div class="verdict ok"><b>BUT THE ISOBUTANE HYPOTHESIS IS DISFAVOURED ANYWAY —
by the gain ladder, not by v(E).</b> Richer isobutane <i>lowers</i> gain steeply
at fixed voltage, so it deepens the T14 amplitude deficit instead of curing it:
matching the bench's actual gain wants <b>{gx['iso_for_gain']:.1f} %</b>
isobutane, i.e. <i>less</i> than 5 %, and that fraction leaves the slope at
<b>{gx['slope_at_iso_for_gain']:.3f}</b> per 10 V — no better than the
{gx['slope_sim10']:.3f} we started from. Gain and slope pull in opposite
directions along the isobutane axis, so no fraction satisfies both.
<span style="font-weight:400">(The {gx['slope_sim10']:.3f} here is the
Townsend-form slope at exactly 490 V; the HV-slope report's
{hv['sim']['dry_head']['slope10']:.3f} is a straight-line fit over 460–490 V.
Same quantity, 5 % apart from the functional form.)</span></div>

<h2>1. The v(E) test</h2>
<h3>What was confronted with what</h3>
<p><b>Measured.</b> The June waveform-first forward-fit v(HV) ladder — 300, 500,
700, 900, 1000, 1100 V — on this same detector and this same saturday scan
(<code>WAVEFORM_FIRST_THREADING.md</code> §14/§17,
v(1000 V) = 36.7 ± 0.3 (fit) ± 0.9 (model) µm/ns). Waveform-derived throughout;
no <code>combined_hits</code> time is read anywhere.</p>
<p><b>Modelled.</b> All {64} Ar/isobutane Magboltz drift-velocity grids in
<code>garfield_sim/results/</code> — the iso 3–8 % × H₂O 0.4–1.1 % 2-D grid, the
iso 10 % × H₂O 0–3 % contamination suite from the July study, and the dry
15/20 % points — put on the bench pressure by the exact E/N scaling
v<sub>p</sub>(E) = v<sub>p₀</sub>(E·p₀/p).</p>

<h3>Why the drift gap is the whole story</h3>
<p>The forward-fit velocity is <b>gap-free</b>: it comes from arrival time
against a depth scale set by the M3 reference track angle, not from filling a
nominal gap. The gap enters only through the field axis, E = V<sub>drift</sub> /
gap — and there it is a real unknown. det3's cathode is dished: the endpoint map
reads 25.7–29.2 mm across the surface and 27.9 ± 0.1 (stat) ± 1.0 (calib) mm
overall, against a 30 mm mechanical nominal and a det2 control that reads its
full 30.5 mm (<code>mx_june_wft/GAP_STUDY_2026-07-30.md</code>). A 30-vs-28 mm
choice moves every field by 7 %, which is the same size as the difference
between the mixtures being tested.</p>
<p>So the fit is run three ways, declared in advance:</p>
<ul>
<li><b>A — gap free</b> over 24–32 mm. The conservative reading: assume only
that the gap is somewhere physical.</li>
<li><b>B — gap tied to det3's own measured charge column.</b> The column
endpoint is a <i>time</i>, 757.5 ± 2.2 ns at 1000 V, measured on this subrun
before any velocity is applied; the published 27.9 mm is that time times a
velocity, so the ±1.0 mm "calibration" systematic <i>is</i> the velocity
ambiguity and must not be counted twice. Here each mixture supplies its own
velocity and the gap follows from the fixed point gap = v(1000 V/gap)·T.</li>
<li><b>C — gap fixed at 30.0 mm with no velocity-scale freedom</b>: the June
convention, included to show where the old conclusion came from.</li>
</ul>
<p>A common velocity scale is carried as a nuisance with a 2.6 % Gaussian prior
in A and B — the size of the calibration's own model bias measured by the June
toy closure (a deliberately wrong sharing truth deflated v by 2.6 %).</p>

<h3>Result</h3>
<table><tr><th>isobutane</th><th>best H₂O</th>
<th>Δχ² (A)</th><th>gap (A)</th><th>Δχ² (B)</th><th>gap (B)</th>
<th>Δχ² (C)</th></tr>
{prof_table("ge500")}</table>
<p class="sub">Points ≥ 500 V (the 300 V point sits on a truncated window and is
shown separately in <code>iso_ve.json</code>); water and every other contaminant
profiled out at each isobutane fraction; gaps in mm.</p>
<p>Under A and B nothing in 3–10 % is separated by more than Δχ² ≈ 7, and the
<i>ordering flips between point sets</i> — the ≥ 500 V points mildly prefer
10 %, the ≥ 700 V points mildly prefer 4–5 %. A preference that reverses when
one measured point is added is not a measurement. Under C the June ordering
does come back, and most sharply in exactly the June configuration — gap 30 mm,
no v-scale, ≥ 700 V only — where 5 % wins by Δχ² = {c7[4.0]:.1f} over 4 %,
{c7[7.0]:.1f} over 7 % and {c7[10.0]:.1f} over 10 %. That is the whole basis of
"iso 5 % clearly preferred", and it does not survive letting the gap be what
det3's own endpoint map says it is. Tightening the per-point error from 3 % to 1 % does not rescue the
test either: it sharpens the ≥ 500 V preference toward <i>higher</i> isobutane
and the ≥ 700 V preference toward <i>lower</i>, i.e. it sharpens the
contradiction rather than the answer.</p>
<p>Two things sharpen this rather than soften it. First, <b>the water grid is
not evenly sampled</b>: 5 % isobutane has 30 mixtures behind it (H₂O stepped at
0.05 % from the June campaign) while 3, 4, 6, 7 and 8 % have four each and 10 %
has twelve. Profiling water out of a seven-times-finer grid can only push the
5 % minimum down, so every variant is <i>biased toward 5 %</i> — and 5 % still
does not win under A or B. Second, variant C's ordering is visibly jagged
(6 % and 7 % spike while 4, 5 and 8 % sit near zero), which is that same
uneven sampling showing through: under a convention with no freedom to absorb
it, whether a mixture has the right water point in the grid decides its χ².</p>
<p>The one thing v(E) does say cleanly: <b>≥ 15 % isobutane is excluded</b>
(Δχ² &gt; 55 in every configuration) — though only as <i>dry</i> 15 %, since no
wet 15 % grid exists.</p>

<h2>2. Does anything on record pin the June mixture?</h2>
<div class="verdict"><b>No. Nothing constrains the June isobutane fraction —
the same epistemic status as the humidity.</b></div>
<p>What was checked, and what is there:</p>
<ul>
<li><b>Run configs.</b> All 22 <code>run_config.json</code> files on the bench
disk carry one free-text field, <code>"gas": "Ar/Iso 95/5"</code> (20 runs; the
other two read He/ethane and Ar/CF₄). It is an operator label, not a
measurement — there is no flow, mixture, pressure or bottle field anywhere in
the schema.</li>
<li><b>The mass-flow mixer exists, but not yet in June.</b> The DAQ has a real
two-channel Bronkhorst MFC mixer that sets the isobutane percentage directly and
logs both channels to a per-day CSV
(<code>nTof_x17_DAQ/gas_mixer_control/</code>). It was commissioned in commits
dated <b>2026-07-07</b>, ten days after the saturday scan, and its log directory
<code>~/beam_july/slow_control/gas_flow</code> is empty on this machine. The
argon bottle-pressure log in the same repo also starts 2026-07-07.</li>
<li><b>No certificate, no e-log.</b> No bottle certificate, gas analysis or
mixture note appears anywhere in <code>nTof_x17</code>,
<code>nTof_x17_DAQ</code> or <code>MX17_Documentation</code>.</li>
<li>What the June bench was actually fed — a pre-mixed bottle or a hand-set
mix — is not recorded in anything reachable from here, and I have not guessed.</li>
</ul>
<div class="verdict info"><b>Actionable:</b> from 2026-07-07 onward the
isobutane fraction is a <i>set and logged</i> quantity. A bench mesh-voltage
ladder taken now, at a known set mixture, would convert the isobutane axis from
unconstrained to constrained — and, taken at two set mixtures, would measure
d ln G/dV against isobutane directly instead of interpolating two simulated
gases. That is a few hours of bench time and it settles the axis.</div>

<h2>3. The T14 X/Y difference — it is not the angle window</h2>
<p>The HV-slope report noted that the frozen T14 data leg reads X = {v['x']['t14_p50']:.0f}
and Y = {v['y']['t14_p50']:.0f} ADC while this ladder gives the two views nearly
equal, and attributed it to the per-view θ window. <b>That attribution was
wrong.</b> Measured properly:</p>
<table><tr><th></th><th>X view</th><th>Y view</th></tr>
<tr><td>detector, 490 V long run, no reco selection — median peak</td>
<td>{v['x']['p50']:.1f}</td><td>{v['y']['p50']:.1f}</td></tr>
<tr><td>same, fraction with peak ≥ 3500 ADC</td>
<td>{v['x']['sat3500']:.3f}</td><td>{v['y']['sat3500']:.3f}</td></tr>
<tr><td>T14 data leg — median peak</td>
<td>{v['x']['t14_p50']:.1f}</td><td>{v['y']['t14_p50']:.1f}</td></tr>
<tr><td>T14 data leg — fraction ≥ 3500 ADC</td>
<td>{v['x']['t14_sat3500']:.3f}</td><td>{v['y']['t14_sat3500']:.3f}</td></tr>
<tr><td>T14 selection / detector</td>
<td>{v['x']['t14_p50'] / v['x']['p50']:.3f}</td>
<td>{v['y']['t14_p50'] / v['y']['p50']:.3f}</td></tr></table>
<p><b>The detector's two views are identical</b> — {v['x']['p50']:.1f} vs
{v['y']['p50']:.1f} ADC, 0.02 % apart, same subrun, same
<code>FeuReader</code>, same M3 fiducial, no threshold. So the 16 % gap in the
T14 tables is made by T14's selection, and three candidate causes are ruled
out:</p>
<ul>
<li><b>The θ window is not it.</b> Median peak amplitude is flat to ±3 % against
the <i>M3 reference</i> angle from 0 to 8°, and both windows
(X ±{wx[1]:.2f}°, Y ±{wy[1]:.2f}°) sit inside that flat region. Applying either
window to either view moves the median by ≤ 2 %.</li>
<li><b>The 2500-event cap is not it.</b> The first 2500 event ids give
{v['x']['cap2500_ratio']:.3f} (X) and {v['y']['cap2500_ratio']:.3f} (Y) of the
full-sample median.</li>
<li><b>Gain drift within the long run is not it.</b> Octiles of event id span
{min(v['x']['octiles']):.0f}–{max(v['x']['octiles']):.0f} ADC in X and
{min(v['y']['octiles']):.0f}–{max(v['y']['octiles']):.0f} in Y, i.e. ±3 % with
no trend.</li>
</ul>
<p>What is left is the per-view reconstruction-quality selection
(<code>{{view}}_ok &amp; {{view}}_quality_ok</code>), and the saturation
fractions show its signature plainly: the detector saturates equally in both
views ({v['x']['sat3500']:.2f} / {v['y']['sat3500']:.2f} of events at
≥ 3500 ADC), but the T14 sample keeps {v['x']['t14_sat3500']:.2f} in X and only
{v['y']['t14_sat3500']:.2f} in Y. <b>The forward fit fails on railed waveforms,
and it fails on them three times more often in Y.</b> T14's Y data leg is
therefore a low-amplitude-biased subsample of the detector, and its Y amplitude
ratio is optimistic: against the unselected detector the Y ratio would be
{v['y']['ratio_vs_unselected']:.3f} rather than {v['y']['t14_ratio']:.3f}
(X barely moves, {v['x']['ratio_vs_unselected']:.3f} vs {v['x']['t14_ratio']:.3f}).</p>
<div class="verdict info"><b>Followed up:</b> the per-view SHAPE targets
(undershoot, FWHM, rise) were measured on these same biased legs, and are
re-derived unbiased in <a href="../xy_shape/report.html">xy_shape/report.html</a>
— the Y undershoot target moves from −12.0 % to −9.3 %, and the legs are fully
accounted for by the θ window plus the rail depletion.</div>
<div class="verdict info"><b>For the T14 owner:</b> this is a like-for-like
question, not a correction — the sim leg passes through the same quality cut,
and the sim saturates far less ({json.load(open(T14))['views']['y']['sat_frac_3500']['sim']:.3f}
in Y), so there is little on the sim side for the cut to remove and the bias
probably does not cancel. Worth checking the sim leg's quality acceptance before
quoting the Y ratio. Nothing in <code>t14_compare/</code> was regenerated here.</div>

<h2>Figures</h2>
{figblocks}

<h2>What this does not rule out</h2>
<ul>
<li><b>It does not clear isobutane on the gain side.</b> The gain argument
interpolates a two-parameter Townsend fit between exactly two simulated gases
(95/5 and 90/10) and extrapolates the 90/10 ladder 40 V below its own range.
It is an ordering argument. The S3 iso ladder measuring d ln G/dV directly at
92.5/7.5 and 90/10 is the real test and supersedes this.</li>
<li><b>It does not measure the drift gap.</b> Variant B assumes the charge
column spans the full mechanical depth — supported by the det2 control reading
its full 30.5 mm, but not proven for det3, whose short column is the very thing
under discussion. A mechanical measurement of det3's cathode plane would break
the whole degeneracy at once.</li>
<li><b>It does not exclude water, N₂ or air.</b> Those were profiled out as
nuisances, never constrained. Trace N₂ remains degenerate at the 0.05 µm/ns
level, as it was in June.</li>
<li><b>It does not test any mixture the grids do not contain.</b> There is no
Ar/iso/H₂O grid above 8 % isobutane at bench pressure, and 15 % and 20 % exist
only as dry, coarse (ncoll = 2) points — so "≥ 15 % excluded" is really "≥ 15 %
<i>dry</i> excluded"; a wet 15 % was never computed.</li>
<li><b>The v(E) ladder is one detector, one day, six points.</b> Two of the six
(300 and 500 V) carry window-truncation caveats, and the whole ladder shares one
reconstruction and one calibration bundle.</li>
</ul>

<h2>Reproducing</h2>
<pre><code>python3 mx17_sim_wft/hv_slope/extract.py --subrun long_run_resist_490V_drift_1000V 490 \\
    --out ~/x17/response_sim/hv_slope/peaks_longrun.parquet
python3 mx17_sim_wft/hv_slope/iso_ve.py
python3 mx17_sim_wft/hv_slope/make_iso_report.py</code></pre>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HV_DIR, "iso_ve"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.dir, "figures"), exist_ok=True)

    ive = json.load(open(os.path.join(a.dir, "iso_ve.json")))
    hv = json.load(open(os.path.join(HV_DIR, "slopes.json")))
    t14 = json.load(open(T14))
    gx = gas_axis(t14["views"]["x"]["peak_amp_med"]["ratio"],
                  hv["data"]["x"]["p50_head"]["slope10"])
    d, xy = xy_check()
    dang = angle_join(d)
    figs = figures(a.dir, ive, gx, xy, dang)

    with open(os.path.join(a.dir, "gas_axis.json"), "w") as f:
        json.dump({k: v for k, v in gx.items() if not k.endswith("_grid")},
                  f, indent=1)
    with open(os.path.join(a.dir, "xy_check.json"), "w") as f:
        json.dump(xy, f, indent=1)
    p = os.path.join(a.dir, "report.html")
    with open(p, "w") as f:
        f.write(build_html(ive, gx, xy, figs))
    print(json.dumps({k: round(v, 4) for k, v in gx.items()
                      if isinstance(v, float)}, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
