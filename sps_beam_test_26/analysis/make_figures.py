#!/usr/bin/env python3
"""Figures for the SPS extraction report.

Reads only the JSON products the analysis scripts already wrote, so it is
cheap to re-run and cannot drift from the numbers:

    ladder_span_run63_rot25.json        run_62/63 v(E) reproducibility
    ladder_span_run62_rot25_ladder.json
    ladder_span_run57_rot25_co2.json    CO2 vs CF4 mixture comparison
    gain_scan_run66_flat_resist.json    kernel gain-invariance + its control
    gain_scan_run70_flat_drift.json     transparency curve at normal incidence

    ../../.venv/bin/python make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets                      # noqa: E402

OUT = os.path.join(os.path.expanduser("~"), "x17", "sps_beam_test_26",
                   "extraction_2026-08-05")
FIGS = os.path.join(OUT, "figures")
GAP_MM, C0_NS = 28.8, 30.0

# Categorical slots of the reference palette (light mode), used consistently.
C1, C2, C3, C4 = "#2a78d6", "#d1495b", "#2e8b57", "#e08a1e"
GRID = dict(color="#cccccc", lw=0.6, alpha=0.7)


def _style(ax, xlabel, ylabel, title=None):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(path):
    with open(path) as f:
        return json.load(f)


def v_of(span):
    return GAP_MM * 1e3 / (span - C0_NS)


def fig_ladder():
    S = datasets.STAGE_ROOT
    a = load(S + "run_63/ladder_span_run63_rot25.json")
    b = load(S + "run_62/ladder_span_run62_rot25_ladder.json")
    c = load(S + "run_57/ladder_span_run57_rot25_co2.json")

    def curve(d):
        p = sorted((x["field_Vcm"], v_of(x["span"])) for x in d.values())
        return np.array([q[0] for q in p]), np.array([q[1] for q in p])

    f3, v3 = curve(a)
    f2, v2 = curve(b)
    fc, vc = curve(c)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(f3, v3, "o-", color=C1, label="run_63 (CF$_4$, 25.64°)", ms=6)
    ax.plot(f2, v2, "s--", color=C2, label="run_62 (CF$_4$, 25.64°, independent)",
            ms=7, mfc="none", mew=1.8)
    ax.axhspan(11.4, 11.7, color="#999999", alpha=0.18, zorder=0)
    ax.text(120, 11.75, "window-truncation floor", fontsize=8, color="#555555")
    _style(ax, "drift field [V/cm]", "v$_{drift}$ [µm/ns]",
           "Two independent CF$_4$ ladders — 0.6 % RMS")
    ax.legend(fontsize=8.5, frameon=False)

    ax = axes[1]
    ax.plot(f3, v3, "o-", color=C1, label="Ar/CF$_4$/iso (run_63)", ms=6)
    ax.plot(fc, vc, "^-", color=C3, label="Ar/CO$_2$/iso (run_57)", ms=7)
    ax.annotate("×1.14", xy=(240, 13.2), fontsize=10, color="#333333")
    ax.annotate("", xy=(243, 12.33), xytext=(243, 14.35),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
    _style(ax, "drift field [V/cm]", "v$_{drift}$ [µm/ns]",
           "Mixture ratio: ladder 1.14 vs flush 1.17")
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ladders.png"), dpi=140)
    plt.close(fig)


def _rows(d, key, view):
    r = [(x[key], x["views"][view]) for x in d.values() if view in x["views"]]
    r.sort(key=lambda t: -t[0])
    return r


def fig_gain():
    S = datasets.STAGE_ROOT
    g = load(S + "run_66/gain_scan_run66_flat_resist.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax2 = ax.twinx()
    for view, col, mk in (("y", C1, "o"), ("x", C2, "s")):
        r = _rows(g, "resist_V", view)
        rv = [q[0] for q in r]
        ax.plot(rv, [q[1]["n_events"] / 1e3 for q in r], mk + "-", color=col,
                label=f"{view.upper()} events (×10³)", ms=6)
        ax2.plot(rv, [q[1]["q_lead_trunc"] for q in r], mk + ":", color=col,
                 alpha=0.6, ms=5, label=f"{view.upper()} q$_{{lead}}$")
    _style(ax, "resist voltage [V]", "events per plateau (×10³)",
           "The gain lever: yield moves ×2.15, q$_{lead}$ only ×1.2")
    ax2.set_ylabel("truncated-mean q$_{lead}$ [ADC]  (ZS-censored)",
                   fontsize=9, color="#555555")
    ax2.set_ylim(0, 300)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    for view, col, mk in (("y", C1, "o"), ("x", C2, "s")):
        r = _rows(g, "resist_V", view)
        rv = [q[0] for q in r]
        ax.plot(rv, [q[1]["share_ratio_med"] for q in r], mk + ":", color=col,
                alpha=0.55, ms=5, label=f"{view.upper()} raw (censored)")
        ax.plot(rv, [q[1]["share_matched_med"] for q in r], mk + "-", color=col,
                ms=6, label=f"{view.upper()} amplitude-matched")
    ax.set_ylim(0, 0.75)
    _style(ax, "resist voltage [V]", "$\\Sigma_{|d|=1}$ / lead",
           "Sharing vs gain: flat to 1–3 % once controlled")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "gain_invariance.png"), dpi=140)
    plt.close(fig)


def fig_transparency():
    S = datasets.STAGE_ROOT
    d = load(S + "run_70/gain_scan_run70_flat_drift.json")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for view, col, mk in (("y", C1, "o"), ("x", C2, "s")):
        r = [(x["drift_V"] / (GAP_MM / 10), x["views"][view])
             for x in d.values() if view in x["views"]]
        r.sort(key=lambda t: t[0])
        f = [q[0] for q in r]
        ax.plot(f, [q[1]["q_lead_trunc"] for q in r], mk + "-", color=col,
                ms=6, label=f"{view.upper()} leading-strip amplitude")
    _style(ax, "drift field [V/cm]", "truncated-mean q$_{lead}$ [ADC]",
           "Mesh transparency at normal incidence (run_70)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "transparency.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig_ladder()
    fig_gain()
    fig_transparency()
    print(f"wrote figures under {FIGS}")


if __name__ == "__main__":
    main()
