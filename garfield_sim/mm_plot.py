#!/usr/bin/env python3
"""
mm_plot.py — Plot gain vs voltage / field results
==================================================
Reads JSON files from results/ and produces publication-quality figures.

Usage:
    python3 mm_plot.py [--outdir DIR] [--format png|pdf|svg]

Figures produced:
    1. gain_vs_voltage.png  — gain vs mesh voltage for all combinations
    2. gain_vs_field.png    — gain vs amplification field (V/cm)
    3. gain_log.png         — log-scale gain vs voltage (useful for exponential regime)
    4. pressure_ratio.png   — gain ratio Saclay/CERN vs voltage (per gas)
    5. gain_distributions/  — histograms of gain distribution at each voltage (optional)

Requires: matplotlib, numpy (no ROOT needed for plotting)
    pip3 install matplotlib numpy
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(__file__))
import mm_config as cfg

# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "legend.fontsize":   10,
    "lines.linewidth":   1.8,
    "lines.markersize":  5,
    "errorbar.capsize":  3,
})

# Colour + marker scheme: one per (gas, pressure) combination
# Keys: (gas_label_substring, pressure_label_substring) → (colour, marker)
STYLE = {
    ("He_C2H6",  "Saclay"): dict(color="#1f77b4", marker="o", ls="-"),
    ("He_C2H6",  "CERN"):   dict(color="#1f77b4", marker="s", ls="--"),
    ("Ar_iC4H10","Saclay"): dict(color="#d62728", marker="^", ls="-"),
    ("Ar_iC4H10","CERN"):   dict(color="#d62728", marker="v", ls="--"),
    # Ne/iC4H10 — three shades of green, one per rP value
    ("rP040",    "Saclay"): dict(color="#006d2c", marker="o", ls="-"),
    ("rP040",    "CERN"):   dict(color="#006d2c", marker="s", ls="--"),
    ("rP050",    "Saclay"): dict(color="#31a354", marker="^", ls="-"),
    ("rP050",    "CERN"):   dict(color="#31a354", marker="v", ls="--"),
    ("rP060",    "Saclay"): dict(color="#74c476", marker="D", ls="-"),
    ("rP060",    "CERN"):   dict(color="#74c476", marker="d", ls="--"),
}

def get_style(gas_label, pressure_label):
    for (gk, pk), sty in STYLE.items():
        if gk in gas_label and pk in pressure_label:
            return sty
    return dict(color="grey", marker="o", ls="-")

def pretty_label(gas_label, pressure_label):
    gas_map = {
        "He_C2H6_96p5_3p5":    "He/C₂H₆ 96.5/3.5%",
        "Ar_iC4H10_95_5":      "Ar/iC₄H₁₀ 95/5%",
        "Ne_iC4H10_95_5_rP040": "Ne/iC₄H₁₀ 95/5% (rP=0.40)",
        "Ne_iC4H10_95_5_rP050": "Ne/iC₄H₁₀ 95/5% (rP=0.50)",
        "Ne_iC4H10_95_5_rP060": "Ne/iC₄H₁₀ 95/5% (rP=0.60)",
    }
    prs_map = {
        "Saclay_160m": "Saclay 160 m",
        "CERN_450m":   "CERN 450 m",
    }
    g = gas_map.get(gas_label, gas_label)
    p = prs_map.get(pressure_label, pressure_label)
    return f"{g}  [{p}]"


# ── Load results ───────────────────────────────────────────────────────────────

def load_all_results(results_dir):
    results = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(fpath) as f:
            results.append(json.load(f))
    return results


# ── Plot helpers ───────────────────────────────────────────────────────────────

def errorbar_kw(sty):
    return dict(
        color=sty["color"],
        marker=sty["marker"],
        linestyle=sty["ls"],
        elinewidth=1.0,
        capsize=3,
    )


def add_pressure_lines(ax, results_dir):
    """Draw vertical dashed lines for the two pressure conditions."""
    # Nothing to draw — pressures show up as separate curves, not lines
    pass


def fit_exponential(volts, gains):
    """
    Fit gain = A * exp(B * V) via log-linear regression.
    Returns (A, B) or None if fewer than 3 valid points.
    """
    mask = gains > 0
    if mask.sum() < 3:
        return None
    try:
        B, logA = np.polyfit(volts[mask], np.log(gains[mask]), 1)
        return np.exp(logA), B
    except Exception:
        return None


# ── Figure 1 & 2: gain vs V and gain vs E (linear + log) ──────────────────────

def plot_gain(results, outdir, fmt):
    fig_lin, ax_lin = plt.subplots(figsize=(7, 5))
    fig_log, ax_log = plt.subplots(figsize=(7, 5))
    fig_e,   ax_e   = plt.subplots(figsize=(7, 5))

    ax_log.set_yscale("log")
    ax_e.set_yscale("log")

    # Collect per-dataset arrays for two-pass rendering
    datasets = []
    for res in results:
        gas    = res["gas"]
        plabel = res["pressure_label"]
        volts  = np.array(res["voltages"])
        fields = np.array(res["fields"])
        mean   = np.array(res["gain_mean"])
        std    = np.array(res["gain_std"])
        surv   = np.array(res["survival"])

        gain_raw = res.get("gain_raw", None)
        if gain_raw is not None:
            n = np.array([max(len(g), 1) for g in gain_raw], dtype=float)
        else:
            n = np.ones(len(mean))
        sem = std / np.sqrt(n)

        mask = (mean > 0) & (surv > 0.1)
        sty  = get_style(gas, plabel)
        datasets.append(dict(
            volts=volts, fields=fields, mean=mean, std=std, sem=sem,
            mask=mask, sty=sty, label=pretty_label(gas, plabel),
            ekw=errorbar_kw(sty), color=sty["color"],
        ))

    # Pass 1: errorbars + exponential fit lines — these drive autoscaling
    all_axes = (ax_lin, ax_log, ax_e)
    for d in datasets:
        mask  = d["mask"]
        x_v   = d["volts"][mask]
        x_e   = d["fields"][mask] / 1e3
        m     = d["mean"][mask]
        se    = d["sem"][mask]
        color = d["color"]

        for ax, x in [(ax_lin, x_v), (ax_log, x_v), (ax_e, x_e)]:
            ax.errorbar(x, m, yerr=se, label=d["label"], **d["ekw"])

        fit = fit_exponential(x_v, m)
        if fit is not None:
            A, B = fit
            v_fine = np.linspace(x_v.min(), x_v.max(), 300)
            e_fine = np.linspace(d["fields"][mask].min(),
                                 d["fields"][mask].max(), 300) / 1e3
            g_fine = A * np.exp(B * v_fine)
            fit_kw = dict(color=color, lw=1.2, alpha=0.65,
                          linestyle=d["sty"]["ls"], zorder=1.5)
            ax_lin.plot(v_fine, g_fine, **fit_kw)
            ax_log.plot(v_fine, g_fine, **fit_kw)
            ax_e.plot(e_fine, g_fine, **fit_kw)

    # Lock zoom based on errorbars + fits; fill_between must not expand limits
    for ax in all_axes:
        ax.relim()
        ax.autoscale_view()
    saved_lims = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in all_axes}

    # Pass 2: semi-transparent STD band (drawn behind data)
    for d in datasets:
        mask  = d["mask"]
        x_v   = d["volts"][mask]
        x_e   = d["fields"][mask] / 1e3
        m     = d["mean"][mask]
        s     = d["std"][mask]
        color = d["color"]
        for ax, x in [(ax_lin, x_v), (ax_log, x_v), (ax_e, x_e)]:
            ax.fill_between(x, np.maximum(m - s, 1e-10), m + s,
                            color=color, alpha=0.15, linewidth=0, zorder=1)

    # Restore limits so the STD band doesn't expand the view
    for ax, (xlim, ylim) in saved_lims.items():
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for ax in all_axes:
        ax.legend(loc="upper left", framealpha=0.85)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("Gas Gain")

    ax_lin.set_xlabel("Mesh Voltage (V)")
    ax_lin.set_title("Micromegas Gain vs Mesh Voltage")
    ax_log.set_xlabel("Mesh Voltage (V)")
    ax_log.set_title("Micromegas Gain vs Mesh Voltage (log scale)")
    ax_e.set_xlabel("Amplification Field (kV/cm)")
    ax_e.set_title("Micromegas Gain vs Amplification Field")

    for fig, name in [
        (fig_lin, "gain_vs_voltage"),
        (fig_log, "gain_vs_voltage_log"),
        (fig_e,   "gain_vs_field_log"),
    ]:
        fig.tight_layout()
        path = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(path)
        print(f"  Saved: {path}")
        plt.close(fig)


# ── Figure 3: pressure ratio ───────────────────────────────────────────────────

def plot_pressure_ratio(results, outdir, fmt):
    """
    For each gas, plot gain(Saclay) / gain(CERN) vs voltage.
    Ratio > 1 means higher gain at lower pressure (Saclay is lower than CERN
    in pressure if... wait — 160m < 450m means Saclay has HIGHER pressure).
    Actually 160m → higher P (closer to sea level) → lower gain.
    CERN at 450m → lower P → higher gain.
    """
    # Group by gas label
    gas_labels = list({r["gas"] for r in results})

    fig, axes = plt.subplots(1, len(gas_labels),
                             figsize=(6 * len(gas_labels), 5),
                             squeeze=False)

    for col, gas in enumerate(sorted(gas_labels)):
        ax = axes[0][col]
        res_by_pres = {r["pressure_label"]: r
                       for r in results if r["gas"] == gas}

        saclay_key = next((k for k in res_by_pres if "Saclay" in k), None)
        cern_key   = next((k for k in res_by_pres if "CERN"   in k), None)

        if saclay_key is None or cern_key is None:
            ax.text(0.5, 0.5, "Incomplete data", transform=ax.transAxes,
                    ha="center")
            continue

        rs = res_by_pres[saclay_key]
        rc = res_by_pres[cern_key]

        # Align on common voltages
        v_s = np.array(rs["voltages"])
        v_c = np.array(rc["voltages"])
        common = np.intersect1d(v_s, v_c)

        idx_s = [np.where(v_s == v)[0][0] for v in common]
        idx_c = [np.where(v_c == v)[0][0] for v in common]

        g_s = np.array(rs["gain_mean"])[idx_s]
        g_c = np.array(rc["gain_mean"])[idx_c]
        std_s = np.array(rs["gain_std"])[idx_s]
        std_c = np.array(rc["gain_std"])[idx_c]

        # Compute SEM from raw counts
        raw_s = rs.get("gain_raw", None)
        raw_c = rc.get("gain_raw", None)
        if raw_s is not None:
            n_s = np.array([max(len(raw_s[i]), 1) for i in idx_s], dtype=float)
        else:
            n_s = np.ones(len(idx_s))
        if raw_c is not None:
            n_c = np.array([max(len(raw_c[i]), 1) for i in idx_c], dtype=float)
        else:
            n_c = np.ones(len(idx_c))
        e_s = std_s / np.sqrt(n_s)
        e_c = std_c / np.sqrt(n_c)

        mask = (g_s > 0) & (g_c > 0)
        ratio = g_s[mask] / g_c[mask]
        # Error propagation for ratio using SEM: σ_r/r = sqrt((sem_a/a)² + (sem_b/b)²)
        ratio_err = ratio * np.sqrt((e_s[mask]/g_s[mask])**2 +
                                    (e_c[mask]/g_c[mask])**2)

        gas_label_pretty = pretty_label(gas, "").strip()
        ax.errorbar(common[mask], ratio, yerr=ratio_err,
                    color="black", marker="o", ls="-",
                    elinewidth=1.0, capsize=3)
        ax.axhline(1.0, color="grey", ls="--", lw=1)
        ax.set_xlabel("Mesh Voltage (V)")
        ax.set_ylabel("Gain(Saclay 160m) / Gain(CERN 450m)")
        ax.set_title(f"Pressure Ratio\n{gas_label_pretty}")
        ax.grid(True, alpha=0.3)

        # Annotate with pressure values
        p_s = rs["pressure_torr"]
        p_c = rc["pressure_torr"]
        ax.text(0.03, 0.05,
                f"Saclay: {p_s:.1f} Torr\nCERN:   {p_c:.1f} Torr",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", alpha=0.1))

    fig.tight_layout()
    path = os.path.join(outdir, f"pressure_ratio.{fmt}")
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 4: gain distributions at selected voltages ─────────────────────────

def plot_distributions(results, outdir, fmt, n_voltages=4):
    """
    For each (gas, pressure) combination, plot histograms of the gain
    distribution at n_voltages evenly-spaced voltage steps.
    """
    dist_dir = os.path.join(outdir, "gain_distributions")
    os.makedirs(dist_dir, exist_ok=True)

    for res in results:
        gas    = res["gas"]
        plabel = res["pressure_label"]
        volts  = res["voltages"]
        raw    = res["gain_raw"]

        # Pick n_voltages evenly spaced
        indices = np.linspace(0, len(volts) - 1, n_voltages, dtype=int)

        fig, axes = plt.subplots(1, n_voltages, figsize=(4 * n_voltages, 4),
                                 sharey=False)

        for ax, idx in zip(axes, indices):
            v    = volts[idx]
            data = np.array(raw[idx], dtype=float)
            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            # Log-scale histogram (gain distributions are approximately log-normal)
            if data.min() > 0:
                log_data = np.log10(data)
                bins = np.linspace(log_data.min(), log_data.max(), 30)
                ax.hist(log_data, bins=bins, color="steelblue", edgecolor="white",
                        alpha=0.8, linewidth=0.4)
                ax.set_xlabel("log₁₀(Gain)")
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: f"$10^{{{x:.1f}}}$"))
            else:
                ax.hist(data, bins=30, color="steelblue", edgecolor="white",
                        alpha=0.8)
                ax.set_xlabel("Gain")

            mean_g = float(np.mean(data))
            ax.set_title(f"V = {v:.0f} V\n⟨G⟩ = {mean_g:.0f}")
            ax.set_ylabel("Events")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{pretty_label(gas, plabel)}",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        safe = f"{gas}_{plabel}"
        path = os.path.join(dist_dir, f"dist_{safe}.{fmt}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)


# ── Figure 5: CERN / CEA equivalence ─────────────────────────────────────────

def plot_equivalence(results, outdir, fmt):
    """
    For each gas, find the CERN mesh voltage that produces the same gain as a
    given CEA (Saclay) mesh voltage, using exponential fits to each dataset.

    Left panel : V_CERN vs V_CEA with a 1:1 reference line.
    Right panel: ΔV = V_CERN − V_CEA vs V_CEA.
    """
    gas_labels = sorted({r["gas"] for r in results})

    for gas in gas_labels:
        res_by_pres = {r["pressure_label"]: r
                       for r in results if r["gas"] == gas}
        cea_key  = next((k for k in res_by_pres if "Saclay" in k), None)
        cern_key = next((k for k in res_by_pres if "CERN"   in k), None)
        if cea_key is None or cern_key is None:
            continue

        r_cea  = res_by_pres[cea_key]
        r_cern = res_by_pres[cern_key]

        v_cea   = np.array(r_cea["voltages"])
        g_cea   = np.array(r_cea["gain_mean"])
        sv_cea  = np.array(r_cea["survival"])
        v_cern  = np.array(r_cern["voltages"])
        g_cern  = np.array(r_cern["gain_mean"])
        sv_cern = np.array(r_cern["survival"])

        mask_cea  = (g_cea  > 0) & (sv_cea  > 0.1)
        mask_cern = (g_cern > 0) & (sv_cern > 0.1)

        fit_cea  = fit_exponential(v_cea[mask_cea],  g_cea[mask_cea])
        fit_cern = fit_exponential(v_cern[mask_cern], g_cern[mask_cern])

        if fit_cea is None or fit_cern is None:
            print(f"  Skipping equivalence for {gas}: insufficient data for fit")
            continue

        A_cea,  B_cea  = fit_cea
        A_cern, B_cern = fit_cern

        # For G_cern(V_cern) = G_cea(V_cea):
        #   A_cern * exp(B_cern * V_cern) = A_cea * exp(B_cea * V_cea)
        #   V_cern = (log(A_cea / A_cern) + B_cea * V_cea) / B_cern
        v_cea_range  = np.linspace(v_cea[mask_cea].min(),
                                   v_cea[mask_cea].max(), 300)
        v_cern_equiv = (np.log(A_cea / A_cern) + B_cea * v_cea_range) / B_cern
        delta_v      = v_cern_equiv - v_cea_range

        gas_pretty = pretty_label(gas, "").strip()
        p_cea  = r_cea["pressure_torr"]
        p_cern = r_cern["pressure_torr"]

        fig, (ax_abs, ax_diff) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: equivalent CERN voltage vs CEA voltage
        ax_abs.plot(v_cea_range, v_cern_equiv, color="steelblue", lw=2,
                    label="Equivalent CERN voltage (fit)")
        diag_lo = min(v_cea_range.min(), v_cern_equiv.min())
        diag_hi = max(v_cea_range.max(), v_cern_equiv.max())
        ax_abs.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
                    color="grey", ls="--", lw=1, label="1:1 reference")
        ax_abs.set_xlabel("CEA (Saclay) Mesh Voltage (V)")
        ax_abs.set_ylabel("Equivalent CERN Mesh Voltage (V)")
        ax_abs.set_title(f"Equivalent CERN Voltage\n{gas_pretty}")
        ax_abs.legend(framealpha=0.85)
        ax_abs.grid(True, alpha=0.3)
        ax_abs.text(
            0.04, 0.96,
            f"CEA fit:  G = {A_cea:.2g} · exp({B_cea*1e3:.3f} · V/1000)\n"
            f"CERN fit: G = {A_cern:.2g} · exp({B_cern*1e3:.3f} · V/1000)\n"
            f"CEA {p_cea:.1f} Torr  |  CERN {p_cern:.1f} Torr",
            transform=ax_abs.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", alpha=0.1),
        )

        # Right: ΔV = V_CERN − V_CEA vs V_CEA
        ax_diff.plot(v_cea_range, delta_v, color="steelblue", lw=2)
        ax_diff.axhline(0, color="grey", ls="--", lw=1)
        ax_diff.set_xlabel("CEA (Saclay) Mesh Voltage (V)")
        ax_diff.set_ylabel(r"$\Delta V = V_\mathrm{CERN} - V_\mathrm{CEA}$ (V)")
        ax_diff.set_title(f"CERN − CEA Voltage Offset for Equal Gain\n{gas_pretty}")
        ax_diff.grid(True, alpha=0.3)

        fig.tight_layout()
        safe_gas = gas.replace("/", "_").replace(" ", "_")
        path = os.path.join(outdir, f"equivalence_{safe_gas}.{fmt}")
        fig.savefig(path)
        print(f"  Saved: {path}")
        plt.close(fig)


# ── Figure 6: overview table ───────────────────────────────────────────────────

def print_summary_table(results):
    print("\nGain Summary (mean ± std)")
    print("-" * 90)
    print(f"{'Gas':<30} {'Pressure':>15} {'V(V)':>6} {'Field(kV/cm)':>13} "
          f"{'Gain':>10} {'σ/μ':>7} {'Surv%':>7}")
    print("-" * 90)

    for res in results:
        gas    = res["gas"]
        plabel = res["pressure_label"]
        for i in range(0, len(res["voltages"]), max(1, len(res["voltages"]) // 5)):
            v    = res["voltages"][i]
            e    = res["fields"][i]
            g    = res["gain_mean"][i]
            s    = res["gain_std"][i]
            rms  = res["gain_rms_rel"][i]
            surv = res["survival"][i] * 100
            rms_str = f"{rms:.2f}" if not np.isnan(rms) else "  —"
            print(f"{gas:<30} {plabel:>15} {v:>6.0f} {e/1e3:>13.2f} "
                  f"{g:>10.0f} {rms_str:>7} {surv:>6.1f}%")
    print("-" * 90)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot Micromegas gain results")
    parser.add_argument("--outdir", default=cfg.RESULTS_DIR,
                        help=f"Output directory (default: {cfg.RESULTS_DIR})")
    parser.add_argument("--format", default="png",
                        choices=["png", "pdf", "svg"],
                        help="Figure format (default: png)")
    parser.add_argument("--no-dist", action="store_true",
                        help="Skip gain distribution histograms")
    args = parser.parse_args()

    results = load_all_results(cfg.RESULTS_DIR)
    if not results:
        print(f"No JSON result files found in {cfg.RESULTS_DIR}")
        print("Run mm_gain_scan.py first.")
        sys.exit(1)

    print(f"Loaded {len(results)} result file(s)")
    print(f"Writing figures to: {args.outdir}")
    print()

    print_summary_table(results)
    print()

    plot_gain(results, args.outdir, args.format)
    plot_pressure_ratio(results, args.outdir, args.format)
    plot_equivalence(results, args.outdir, args.format)
    if not args.no_dist:
        plot_distributions(results, args.outdir, args.format)

    print("\nDone.")


if __name__ == "__main__":
    main()
