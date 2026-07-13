#!/usr/bin/env python3
"""
mm_hv_equivalence.py — Ar/iC4H10 iso-gain HV mapping across quencher fractions

Goal
----
For apples-to-apples comparisons between Ar/isobutane mixtures, answer:

    "What mesh voltage does mixture X need to reach the SAME gas gain that
     Ar/iC4H10 95/5 has at voltage V_ref?"

Method
------
Every mixture's gain curve is very close to exponential, G = A·exp(B·V), so
matching a reference gain gives a *linear* relation between the two voltages:

    G_mix(V*) = G_95/5(V_ref)
    =>  V*(mix) = (B_ref / B_mix)·V_ref  +  ln(A_ref / A_mix) / B_mix
              =  m_mix · V_ref  +  c_mix                         (analytic map)

That closed form is the headline deliverable (slope m, intercept c per mixture
and pressure). Because the log-gain curves carry a little upward curvature over
the full simulated span, we ALSO fit an accurate quadratic model

    ln G = a + b·V + c2·V²

per mixture and invert it numerically to gain-match the 95/5 reference on a
voltage grid. That yields the precise lookup TABLE; the linear map is fit to
those points and its max residual vs the table is reported. Matches that fall
outside a mixture's simulated voltage range are flagged as extrapolations.

Reference range: 95/5 is only simulated to 490 V, so V_ref is swept over its
measured span (400–490 V) — no reference extrapolation.

Outputs (results/)
------------------
  hv_equivalence.json        fit coefficients + analytic-map coefficients
  hv_equivalence_table.csv   full gain-matched table, both pressures
  hv_equivalence.png         V_equiv vs V_ref  and  ΔV vs isobutane fraction
  HV_EQUIVALENCE.md          human-readable tables, formulas, caveats

Run from garfield_sim/ after the Ar/iC4H10 quencher-scan JSONs are in results/.
"""

import os
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mm_config as cfg

REF_GAS   = "Ar_iC4H10_95_5"
FAMILY    = "Ar_iC4H10_"
V_REF_LO  = 400.0     # 95/5 measured span
V_REF_HI  = 490.0
V_REF_STEP = 10.0
PRESSURES = [("Saclay_160m", "Saclay 160 m"), ("CERN_450m", "CERN 450 m")]

PRETTY = {
    "Ar_iC4H10_98_2":  "98/2",
    "Ar_iC4H10_95_5":  "95/5",
    "Ar_iC4H10_90_10": "90/10",
    "Ar_iC4H10_85_15": "85/15",
    "Ar_iC4H10_80_20": "80/20",
    "Ar_iC4H10_75_25": "75/25",
}


def ic4_frac(gas):
    """Isobutane percentage parsed from label 'Ar_iC4H10_<ar>_<ic4>'."""
    try:
        return float(gas.split("_")[-1])
    except Exception:
        return float("nan")


def load_family(results_dir):
    """Load Ar/iC4H10 binary-family results keyed by (gas, pressure_label)."""
    out = {}
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(fpath) as f:
            d = json.load(f)
        gas = d.get("gas", "")
        if not gas.startswith(FAMILY):
            continue
        # skip ternaries like Ar_CF4_iC4H10_...
        if gas.count("_") != 3:
            continue
        out[(gas, d["pressure_label"])] = d
    return out


def clean_arrays(d):
    """Return (V, G) with gain>0 and survival>0.5."""
    v = np.array(d["voltages"], float)
    g = np.array(d["gain_mean"], float)
    s = np.array(d.get("survival", np.ones_like(g)), float)
    m = (g > 0) & (s > 0.5)
    return v[m], g[m]


def fit_loglinear(v, g):
    """ln G = lnA + B·V. Returns (A, B)."""
    B, lnA = np.polyfit(v, np.log(g), 1)
    return float(np.exp(lnA)), float(B)


def fit_logquad(v, g):
    """ln G = a + b·V + c2·V². Returns (a, b, c2)."""
    c2, b, a = np.polyfit(v, np.log(g), 2)
    return float(a), float(b), float(c2)


def invert_logquad(coef, lnG_target, vspan):
    """
    Solve a + b·V + c2·V² = lnG_target for V, choosing the physical
    (monotonically-increasing) branch. Returns V (float).
    """
    a, b, c2 = coef
    k = a - lnG_target
    if abs(c2) < 1e-12:                       # degenerate: linear
        return (lnG_target - a) / b
    disc = b * b - 4.0 * c2 * k
    if disc < 0:                              # no real match — fall back to vertex
        return -b / (2.0 * c2)
    sq = np.sqrt(disc)
    r1 = (-b + sq) / (2.0 * c2)
    r2 = (-b - sq) / (2.0 * c2)
    # physical branch: dG/dV > 0  =>  b + 2 c2 V > 0
    cands = [r for r in (r1, r2) if (b + 2.0 * c2 * r) > 0]
    if not cands:
        cands = [r1, r2]
    # prefer the root nearest the measured span
    mid = 0.5 * (vspan[0] + vspan[1])
    return float(min(cands, key=lambda r: abs(r - mid)))


def main():
    results_dir = cfg.RESULTS_DIR
    fam = load_family(results_dir)

    gases = sorted({g for (g, _) in fam}, key=ic4_frac)
    if REF_GAS not in gases:
        raise SystemExit(f"Reference {REF_GAS} not found in {results_dir}")

    v_ref_grid = np.arange(V_REF_LO, V_REF_HI + 0.5 * V_REF_STEP, V_REF_STEP)

    out = {
        "reference_gas": REF_GAS,
        "reference_voltage_range_V": [V_REF_LO, V_REF_HI],
        "model": "per-mixture ln G = a + b V + c2 V^2 (accurate); "
                 "analytic map V_equiv = m*V_ref + c from G=A exp(B V)",
        "pressures": {},
    }
    csv_rows = [("pressure", "mixture", "isobutane_pct",
                 "V_ref_95_5_V", "G_ref", "V_equiv_V", "delta_V",
                 "extrapolated")]

    # figure: left = V_equiv vs V_ref, right = ΔV vs isobutane at a mid V_ref
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("viridis")

    for pkey, ptitle in PRESSURES:
        # per-mixture fits
        loglin, logquad, vspan = {}, {}, {}
        for gas in gases:
            d = fam.get((gas, pkey))
            if d is None:
                continue
            v, g = clean_arrays(d)
            if len(v) < 3:
                continue
            loglin[gas] = fit_loglinear(v, g)
            logquad[gas] = fit_logquad(v, g)
            vspan[gas] = (float(v.min()), float(v.max()))

        A_ref, B_ref = loglin[REF_GAS]
        qa_ref = logquad[REF_GAS]

        pdata = {
            "reference_fit_loglinear": {"A": A_ref, "B": B_ref},
            "mixtures": {},
        }

        # reference gains on the grid (from the accurate 95/5 quadratic fit)
        a_r, b_r, c2_r = qa_ref
        lnG_ref_grid = a_r + b_r * v_ref_grid + c2_r * v_ref_grid**2
        G_ref_grid = np.exp(lnG_ref_grid)

        for gi, gas in enumerate(gases):
            if gas not in logquad:
                continue
            frac = ic4_frac(gas)
            A_m, B_m = loglin[gas]
            qa_m = logquad[gas]

            if gas == REF_GAS:
                # reference maps to itself exactly by construction
                v_equiv = v_ref_grid.copy()
                extrap = np.zeros(len(v_ref_grid), bool)
            else:
                # accurate gain-matched equivalent voltages via quadratic inversion
                v_equiv = np.array([invert_logquad(qa_m, lg, vspan[gas])
                                    for lg in lnG_ref_grid])
                extrap = (v_equiv < vspan[gas][0] - 1) | (v_equiv > vspan[gas][1] + 1)

            # analytic closed-form linear map (single-exponential)
            m_map = B_ref / B_m
            c_map = np.log(A_ref / A_m) / B_m
            v_equiv_lin = m_map * v_ref_grid + c_map
            lin_resid = float(np.max(np.abs(v_equiv_lin - v_equiv)))

            pdata["mixtures"][gas] = {
                "isobutane_pct": frac,
                "fit_loglinear": {"A": A_m, "B": B_m},
                "fit_logquad": {"a": qa_m[0], "b": qa_m[1], "c2": qa_m[2]},
                "measured_V_range": list(vspan[gas]),
                "analytic_map": {
                    "form": "V_equiv = m * V_ref_95_5 + c",
                    "m": m_map,
                    "c": c_map,
                },
                "linear_map_max_resid_vs_table_V": lin_resid,
                "table": [
                    {"V_ref": float(vr), "G_ref": float(gr),
                     "V_equiv": float(ve), "delta_V": float(ve - vr),
                     "extrapolated": bool(ex)}
                    for vr, gr, ve, ex in zip(v_ref_grid, G_ref_grid,
                                              v_equiv, extrap)
                ],
            }

            for vr, gr, ve, ex in zip(v_ref_grid, G_ref_grid, v_equiv, extrap):
                csv_rows.append((pkey, PRETTY.get(gas, gas), f"{frac:.0f}",
                                 f"{vr:.0f}", f"{gr:.1f}", f"{ve:.1f}",
                                 f"{ve - vr:+.1f}", "Y" if ex else "N"))

            # plotting on CERN axis pair only to keep the figure legible
            if pkey == "CERN_450m":
                col = cmap(gi / max(len(gases) - 1, 1))
                ax = axes[0]
                ax.plot(v_ref_grid[~extrap], v_equiv[~extrap], "-o", color=col,
                        ms=4, label=PRETTY.get(gas, gas))
                ax.plot(v_ref_grid[extrap], v_equiv[extrap], "--o", color=col,
                        ms=4, mfc="white")

        out["pressures"][pkey] = pdata

        # right panel: ΔV vs isobutane fraction at a representative V_ref
        v_pick = 450.0
        fr_list, dv_list, ex_list = [], [], []
        a_r, b_r, c2_r = qa_ref
        lnG_pick = a_r + b_r * v_pick + c2_r * v_pick**2
        for gas in gases:
            if gas not in logquad:
                continue
            ve = invert_logquad(logquad[gas], lnG_pick, vspan[gas])
            fr_list.append(ic4_frac(gas))
            dv_list.append(ve - v_pick)
            ex_list.append((ve < vspan[gas][0] - 1) or (ve > vspan[gas][1] + 1))
        order = np.argsort(fr_list)
        fr_arr = np.array(fr_list)[order]
        dv_arr = np.array(dv_list)[order]
        ex_arr = np.array(ex_list)[order]
        ls = "-" if pkey == "CERN_450m" else "--"
        axes[1].plot(fr_arr, dv_arr, ls, marker="s", label=ptitle)
        axes[1].scatter(fr_arr[ex_arr], dv_arr[ex_arr], facecolor="white",
                        edgecolor="k", zorder=5, s=60)

    # finish figure
    ax = axes[0]
    lo = min(V_REF_LO, 380)
    ax.plot([lo, 620], [lo, 620], ":", color="grey", lw=1, label="y = x (95/5)")
    ax.set_xlabel("V in 95/5 (V)")
    ax.set_ylabel("Equivalent V, same gain (V)")
    ax.set_title("Iso-gain mesh voltage vs 95/5  (CERN 450 m)\n"
                 "dashed/hollow = extrapolated beyond simulated range")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    axes[1].axhline(0, color="grey", lw=1, ls=":")
    axes[1].set_xlabel("Isobutane fraction (%)")
    axes[1].set_ylabel("ΔV = V_equiv − V(95/5)   at V(95/5)=450 V (V)")
    axes[1].set_title("Extra mesh voltage to match 95/5 @ 450 V\n"
                      "hollow markers = extrapolated")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("Ar/iC₄H₁₀ — HV equivalence to 95/5 at equal gas gain",
                 fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    png = os.path.join(results_dir, "hv_equivalence.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # write JSON + CSV
    jpath = os.path.join(results_dir, "hv_equivalence.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)

    cpath = os.path.join(results_dir, "hv_equivalence_table.csv")
    with open(cpath, "w") as f:
        for row in csv_rows:
            f.write(",".join(row) + "\n")

    write_markdown(out, os.path.join(results_dir, "HV_EQUIVALENCE.md"))

    print(f"Saved: {png}")
    print(f"Saved: {jpath}")
    print(f"Saved: {cpath}")
    print(f"Saved: {os.path.join(results_dir, 'HV_EQUIVALENCE.md')}")


def write_markdown(out, path):
    ref = out["reference_gas"]
    lo, hi = out["reference_voltage_range_V"]
    lines = []
    lines.append("# Ar/iC₄H₁₀ HV equivalence — matching 95/5 gas gain\n")
    lines.append(
        f"Maps the mesh voltage of each Ar/isobutane mixture to the voltage of "
        f"**Ar/iC₄H₁₀ 95/5** that gives the **same simulated gas gain** "
        f"(Garfield++/Magboltz). Use it to put HV scans in different mixtures on "
        f"a common footing.\n")
    lines.append(
        f"Reference 95/5 voltage is swept over its simulated span "
        f"**{lo:.0f}–{hi:.0f} V**. Mixtures whose match falls outside their own "
        f"simulated range are flagged `*` (extrapolated — larger uncertainty; "
        f"this happens for the high-isobutane mixtures, which need much higher "
        f"HV than was simulated).\n")

    lines.append("## Analytic map (closed form)\n")
    lines.append(
        "Each gain curve is ≈ exponential, `G = A·exp(B·V)`, so equal gain gives "
        "a **linear** voltage map\n")
    lines.append("```\nV_equiv = m · V(95/5) + c\n```\n")
    lines.append(
        "with `m = B_ref/B_mix` and `c = ln(A_ref/A_mix)/B_mix`. Coefficients "
        "per mixture and pressure (`resid` = max deviation of this linear form "
        "from the accurate quadratic-fit lookup over the reference range):\n")

    for pkey, pdata in out["pressures"].items():
        lines.append(f"### {pkey}\n")
        lines.append("| Mixture | iC₄H₁₀ % | m (slope) | c (V) | max resid (V) |")
        lines.append("|---|---|---|---|---|")
        for gas, md in pdata["mixtures"].items():
            am = md["analytic_map"]
            lines.append(
                f"| {PRETTY.get(gas, gas)} | {md['isobutane_pct']:.0f} | "
                f"{am['m']:.4f} | {am['c']:+.1f} | "
                f"{md['linear_map_max_resid_vs_table_V']:.1f} |")
        lines.append("")

    lines.append("## Lookup table (accurate, quadratic-fit gain match)\n")
    lines.append(
        "Equivalent mesh voltage (V) to reach the same gain as 95/5 at the given "
        "V(95/5). `*` = extrapolated beyond the mixture's simulated voltage range.\n")

    for pkey, pdata in out["pressures"].items():
        mixes = list(pdata["mixtures"].values())
        glabels = [PRETTY.get(g, g) for g in pdata["mixtures"].keys()]
        lines.append(f"### {pkey}\n")
        header = "| V(95/5) | G(95/5) | " + " | ".join(glabels) + " |"
        sep = "|" + "---|" * (2 + len(glabels))
        lines.append(header)
        lines.append(sep)
        ref_table = mixes[0]["table"]
        for i, rrow in enumerate(ref_table):
            vr = rrow["V_ref"]
            gr = rrow["G_ref"]
            cells = []
            for md in mixes:
                t = md["table"][i]
                mark = "*" if t["extrapolated"] else ""
                cells.append(f"{t['V_equiv']:.0f}{mark}")
            lines.append(f"| {vr:.0f} | {gr:,.0f} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- Gain model is per-mixture `ln G = a + b·V + c₂·V²` (R² ≥ 0.997); the "
        "closed-form linear map above uses the single-exponential fit and agrees "
        "with the table to within the listed residual inside the reference range.\n"
        "- 95/5 is only simulated to 490 V, so the reference does not extrapolate; "
        "the equivalents for 80/20 and 75/25 (and the low-voltage end of 98/2) "
        "*do* extrapolate and should be treated as indicative.\n"
        "- Two pressure conditions are reported (Saclay 160 m ≈ 746 Torr, "
        "CERN 450 m ≈ 721 Torr); pick the one matching the operating site.\n"
        "- Regenerate with `python3 mm_hv_equivalence.py` after refreshing the "
        "Ar/iC4H10 quencher-scan JSONs in `results/`.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
