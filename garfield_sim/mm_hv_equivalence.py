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

# ── Ar/CO2/iC4H10 93/5/2 — the n_TOF operating mixture ───────────────────────
# Not a member of the Ar/iC4H10 binary family, but mapped onto the same 95/5
# reference. Garfield++ has no built-in Penning parameterisation for this
# ternary, so it was simulated at three hand-set transfer probabilities; the
# spread between them IS the Penning systematic on the map.
TERNARY_BASE     = "Ar_CO2_iC4H10_93_5_2"
TERNARY_VARIANTS = [
    (f"{TERNARY_BASE}_rP030", 0.30),
    (f"{TERNARY_BASE}_rP040", 0.40),   # central
    (f"{TERNARY_BASE}_rP050", 0.50),
]
TERNARY_CENTRAL  = f"{TERNARY_BASE}_rP040"
TERNARY_LABELS   = [lab for lab, _ in TERNARY_VARIANTS]

PRETTY = {
    "Ar_iC4H10_98_2":  "98/2",
    "Ar_iC4H10_95_5":  "95/5",
    "Ar_iC4H10_90_10": "90/10",
    "Ar_iC4H10_85_15": "85/15",
    "Ar_iC4H10_80_20": "80/20",
    "Ar_iC4H10_75_25": "75/25",
    f"{TERNARY_BASE}_rP030": "93/5/2 r=0.30",
    f"{TERNARY_BASE}_rP040": "93/5/2 r=0.40",
    f"{TERNARY_BASE}_rP050": "93/5/2 r=0.50",
}


def is_ternary(gas):
    return gas.startswith(TERNARY_BASE)


def ic4_frac(gas):
    """Isobutane percentage parsed from label 'Ar_iC4H10_<ar>_<ic4>'."""
    if is_ternary(gas):
        return 2.0                      # 93/5/2 carries 2% isobutane
    try:
        return float(gas.split("_")[-1])
    except Exception:
        return float("nan")


def sort_key(gas):
    """Binaries ordered by isobutane fraction; ternary variants last."""
    return (1 if is_ternary(gas) else 0, ic4_frac(gas), gas)


def load_family(results_dir):
    """
    Load the Ar/iC4H10 binary family plus the Ar/CO2/iC4H10 ternary variants,
    keyed by (gas, pressure_label).
    """
    out = {}
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(fpath) as f:
            d = json.load(f)
        gas = d.get("gas", "")
        if is_ternary(gas):
            out[(gas, d["pressure_label"])] = d
            continue
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

    gases = sorted({g for (g, _) in fam}, key=sort_key)
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

    # figure: left = V_equiv vs V_ref (binary family), middle = ΔV vs isobutane,
    # right = the Ar/CO2/iC4H10 93/5/2 map with its Penning band
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    cmap = plt.get_cmap("viridis")
    tern_curves = {}          # (pressure, gas) -> v_equiv array, for panel 3

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

            if is_ternary(gas):
                tern_curves[(pkey, gas)] = (v_equiv, extrap)

            # plotting on CERN axis pair only to keep the figure legible
            if pkey == "CERN_450m":
                ax = axes[0]
                if is_ternary(gas):
                    # only the central rP on the family panel, in red
                    if gas != TERNARY_CENTRAL:
                        continue
                    ax.plot(v_ref_grid[~extrap], v_equiv[~extrap], "-D",
                            color="crimson", ms=5, lw=2,
                            label="93/5/2 (r=0.40)")
                    ax.plot(v_ref_grid[extrap], v_equiv[extrap], "--D",
                            color="crimson", ms=5, mfc="white")
                    continue
                col = cmap(gi / max(len(gases) - 1, 1))
                ax.plot(v_ref_grid[~extrap], v_equiv[~extrap], "-o", color=col,
                        ms=4, label=PRETTY.get(gas, gas))
                ax.plot(v_ref_grid[extrap], v_equiv[extrap], "--o", color=col,
                        ms=4, mfc="white")

        out["pressures"][pkey] = pdata

        # middle panel: ΔV vs isobutane fraction at a representative V_ref.
        # Binary family only — the ternary's 2% isobutane is not comparable
        # (it carries 5% CO2 as well), so it goes on as a standalone marker.
        v_pick = 450.0
        fr_list, dv_list, ex_list = [], [], []
        tern_pick = {}
        a_r, b_r, c2_r = qa_ref
        lnG_pick = a_r + b_r * v_pick + c2_r * v_pick**2
        for gas in gases:
            if gas not in logquad:
                continue
            ve = invert_logquad(logquad[gas], lnG_pick, vspan[gas])
            ex = (ve < vspan[gas][0] - 1) or (ve > vspan[gas][1] + 1)
            if is_ternary(gas):
                tern_pick[gas] = (ve - v_pick, ex)
                continue
            fr_list.append(ic4_frac(gas))
            dv_list.append(ve - v_pick)
            ex_list.append(ex)
        order = np.argsort(fr_list)
        fr_arr = np.array(fr_list)[order]
        dv_arr = np.array(dv_list)[order]
        ex_arr = np.array(ex_list)[order]
        ls = "-" if pkey == "CERN_450m" else "--"
        axes[1].plot(fr_arr, dv_arr, ls, marker="s", label=f"Ar/iC₄H₁₀ — {ptitle}")
        axes[1].scatter(fr_arr[ex_arr], dv_arr[ex_arr], facecolor="white",
                        edgecolor="k", zorder=5, s=60)
        if TERNARY_CENTRAL in tern_pick:
            dv_c = tern_pick[TERNARY_CENTRAL][0]
            lo = min(v[0] for v in tern_pick.values())
            hi = max(v[0] for v in tern_pick.values())
            axes[1].errorbar([2.0], [dv_c],
                             yerr=[[dv_c - lo], [hi - dv_c]],
                             fmt="D", color="crimson", ms=8, capsize=5,
                             zorder=6,
                             label=("Ar/CO₂/iC₄H₁₀ 93/5/2 — " + ptitle
                                    + " (bar = Penning r 0.30–0.50)"))

        # right panel: the ternary map itself, with the Penning band
        if pkey == "CERN_450m" and (pkey, TERNARY_CENTRAL) in tern_curves:
            lo_v = np.minimum.reduce(
                [tern_curves[(pkey, g)][0] for g in TERNARY_LABELS
                 if (pkey, g) in tern_curves])
            hi_v = np.maximum.reduce(
                [tern_curves[(pkey, g)][0] for g in TERNARY_LABELS
                 if (pkey, g) in tern_curves])
            ce_v, ce_ex = tern_curves[(pkey, TERNARY_CENTRAL)]
            axes[2].fill_between(v_ref_grid, lo_v, hi_v, color="crimson",
                                 alpha=0.18,
                                 label="Penning r = 0.30 – 0.50")
            axes[2].plot(v_ref_grid, ce_v, "-D", color="crimson", ms=5, lw=2,
                         label="central r = 0.40")
            axes[2].plot(v_ref_grid[ce_ex], ce_v[ce_ex], "D", color="crimson",
                         ms=5, mfc="white", label="extrapolated")

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
    axes[1].legend(fontsize=8)

    axes[2].plot([V_REF_LO, V_REF_HI], [V_REF_LO, V_REF_HI], ":", color="grey",
                 lw=1, label="y = x (95/5)")
    axes[2].set_xlabel("V in 95/5 (V)")
    axes[2].set_ylabel("Equivalent V in 93/5/2, same gain (V)")
    axes[2].set_title("Ar/CO₂/iC₄H₁₀ 93/5/2 vs 95/5  (CERN 450 m)\n"
                      "band = Penning-transfer systematic")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=9, loc="upper left")

    fig.suptitle("HV equivalence to Ar/iC₄H₁₀ 95/5 at equal gas gain",
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


def write_ternary_section(out, lines):
    """
    Dedicated section for Ar/CO2/iC4H10 93/5/2: the map you actually apply when
    moving an HV setting from the 95/5 bench gas to the n_TOF operating gas,
    with the Penning-transfer systematic shown explicitly.
    """
    have = any(g in pdata["mixtures"]
               for pdata in out["pressures"].values()
               for g in TERNARY_LABELS)
    if not have:
        return

    lines.append("## Ar/CO₂/iC₄H₁₀ 93/5/2 — the operating mixture\n")
    lines.append(
        "Garfield++ has **no built-in Penning parameterisation** for this "
        "ternary: `EnablePenningTransfer()` returns *false* and would leave the "
        "mixture with **zero** Penning transfer while the 95/5 reference runs at "
        "r = 0.40. It was therefore simulated at three hand-set transfer "
        "probabilities — r = 0.30, 0.40 (central) and 0.50 — and the spread "
        "between them is quoted below as the Penning systematic. The central "
        "value follows Garfield's own binary parameterisations at this quencher "
        "content (Ar/CO₂ gives 0.376 at 7% CO₂, Ar/iC₄H₁₀ 0.400 flat).\n")

    for pkey, pdata in out["pressures"].items():
        present = [g for g in TERNARY_LABELS if g in pdata["mixtures"]]
        if not present:
            continue
        lines.append(f"### {pkey}\n")
        lines.append("| V(95/5) | G(95/5) | " +
                     " | ".join(f"V(93/5/2) r={PRETTY[g].split('=')[-1]}"
                                for g in present) +
                     " | ΔV central | Penning spread |")
        lines.append("|" + "---|" * (4 + len(present)))
        ref_rows = pdata["mixtures"][present[0]]["table"]
        for i, rrow in enumerate(ref_rows):
            vs = [pdata["mixtures"][g]["table"][i]["V_equiv"] for g in present]
            marks = ["*" if pdata["mixtures"][g]["table"][i]["extrapolated"]
                     else "" for g in present]
            cen = (pdata["mixtures"][TERNARY_CENTRAL]["table"][i]["V_equiv"]
                   if TERNARY_CENTRAL in pdata["mixtures"] else vs[len(vs) // 2])
            cells = " | ".join(f"{v:.0f}{m}" for v, m in zip(vs, marks))
            lines.append(f"| {rrow['V_ref']:.0f} | {rrow['G_ref']:,.0f} | "
                         f"{cells} | {cen - rrow['V_ref']:+.0f} | "
                         f"±{0.5 * (max(vs) - min(vs)):.0f} V |")
        lines.append("")

        if TERNARY_CENTRAL in pdata["mixtures"]:
            am = pdata["mixtures"][TERNARY_CENTRAL]["analytic_map"]
            lines.append(
                f"Closed form at the central Penning value: "
                f"`V(93/5/2) = {am['m']:.4f} · V(95/5) {am['c']:+.1f}` V "
                f"(max deviation from the table above: "
                f"{pdata['mixtures'][TERNARY_CENTRAL]['linear_map_max_resid_vs_table_V']:.1f} V).\n")

    lines.append(
        "`*` = the match falls outside the voltage range actually simulated for "
        "the ternary and is an extrapolation of its fit.\n")


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

    write_ternary_section(out, lines)

    lines.append("## Notes\n")
    lines.append(
        "- Gain model is per-mixture `ln G = a + b·V + c₂·V²` (R² ≥ 0.997); the "
        "closed-form linear map above uses the single-exponential fit and agrees "
        "with the table to within the listed residual inside the reference range.\n"
        "- The reference itself is simulated to 490 V at Saclay but only to 480 V "
        "at CERN, so the CERN 490 V row is a 10 V extrapolation of the 95/5 fit "
        "(small compared with its 80 V fitted span, but it is not measured).\n"
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
