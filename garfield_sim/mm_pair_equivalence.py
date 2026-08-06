#!/usr/bin/env python3
"""
mm_pair_equivalence.py — iso-gain HV map between any two mixtures
=================================================================
mm_hv_equivalence.py maps a whole family against the fixed Ar/iC4H10 95/5
reference. This does the same arithmetic for an arbitrary *pair*: pick any
simulated mixture as the reference and any other as the target, optionally with
a bracket of Penning variants, and get the mesh voltage the target needs to sit
at the reference's gain.

    python3 mm_pair_equivalence.py \\
        --ref Ar_CO2_70_30 \\
        --target Ne_CF4_C2H6_80_10_10 \\
        --variants rP040,rP050,rP060 --central rP050

Both mixtures must already have results/<label>_<pressure>.json from
mm_condor_collect.py. Voltages outside a mixture's simulated span are marked
with * in the tables and dashed in the plot — the fit is a quadratic in ln G, so
a few tens of volts of extrapolation is reasonable and a hundred is not.

Outputs (into results/, stem from --out-stem):
    <stem>.png    gain curves + the HV map with the Penning band
    <stem>.json   machine-readable map
    <stem>.csv    the lookup table
    <stem>.md     the write-up, including the closed-form linear map
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_config as cfg
from mm_hv_equivalence import (clean_arrays, fit_loglinear,
                               fit_logquad, invert_logquad)

PRETTY = {
    "Ar_CO2_70_30":         "Ar/CO₂ 70/30",
    "Ne_CF4_C2H6_80_10_10": "Ne/CF₄/C₂H₆ 80/10/10",
    "Ar_iC4H10_95_5":       "Ar/iC₄H₁₀ 95/5",
    "Ar_CO2_iC4H10_93_5_2": "Ar/CO₂/iC₄H₁₀ 93/5/2",
}


def pretty(label):
    # Longest match first: "Ne_CF4_C2H6_80_10_10" is a prefix of the uRWELL
    # labels too, and matching the short one would drop the geometry suffix.
    for base in sorted(PRETTY, key=len, reverse=True):
        if not label.startswith(base):
            continue
        name = PRETTY[base]
        tail = label[len(base):].strip("_")
        if not tail:
            return name
        # The rP token can sit anywhere in the tail (e.g. "uRW50_rP040"), so
        # pick it out rather than assuming it leads.
        bits = [b for b in tail.split("_") if b]
        rp = next((b for b in bits if b.startswith("rP") and b[2:].isdigit()), None)
        rest = [b for b in bits if b != rp]
        out = name
        if rest:
            out += " " + " ".join(rest)
        if rp:
            out += f" r={int(rp[2:]) / 100:.2f}"
        return out
    return label


def parse_args():
    p = argparse.ArgumentParser(description="Iso-gain HV map between two mixtures")
    p.add_argument("--ref",       required=True,
                   help="Reference gas label (the axis you read V from)")
    p.add_argument("--target",    required=True,
                   help="Target gas label, or the base label if --variants is used")
    p.add_argument("--variants",  default="",
                   help="Comma-separated Penning suffixes on the target, "
                        "e.g. rP040,rP050,rP060. Empty = target used as-is.")
    p.add_argument("--central",   default="",
                   help="Which variant is the central value (default: middle one)")
    p.add_argument("--ref-variants", default="",
                   help="As --variants but for the reference side")
    p.add_argument("--pressures", default="CERN_450m,Saclay_160m")
    p.add_argument("--npoints",   type=int, default=14,
                   help="Rows in the lookup table")
    p.add_argument("--out-stem",  default=None,
                   help="Output filename stem (default: equivalence_<ref>_to_<target>)")
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


def load(results_dir, label, pressure):
    path = os.path.join(results_dir, f"{label}_{pressure}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    v, g = clean_arrays(d)
    if len(v) < 3:
        return None
    return {
        "label": label, "pressure": pressure, "V": v, "G": g,
        # gap from the result file, not cfg.GAP_CM — the 50 µm uRWELL and the
        # 150 µm Micromegas branches both come through here
        "gap_cm": float(d.get("gap_cm", cfg.GAP_CM)),
        "span": (float(v.min()), float(v.max())),
        "quad": fit_logquad(v, g), "lin": fit_loglinear(v, g),
        "penning": d.get("penning", {}), "n_events": d.get("n_events"),
        "partial": d.get("partial", False),
    }


def expand(base, variants):
    return [base] if not variants else [f"{base}_{s}" for s in variants]


def overlap_grid(ref, targets, npoints):
    """
    Voltage rows on the reference, chosen so the *gain* they ask for is one the
    target curves actually reached. Extrapolating the target is what goes wrong
    first, so the grid is built in ln G on the intersection of the spans.
    """
    def lnG(m, v):
        a, b, c2 = m["quad"]
        return a + b * v + c2 * v * v

    lo = lnG(ref, ref["span"][0])
    hi = lnG(ref, ref["span"][1])
    for t in targets:
        lo = max(lo, lnG(t, t["span"][0]))
        hi = min(hi, lnG(t, t["span"][1]))
    if hi <= lo:                      # no common gain range — fall back to the reference
        lo, hi = lnG(ref, ref["span"][0]), lnG(ref, ref["span"][1])
    lnGs = np.linspace(lo, hi, npoints)
    vref = np.array([invert_logquad(ref["quad"], x, ref["span"]) for x in lnGs])
    return vref, np.exp(lnGs)


def main():
    args = parse_args()
    results_dir = args.results_dir or cfg.RESULTS_DIR
    variants     = [s for s in args.variants.split(",") if s]
    ref_variants = [s for s in args.ref_variants.split(",") if s]
    pressures    = [s for s in args.pressures.split(",") if s]

    tgt_labels = expand(args.target, variants)
    ref_labels = expand(args.ref, ref_variants)
    central_t  = (f"{args.target}_{args.central}" if args.central
                  else tgt_labels[len(tgt_labels) // 2])
    ref_label  = ref_labels[len(ref_labels) // 2]

    stem = args.out_stem or f"equivalence_{args.ref}_to_{args.target}"

    out = {"reference": ref_label, "target_base": args.target,
           "target_variants": tgt_labels, "target_central": central_t,
           "gap_cm": None, "pressures": {}}

    missing = []
    for pres in pressures:
        ref = load(results_dir, ref_label, pres)
        if ref is None:
            missing.append(f"{ref_label}_{pres}")
            continue
        tgts = []
        for lab in tgt_labels:
            t = load(results_dir, lab, pres)
            if t is None:
                missing.append(f"{lab}_{pres}")
            else:
                tgts.append(t)
        if not tgts:
            continue

        gaps = {round(ref["gap_cm"], 8)} | {round(t["gap_cm"], 8) for t in tgts}
        if len(gaps) > 1:
            sys.exit(f"[pair] refusing to map across different amplification "
                     f"gaps: {sorted(g*1e4 for g in gaps)} µm. An equal-gain map "
                     f"is only meaningful within one geometry.")
        gap_cm = gaps.pop()
        if out["gap_cm"] is None:
            out["gap_cm"] = gap_cm

        vref, gref = overlap_grid(ref, tgts, args.npoints)
        rows = []
        for i, (vr, gr) in enumerate(zip(vref, gref)):
            row = {"V_ref": float(vr), "G": float(gr),
                   "V_ref_extrap": not (ref["span"][0] <= vr <= ref["span"][1])}
            for t in tgts:
                vt = invert_logquad(t["quad"], np.log(gr), t["span"])
                row[t["label"]] = float(vt)
                row[t["label"] + "_extrap"] = not (t["span"][0] <= vt <= t["span"][1])
            rows.append(row)

        # Closed-form linear map V_target = m·V_ref + c from G = A·exp(B·V)
        lin = {}
        A_r, B_r = ref["lin"]
        for t in tgts:
            A_t, B_t = t["lin"]
            m = B_r / B_t
            c = np.log(A_r / A_t) / B_t
            resid = max(abs(m * r["V_ref"] + c - r[t["label"]]) for r in rows)
            lin[t["label"]] = {"m": float(m), "c": float(c),
                               "max_resid_V": float(resid)}

        out["pressures"][pres] = {
            "ref": {"span": ref["span"], "n_events": ref["n_events"],
                    "penning": ref["penning"], "partial": ref["partial"]},
            "targets": {t["label"]: {"span": t["span"], "n_events": t["n_events"],
                                     "penning": t["penning"],
                                     "partial": t["partial"]} for t in tgts},
            "rows": rows, "linear_map": lin,
            "_curves": {"ref": (ref["V"].tolist(), ref["G"].tolist()),
                        **{t["label"]: (t["V"].tolist(), t["G"].tolist())
                           for t in tgts}},
        }

    if not out["pressures"]:
        sys.exit(f"[pair] nothing to map — missing: {', '.join(missing)}")
    if missing:
        print(f"[pair] WARNING missing results: {', '.join(missing)}")

    write_plot(out, args, os.path.join(results_dir, stem + ".png"))
    write_csv(out, os.path.join(results_dir, stem + ".csv"))
    write_markdown(out, args, missing, os.path.join(results_dir, stem + ".md"))

    slim = json.loads(json.dumps(out))
    for pres in slim["pressures"]:
        slim["pressures"][pres].pop("_curves", None)
    with open(os.path.join(results_dir, stem + ".json"), "w") as f:
        json.dump(slim, f, indent=2)

    for ext in ("png", "csv", "md", "json"):
        print(f"Saved: {os.path.join(results_dir, stem + '.' + ext)}")


# ── output ────────────────────────────────────────────────────────────────────

def write_plot(out, args, path):
    pressures = list(out["pressures"])
    n = len(pressures)
    fig, axes = plt.subplots(2, n, figsize=(7.5 * n, 10), squeeze=False)

    central = out["target_central"]
    band = plt.cm.viridis(np.linspace(0.15, 0.8, len(out["target_variants"])))

    for j, pres in enumerate(pressures):
        blk = out["pressures"][pres]
        cur = blk["_curves"]

        # top: gain curves
        ax = axes[0][j]
        vr, gr = cur["ref"]
        ax.semilogy(vr, gr, "o-", color="#333333", lw=2, ms=5,
                    label=pretty(out["reference"]))
        for k, lab in enumerate(out["target_variants"]):
            if lab not in cur:
                continue
            vt, gt = cur[lab]
            ax.semilogy(vt, gt, "s--" if lab != central else "s-",
                        color=band[k], lw=2.2 if lab == central else 1.4,
                        ms=5, alpha=1.0 if lab == central else 0.75,
                        label=pretty(lab))
        ax.set_xlabel("Mesh voltage (V)")
        ax.set_ylabel("Simulated gas gain")
        ax.set_title(f"{pres} — gain curves")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

        # bottom: the map
        ax = axes[1][j]
        rows = blk["rows"]
        x = [r["V_ref"] for r in rows]
        present = [l for l in out["target_variants"] if l in blk["targets"]]
        if len(present) > 1:
            lo = [min(r[l] for l in present) for r in rows]
            hi = [max(r[l] for l in present) for r in rows]
            ax.fill_between(x, lo, hi, color="crimson", alpha=0.18,
                            label="Penning bracket")
        if central in blk["targets"]:
            ax.plot(x, [r[central] for r in rows], "D-", color="crimson",
                    lw=2, ms=5, label=pretty(central))
        for l in present:
            if l == central:
                continue
            ax.plot(x, [r[l] for r in rows], "--", color="crimson",
                    lw=1, alpha=0.5)
        ax.plot(x, x, ":", color="grey", lw=1, label="equal voltage")
        ax.set_xlabel(f"{pretty(out['reference'])} mesh voltage (V)")
        ax.set_ylabel(f"{pretty(args.target)} mesh voltage (V)")
        ax.set_title(f"{pres} — equal-gain mesh voltage")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Equal-gain mesh voltage: {pretty(out['reference'])} → "
                 f"{pretty(args.target)}   ({out['gap_cm']*1e4:.0f} µm gap)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(out, path):
    labs = out["target_variants"]
    with open(path, "w") as f:
        f.write("pressure,V_ref,gain," + ",".join(labs) + "\n")
        for pres, blk in out["pressures"].items():
            for r in blk["rows"]:
                vals = ",".join(f"{r[l]:.1f}" if l in r else "" for l in labs)
                f.write(f"{pres},{r['V_ref']:.1f},{r['G']:.4g},{vals}\n")


def write_markdown(out, args, missing, path):
    L = []
    ref_n, tgt_n = pretty(out["reference"]), pretty(args.target)
    L.append(f"# Equal-gain mesh voltage: {ref_n} → {tgt_n}")
    L.append("")
    L.append(f"Gap {out['gap_cm']*1e4:.0f} µm, T = {cfg.TEMP_K:.2f} K. Each row is one "
             f"gain: read the {ref_n} voltage on the left and the {tgt_n} voltage "
             f"that reaches the same simulated gain on the right. `*` marks a "
             f"voltage outside that mixture's simulated span.")
    L.append("")
    # Penning provenance: "auto" means Garfield++ found a built-in curve for the
    # mixture; "manual" means someone chose the number, and then the bracket is
    # usually the largest single uncertainty on the map.
    modes = {}
    for blk in out["pressures"].values():
        modes[out["reference"]] = blk["ref"]["penning"].get("mode", "?")
        for l, t in blk["targets"].items():
            modes[l] = t["penning"].get("mode", "?")
    auto = [l for l, m in modes.items() if m == "auto"]
    man  = [l for l, m in modes.items() if m == "manual"]
    L.append("**Penning.** " + "; ".join(
        [f"`{l}` uses Garfield++'s built-in parameterisation (auto)" for l in auto]
        + [f"`{l}` is hand-set — Garfield++ has no curve for it" for l in man]) + ".")
    if man and auto:
        L.append("")
        L.append("The two sides are therefore not on equal footing: one is a "
                 "measurement Garfield++ ships, the other is a choice. Where a "
                 "bracket is shown it is an assumption, not an uncertainty "
                 "propagated from data.")
        # How much that assumption is actually worth, in volts. Do not assert
        # that it dominates -- in a high-alpha mixture the avalanche is carried
        # by direct ionisation and Penning is a small perturbation, so this can
        # come out far smaller than the geometry term.
        worst = 0.0
        for blk in out["pressures"].values():
            labs = [l for l in out["target_variants"] if l in blk["targets"]]
            if len(labs) > 1:
                for r in blk["rows"]:
                    worst = max(worst,
                                (max(r[l] for l in labs)
                                 - min(r[l] for l in labs)) / 2.0)
        if worst:
            L.append("")
            L.append(f"Across the full bracket that assumption is worth at most "
                     f"**±{worst:.0f} V** on this map (largest half-spread in the "
                     f"tables below). Judge it against the other error terms "
                     f"before deciding it is the one that matters.")
    L.append("")
    if missing:
        L.append(f"> Missing results at write time: `{'`, `'.join(missing)}`")
        L.append("")

    for pres, blk in out["pressures"].items():
        labs = [l for l in out["target_variants"] if l in blk["targets"]]
        L.append(f"## {pres}")
        L.append("")
        parts = [f"{ref_n} {blk['ref']['span'][0]:.0f}–{blk['ref']['span'][1]:.0f} V "
                 f"({blk['ref']['n_events']} events/point)"]
        for l in labs:
            t = blk["targets"][l]
            parts.append(f"{pretty(l)} {t['span'][0]:.0f}–{t['span'][1]:.0f} V "
                         f"({t['n_events']} events/point)")
        L.append("Simulated spans: " + "; ".join(parts) + ".")
        if blk["ref"]["partial"] or any(blk["targets"][l]["partial"] for l in labs):
            L.append("")
            L.append("> ⚠ At least one curve is built from a **partial** fragment set.")
        L.append("")
        L.append("| V(ref) | gain | " + " | ".join(pretty(l) for l in labs)
                 + " | ΔV central | bracket |")
        L.append("|---|---|" + "---|" * (len(labs) + 2))
        central = out["target_central"]
        for r in blk["rows"]:
            cells = []
            for l in labs:
                cells.append(f"{r[l]:.0f}" + ("*" if r[l + '_extrap'] else ""))
            spread = (max(r[l] for l in labs) - min(r[l] for l in labs)) / 2.0
            dv = r[central] - r["V_ref"] if central in r else float("nan")
            L.append(f"| {r['V_ref']:.0f}{'*' if r['V_ref_extrap'] else ''} "
                     f"| {r['G']:,.0f} | " + " | ".join(cells)
                     + f" | {dv:+.0f} V | ±{spread:.0f} V |")
        L.append("")
        L.append(f"Same table in **field**, which is the form that travels to a "
                 f"detector with a different amplification gap. Equal gain means "
                 f"equal effective Townsend coefficient, and that condition has "
                 f"no gap in it — so divide out the {out['gap_cm']*1e4:.0f} µm gap "
                 f"these numbers were simulated at, and multiply back in by "
                 f"whatever gap the other detector has. See `mm_gap_scaling.py` "
                 f"for how far that actually holds — it was checked around the "
                 f"150 µm Micromegas case, where rebuilding the map at 128 vs "
                 f"150 µm moved it only a few volts, and it degraded outside "
                 f"that range. It has NOT been checked around this geometry.")
        L.append("")
        L.append("| E(ref) kV/cm | gain | " + " | ".join(pretty(l) + " kV/cm"
                                                         for l in labs) + " |")
        L.append("|---|---|" + "---|" * len(labs))
        for r in blk["rows"]:
            cells = " | ".join(f"{r[l] / out['gap_cm'] / 1e3:.2f}" for l in labs)
            L.append(f"| {r['V_ref'] / out['gap_cm'] / 1e3:.2f} "
                     f"| {r['G']:,.0f} | {cells} |")
        L.append("")
        L.append("Closed-form linear map, `V_target = m·V_ref + c` "
                 "(from G = A·e^(B·V) on each curve):")
        L.append("")
        L.append("| variant | m | c (V) | max resid vs table |")
        L.append("|---|---|---|---|")
        for l in labs:
            d = blk["linear_map"][l]
            L.append(f"| {pretty(l)} | {d['m']:.4f} | {d['c']:+.1f} "
                     f"| {d['max_resid_V']:.1f} V |")
        L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
