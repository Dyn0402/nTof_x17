"""
Build the shareable Config A vs B significance report PDF.

Combines the geometry top-down plots, the significance study figure and the
enhancement scan into results/significance/Config_A_vs_B_report.pdf with an
executive summary and caveats.

The numbers in the summary table are transcribed from the
run_significance_study.py output of 2026-06-10 (200k MC windows per
config/source, Geant4 response from the OLD-geometry dataset).  Re-run that
script and update _RESULTS below if inputs change.

Run:  venv/bin/python make_config_report.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT_PDF = os.path.join(RESULTS, "significance", "Config_A_vs_B_report.pdf")

# run_significance_study.py output, 2026-06-10  (30 days, 578,700 pulses,
# 540 X17 / 21,605 IPC pairs produced; 30 ns time resolution with 200 ns
# coincidence window)
_RESULTS = [
    # label                                          A          B
    ("X17 pairs, angle channel (double trigger)",   "67",      "40"),
    ("IPC pairs, angle channel",                    "2110",    "1381"),
    ("X17 pairs, mass channel (both back-scint)",   "18",      "29"),
    ("IPC pairs, mass channel",                     "1388",    "2022"),
    ("Both-MM pair efficiency",                     "13.0%",   "13.1%"),
    ("Trigger efficiency (of both-MM)",             "94.5%",   "57.6%"),
    ("Full calorimetry (of both-MM)",               "26.5%",   "42.2%"),
    ("Z — angle, IPC shape fixed (stat only)",      "3.02 σ",  "2.27 σ"),
    ("Z — angle, IPC normalisation fit from data",  "2.87 σ",  "2.17 σ"),
    ("Z — angle, IPC norm+tilt+curvature fit",      "2.30 σ",  "1.74 σ"),
    ("Z — mass, IPC shape fixed (stat only)",       "1.29 σ",  "1.57 σ"),
    ("Z — mass, IPC norm+tilt+curvature fit",       "1.00 σ",  "1.18 σ"),
]

PAGE = (11.69, 8.27)   # A4 landscape


def _text_page(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.5, 0.95, "MX17: Configuration A vs B — X17 Discovery Significance",
             ha="center", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.905,
             "Fast-MC study with Geant4-derived detector response  ·  "
             "30-day exposure (578,700 pulses)  ·  2026-06-10",
             ha="center", fontsize=10, color="0.35")

    # ── Executive summary ────────────────────────────────────────────────
    summary = (
        "Config A (standard stack: MM → trigger scint → LS1 → LS2 → back-scint) "
        "reaches Z = 3.0σ if the IPC spectrum shape\n"
        "is taken as perfectly known, and Z = 2.3σ when the IPC normalisation and "
        "shape (tilt + curvature) are instead fit\n"
        "from the data itself.  Config B (back-scint before the LS layers, used as "
        "trigger) reaches 2.3σ / 1.7σ on the same ladder.\n\n"
        "B's better calorimetric coverage (42% vs 26% full calorimetry) raises its "
        "mass-channel significance by only +0.3σ,\n"
        "while its trigger loss (58% vs 94% of both-MM pairs) costs −0.7σ in the "
        "angle channel.  With the realistic angular\n"
        "resolution (σ68 ≈ 14°, dominated by multiple scattering in the He-3 target "
        "walls) and mass resolution (σ ≈ 8 MeV),\n"
        "the invariant mass adds almost no discrimination beyond the opening angle "
        "it is built from — the experiment is\n"
        "effectively an angular-spectrum measurement, and Config A maximises it. "
        "The A > B ranking holds at every rung of the\n"
        "background-knowledge ladder.  The X17 excess is a broad coherent surplus "
        "over 100–170° (see stacked spectra), so the\n"
        "discovery power rests on how well the IPC tail shape can be constrained — "
        "by theory, sidebands, and control samples."
    )
    fig.text(0.06, 0.84, "Executive summary", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.815, summary, fontsize=10, va="top", linespacing=1.45)

    # ── Results table ────────────────────────────────────────────────────
    fig.text(0.06, 0.555, "Results (Asimov profile-likelihood significance)",
             fontsize=13, fontweight="bold")
    tbl_ax = fig.add_axes([0.06, 0.22, 0.55, 0.32])
    tbl_ax.axis("off")
    cells  = [[lbl, a, b] for lbl, a, b in _RESULTS]
    table = tbl_ax.table(cellText=cells,
                         colLabels=["Metric", "Config A", "Config B"],
                         colWidths=[0.62, 0.19, 0.19],
                         cellLoc="left", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.22)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eef7")
        if c > 0:
            cell.get_text().set_ha("center")
        if r > 0 and "Z —" in cells[r - 1][0]:
            cell.set_facecolor("#fdf3e3")

    # ── Verdict box ──────────────────────────────────────────────────────
    fig.text(0.645, 0.52,
             "Verdict\n\n"
             "Config A is preferred:\n"
             "Z = 3.0σ vs 2.3σ (IPC shape\n"
             "known), 2.3σ vs 1.7σ (IPC\n"
             "shape fit from data).\n\n"
             "The trigger loss of Config B\n"
             "is not repaid by its\n"
             "calorimetry gain, at every\n"
             "level of bkg knowledge.",
             fontsize=11, va="top",
             bbox=dict(boxstyle="round,pad=0.6", fc="#e7f4e7", ec="#2ca02c", lw=1.5))

    # ── Caveats ──────────────────────────────────────────────────────────
    import textwrap
    _raw_caveats = [
        "Detector response derived from the OLD-geometry Geant4 production "
        "(22 cm arms, 300 bar He-3, 0.9 mm CFRP + 0.5 mm Al target walls, ArIso "
        "drift gas), applied to the 25 cm fast-MC layout. Regenerate the response "
        "after the new-geometry Geant4 run before quoting absolute significances.",
        "The X17 excess is broad (no visible bump): the stat-only Z assumes the "
        "IPC spectrum shape is exactly known. The 'fit from data' rows profile an "
        "exp(polynomial) background over the full range — the truth lies between, "
        "depending on theory/sideband constraints on the IPC tail.",
        "Config B's calorimeter energy response reuses the standard-stack "
        "(Config A) tables — adequate for this relative comparison only.",
        "Rates: IPC = 1.12e-2/pulse ÷ 0.3 efficiency scale (Alberto), "
        "X17/IPC = 2.5%. Z scales as √(exposure) and linearly with the X17 "
        "fraction. Timing: 30 ns resolution, 200 ns coincidence window (lossless).",
        "No cosmic/beam-related backgrounds beyond IPC.",
    ]
    caveats = "Caveats — read before quoting numbers\n" + "\n".join(
        textwrap.fill(f"•  {c}", width=150, subsequent_indent="    ")
        for c in _raw_caveats)
    fig.text(0.06, 0.175, caveats, fontsize=8, va="top", linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.5", fc="#fdeaea", ec="#d62728", lw=1.2))
    pdf.savefig(fig)
    plt.close(fig)


def _image_page(pdf, paths, titles, suptitle, note=None):
    n = len(paths)
    fig, axes = plt.subplots(n, 1, figsize=PAGE)
    if n == 1:
        axes = [axes]
    for ax, p, t in zip(axes, paths, titles):
        ax.imshow(mpimg.imread(p))
        ax.set_title(t, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def main():
    geo = os.path.join(RESULTS, "geometry")
    sig = os.path.join(RESULTS, "significance")
    with PdfPages(OUT_PDF) as pdf:
        _text_page(pdf)
        _image_page(
            pdf,
            [os.path.join(geo, "geometry_detail_A.png"),
             os.path.join(geo, "geometry_detail_B.png")],
            ["Config A — standard:  MM → trigger scint → LS-1 → LS-2 → back-scint",
             "Config B — back-scint first:  MM → trigger scint → back-scint → LS-1 → LS-2"],
            "Detector stack geometry (top-down cross-section + face-on view)",
            note="Active volumes only; dimensions in cm; beam along +Y.")
        _image_page(
            pdf,
            [os.path.join(sig, "significance_study.png")],
            [""],
            "Expected spectra and significance — 30-day exposure",
            note="Top row: Config A.  Bottom row: Config B.  X17 histograms scaled "
                 "×10 for visibility (annotated). Geant4-response smearing applied "
                 "(angular σ68 ≈ 14°, mass σ ≈ 8 MeV).")
        _image_page(
            pdf,
            [os.path.join(sig, "significance_stacked.png")],
            [""],
            "Unscaled stacked spectra — the X17 excess as it appears in data",
            note="X17 stacked on IPC with NO scaling; error bars are the expected "
                 "±√N statistical errors. The excess is a broad coherent surplus "
                 "over ~100–170° (per-bin S/√B ≤ 1.1), not a visible bump — the "
                 "quoted Z sums this excess across bins, and therefore depends on "
                 "how well the IPC tail shape is known (see ladder in the table).")
        _image_page(
            pdf,
            [os.path.join(sig, "enhancement_scan.png")],
            [""],
            "Enhancement scan — symmetric-energy selection (no significant gain)",
            note="A min-reconstructed-energy cut buys <10% in Z: the geometric "
                 "acceptance already removes soft asymmetric pairs. The strongest "
                 "remaining levers are target-wall material (half walls: "
                 "Z 2.9 → 3.2σ in toy) and run time (Z ∝ √days).")
        d = pdf.infodict()
        d["Title"]   = "MX17 Config A vs B — X17 discovery significance"
        d["Author"]  = "Dylan Neff (fast-MC + Geant4 response study)"
        d["Subject"] = "Configuration decision study, old-geometry Geant4 response"
    print(f"[Output] {OUT_PDF}")


if __name__ == "__main__":
    main()
