#!/usr/bin/env python3
"""Side-view schematic of the Saclay cosmic test bench.

Geometry from run_config.json (bench_geometry + detectors, mm):
  M3 reference planes (50x50 cm): z = 24, 144 (below) and 1185, 1302 (above)
  Test slots (MX17 chambers, ~40x40 cm active): P1 z = 227, P2 z = 697
  Trigger scintillators (60x60 cm): one above, one below the full stack.

Output: figures/00-cosmic-bench-schematic.{png,pdf}

Usage:  ../../.venv/bin/python make_bench_diagram.py   (from engineer_package/)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "00-cosmic-bench-schematic"

ACCENT = "#2E598C"      # MX17 chambers
M3_C = "#8FB4D9"        # M3 planes
SCINT_C = "#E4B04A"     # scintillators
MUON_C = "#C23B3B"

# z positions (mm), from run_config.json
M3_Z = [24, 144, 1185, 1302]
SLOT_Z = {"P1 (bottom slot)": 227, "P2 (top slot)": 697}
SCINT_Z = {"bottom": -110, "top": 1420}   # drawn just outside the stack

M3_W, MX_W, SC_W = 500, 400, 600          # widths (mm)
M3_T, MX_T, SC_T = 22, 45, 40             # drawn thicknesses (mm)

fig, ax = plt.subplots(figsize=(7.2, 9.0))

label_x = 380  # where the label arrows land


def slab(zc, w, t, color, alpha=1.0, ec="black"):
    ax.add_patch(Rectangle((-w / 2, zc - t / 2), w, t, facecolor=color,
                           edgecolor=ec, linewidth=1.1, alpha=alpha, zorder=3))


def label(zc, text, color, dy=0):
    ax.annotate(text, xy=(label_x - 40, zc), xytext=(label_x, zc + dy),
                fontsize=11.5, va="center", ha="left", color="black",
                arrowprops=dict(arrowstyle="-", color="0.45", lw=1.0))


# scintillators
for side, z in SCINT_Z.items():
    slab(z, SC_W, SC_T, SCINT_C)
label(SCINT_Z["top"], "Trigger scintillator\n(60 × 60 cm)", SCINT_C)
label(SCINT_Z["bottom"], "Trigger scintillator\n(60 × 60 cm)", SCINT_C)

# M3 planes
for z in M3_Z:
    slab(z, M3_W, M3_T, M3_C)
label((M3_Z[2] + M3_Z[3]) / 2, "Reference tracker:\n2 × M3 Micromegas\n(50 × 50 cm)", M3_C)
label((M3_Z[0] + M3_Z[1]) / 2, "Reference tracker:\n2 × M3 Micromegas\n(50 × 50 cm)", M3_C)

# test chambers
for name, z in SLOT_Z.items():
    slab(z, MX_W, MX_T, ACCENT)
label(SLOT_Z["P2 (top slot)"], "MX17 chamber under test\n(top slot)", ACCENT)
label(SLOT_Z["P1 (bottom slot)"], "MX17 chamber under test\n(bottom slot)", ACCENT)

# cosmic muon track (slightly inclined, downward)
z_hi, z_lo = 1530, -220
x_hi, x_lo = -95, 65
ax.add_patch(FancyArrowPatch((x_hi, z_hi), (x_lo, z_lo), arrowstyle="-|>",
                             mutation_scale=22, color=MUON_C, lw=2.4, zorder=4))
ax.text(x_hi - 18, z_hi - 8, "cosmic muon", fontsize=12.5, color=MUON_C,
        ha="right", va="top", fontstyle="italic")

# height scale bar
ax.annotate("", xy=(-340, 1302), xytext=(-340, 24),
            arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
ax.text(-355, (1302 + 24) / 2, "1.28 m", rotation=90, fontsize=11,
        va="center", ha="right", color="0.25")

# coincidence note
ax.text(0, -300, "Readout triggered by top + bottom scintillator coincidence;\n"
        "the four M3 planes reconstruct each muon independently of the chamber under test.",
        fontsize=10.5, ha="center", va="top", color="0.3")

ax.set_xlim(-430, 660)
ax.set_ylim(-390, 1620)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Saclay cosmic test bench (side view, to scale in height)",
             fontsize=14.5, fontweight="bold", pad=12)

fig.tight_layout()
fig.savefig(OUT.with_suffix(".png"), dpi=220, facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
print(f"wrote {OUT}.png/.pdf")
