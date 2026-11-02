#!/usr/bin/env python3
"""Side-view schematic of the Saclay cosmic test bench (Paper I fig. bench_setup).

Geometry from the run configs (run_config.json: `detectors[].det_center_coords` and `bench_geometry`, mm):
  M3 reference planes (50x50 cm): z = 24, 144 (below) and 1185, 1302 (above)
  Test slots (MX17 chambers, ~40x40 cm active): chamber centre z = 232 (bottom) and 702 (top);
    the slot plates are at bench_geometry p1_z = 227 / p2_z = 697 (+5 mm board thickness).
  Trigger scintillators (60x60 cm): one above, one below the full stack (drawn, not to a measured z).

det3 golden rows: both ran det3 (mx17_3) in the TOP slot, det_center_coords z = 702 --
  cosmic_bench/det3/mx17_det3_p2_det1_overnight_6-27-26/run_config.json   (6-27 p2 long_run, the quoted row)
  cosmic_bench/det3/mx17_det3_saturday_scan_6-27-26/run_config.json       (sat_det3, 6-27 long run 490 V)
  (qa_config.py g_det3_wknd / sat_det3: det_z = 702, FEU 7/8.)

Style: papers/paper_style.py (det3 colour for the det3 slot).
Output: figures/00-cosmic-bench-schematic.{png,pdf}

Usage:  ../../.venv/bin/python make_bench_diagram.py   (from engineer_package/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "papers"))
import paper_style as ps  # noqa: E402

OUT = HERE / "figures" / "00-cosmic-bench-schematic"

ps.use()
plt.rcParams["axes.grid"] = False

DET3_C = ps.color("det3")   # the det3 (golden-row) slot
OTHER_C = "#9bb0c8"         # the other test slot
M3_C = "#cfd8e3"            # M3 planes
SCINT_C = "#f0d58c"         # scintillators
MUON_C = ps.ACCENT

# z positions (mm), from the run configs (det_center_coords)
M3_Z = [24, 144, 1185, 1302]
SLOT_Z = {"bottom": 232, "top": 702}
SCINT_Z = {"bottom": -110, "top": 1420}   # drawn just outside the stack (schematic)

M3_W, MX_W, SC_W = 500, 400, 600          # widths (mm)
M3_T, MX_T, SC_T = 22, 45, 40             # drawn thicknesses (mm; the slab thicknesses are not to scale)

# Printed at 4.3 in (publication checklist 2026-11-02). The z positions are to scale; the horizontal scale is not the
# vertical one (the caption says so, with the trigger and the golden-row dates that left the figure).
fig, ax = ps.fig_for(4.3, height=3.7, layout=None)
fig.subplots_adjust(0, 0, 1, 1)
FS = ps.BASE_PT
label_x = 360   # where the label leaders land (right)
zlab_x = -330   # z labels (left)


def slab(zc, w, t, color, ec="#333333", lw=0.8):
    ax.add_patch(Rectangle((-w / 2, zc - t / 2), w, t, facecolor=color,
                           edgecolor=ec, linewidth=lw, zorder=3))


def label(zc, text, dy=0, color=ps.INK, weight="normal"):
    ax.annotate(text, xy=(label_x - 50, zc), xytext=(label_x, zc + dy),
                fontsize=FS, va="center", ha="left", color=color, fontweight=weight,
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7))


def zlabel(zc, w, color=ps.INK, weight="normal"):
    ax.plot([zlab_x + 10, -w / 2 - 8], [zc, zc], color="0.7", lw=0.6, ls=":", zorder=2)
    ax.text(zlab_x, zc, f"{zc:g}", fontsize=FS, ha="right", va="center", color=color, fontweight=weight)


# scintillators
for side, z in SCINT_Z.items():
    slab(z, SC_W, SC_T, SCINT_C)
label(SCINT_Z["top"], "trigger scintillator")
label(SCINT_Z["bottom"], "trigger scintillator")

# M3 planes
for z in M3_Z:
    slab(z, M3_W, M3_T, M3_C)
    zlabel(z, M3_W)
label((M3_Z[2] + M3_Z[3]) / 2, "2 M3 reference planes")
label((M3_Z[0] + M3_Z[1]) / 2, "2 M3 reference planes")

# test chambers
slab(SLOT_Z["top"], MX_W, MX_T, DET3_C, lw=1.0)
slab(SLOT_Z["bottom"], MX_W, MX_T, OTHER_C)
zlabel(SLOT_Z["top"], MX_W, color=DET3_C, weight="bold")
zlabel(SLOT_Z["bottom"], MX_W)
label(SLOT_Z["top"], "det3 (top slot)", color=DET3_C, weight="bold")
label(SLOT_Z["bottom"], "MX17 chamber (bottom slot)")

# z axis title on the left
ax.text(zlab_x, 1530, "$z$ [mm]", fontsize=FS, ha="right", va="bottom", color=ps.INK)

# cosmic muon track (slightly inclined, downward)
z_hi, z_lo = 1530, -220
x_hi, x_lo = -95, 65
ax.add_patch(FancyArrowPatch((x_hi, z_hi), (x_lo, z_lo), arrowstyle="-|>",
                             mutation_scale=12, color=MUON_C, lw=1.5, zorder=4))
ax.text(x_hi + 22, z_hi - 8, "cosmic muon", fontsize=FS, color=MUON_C,
        ha="left", va="top", fontstyle="italic")

ax.set_xlim(-490, 1040)
ax.set_ylim(-240, 1600)
ax.axis("off")

import pandas as pd  # noqa: E402
geom = pd.DataFrame(
    [dict(element="scintillator", z_mm=z, width_mm=SC_W, z_measured=False) for z in SCINT_Z.values()]
    + [dict(element="M3", z_mm=z, width_mm=M3_W, z_measured=True) for z in M3_Z]
    + [dict(element=f"MX17 {k} slot", z_mm=z, width_mm=MX_W, z_measured=True) for k, z in SLOT_Z.items()])
ps.save(fig, OUT, data=geom)
print(f"wrote {OUT}.png/.pdf/.csv")
