#!/usr/bin/env python3
"""One figure: det4's efficiency map at H4, and what the beam says about the
stripe pattern that the June cosmics could only blur."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = ("/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/"
        "mapping_check/")
RUNS = [
    ("_run53", "505-535 V, flat", "#9ecae1"),
    ("_run56", "610-620 V, flat", "#4292c6"),
    ("_run57", "655-670 V, rotated 25°", "#08306b"),
]
J = np.load("/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
            "det4_sps_assessment/stripes_g_det4.npz")

cmap = LinearSegmentedColormap.from_list(
    "eff", ["#241a33", "#3b5b92", "#3f9e9e", "#8ecf6a", "#f7f056"])
cmap.set_bad("#eaeaea")

fig = plt.figure(figsize=(14.5, 11.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.95],
                      width_ratios=[1.45, 1], hspace=0.5, wspace=0.24)

# --- A: the map, flat mount at its best point -------------------------------
M = np.load(BASE + "det4_efficiency_map_run56.npz", allow_pickle=True)
ax = fig.add_subplot(gs[0, 0])
im = ax.pcolormesh(M["ex"], M["ey"], M["eff"].T, cmap=cmap, vmin=0, vmax=1)
ax.set(xlabel="detector-local X [mm]", ylabel="detector-local Y [mm]",
       title="A. det4 efficiency map — run 56, 610-620 V, flat mount\n"
             "3 mm cells; reconstructed within 5 mm of the uRWELL track")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, fraction=0.036, label="efficiency")

# --- B: along the strips, inside one live band ------------------------------
ax = fig.add_subplot(gs[0, 1])
for tag, lab, col in RUNS:
    m = np.load(BASE + f"det4_efficiency_map{tag}.npz", allow_pickle=True)
    ax.plot(m["prof_y_centres"], m["prof_y"], color=col, lw=1.6, label=lab)
ax.set(xlabel="detector-local Y [mm]", ylabel="efficiency", ylim=(0, 1.0),
       title="B. along the strips, inside the X 149-161 band\n"
             "80 % and flat — the bands are genuinely 1-D")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)

# --- C: the X profile, the whole point --------------------------------------
ax = fig.add_subplot(gs[1, :])
for lo, hi in J["bands"]:
    ax.axvspan(lo, hi, color="tab:green", alpha=0.15, zorder=0)
for tag, lab, col in RUNS:
    m = np.load(BASE + f"det4_efficiency_map{tag}.npz", allow_pickle=True)
    ax.plot(m["prof_x_centres"], m["prof_x"], color=col, lw=1.7, label=lab)
ax2 = ax.twinx()
ax2.semilogy(J["c"], np.maximum(J["med"], 1), color="tab:orange", lw=1.1, alpha=0.85)
ax2.set_ylabel("June cosmic median charge [ADC]", color="tab:orange")
ax2.tick_params(axis="y", colors="tab:orange")
mm = np.load(BASE + "det4_efficiency_map_run56.npz", allow_pickle=True)
ax.set(xlim=(mm["prof_x_centres"][0], mm["prof_x_centres"][-1]), ylim=(0, 1.05),
       xlabel="detector-local X [mm]", ylabel="efficiency within 5 mm",
       title="C. the striped coordinate, 1 mm bins — green = June cosmic live bands, "
             "orange = June median charge")
ax.legend(loc="upper center", fontsize=8.5, ncol=3, title="resist voltage",
          title_fontsize=8, framealpha=0.95)
ax.grid(alpha=0.25)
ax.annotate("dead notch at 188-199, inside what\nJune called one 38 mm band",
            xy=(194, 0.03), xytext=(206, 0.45), fontsize=8.5, ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"))
ax.annotate("the band to aim at:\n149-161, 80 % efficient", xy=(155, 0.83),
            xytext=(163, 0.62), fontsize=8.5, ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.9, color="0.25"))

# --- D: the numbers ---------------------------------------------------------
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
head = ["resist [V]", "505-535", "610-620", "655-670", "June bench, 495 V"]
rows = [
    ["mount", "flat", "flat", "rotated 25°", "cosmic bench"],
    ["clean reference tracks", "398,397", "30,712", "46,506", "12.9 k rays"],
    ["det4 fired at all", "27.6 %", "44.4 %", "49.8 %", "95.6 %"],
    ["within 5 mm, whole beam spot", "10.9 %", "26.2 %", "30.4 %", "40.1 % (whole chamber)"],
    ["within 5 mm, in June bands", "15.6 %", "37.8 %", "44.5 %", "77.4 %"],
    ["within 5 mm, between bands", "0.0 %", "0.1 %", "0.1 %", "15.4 %"],
    ["within 5 mm, band X 149-161", "21 %", "75 %", "80 %", "80.0 % (this band)"],
    ["best 10 mm window", "—", "83 % @ 150-160", "85 % @ 150-160", "97 % @ 205-213"],
    ["discharge-flagged", "0.1 %", "0.7 %", "1.8 %", "8.2 %"],
]
t = ax.table(cellText=rows, colLabels=head, loc="upper center", cellLoc="center",
             bbox=[0, 0, 1, 0.92])
t.auto_set_font_size(False)
t.set_fontsize(9)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("0.85")
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#eef2f7")
    if c == 0:
        cell.set_text_props(ha="left")
    if c == 4:
        cell.set_facecolor("#fdf3e6")
    if r == 7:
        cell.set_facecolor("#e6f3e6" if c != 4 else "#f0f3e0")
ax.text(0.5, 1.0, "D. det4 at H4 against its own June cosmic numbers — inside a "
        "good band it already matches the bench",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5)

fig.suptitle("det4 (mx17_E) efficiency at SPS H4, referenced to the uRWELL telescope "
             "— 2026-08-01", fontsize=13.5, y=0.98)
fig.savefig(BASE + "det4_efficiency_summary.png", dpi=115, bbox_inches="tight")
print("wrote", BASE + "det4_efficiency_summary.png")
