"""The dimensioned top-down (plan) view of the EAR2 station.

Drawn from the **as-built Geant4 model**, not from a sketch: every coordinate
below was dumped out of `MX17_Full_Geant` by replaying `DetectorConstruction`'s
own depth-chain arithmetic against `SimConfig.hh` and `MX17ModuleGeometry.hh`,
so the drawing and the simulation are the same geometry.

Frame (`GEOMETRY_COORDINATE_CONVENTION.md`):

* origin = centre of the ³He capsule
* **the beam is +Y — vertical, floor to ceiling** — so in this plan view it
  comes straight out of the page and the four arms lie in the plane of it
* screen x = world Z (East, right), screen y = world X (North, up)
* arm **A = +Z**, **B = −X**, **C = −Z**, **D = +X**, a clockwise pinwheel

Two numbers on the drawing carry a caveat, both stated in the report text:

* the capsule-to-strip-plane distance is **24.0 cm in this as-built model** but
  **23.5 cm in the reconstruction frame** (`ntof_tracking/reco/geometry.py`),
  because the surveyed 40.8/40.9 cm window separation may have been referenced
  to the flange front rather than the mylar.  `docs/HANDOFF_ARM_GEOMETRY.md`
  has it open.  The drawing shows the model and the report says both.
* the liquid cells are drawn at their **modelled** 6.5 L bulged profile.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

# ---------------------------------------------------------------- the model
# Depths are measured from each arm's mylar window front face, along the
# outward normal w; transverse positions are along the tangent u.
W_WINDOW = {"A": 204.5, "C": 204.5, "B": 204.0, "D": 204.0}  # survey
W_DRIFT_FRONT, W_DRIFT_BACK = 5.1191, 35.1101   # drift gas, 29.99 mm of it
W_STRIP = 36.2293                                # L6 X-strip copper
W_FRAME = 49.8352                                # module back
W_WALL_C_F, W_WALL_C_B = 110.0, 145.0            # SiPM bar container, 35 mm
W_WALL_F = (W_WALL_C_F + W_WALL_C_B) / 2 - 1.5   # active scint, 3 mm, centred
W_WALL_B = (W_WALL_C_F + W_WALL_C_B) / 2 + 1.5   # in the container
W_PLASTIC_F, W_PLASTIC_B = {                     # 20 mm PVT, per-arm gap
    "A": (208.22, 228.22), "B": (206.22, 226.22),
    "C": (206.22, 226.22), "D": (210.22, 230.22),
}["A"], None

PLASTIC = {"A": (208.22, 228.22), "B": (206.22, 226.22),
           "C": (206.22, 226.22), "D": (210.22, 230.22)}
LIQUID = {"A": (268.0, 289.2), "B": (272.0, 293.2),
          "C": (272.0, 293.2), "D": (268.0, 289.2)}
LIQUID_BULGE = 12.511

# Pinwheel: the module is slid along -u relative to the mechanical structure.
SHIFT = {"A": 16.35, "B": 15.75, "C": 17.30, "D": 15.50}

ACTIVE_W = 399.0      # measured active width, 39.9 cm
FRAME_W = 440.0       # gas frame / flange outer square
WALL_W = 500.0        # 20 bars x 25 mm
BAR_W = 25.0
LIVE_BARS = range(1, 17)      # 0 and 17-19 are not instrumented
PLASTIC_U = 101.72            # bar centre offset, on the module axis
PLASTIC_HW = 100.22
LIQUID_HW = 225.6

CAPSULE = [(11.5, "#9aa7b1", "CFRP"), (10.6, "#c8ced4", "Al"),
           (10.0, "#e8b04b", "³He")]

# (outward unit vector, tangent unit vector) in plan-view (screen x=Z, y=X)
ARM = {
    "A": ((0.0, 1.0), (1.0, 0.0)),     # w = +Z (right),  u = +X (up)
    "C": ((0.0, -1.0), (-1.0, 0.0)),   # w = -Z (left),   u = -X (down)
    "D": ((1.0, 0.0), (0.0, -1.0)),    # w = +X (up),     u = -Z (left)
    "B": ((-1.0, 0.0), (0.0, 1.0)),    # w = -X (down),   u = +Z (right)
}
# NOTE screen coords: point = w_hat * depth + u_hat * transverse, where
# w_hat/u_hat are written (screen_x, screen_y) = (world Z, world X).
ARM_SCREEN = {
    "A": ((1.0, 0.0), (0.0, 1.0)),
    "C": ((-1.0, 0.0), (0.0, -1.0)),
    "D": ((0.0, 1.0), (-1.0, 0.0)),
    "B": ((0.0, -1.0), (1.0, 0.0)),
}

COL = {
    "drift": "#5b9279", "strip": "#8a4b2a", "frame": "#98a2ac",
    "wall": "#2f6f5e", "plastic": "#e0a458", "liquid": "#8aa5c4",
    "dim": "#3a3f45",
}


def _slab(ax, arm, w0, w1, u0, u1, *, shift_u=0.0, **kw):
    """A rectangle given in (depth, transverse) for one arm, in plan view."""
    wh, uh = ARM_SCREEN[arm]
    w0 += W_WINDOW[arm]
    w1 += W_WINDOW[arm]
    u0 += shift_u
    u1 += shift_u
    corners = [(w * wh[0] + u * uh[0], w * wh[1] + u * uh[1])
               for w, u in ((w0, u0), (w1, u0), (w1, u1), (w0, u1))]
    ax.add_patch(plt.Polygon(corners, closed=True, **kw))


def _pt(arm, w, u, *, shift_u=0.0):
    wh, uh = ARM_SCREEN[arm]
    w += W_WINDOW[arm]
    u += shift_u
    return (w * wh[0] + u * uh[0], w * wh[1] + u * uh[1])


def _dim(ax, p0, p1, label, *, off=0.0, perp=(0, 1), fs=9.5, color=None):
    """A double-headed dimension line with a label at its middle."""
    color = color or COL["dim"]
    p0 = (p0[0] + perp[0] * off, p0[1] + perp[1] * off)
    p1 = (p1[0] + perp[0] * off, p1[1] + perp[1] * off)
    ax.annotate("", xy=p0, xytext=p1,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.1,
                                shrinkA=0, shrinkB=0))
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    rot = 90 if abs(p1[0] - p0[0]) < abs(p1[1] - p0[1]) else 0
    ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
            rotation=rot, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                      alpha=0.92))


def topdown(out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(10.4, 10.4))

    for arm in ("A", "B", "C", "D"):
        s = SHIFT[arm]
        half = ACTIVE_W / 2
        # ---- module: frame, drift gas, strip plane
        _slab(ax, arm, 0.0, W_FRAME, -FRAME_W / 2, FRAME_W / 2, shift_u=-s,
              fc="none", ec=COL["frame"], lw=1.0, ls="--", zorder=2)
        _slab(ax, arm, W_DRIFT_FRONT, W_DRIFT_BACK, -half, half, shift_u=-s,
              fc=COL["drift"], ec="none", alpha=0.55, zorder=3)
        _slab(ax, arm, W_STRIP - 0.9, W_STRIP + 0.9, -half, half, shift_u=-s,
              fc=COL["strip"], ec="none", zorder=4)

        # ---- SiPM wall: 20 bars, 16 instrumented, centred on the STRUCTURE.
        # Drawn at the true 3 mm active-scint depth, centred in the 35 mm bar
        # container (W_WALL_C_F/B) it sits in.
        for i in range(20):
            u0 = -WALL_W / 2 + i * BAR_W
            live = i in LIVE_BARS
            _slab(ax, arm, W_WALL_F, W_WALL_B, u0 + 1.6, u0 + BAR_W - 1.6,
                  fc=COL["wall"] if live else "#c9ced3", ec="none",
                  alpha=0.95 if live else 0.5, zorder=4)

        # ---- two plastic bars, centred on the MODULE axis
        for sgn in (-1, 1):
            c = sgn * PLASTIC_U
            _slab(ax, arm, *PLASTIC[arm], c - PLASTIC_HW, c + PLASTIC_HW,
                  shift_u=-s, fc=COL["plastic"], ec="#b07a34", lw=0.7,
                  zorder=4)

        # ---- liquid cell: flat slab plus the modelled bulge
        lw0, lw1 = LIQUID[arm]
        _slab(ax, arm, lw0, lw1, -LIQUID_HW, LIQUID_HW,
              fc=COL["liquid"], ec="#5f7fa4", lw=0.8, alpha=0.85, zorder=3)
        wh, uh = ARM_SCREEN[arm]
        t = np.linspace(-1, 1, 60)
        for w_edge, sgn in ((lw0, -1), (lw1, +1)):
            wcurve = w_edge + sgn * LIQUID_BULGE * np.sqrt(np.clip(1 - t**2, 0, 1))
            ucurve = t * LIQUID_HW
            xs = (wcurve + W_WINDOW[arm]) * wh[0] + ucurve * uh[0]
            ys = (wcurve + W_WINDOW[arm]) * wh[1] + ucurve * uh[1]
            ax.plot(xs, ys, color="#5f7fa4", lw=0.8, zorder=3)

        # ---- arm label, outside everything
        lx, ly = _pt(arm, 330.0, 0.0)
        ax.text(lx, ly, arm, fontsize=22, fontweight="bold", ha="center",
                va="center", color="#1b1f24", alpha=0.75,
                bbox=dict(boxstyle="circle,pad=0.30", fc="white",
                          ec="#c2c8ce", lw=1.0))

    # ------------------------------------------------------------- capsule
    for r, c, _ in CAPSULE:
        ax.add_patch(Circle((0, 0), r, fc=c, ec="#5a636e", lw=0.8, zorder=6))
    ax.plot([0], [0], marker="o", ms=3.0, color="#1b1f24", zorder=7)
    ax.text(20, -6, "³He capsule", fontsize=9.5, color="#1b1f24",
            ha="left", va="top", zorder=9)
    ax.text(524, 618,
            "³He capsule\nØ 20 mm of gas at 500 bar, in a 0.6 mm Al wall\n"
            "and a 0.9 mm CFRP wrap — Ø 23 mm overall.\n"
            "The neutron beam runs vertically, out of the page.",
            fontsize=9.5, color="#1b1f24", ha="right", va="top",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f6f8f9", ec="#d6dbe0"),
            zorder=9)

    # distance ladder, in an empty diagonal corner rather than as leader lines
    rows = [("strip read-out plane", (W_WINDOW["A"] + W_STRIP) / 10),
            ("SiPM wall", (W_WINDOW["A"] + W_WALL_F) / 10),
            ("plastic bars", 41.3), ("liquid cell", 47.2)]
    txt = "from the capsule centre outward\n" + "\n".join(
        f"{lab:.<22}{d:>5.1f} cm" for lab, d in rows)
    ax.text(-530, 618, txt, fontsize=9.5, family="monospace", va="top",
            ha="left", color="#3a3f45", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f6f8f9", ec="#d6dbe0"),
            zorder=9)

    # ---------------------------------------------------- dimension lines
    # window-to-window across the beam, both pairs (the surveyed numbers)
    _dim(ax, (204.5, -300), (-204.5, -300), "40.9 cm", fs=11)
    _dim(ax, (300, 204.0), (300, -204.0), "40.8 cm", fs=11)

    # capsule centre to arm A's strip plane
    _dim(ax, (0, 55), (W_WINDOW["A"] + W_STRIP, 55),
         f"{(W_WINDOW['A'] + W_STRIP) / 10:.1f} cm", fs=11)

    # the 30 mm drift gap, called out on arm A where there is room
    ax.annotate("30 mm drift gap",
                xy=_pt("A", (W_DRIFT_FRONT + W_DRIFT_BACK) / 2, 150,
                       shift_u=-SHIFT["A"]),
                xytext=(292, 292), fontsize=9.5, color="#2f5c4c",
                ha="left",
                arrowprops=dict(arrowstyle="->", color="#2f5c4c", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none",
                          alpha=0.9), zorder=9)

    # the active width, on arm B
    _dim(ax, _pt("B", W_STRIP + 26, -ACTIVE_W / 2, shift_u=-SHIFT["B"]),
         _pt("B", W_STRIP + 26, ACTIVE_W / 2, shift_u=-SHIFT["B"]),
         "39.9 cm active", fs=10)

    # ------------------------------------------------------------- legend
    handles = [
        Rectangle((0, 0), 1, 1, fc=COL["drift"], alpha=0.55,
                  label="micromegas drift volume (30 mm)"),
        Rectangle((0, 0), 1, 1, fc=COL["strip"],
                  label="strip read-out plane (512 × 778.5 µm per view)"),
        Rectangle((0, 0), 1, 1, fc=COL["wall"],
                  label="SiPM wall — 16 instrumented bars of 2.5 cm"),
        Rectangle((0, 0), 1, 1, fc="#c9ced3", alpha=0.55,
                  label="wall bars not instrumented"),
        Rectangle((0, 0), 1, 1, fc=COL["plastic"], ec="#b07a34",
                  label="plastic bars on PMTs (2 cm)"),
        Rectangle((0, 0), 1, 1, fc=COL["liquid"], ec="#5f7fa4", alpha=0.85,
                  label="liquid-scintillator cell (never in the trigger)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015),
              ncol=2, fontsize=10, frameon=False)

    lim = 545
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, 625)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("The station seen from above, to scale\n"
                 "from the as-built Geant4 model",
                 fontsize=13.5, fontweight="bold", loc="left", pad=14)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "window_sep_cm": (40.8, 40.9),
        "pinwheel_shift_cm": (min(SHIFT.values()) / 10, max(SHIFT.values()) / 10),
        "strip_plane_cm_model": round((W_WINDOW["A"] + W_STRIP) / 10, 2),
        "strip_plane_cm_reco": 23.5,
        "drift_gap_mm": round(W_DRIFT_BACK - W_DRIFT_FRONT, 2),
        "active_cm": (39.9, 36.0),
        "wall_cm": round((W_WINDOW["A"] + W_WALL_F) / 10, 1),
        "plastic_cm": round((W_WINDOW["A"] + PLASTIC["A"][0]) / 10, 1),
        "liquid_cm": round((W_WINDOW["A"] + LIQUID["A"][0]) / 10, 1),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(topdown(Path(__file__).parent / "figures/setup_topdown.png"),
                     indent=2))
