"""Raw evidence for the HV-supply-current method of measuring the flash charge.

Section 5 of the report quotes a gamma-flash charge measured two ways on the
same detector, the same evening and the same 25 working points: from the
resistive-layer HV supply current (`imon`), and from direct 1 GS/s integration
of chamber A's strip 32 patched into the n_TOF DAQ.  This figure is the raw
input to the first of the two — the supply current itself, right through the
scan, so a reader can see the plateau structure and the size of the beam-induced
step before any estimator is applied.

The method, established in `ntof_july_analysis/flash_charge/` and applied point
for point to this scan by `ntof_processing/mm_flash/imon_scan.py`, is

    Q_pulse = ( mean(imon) - median(imon) ) / f_pulse

The CAEN readback runs at ~1 Hz while n_TOF pulses arrive every few seconds, so
most samples sit at the standing leakage current: the per-plateau **median** is
the leakage at that voltage and the **mean** carries the beam-induced part.
Both are drawn here, so the quantity the charge is built out of is visible as
the gap between two lines rather than asserted.

Data source
-----------
`/media/dylan/data/x17/ntof_mm_flash/imon_224709.csv` — the detector-A
resistive-layer and drift channels of the DAQ's own `hv_monitor.csv`, pulled out
of local runs run_160/161 and trimmed to the scan, 1 Hz.
`/media/dylan/data/x17/ntof_mm_flash/hv_plateaus_224709.csv` — the 25 plateau
boundaries and their (drift, amplification) setpoints, from the same monitor.

n_TOF run 224709, 9 August 2026, 17:10-19:37, detector A (mx17_3), Ar/iC4H10
90/10.  The scan walks the amplification voltage *down* inside each block of
constant drift field, which is why the current steps down across a block and
jumps back up at each block boundary.

The current axis is logarithmic because the standing leakage is ~30x higher in
the drift-600 and drift-500 blocks than in the first drift-700 block; that
offset is a property of the channel, not of the beam, and is exactly what the
per-plateau median removes.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

MM_FLASH = Path("/media/dylan/data/x17/ntof_mm_flash")
IMON_CSV = MM_FLASH / "imon_224709.csv"
PLATEAU_CSV = MM_FLASH / "hv_plateaus_224709.csv"

# Seconds discarded after each setpoint change, so a plateau's statistics are
# not contaminated by the ramp.  Same value as mm_flash/imon_scan.py.
SETTLE_S = 45

DRIFT_COLOUR = {700: "#dfe8f1", 600: "#efe4d6", 500: "#e2ecdf"}


def _load_imon() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(datetime64 timestamps, detector-A resist imon [uA], resist vmon [V])."""
    ts, cur, vol = [], [], []
    with IMON_CSV.open() as fh:
        for r in csv.DictReader(fh):
            try:
                i = float(r["A_resist_imon"])
                v = float(r["A_resist_vmon"])
            except (TypeError, ValueError):
                continue
            ts.append(np.datetime64(r["timestamp"].replace(" ", "T")))
            cur.append(i)
            vol.append(v)
    return np.array(ts), np.array(cur), np.array(vol)


def _load_plateaus() -> list[dict]:
    out = []
    with PLATEAU_CSV.open() as fh:
        for r in csv.DictReader(fh):
            out.append(
                dict(
                    t0=np.datetime64(r["start"].replace(" ", "T")),
                    t1=np.datetime64(r["end"].replace(" ", "T")),
                    drift=int(r["A_drift_V"]),
                    resist=int(r["A_resist_V"]),
                )
            )
    return out


def hv_current_scan(out: Path) -> dict:
    """Detector-A resistive-layer supply current across the 25-plateau HV scan.

    Top panel: every 1 Hz `imon` sample of the scan, with each plateau's median
    (the leakage at that voltage) and mean (leakage + beam) drawn over it, and
    the amplification setpoint labelled.  Background shading groups the
    plateaus into their four blocks of constant drift voltage.

    Bottom panel: the amplification setpoint itself, so the staircase that
    drives the current is unambiguous.

    Returns the per-plateau table (mean, median, difference) plus the scan-wide
    summary numbers quoted in the report.
    """
    t, i_ua, v_res = _load_imon()
    plateaus = _load_plateaus()

    rows = []
    for p in plateaus:
        lo = p["t0"] + np.timedelta64(SETTLE_S, "s")
        m = (t >= lo) & (t <= p["t1"])
        if m.sum() < 60:
            continue
        med = float(np.median(i_ua[m]))
        mean = float(np.mean(i_ua[m]))
        rows.append(
            dict(
                drift_v=p["drift"],
                resist_v=p["resist"],
                t0=str(p["t0"]),
                t1=str(p["t1"]),
                n_samples=int(m.sum()),
                i_median_ua=med,
                i_mean_ua=mean,
                di_ua=mean - med,
                i_max_ua=float(np.max(i_ua[m])),
                _lo=lo,
                _hi=p["t1"],
            )
        )

    fig, (ax, axd, axv) = plt.subplots(
        3, 1, figsize=(12.0, 7.8), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1.5, 0.9], hspace=0.09),
    )

    # --- blocks of constant drift voltage, as background bands ---------------
    blocks = []
    for p in plateaus:
        if blocks and blocks[-1][0] == p["drift"]:
            blocks[-1][2] = p["t1"]
        else:
            blocks.append([p["drift"], p["t0"], p["t1"]])
    for drift, b0, b1 in blocks:
        for a in (ax, axd, axv):
            a.axvspan(b0, b1, color=DRIFT_COLOUR.get(drift, "#eeeeee"), zorder=0)
        ax.annotate(
            f"drift {drift} V",
            xy=(b0 + (b1 - b0) / 2, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=9.5, color="#3a3f45",
            fontweight="bold",
        )

    # --- the raw readback ----------------------------------------------------
    ax.plot(t, i_ua, lw=0.45, color="#2f6f9f", alpha=0.85, zorder=2,
            label="imon readback, 1 Hz")

    # --- what the estimator takes off each plateau ---------------------------
    for k, r in enumerate(rows):
        span = [r["_lo"], r["_hi"]]
        ax.plot(span, [r["i_median_ua"]] * 2, color="#2e7d4f", lw=2.0, zorder=4,
                label="plateau median = leakage" if k == 0 else None)
        ax.plot(span, [r["i_mean_ua"]] * 2, color="#c0632c", lw=2.0, ls="--",
                zorder=5, label="plateau mean = leakage + beam" if k == 0 else None)

    ax.set_yscale("log")
    ax.set_ylim(8e-3, 9.0)
    ax.set_ylabel("detector A resistive-layer\nsupply current  [µA]")
    ax.set_title(
        "The gamma-flash charge, as the HV supply sees it — "
        "25-plateau scan, n_TOF run 224709, 9 August 2026",
        fontsize=12.5, fontweight="bold", loc="left", pad=18,
    )
    ax.grid(axis="y", alpha=0.3, which="major")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, ncol=3)

    # --- the beam-induced part, plateau by plateau ---------------------------
    # This is the quantity the charge is built out of: what the supply delivers
    # over and above its own leakage, i.e. the avalanche ion current.
    for r in rows:
        w = r["_hi"] - r["_lo"]
        axd.bar(r["_lo"] + w / 2, r["di_ua"] * 1e3, width=w,
                color="#c0632c", alpha=0.85, zorder=3, edgecolor="white",
                linewidth=0.6)
        axd.annotate(
            f"{r['resist_v']}", xy=(r["_lo"] + w / 2, r["di_ua"] * 1e3),
            xytext=(0, 3), textcoords="offset points", ha="center",
            va="bottom", fontsize=7.4, rotation=90, color="#5a6169",
        )
    axd.set_ylim(0, max(r["di_ua"] for r in rows) * 1e3 * 1.42)
    axd.set_ylabel("beam-induced current\nmean − median  [nA]")
    axd.annotate(
        "numbers are the amplification setpoint of the plateau  [V]",
        xy=(0.005, 0.94), xycoords="axes fraction", ha="left", va="top",
        fontsize=8, color="#5a6169", style="italic",
    )
    axd.grid(axis="y", alpha=0.3)
    axd.set_axisbelow(True)

    # --- the staircase -------------------------------------------------------
    axv.plot(t, v_res, lw=1.3, color="#7a4b8a", zorder=3)
    axv.set_ylabel("amplification\n[V]")
    axv.set_xlabel("wall-clock time, 9 August 2026")
    axv.grid(alpha=0.3)
    axv.set_axisbelow(True)
    axv.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axv.xaxis.set_major_locator(mdates.MinuteLocator(byminute=(0, 20, 40)))
    axv.set_xlim(t.min(), t.max())

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    for r in rows:
        r.pop("_lo")
        r.pop("_hi")

    hi = max(rows, key=lambda r: r["di_ua"])
    ref = [r for r in rows if r["drift_v"] == 700 and r["resist_v"] == 540]
    return {
        "ntof_run": 224709,
        "detector": "A (mx17_3)",
        "gas": "Ar/iC4H10 90/10",
        "source_imon": str(IMON_CSV),
        "source_plateaus": str(PLATEAU_CSV),
        "t_start": str(t.min()),
        "t_end": str(t.max()),
        "n_imon_samples": int(t.size),
        "n_plateaus": len(rows),
        "drift_v_used": sorted({r["drift_v"] for r in rows}),
        "resist_v_range": [min(r["resist_v"] for r in rows),
                           max(r["resist_v"] for r in rows)],
        "leakage_range_ua": [min(r["i_median_ua"] for r in rows),
                             max(r["i_median_ua"] for r in rows)],
        "di_range_ua": [min(r["di_ua"] for r in rows),
                        max(r["di_ua"] for r in rows)],
        "di_max_plateau": {k: hi[k] for k in
                           ("drift_v", "resist_v", "i_median_ua",
                            "i_mean_ua", "di_ua")},
        "working_point_700_540": (
            {k: ref[0][k] for k in ("i_median_ua", "i_mean_ua", "di_ua")}
            if ref else None),
        "peak_sample_ua": float(np.max(i_ua)),
        "plateaus": rows,
    }


if __name__ == "__main__":
    import json

    here = Path(__file__).resolve().parent
    (here / "figures").mkdir(exist_ok=True)
    s = hv_current_scan(here / "figures/hv_current_scan.png")
    s.pop("plateaus")
    print(json.dumps(s, indent=1))
