"""The two figures that only this report makes.

`beam_availability` — the beam record for the whole run, from the DAQ's own
NXCALS logger (`FTN.BCT477`, protons on the n_TOF target).  The logger writes a
one-minute classification: `on` if n_TOF cycles delivered protons in that
minute, otherwise `off_ps` / `off_ntof` according to whether the PS was
delivering to anybody else at the time.

**The off_ps / off_ntof split is not reliable and the report does not use it.**
Scheduled-but-not-delivered and delivered-to-others are read off a supercycle
snapshot that mislabels short stops and every access.  The *total* is a direct
statement about protons on target and is trustworthy; the blame is not.  This
figure therefore shows one number per day — the fraction of the day with beam.

`timeline` — the four phases, drawn from the logbook dates.
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BEAM_LOG = Path(os.path.expanduser("~/x17/beam_july/slow_control/beam_intensity"))

PHASE_COLOUR = {
    "setup": "#8aa5c4",       # 28 Jun - 8 Jul
    "trigger": "#e0a458",     # 9 Jul - 25 Jul
    "production": "#5b9279",  # 26 Jul - 10 Aug
}


def _load_minutes():
    rows = []
    for f in sorted(glob.glob(str(BEAM_LOG / "beam_class_*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append((r["timestamp"][:10], r["beam_class"], int(r["ntof_beam"])))
    return rows


def beam_availability(out: Path) -> dict:
    rows = _load_minutes()
    per_day = defaultdict(lambda: [0, 0])  # day -> [on, total]
    pulses = defaultdict(int)
    for day, cls, npulse in rows:
        per_day[day][1] += 1
        if cls == "on":
            per_day[day][0] += 1
        pulses[day] += npulse

    days = sorted(per_day)
    x = [np.datetime64(d) for d in days]
    frac = np.array([per_day[d][0] / per_day[d][1] for d in days]) * 100

    colours = []
    for d in days:
        if d < "2026-07-09":
            colours.append(PHASE_COLOUR["setup"])
        elif d < "2026-07-26":
            colours.append(PHASE_COLOUR["trigger"])
        else:
            colours.append(PHASE_COLOUR["production"])

    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.bar(x, frac, width=0.82, color=colours, edgecolor="none")
    total_on = sum(per_day[d][0] for d in days)
    total = sum(per_day[d][1] for d in days)
    ax.axhline(100 * total_on / total, color="#444", ls="--", lw=1.2,
               label=f"campaign mean  {100*total_on/total:.1f} %")
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("fraction of the day with protons\non the n_TOF target  [%]")
    ax.set_title("Beam availability, 1 July – 10 August 2026",
                 fontsize=13, fontweight="bold", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    handles = [
        Patch(color=PHASE_COLOUR["setup"], label="hardware set-up"),
        Patch(color=PHASE_COLOUR["trigger"], label="thermal trigger build-up"),
        Patch(color=PHASE_COLOUR["production"], label="production data taking"),
    ]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=9, ncol=4, frameon=False)
    ax.annotate("1 Jul: 5 % — first beam of the run", xy=(x[0], 7),
                xytext=(x[1], 118), fontsize=8.5, color="#555", va="center",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))
    ax.annotate("10 Aug: partial day,\nHV off 09:15", xy=(x[-1], 60),
                xytext=(x[-7], 112), fontsize=8.5, color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    prod = [d for d in days if d >= "2026-07-26"]
    return {
        "days": len(days),
        "hours": total / 60,
        "on_pct": 100 * total_on / total,
        "prod_on_pct": 100
        * sum(per_day[d][0] for d in prod)
        / sum(per_day[d][1] for d in prod),
        "prod_hours": sum(per_day[d][1] for d in prod) / 60,
        "pulses": sum(pulses.values()),
        "worst_day": min(days, key=lambda d: per_day[d][0] / per_day[d][1]),
        "worst_frac": 100 * min(per_day[d][0] / per_day[d][1] for d in days),
    }
