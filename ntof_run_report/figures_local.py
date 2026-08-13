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

# every DREAM sub-run's event count, scanned off EOS once (see count_events.py)
EVENT_CENSUS = Path(__file__).resolve().parent / "data/events_per_subrun.csv"

PHASE_COLOUR = {
    "setup": "#8aa5c4",       # 28 Jun - 13 Jul
    "trigger": "#e0a458",     # 14 Jul - 25 Jul
    "production": "#5b9279",  # 26 Jul - 10 Aug
}

# The set-up phase ends when the scintillator system is complete: the fourth
# liquid cell went in during the 14 July access, which is also the access that
# removed the mesh charge-injection circuits and swapped the trigger from the
# liquids to the plastics.  Production starts with run_79.
SETUP_END = "2026-07-14"
PRODUCTION_START = "2026-07-26"


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
        if d < SETUP_END:
            colours.append(PHASE_COLOUR["setup"])
        elif d < PRODUCTION_START:
            colours.append(PHASE_COLOUR["trigger"])
        else:
            colours.append(PHASE_COLOUR["production"])

    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.bar(x, frac, width=0.82, color=colours, edgecolor="none")
    total_on = sum(per_day[d][0] for d in days)
    total = sum(per_day[d][1] for d in days)
    ax.axhline(100 * total_on / total, color="#444", ls="--", lw=1.2,
               label=f"campaign mean  {100*total_on/total:.1f} %")

    # the two phase boundaries, drawn between bars rather than through one
    for boundary, label in ((SETUP_END, "14 Jul — scintillator system\n"
                                        "complete, 4th liquid cell in"),
                            (PRODUCTION_START, "26 Jul — run_79, final\n"
                                               "configuration frozen")):
        xb = np.datetime64(boundary) - np.timedelta64(12, "h")
        ax.axvline(xb, color="#3a3f45", lw=1.6, zorder=5)
        ax.annotate(label, xy=(xb, 104), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color="#3a3f45", va="bottom", ha="left")
    ax.set_ylim(0, 134)
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
    ax.annotate("10 Aug — partial day,\nHV off 09:15", xy=(x[-1], 62),
                xytext=(x[-5], 112), fontsize=8.5, color="#666", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.9))
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


def events_collected(out: Path) -> dict:
    """Total DREAM events recorded per day, over the whole campaign.

    Counted from `events_per_subrun.csv`, which is one row per (sub-run, file
    tag) with the number of entries in that tag's `nt` tree — one entry per
    triggered event.  FEU 1 stands for the sub-run: every FEU sees the same
    global triggers.  The timestamp comes out of the DAQ's own file name, so
    this figure does not depend on the n_TOF side, on the clock fit, or on
    anything having been matched.  Beam vs cosmics is each run's `beam_type`
    from its `run_config.json`.

    Produced by `ntof_run_report/count_events.py`, which has to run on lxplus
    against `/eos/experiment/ntof/data/x17/july_beam/runs`; the CSV is committed
    because that scan takes ~8 minutes and needs EOS.
    """
    rows = list(csv.DictReader(EVENT_CENSUS.open()))
    KINDS = ("neutrons", "cosmics", "pulser")
    per_day = defaultdict(lambda: dict.fromkeys(KINDS, 0))
    bad = 0
    for r in rows:
        if not r["events"] or not r["stamp"]:
            bad += 1
            continue
        kind = r["beam_type"] if r["beam_type"] in KINDS else "pulser"
        per_day[r["stamp"][:10]][kind] += int(r["events"])

    days = sorted(per_day)
    x = [np.datetime64(d) for d in days]
    beam, cos, pul = (
        np.array([per_day[d][k] for d in days], float) for k in KINDS
    )

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.bar(x, beam / 1e6, width=0.82, color=PHASE_COLOUR["production"],
           label="beam")
    ax.bar(x, cos / 1e6, width=0.82, bottom=beam / 1e6, color="#b9bfc6",
           label="cosmics (beam-off reference)")
    ax.bar(x, pul / 1e6, width=0.82, bottom=(beam + cos) / 1e6, color="#e0a458",
           label="pulser (DAQ characterisation)")
    ax.set_ylabel("DREAM events recorded per day\n[millions]")
    import datetime as _dt
    span = (f"{_dt.date.fromisoformat(days[0]):%-d %B} – "
            f"{_dt.date.fromisoformat(days[-1]):%-d %B %Y}")
    ax.set_title(f"DREAM events recorded, {span}",
                 fontsize=13, fontweight="bold", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    for boundary, label in ((SETUP_END, "14 Jul"), (PRODUCTION_START, "26 Jul")):
        xb = np.datetime64(boundary) - np.timedelta64(12, "h")
        ax.axvline(xb, color="#3a3f45", lw=1.4, zorder=5)

    ax2 = ax.twinx()
    cum = np.cumsum(beam + cos + pul) / 1e6
    ax2.plot(x, cum, color="#8a4b2a", lw=2.0)
    ax2.set_ylabel("cumulative [millions]", color="#8a4b2a")
    ax2.tick_params(axis="y", colors="#8a4b2a")
    ax2.set_ylim(0, cum[-1] * 1.25)
    ax2.annotate(f"{cum[-1]:.1f} M events total", xy=(x[-1], cum[-1]),
                 xytext=(-6, 8), textcoords="offset points", ha="right",
                 fontsize=10.5, color="#8a4b2a", fontweight="bold")
    ax2.plot([], [], color="#8a4b2a", lw=2.0, label="cumulative, all")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=9, frameon=False)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    return {
        "total": int(beam.sum() + cos.sum() + pul.sum()),
        "beam": int(beam.sum()),
        "cosmic": int(cos.sum()),
        "pulser": int(pul.sum()),
        "days": len(days),
        "best_day": max(days, key=lambda d: sum(per_day[d].values())),
        "best_day_events": int(max(sum(per_day[d].values()) for d in days)),
        "unreadable_tags": bad,
        "tags": len(rows),
    }
