"""Freeze the pedestal history into the JSON the CERN site's chart page reads.

The site draws its own charts in canvas from a static payload, so this writes
one file and nothing else:

    ~/PycharmProjects/dylan-cern-site/data/x17-pedestals.json

    ../.venv/bin/python -m ntof_pedestal_qa.export_site
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import figures as F
from . import make_report as R
from . import pedestals as P

SITE = os.path.expanduser("~/PycharmProjects/dylan-cern-site")
OUT = os.path.join(SITE, "data", "x17-pedestals.json")


def r3(x):
    """Three significant figures; the payload is served to phones."""
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.3g}")


def main():
    a = R.analyse()
    sets, rows = a["sets"], a["rows"]
    order = F.ORDER

    stamps = [s.when.isoformat(timespec="minutes") for s in sets]

    feus = []
    for feu in order:
        det, view = P.FEU_DET[feu]
        seq = {r["i"]: r for r in rows if r["feu"] == feu}
        feus.append(dict(
            feu=feu, det=det, view=view, label=f"{det}{view}",
            cm=[r3(seq[i]["med_cm"]) if i in seq else None for i in range(len(sets))],
            res=[r3(seq[i]["med_res"]) if i in seq else None for i in range(len(sets))],
            raw=[r3(seq[i]["med_raw"]) if i in seq else None for i in range(len(sets))],
            mean=[r3(seq[i]["med_mean"]) if i in seq else None for i in range(len(sets))],
            thr=[r3(seq[i]["med_thr"] - 256) if i in seq else None for i in range(len(sets))],
            dead=[seq[i]["n_dead"] if i in seq else None for i in range(len(sets))],
            noisy=[seq[i]["n_noisy"] if i in seq else None for i in range(len(sets))],
        ))

    # the 64 x n_sets chip grids, and each chip's own campaign median, so the
    # page can show absolute or relative without shipping both grids
    grids = {}
    for kind in ("cm", "res"):
        g = F._chip_grid(sets, kind)
        grids[kind] = dict(
            values=[[r3(v) for v in row] for row in g],
            median=[r3(v) for v in np.nanmedian(g, axis=1)],
        )

    events = []
    for e in a["episodes"]:
        runs = e["runs"]
        events.append(dict(
            start=e["first"].isoformat(timespec="minutes"),
            end=e["last_seen"].isoformat(timespec="minutes"),
            where=f'{e["det"]}{e["view"]} connector {e["chip"] + 1}',
            det=e["det"], kind="connector",
            what=f'64 channels ({e["chip"] * 64}–{e["chip"] * 64 + 63}) electrically silent',
            covers=(f'run_{runs[0]}–run_{runs[-1]}' if len(runs) > 1
                    else (f'run_{runs[0]}' if runs else '—')),
            subruns=e["subruns"], recovered=True))
    events.append(dict(
        start=R.A_START.isoformat(timespec="minutes"),
        end=R.A_END.isoformat(timespec="minutes"),
        where="A, both views", det="A", kind="common-mode",
        what=f'common mode \u00d7{np.mean(list(a["a_cm_ratio"].values())):.1f} on all 16 chips',
        covers="run_64–run_82", subruns=415, recovered=True))
    events.append(dict(
        start=R.CLOCK_STEP.isoformat(timespec="minutes"), end=None,
        where="all four chambers", det=None, kind="config",
        what=f'residual noise \u00d7{np.median(list(a["res_ratio"].values())):.1f}, '
             f'common mode unchanged',
        covers="run_69–run_162", subruns=822, recovered=False))
    events.sort(key=lambda e: e["start"])

    epochs = []
    for name, ea, eb in F.EPOCHS:
        sel = [r for r in rows if ea <= r["when"] < eb]
        if not sel:
            continue
        epochs.append(dict(
            name=name, start=ea.isoformat(), end=eb.isoformat(),
            n=len({r["stamp"] for r in sel}),
            raw=r3(np.median([r["med_raw"] for r in sel])),
            cm=r3(np.median([r["med_cm"] for r in sel])),
            res=r3(np.median([r["med_res"] for r in sel]))))

    last = []
    for feu in order:
        det, view = P.FEU_DET[feu]
        r = a["last"][feu]
        last.append(dict(label=f"{det}{view}", det=det, feu=feu,
                         mean=r3(r["med_mean"]), spread=r3(r["spread_mean"]),
                         raw=r3(r["med_raw"]), cm=r3(r["med_cm"]),
                         res=r3(r["med_res"]),
                         coherent=r3(r["med_raw"] / r["med_res"]),
                         thr=r3(r["med_thr"] - 256),
                         dead=r["n_dead"], noisy=r["n_noisy"]))

    step = []
    for feu in order:
        det, view = P.FEU_DET[feu]
        b, c = a["before"][feu], a["after"][feu]
        step.append(dict(label=f"{det}{view}", det=det,
                         res_before=r3(b["med_res"]), res_after=r3(c["med_res"]),
                         res_step=r3(c["med_res"] / b["med_res"]),
                         cm_before=r3(b["med_cm"]), cm_after=r3(c["med_cm"]),
                         cm_step=r3(c["med_cm"] / b["med_cm"])))

    age = a["age_h"]
    payload = dict(
        generated=None,                    # stamped by the caller if wanted
        source="/eos/experiment/ntof/data/x17/july_beam/pedestals",
        n_sets=a["n_sets"], n_used=a["n_used"], n_subruns=a["n_subruns"],
        n_channels=4096,
        span=[sets[0].when.isoformat(timespec="minutes"),
              sets[-1].when.isoformat(timespec="minutes")],
        stamps=stamps, feus=feus, grids=grids, events=events, epochs=epochs,
        last=last, step=step,
        clock_step=R.CLOCK_STEP.isoformat(timespec="minutes"),
        res_step=round(float(np.median(list(a["res_ratio"].values()))), 1),
        cm_step=round(float(np.median(list(a["cm_ratio"].values()))), 2),
        a_cm_ratio=round(float(np.mean(list(a["a_cm_ratio"].values()))), 1),
        age=dict(median=r3(np.median(age)), max=r3(age.max()),
                 frac_over_24h=r3(float((age > 24).mean()))),
        fw=dict(r_median=r3(a["fw_agreement"]["r_median"]),
                frac_high=r3(a["fw_agreement"]["frac_high"])),
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
