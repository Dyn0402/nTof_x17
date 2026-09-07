"""Loading and derived quantities for the DREAM pedestal history.

The raw material is `data/ped_stats.npz`, written on lxplus by
`lxplus/extract_pedestals.py`: for every pedestal acquisition of the campaign,
for each of the eight FEUs, per-channel `mean`, `raw_sigma`, `cns_sigma` and
`shape_rms`, per-chip `cm_rms`, and the firmware's own numbers parsed out of
the `_ped.aux` / `_thr.aux` files.

Vocabulary, fixed here and used everywhere downstream:

    raw sigma     per-channel std of the raw ADC over all samples
    common mode   per chip, per time sample, the median over live channels of
                  (amplitude - channel mean); `cm_rms` is its std
    residual      what is left of a channel once its chip's common mode is
                  subtracted; `cns_sigma` is its std

This is the same decomposition the end-of-run report uses, so the numbers on
the two pages are the same numbers.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NCH, BLK, NBLK = 512, 64, 8

# FEU -> (chamber, view).  ntof_active_area/clusters.py is the authority.
FEU_DET = {3: ("A", "x"), 4: ("A", "y"),
           5: ("B", "x"), 6: ("B", "y"),
           7: ("C", "x"), 8: ("C", "y"),
           1: ("D", "x"), 2: ("D", "y")}
DETS = ["A", "B", "C", "D"]
DET_FEUS = {d: [f for f, (dd, _) in sorted(FEU_DET.items()) if dd == d]
            for d in DETS}

# A channel whose strip has come off the preamp input stops picking up what its
# neighbours pick up, so its *raw* noise collapses while the rest of the FEU
# keeps swinging with the common mode.  That is the disconnection signature, and
# it is the raw sigma that carries it -- the residual barely moves, because the
# residual is what is left after the common mode is taken out either way.
#
# Loudness, by contrast, only shows once the common mode is gone, so it is cut
# on the residual.  Both cuts are relative to the FEU's own median in the same
# pedestal run: the absolute ADC scale moved by 2x mid-campaign when the readout
# clock changed, and a fixed threshold would have read that as 4 000 new dead
# channels overnight.
DEAD_FRAC = 0.12        # raw sigma below this fraction of the FEU median raw
DEAD_ABS = 8.0          # ...or below this in ADC outright, whatever the FEU does
NOISY_FRAC = 3.0        # residual above this multiple of the FEU median residual


@dataclass
class PedSet:
    """One pedestal acquisition: eight FEUs, 512 channels each."""
    stamp: str                    # '260701_19H52'
    when: datetime
    directory: str
    feus: dict                    # feu -> dict of per-channel arrays

    @property
    def iso(self):
        return self.when.strftime("%Y-%m-%dT%H:%M")

    @property
    def day(self):
        return self.when.strftime("%Y-%m-%d")


def _stamp_dt(stamp):
    d, t = stamp.split("_")
    return datetime(2000 + int(d[0:2]), int(d[2:4]), int(d[4:6]),
                    int(t[0:2]), int(t[3:5]))


def load(path=None):
    """Every pedestal acquisition, oldest first."""
    path = path or os.path.join(DATA, "ped_stats.npz")
    z = np.load(path, allow_pickle=True)
    meta = z["meta"]
    sets = []
    for stamp, _iso, dirname, feulist in meta:
        feus = {}
        for f in feulist.split(","):
            feu = int(f)
            pre = f"{stamp}/{feu:02d}"
            keys = [k for k in z.files if k.startswith(pre + "/")]
            feus[feu] = {k.rsplit("/", 1)[1]: z[k] for k in keys}
        sets.append(PedSet(stamp, _stamp_dt(stamp), dirname, feus))
    sets.sort(key=lambda s: s.when)
    return sets


def live_mask(d):
    """Channels usable for a median: finite and not railed."""
    return np.isfinite(d["cns_sigma"]) & (d["cns_sigma"] > 0)


def classify(d):
    """(disconnected, noisy, median residual) for one FEU of one pedestal run."""
    raw, res = d["raw_sigma"], d["cns_sigma"]
    ok = live_mask(d)
    med_raw = np.median(raw[ok]) if ok.any() else np.nan
    med_res = np.median(res[ok]) if ok.any() else np.nan
    dead = ok & ((raw < DEAD_FRAC * med_raw) | (raw < DEAD_ABS))
    noisy = ok & (res > NOISY_FRAC * med_res)
    return dead, noisy, med_res


def silent_connectors(sets, frac=DEAD_FRAC):
    """Whole 64-channel connectors that went electrically silent, with dates.

    A connector whose median raw noise sits below `frac` of its FEU's median has
    stopped seeing the detector; grouped into contiguous episodes so the answer
    is "this connector, these dates" rather than a list of pedestal runs.
    """
    state = {}
    episodes = []
    for s in sets:
        now = set()
        for feu, d in s.feus.items():
            fm = np.median(d["raw_sigma"])
            for c in range(NBLK):
                cm_ = np.median(d["raw_sigma"][c * BLK:(c + 1) * BLK])
                if cm_ < frac * fm or cm_ < DEAD_ABS:
                    now.add((feu, c))
        for key in now - set(state):
            state[key] = s
        for key in set(state) - now:
            episodes.append(dict(feu=key[0], chip=key[1],
                                 det=FEU_DET[key[0]][0], view=FEU_DET[key[0]][1],
                                 first=state[key].when, last_seen=s.when,
                                 recovered=True))
            del state[key]
    for key, first in state.items():
        episodes.append(dict(feu=key[0], chip=key[1],
                             det=FEU_DET[key[0]][0], view=FEU_DET[key[0]][1],
                             first=first.when, last_seen=sets[-1].when,
                             recovered=False))
    episodes.sort(key=lambda e: e["first"])
    return episodes


def feu_summary(d):
    """Scalar summary of one FEU in one pedestal run."""
    dead, noisy, med_res = classify(d)
    good = live_mask(d) & ~dead & ~noisy
    m = d["mean"]
    return dict(
        med_mean=float(np.median(m)),
        spread_mean=float(np.percentile(m, 84) - np.percentile(m, 16)),
        med_raw=float(np.median(d["raw_sigma"])),
        med_res=float(med_res),
        med_res_good=float(np.median(d["cns_sigma"][good])) if good.any() else np.nan,
        cm_rms=np.asarray(d["cm_rms"], float),
        med_cm=float(np.median(d["cm_rms"])),
        n_dead=int(dead.sum()),
        n_noisy=int(noisy.sum()),
        med_shape=float(np.median(d["shape_rms"])),
        med_thr=float(np.median(d["fw_thr"])) if "fw_thr" in d else np.nan,
        med_fw_zs=float(np.median(d["fw_zs_std"])) if "fw_zs_std" in d else np.nan,
        dead=dead, noisy=noisy,
    )


def series(sets):
    """Tidy per-(set, FEU) table plus the per-chip common-mode cube.

    Returns (rows, cm) where rows is a list of dicts and cm has shape
    (n_sets, 8 FEUs, 8 chips) indexed by FEU order 1..8.
    """
    rows = []
    cm = np.full((len(sets), 8, NBLK), np.nan)
    for i, s in enumerate(sets):
        for feu, d in s.feus.items():
            su = feu_summary(d)
            cm[i, feu - 1] = su["cm_rms"]
            det, view = FEU_DET[feu]
            rows.append(dict(i=i, stamp=s.stamp, when=s.when, feu=feu,
                             det=det, view=view,
                             **{k: v for k, v in su.items()
                                if k not in ("cm_rms", "dead", "noisy")}))
    return rows, cm


def load_usage(path=None):
    """Sub-run -> pedestal-in-force, as written by lxplus/extract_usage.py."""
    path = path or os.path.join(DATA, "ped_usage.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["start"] = (datetime.strptime(r["subrun_start"], "%y%m%d_%HH%M")
                      if r["subrun_start"] else None)
        r["ped_dt"] = (datetime.fromisoformat(r["pedestal_time"])
                       if r["pedestal_time"] else None)
    return rows


def step_events(rows, key="med_cm", rel=0.5, abs_floor=2.0):
    """Consecutive-pedestal steps in `key` big enough to be worth naming.

    A step is reported when the value changes by more than `rel` of the smaller
    of the two *and* by more than `abs_floor` ADC, so that a jitter on a large
    number and a doubling of a tiny one are both excluded.
    """
    out = []
    for feu in sorted(FEU_DET):
        seq = [r for r in rows if r["feu"] == feu]
        seq.sort(key=lambda r: r["when"])
        for a, b in zip(seq, seq[1:]):
            va, vb = a[key], b[key]
            if not (np.isfinite(va) and np.isfinite(vb)):
                continue
            d = vb - va
            if abs(d) < abs_floor or abs(d) < rel * min(va, vb):
                continue
            out.append(dict(feu=feu, det=a["det"], view=a["view"],
                            when=b["when"], prev_when=a["when"],
                            key=key, before=va, after=vb, delta=d))
    out.sort(key=lambda r: r["when"])
    return out
