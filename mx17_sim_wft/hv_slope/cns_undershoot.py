#!/usr/bin/env python3
"""
Is the amplitude-limited Y undershoot made by common-mode subtraction?

`xy_shape.py` found that det3's Y-view undershoot saturates at about -290 ADC
in absolute terms while X's stays proportional to the signal, which is what
makes the Y undershoot *fraction* fall with amplitude and what made T14's
rail-depleted Y leg look 2.7 points deeper than the detector. The obvious
suspect is CNS: `wft.io.FeuReader` subtracts a per-sample median over each
64-channel block, and FEU 8 (the Y view here) is one of the two FEUs with a
large raw common mode.

NO REPROCESSING IS NEEDED. The saturday scan was written with
`pedestal_subtraction: false` and `common_noise_subtraction: false`, so
decoded_root holds RAW samples and FeuReader does pedestal + CNS itself, in
software, on every read. Toggling CNS is therefore a switch on the read, not an
mm_processor run — and because FeuReader computes its pedestal from the raw
stack BEFORE the CNS step, the two legs share one pedestal exactly. (The
mm_processor gotcha that pedestal RMS is always post-CNS does not apply here:
nothing in this path uses mm_processor pedestals.)

`_Reader` below mirrors wft.io.FeuReader's recipe with `cns` switchable, and
`--verify` checks it reproduces FeuReader bit-for-bit at cns=True so the
duplication cannot silently drift.

    python3 mx17_sim_wft/hv_slope/cns_undershoot.py --verify
    python3 mx17_sim_wft/hv_slope/cns_undershoot.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path[:0] = [_HERE, _REPO, os.path.join(_REPO, "mx_june_cosmic_qa"),
                os.path.join(_REPO, "cosmic_bench_analysis")]

RUN = "mx17_det3_saturday_scan_6-27-26"
SUB = "long_run_resist_490V_drift_1000V"
BASE = "/home/dylan/x17/cosmic_bench/det3/"
DET_Z = 702.0
FID_X, FID_Y = (-190.0, 115.0), (-190.0, 165.0)
CNS_BLOCK = 64
N_PED = 300
ALIGN = 12          # sample the common-mode average is aligned to
NAVG = 44


class _Reader:
    """wft.io.FeuReader's pedestal/CNS recipe, with CNS switchable and the
    per-block common mode exposed. Verified bit-identical at cns=True."""

    def __init__(self, path, cns=True, n_ped=N_PED):
        import uproot
        self.cns = cns
        self.tree = uproot.open(path)["nt"]
        self.n_entries = self.tree.num_entries
        a0 = self.tree.arrays(["amplitude"], entry_stop=min(n_ped, self.n_entries),
                              library="np")["amplitude"]
        lens = np.array([len(a) // 512 for a in a0])
        self.n_sample = int(np.bincount(lens).argmax())
        stack = np.stack([a.reshape(self.n_sample, 512)
                          for a, l in zip(a0, lens) if l == self.n_sample]
                         ).astype(np.float32)
        self.ped = np.median(stack, axis=(0, 1))          # pre-CNS, both legs
        sub = stack - self.ped[None, None, :]
        nblk = 512 // CNS_BLOCK
        if cns:
            cms = np.median(sub.reshape(len(stack), self.n_sample, nblk,
                                        CNS_BLOCK), axis=3)
            sub -= np.repeat(cms, CNS_BLOCK, axis=2)
        self.noise = (1.4826 * np.median(np.abs(sub), axis=(0, 1))).astype(np.float32)
        self.event_ids = self.tree.arrays(["eventId"], library="np")["eventId"]

    def iter_events(self, wanted=None):
        """Yield (eventId, W[512, n_sample], CM[nblk, n_sample])."""
        idx = (np.where(np.isin(self.event_ids,
                                np.fromiter(wanted, dtype=np.int64)))[0]
               if wanted is not None else np.arange(self.n_entries))
        nblk = 512 // CNS_BLOCK
        for lo in range(0, len(idx), 400):
            block = idx[lo:lo + 400]
            arr = self.tree.arrays(["eventId", "amplitude"],
                                   entry_start=int(block[0]),
                                   entry_stop=int(block[-1]) + 1, library="np")
            base = int(block[0])
            for i in block:
                j = i - base
                wfm = arr["amplitude"][j].reshape(-1, 512).astype(np.float32) - self.ped
                ns = wfm.shape[0]
                cms = np.median(wfm.reshape(ns, nblk, CNS_BLOCK), axis=2)
                if self.cns:
                    wfm = wfm - np.repeat(cms, CNS_BLOCK, axis=1)
                yield int(arr["eventId"][j]), wfm.T, cms.T


def verify(path):
    from wft.io import FeuReader
    ref, mine = FeuReader(path), _Reader(path, cns=True)
    assert np.array_equal(ref.ped, mine.ped), "pedestal differs"
    assert np.array_equal(ref.noise, mine.noise), "noise differs"
    want = set(int(e) for e in ref.event_ids[:40])
    a = {e: w for e, _f, w in ref.iter_events(want)}
    b = {e: w for e, w, _c in mine.iter_events(want)}
    assert set(a) == set(b), "event sets differ"
    worst = max(float(np.abs(a[e] - b[e]).max()) for e in a)
    print(f"[verify] {len(a)} events, max |FeuReader - _Reader| = {worst:.3e}")
    assert worst == 0.0, "waveforms differ"
    print("[verify] bit-identical at cns=True")


def refs_in_fiducial():
    from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
    setup_paths()
    from M3RefTracking import M3RefTracking
    cfg = get_config("sat_det3")
    cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = BASE, RUN, SUB
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    x, y, evn = rays.get_xy_positions(DET_Z)
    x, y, evn = np.asarray(x), np.asarray(y), np.asarray(evn)
    ok = (np.isfinite(x) & np.isfinite(y) & (x > FID_X[0]) & (x < FID_X[1])
          & (y > FID_Y[0]) & (y < FID_Y[1]))
    return cfg, set(int(e) for e in evn[ok])


def run(cfg, feu, pos, want, view, cns):
    valid = ~np.isnan(pos)
    rows = []
    acc_cm = np.zeros(NAVG)
    n_acc = 0
    for path in sorted(glob.glob(os.path.join(
            BASE, RUN, SUB, "decoded_root", f"*_{feu:02d}.root"))):
        rdr = _Reader(path, cns=cns)
        here = want & set(int(e) for e in rdr.event_ids)
        if not here:
            continue
        for eid, wfm, cm in rdr.iter_events(here):
            amp = np.where(valid, wfm.max(axis=1), -np.inf)
            pk = int(np.argmax(amp))
            w = wfm[pk]
            ipk = int(np.argmax(w))
            a = float(w[ipk])
            tail = w[ipk + 1:]
            blk = cm[pk // CNS_BLOCK]           # the block this strip sits in
            rows.append(dict(
                view=view, cns=cns, event_id=eid, peak_amp=a, peak_sample=ipk,
                undershoot=float(tail.min() / a) if len(tail) else np.nan,
                undershoot_adc=float(tail.min()) if len(tail) else np.nan,
                # what CNS removed from this strip, at the peak and in the tail
                cm_at_peak=float(blk[ipk]),
                cm_tail_min=float(blk[ipk + 1:].min()) if ipk + 1 < len(blk) else np.nan))
            lo = ALIGN - ipk
            if 0 <= lo and lo + len(blk) <= NAVG:
                acc_cm[lo:lo + len(blk)] += blk
                n_acc += 1
    return rows, acc_cm / max(n_acc, 1), n_acc


BINS = [(0, 800), (800, 1500), (1500, 2200), (2200, 3000), (3000, 3500),
        (3500, 1e9)]


def profile(g):
    out = []
    for lo, hi in BINS:
        m = (g.peak_amp >= lo) & (g.peak_amp < hi)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                        peak=float(g.peak_amp[m].median()),
                        frac=float(g.undershoot[m].median()),
                        adc=float(g.undershoot_adc[m].median()),
                        cm_tail_min=float(g.cm_tail_min[m].median())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/x17/response_sim/hv_slope/cns"))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, "cns_ab.parquet")

    cfg, want = refs_in_fiducial()
    if a.verify:
        p = sorted(glob.glob(os.path.join(BASE, RUN, SUB, "decoded_root",
                                          "*_07.root")))[0]
        verify(p)
        return

    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        cmavg = json.load(open(os.path.join(a.out_dir, "cm_avg.json")))
    else:
        from wft.io import strip_position_map
        pos_maps = strip_position_map(cfg)
        rows, cmavg = [], {}
        for view, feu in (("x", cfg.MX17_FEU_X), ("y", cfg.MX17_FEU_Y)):
            for cns in (True, False):
                r, cm, n = run(cfg, feu, pos_maps[feu], want, view, cns)
                print(f"  {view} cns={cns}: {len(r):,} events", flush=True)
                rows += r
                if cns:      # the common mode is the same either way
                    cmavg[view] = dict(feu=int(feu), n=int(n), avg=cm.tolist())
        df = pd.DataFrame(rows)
        df.to_parquet(cache)
        json.dump(cmavg, open(os.path.join(a.out_dir, "cm_avg.json"), "w"),
                  indent=1)

    df = df[df.peak_sample <= 20]
    out = dict(run=RUN, sub_run=SUB, profiles={}, summary={}, cm_avg=cmavg)
    for view in ("x", "y"):
        for cns in (True, False):
            g = df[(df.view == view) & (df.cns == cns)]
            k = f"{view}_cns{int(cns)}"
            out["profiles"][k] = profile(g)
            out["summary"][k] = dict(
                n=int(len(g)), peak_p50=float(g.peak_amp.median()),
                undershoot_p50=float(g.undershoot.median()),
                undershoot_adc_p50=float(g.undershoot_adc.median()))
    json.dump(out, open(os.path.join(a.out_dir, "cns_undershoot.json"), "w"),
              indent=1)

    print(f"\n{'':22s} {'peak':>7} {'under %':>8} {'under ADC':>10} "
          f"{'CM tail min':>12}")
    for view in ("x", "y"):
        print(f"--- {view.upper()} view (FEU {cmavg[view]['feu']}), "
              f"median raw noise ratio shown in the report")
        for cns in (True, False):
            g = df[(df.view == view) & (df.cns == cns)]
            print(f"  CNS {'ON ' if cns else 'OFF'} overall     "
                  f"{g.peak_amp.median():7.0f} "
                  f"{100 * g.undershoot.median():7.1f}% "
                  f"{g.undershoot_adc.median():10.0f}")
            for p in out["profiles"][f"{view}_cns{int(cns)}"]:
                print(f"      {p['lo']:5.0f}-{min(p['hi'], 4200):5.0f} "
                      f"n={p['n']:5d} {p['peak']:7.0f} {100 * p['frac']:7.1f}% "
                      f"{p['adc']:10.0f} {p['cm_tail_min']:12.1f}")
    print("\nwrote", os.path.join(a.out_dir, "cns_undershoot.json"))


if __name__ == "__main__":
    main()
