#!/usr/bin/env python3
"""
Does the Y undershoot cap sit at a fixed ADC level, or does it track the gain?

`xy_shape.py` found det3's Y-view undershoot saturating at about -290 ADC while
X's stays proportional to signal, and `cns_undershoot.py` excluded common-mode
subtraction as the cause. The remaining candidates are an electronics clip in
the return path (a ceiling on the amplifier OUTPUT, i.e. on ADC) and a
detector-side effect (a ceiling on the CHARGE the resistive layer returns).

Two ladders, because they answer different halves:

  MESH ladder (gain varies, drift fixed).  Asks whether undershoot(A) is the
  same function of peak amplitude at different gains. Note what this CANNOT do:
  at a fixed drift field, peak amplitude and collected charge are proportional
  through the gain, so an amplitude ceiling and a charge ceiling predict the
  same universal curve. The mesh ladder tests universality, not mechanism.

  DRIFT ladder (drift varies, mesh fixed at 490 V).  This is the
  degeneracy-breaker. The gas gain and the primary charge are unchanged, but
  stretching the arrival time in a slower drift field halves the peak
  amplitude: at 300 V the median peak is 1291 ADC against 2535 at 1100 V for
  the same muons. So at a FIXED peak amplitude the low-drift events carry
  roughly twice the charge. An amplitude clip predicts the same undershoot;
  a charge-driven return predicts the low-drift events undershooting deeper.

Bins are declared in ABSOLUTE ADC and shared across every voltage so the
comparison is like-for-like, and every railed event (peak >= 3500) is dropped
before anything is measured.

    python3 mx17_sim_wft/hv_slope/cap_scan.py
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

from cns_undershoot import _Reader, CNS_BLOCK              # noqa: E402
from xy_shape import _cross                                # noqa: E402

RUN = "mx17_det3_saturday_scan_6-27-26"
BASE = "/home/dylan/x17/cosmic_bench/det3/"
DET_Z = 702.0
FID_X, FID_Y = (-190.0, 115.0), (-190.0, 165.0)
SIGMA = 5.0
RAIL = 3500.0

MESH = [("hv_scan2_resist_460V_drift_1000V", 460),
        ("hv_scan_resist_475V_drift_1000V", 475),
        ("hv_scan2_resist_490V_drift_1000V", 490),
        ("hv_scan2_resist_500V_drift_1000V", 500)]
DRIFT = [("drift_scan_resist_490V_drift_300V", 300),
         ("drift_scan_resist_490V_drift_500V", 500),
         ("drift_scan_resist_490V_drift_700V", 700),
         ("drift_scan_resist_490V_drift_900V", 900),
         ("drift_scan_resist_490V_drift_1100V", 1100)]

# absolute, shared across every voltage
BINS = [(400, 800), (800, 1200), (1200, 1700), (1700, 2300), (2300, 3000),
        (3000, 3500)]


def refs(sub):
    from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
    setup_paths()
    from M3RefTracking import M3RefTracking
    cfg = get_config("sat_det3")
    cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = BASE, RUN, sub
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    x, y, evn = rays.get_xy_positions(DET_Z)
    x, y, evn = np.asarray(x), np.asarray(y), np.asarray(evn)
    ok = (np.isfinite(x) & np.isfinite(y) & (x > FID_X[0]) & (x < FID_X[1])
          & (y > FID_Y[0]) & (y < FID_Y[1]))
    return cfg, set(int(e) for e in evn[ok])


def one(sub, feu, pos, want, view, ladder, volt):
    valid = ~np.isnan(pos)
    rows = []
    for path in sorted(glob.glob(os.path.join(
            BASE, RUN, sub, "decoded_root", f"*_{feu:02d}.root"))):
        rdr = _Reader(path, cns=True)          # production recipe
        noise = np.where(rdr.noise > 0, rdr.noise, np.inf)
        here = want & set(int(e) for e in rdr.event_ids)
        if not here:
            continue
        for eid, wfm, _cm in rdr.iter_events(here):
            amp = wfm.max(axis=1)
            av = np.where(valid, amp, -np.inf)
            pk = int(np.argmax(av))
            w = wfm[pk]
            ipk = int(np.argmax(w))
            a = float(w[ipk])
            if ipk > 20:                        # need a tail to look into
                continue
            over = valid & (amp / noise >= SIGMA)
            rows.append(dict(
                ladder=ladder, volt=volt, view=view, event_id=eid,
                peak_amp=a, peak_block=pk // CNS_BLOCK, peak_ch=pk,
                undershoot_adc=float(w[ipk + 1:].min()),
                fwhm_ns=((hr - hl) * 60.0
                         if (hl := _cross(w, ipk, a, 0.50, -1)) is not None
                         and (hr := _cross(w, ipk, a, 0.50, +1)) is not None
                         else np.nan),
                q_event=float(amp[over].sum()) if over.any() else 0.0))
    return rows


def profile(g):
    out = []
    for lo, hi in BINS:
        m = (g.peak_amp >= lo) & (g.peak_amp < hi)
        out.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                        peak=float(g.peak_amp[m].median()) if m.any() else None,
                        adc=float(g.undershoot_adc[m].median()) if m.sum() >= 15 else None,
                        q=float(g.q_event[m].median()) if m.any() else None,
                        fwhm=float(g.fwhm_ns[m].median()) if m.sum() >= 15 else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/x17/response_sim/hv_slope/cns"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, "cap_scan.parquet")

    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        print(f"loaded {cache} ({len(df):,} rows)")
    else:
        from wft.io import strip_position_map
        rows = []
        for ladder, points in (("mesh", MESH), ("drift", DRIFT)):
            for sub, volt in points:
                cfg, want = refs(sub)
                pos_maps = strip_position_map(cfg)
                for view, feu in (("x", cfg.MX17_FEU_X), ("y", cfg.MX17_FEU_Y)):
                    r = one(sub, feu, pos_maps[feu], want, view, ladder, volt)
                    rows += r
                print(f"[{ladder} {volt} V] {len(want):,} refs", flush=True)
        df = pd.DataFrame(rows)
        df.to_parquet(cache)

    un = df[df.peak_amp < RAIL]                 # never measure on a railed peak
    out = dict(bins=BINS, rail=RAIL, mesh={}, drift={}, blocks={})
    for ladder in ("mesh", "drift"):
        for view in ("x", "y"):
            for volt in sorted(un[un.ladder == ladder].volt.unique()):
                g = un[(un.ladder == ladder) & (un.view == view)
                       & (un.volt == volt)]
                out[ladder][f"{view}_{volt}"] = profile(g)

    # secondary: is the cap level uniform across the 64-channel blocks?
    lr = un[(un.ladder == "mesh") & (un.volt == 490) & (un.peak_amp >= 2300)]
    for view in ("x", "y"):
        g = lr[lr.view == view]
        out["blocks"][view] = [
            dict(block=int(b), n=int((g.peak_block == b).sum()),
                 adc=float(g.undershoot_adc[g.peak_block == b].median()),
                 peak=float(g.peak_amp[g.peak_block == b].median()))
            for b in sorted(g.peak_block.unique())
            if (g.peak_block == b).sum() >= 15]

    # Does the drift ladder conserve charge? If the low-drift amplitude loss is
    # time-spreading, peak x FWHM is flat and the ladder breaks the
    # amplitude/charge degeneracy. If the peak falls with the width unchanged,
    # it is charge loss, amplitude and charge stay proportional, and the drift
    # ladder discriminates nothing.
    out["drift_charge_check"] = {}
    for view in ("x", "y"):
        rowsv = []
        for volt in sorted(un[un.ladder == "drift"].volt.unique()):
            g = df[(df.ladder == "drift") & (df.view == view)
                   & (df.volt == volt)]
            pk, fw = g.peak_amp.median(), g.fwhm_ns.median()
            rowsv.append(dict(volt=int(volt), n=int(len(g)), peak=float(pk),
                              fwhm=float(fw), area=float(pk * fw / 1000.0)))
        out["drift_charge_check"][view] = rowsv

    json.dump(out, open(os.path.join(a.out_dir, "cap_scan.json"), "w"), indent=1)

    for ladder, lab in (("mesh", "MESH ladder (gain varies, drift 1000 V)"),
                        ("drift", "DRIFT ladder (mesh fixed 490 V)")):
        print(f"\n===== {lab} — median undershoot [ADC] in fixed amplitude bins")
        volts = sorted(un[un.ladder == ladder].volt.unique())
        print(f"{'bin [ADC]':>14} | " + " | ".join(
            f"{v:>5} V" for v in volts for _ in ("x",)) + "     (X view)")
        for i, (lo, hi) in enumerate(BINS):
            cells = []
            for v in volts:
                p = out[ladder][f"x_{v}"][i]
                cells.append(f"{p['adc']:7.0f}" if p["adc"] is not None else "      -")
            print(f"{lo:5d}-{hi:5d}  | " + " | ".join(cells))
        print(f"{'':>14} | " + " | ".join(f"{v:>5} V" for v in volts)
              + "     (Y view)")
        for i, (lo, hi) in enumerate(BINS):
            cells = []
            for v in volts:
                p = out[ladder][f"y_{v}"][i]
                cells.append(f"{p['adc']:7.0f}" if p["adc"] is not None else "      -")
            print(f"{lo:5d}-{hi:5d}  | " + " | ".join(cells))
        # the charge each bin actually carries, for the drift argument
        print(f"{'q_event p50':>14} | " + " | ".join(
            f"{out[ladder][f'y_{v}'][3]['q'] or 0:7.0f}" for v in volts)
            + "   (Y, 1700-2300 ADC bin)")

    print("\n----- drift ladder: is the amplitude loss time-spreading?")
    for view in ("x", "y"):
        print(f"  {view}: " + "  ".join(
            f"{r['volt']}V peak {r['peak']:.0f} fwhm {r['fwhm']:.0f} "
            f"area {r['area']:.0f}" for r in out["drift_charge_check"][view]))

    print("\n----- cap level by 64-channel block (490 V, peak >= 2300 ADC)")
    for view in ("x", "y"):
        print(f"  {view}: " + "  ".join(
            f"b{b['block']}={b['adc']:.0f}({b['n']})" for b in out["blocks"][view]))
    print("\nwrote", os.path.join(a.out_dir, "cap_scan.json"))


if __name__ == "__main__":
    main()
