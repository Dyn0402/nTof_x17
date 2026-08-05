#!/usr/bin/env python3
"""run_60: the gas-flush transient, measured.

The CO2->CF4 switch happened in the 20:24-21:10 access (TAX-dated,
GAS_FLUSH_TIMELINE.md); run_60 then took 24 x 30 min sub-runs at FIXED HV
(drift 700.5 V, resist 649.75 V) and fixed mount (25.64 deg) while the
mixture exchanged at ~2 ln/h into ~4.6 L. Each sub-run is one point on the
flush curve; SPS beam died at ~04:50 so only overnight_00..14 carry beam.

Per sub-run, det4 (FEU3) hits only — no reference needed:

  gain proxy   median per-event leading-strip amplitude, per view
  occupancy    mean in-time hits/event, per view
  drift span   t10-t90 of the hit-time distribution (v_drift tracker)
  rate         events with >=1 in-time det4 hit / wall-clock

Each observable is fitted with A(t) = A_inf + (A0 - A_inf) exp(-t/tau);
agreement of tau across observables = the flush constant of this chamber on
this line. run_59 (detE_long_00, last CO2 dataset, same HV) provides the
t<0 anchor once decoded — plotted, not fitted.

  ../../.venv/bin/python flush_run60.py [--fit-lo 0] [--nmax 15]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "det4_sps_assessment"))
from det4_sps_map import VIEW                              # noqa: E402

STAGE = "/media/dylan/data/x17/sps_run53_det4_check/staging/"
T_SWITCH = datetime(2026, 8, 1, 20, 45, 0)   # +-20 min, mid-access
# in-time gate for 64x60ns with latency 32: prompt ~ sample 8-9, ladder a few us
TIME_LO, TIME_HI = 300.0, 3700.0             # ns, generous in-window gate
AMP_LO, AMP_HI = 60.0, 3000.0                # leading-strip acceptance


def subrun_bounds(run_dir):
    """(name -> (start_dt, end_dt)) from dream_daq.log."""
    out, start = {}, {}
    pat_s = re.compile(r"([\d-]+ [\d:]+),\d+ INFO: Subrun started: (\S+)")
    pat_f = re.compile(r"([\d-]+ [\d:]+),\d+ INFO: Subrun finished: (\S+)")
    with open(os.path.join(run_dir, "dream_daq.log")) as f:
        for line in f:
            m = pat_s.search(line)
            if m:
                start[m.group(2)] = datetime.strptime(m.group(1),
                                                      "%Y-%m-%d %H:%M:%S")
            m = pat_f.search(line)
            if m and m.group(2) in start:
                out[m.group(2)] = (start[m.group(2)],
                                   datetime.strptime(m.group(1),
                                                     "%Y-%m-%d %H:%M:%S"))
    return out


def hv_of(subdir):
    """(drift vmon, resist vmon) medians and peak-to-peak from hv_monitor."""
    p = os.path.join(subdir, "hv_monitor.csv")
    if not os.path.exists(p):
        return None
    dr, re_ = [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                dr.append(float(r["8:8 vmon"]))
                re_.append(float(r["12:2 vmon"]))
            except (KeyError, ValueError):
                pass
    if not dr:
        return None
    return (float(np.median(dr)), float(np.ptp(dr)),
            float(np.median(re_)), float(np.ptp(re_)))


def analyse_subrun(subdir, sub):
    """Aggregate observables over all hits_*_03.root files of one sub-run."""
    files = sorted(glob.glob(os.path.join(subdir, f"hits_{sub}_*_03.root")))
    if not files:
        return None
    lead_x, lead_y, nx, ny, tqs = [], [], [], [], []
    n_events = 0
    for fn in files:
        with uproot.open(fn) as f:
            if "hits" not in f:
                continue
            t = f["hits"]
            if t.num_entries == 0:
                continue
            a = t.arrays(["eventId", "channel", "amplitude", "time"],
                         library="np")
        ev, ch = a["eventId"], a["channel"].astype(int)
        amp, tt = a["amplitude"], a["time"]
        ok = (tt >= TIME_LO) & (tt <= TIME_HI) & (amp >= AMP_LO)
        ev, ch, amp, tt = ev[ok], ch[ok], amp[ok], tt[ok]
        if len(ev) == 0:
            continue
        vx = VIEW[ch] == "x"
        n_events += len(np.unique(ev))
        for vmask, leads, ns in ((vx, lead_x, nx), (~vx, lead_y, ny)):
            e_, a_ = ev[vmask], amp[vmask]
            if len(e_) == 0:
                continue
            o = np.lexsort((-a_, e_))
            es, as_ = e_[o], a_[o]
            first = np.r_[True, es[1:] != es[:-1]]
            lamp = as_[first]
            leads.append(lamp[(lamp >= AMP_LO) & (lamp <= AMP_HI)])
            ns.append(np.bincount(np.searchsorted(np.unique(e_), e_)))
        tqs.append(tt[amp >= 150])
    if n_events == 0:
        return None
    lead_x = np.concatenate(lead_x) if lead_x else np.array([np.nan])
    lead_y = np.concatenate(lead_y) if lead_y else np.array([np.nan])
    nx = np.concatenate(nx) if nx else np.array([np.nan])
    ny = np.concatenate(ny) if ny else np.array([np.nan])
    tq = np.concatenate(tqs) if tqs else np.array([np.nan])
    q10, q90 = (np.nanpercentile(tq, 10), np.nanpercentile(tq, 90)) \
        if len(tq) > 100 else (np.nan, np.nan)
    return dict(n_events=n_events,
                gain_x=float(np.nanmedian(lead_x)),
                gain_y=float(np.nanmedian(lead_y)),
                occ_x=float(np.nanmean(nx)), occ_y=float(np.nanmean(ny)),
                span=float(q90 - q10),
                n_lead=int(len(lead_y)))


def expfit(t, y):
    """Lagged exponential: the new mixture reaches the chamber only after the
    line volume has been pushed through (measured: lag 1.7 h at 2 ln/h), then
    exchanges with constant tau. Fitting a plain exponential to a transient
    with a lag is what produced the spurious 12-15 h taus of the first pass."""
    def f(t, ainf, a0, tlag, tau):
        return np.where(t < tlag, a0,
                        ainf + (a0 - ainf) * np.exp(-(t - tlag) / tau))
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return None
    p0 = [y[ok][-1], y[ok][0], 1.7, 3.5]
    try:
        p, cov = curve_fit(f, t[ok], y[ok], p0=p0,
                           bounds=([-np.inf, -np.inf, 0.0, 0.2],
                                   [np.inf, np.inf, 4.0, 30.0]),
                           maxfev=50000)
        err = np.sqrt(np.abs(np.diag(cov)))
        return p, err, f
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=STAGE + "run_60/")
    ap.add_argument("--nmax", type=int, default=15)
    ap.add_argument("--out", default=STAGE + "run_60/")
    args = ap.parse_args()

    bounds = subrun_bounds(args.run_dir)
    rows = []
    for i in range(args.nmax):
        sub = f"overnight_{i:02d}"
        subdir = os.path.join(args.run_dir, sub)
        if sub not in bounds or not os.path.isdir(subdir):
            continue
        s, e = bounds[sub]
        r = analyse_subrun(subdir, sub)
        if r is None:
            print(f"  {sub}: no decoded hits yet, skipped")
            continue
        hv = hv_of(subdir)
        tmid = ((s + (e - s) / 2) - T_SWITCH).total_seconds() / 3600.0
        r.update(sub=sub, t=tmid, hv=hv)
        flag = ""
        if hv and (hv[1] > 2.0 or hv[3] > 2.0):
            flag = f"  !! HV moved: drift ptp {hv[1]:.1f} V, resist ptp {hv[3]:.1f} V"
        print(f"  {sub}  t={tmid:+5.2f} h  ev {r['n_events']:6d}  "
              f"gainY {r['gain_y']:6.1f}  occY {r['occ_y']:5.2f}  "
              f"span {r['span']:6.0f} ns"
              + (f"  HV {hv[0]:.1f}/{hv[2]:.1f}" if hv else "") + flag)
        rows.append(r)

    if len(rows) < 5:
        print("not enough sub-runs decoded to fit; rerun when decode finishes")
        return
    # drop low-beam sub-runs (SPS dips): their "gain" is sparks and noise
    n_med = np.median([r["n_events"] for r in rows])
    kept = [r for r in rows if r["n_events"] > 0.4 * n_med]
    if len(kept) < len(rows):
        print(f"  (dropping {len(rows) - len(kept)} low-beam sub-runs "
              f"from the fit)")
    rows = kept
    t = np.array([r["t"] for r in rows])

    # Cross-run anchors, SPAN ONLY (the amp>=150 hit-time span survives the
    # per-run ZS-threshold changes; the gain/occ estimators do not):
    #   run_59 detE_long_00  (CO2 mixture, t = -0.7 h, same drift HV)
    #   run_63 operating_03  (fully-exchanged CF4 mixture, t = +28.5 h)
    # Without them run_60's own 13 points cannot separate (lag, tau, Ainf) --
    # the transient is near-linear over the beam-on window.
    SPAN_ANCHORS = [(-0.7, 2355.0), (28.5, 2023.0)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fits = {}
    for ax, key, lab in ((axes[1, 1], "span", "hit-time t10-t90 span [ns]"),
                         (axes[0, 0], "gain_y", "median leading amp Y [ADC]"),
                         (axes[0, 1], "gain_x", "median leading amp X [ADC]"),
                         (axes[1, 0], "occ_y", "mean in-time hits/event Y")):
        y = np.array([r[key] for r in rows])
        ax.plot(t, y, "o", color="#1f4e79")
        tf, yf = t, y
        if key == "span":
            ta = np.array([a[0] for a in SPAN_ANCHORS])
            ya = np.array([a[1] for a in SPAN_ANCHORS])
            ax.plot(ta, ya, "s", color="#2e7d32", ms=7,
                    label="run_59 / run_63 anchors")
            tf, yf = np.r_[ta[:1], t, ta[1:]], np.r_[ya[:1], y, ya[1:]]
        elif "span" in fits:
            # the mixture timescale is one number: pin (lag, tau) from span
            lg, tu = fits["span"]["lag"], fits["span"]["tau"]

            def f2(t, ainf, a0):
                return np.where(t < lg, a0,
                                ainf + (a0 - ainf) * np.exp(-(t - lg) / tu))
            try:
                p2, cov2 = curve_fit(f2, t, y, p0=[y[-1], y[0]], maxfev=20000)
                tt = np.linspace(t.min(), t.max(), 200)
                ax.plot(tt, f2(tt, *p2), "-", color="#c1440e", lw=1.2,
                        label=f"span timescale: A0 {p2[1]:.0f} -> "
                              f"Ainf {p2[0]:.0f}")
                ax.legend(fontsize=9)
                fits[key] = dict(tau=float(tu), tau_err=0.0, lag=float(lg),
                                 a0=float(p2[1]), ainf=float(p2[0]),
                                 timescale="pinned from span")
            except Exception:
                pass
            ax.set(ylabel=lab)
            ax.grid(alpha=0.3)
            continue
        r = expfit(tf, yf)
        if r:
            p, err, f = r
            tt = np.linspace(tf.min(), tf.max(), 200)
            ax.plot(tt, f(tt, *p), "-", color="#c1440e", lw=1.2,
                    label=f"lag {p[2]:.1f} h, tau = {p[3]:.2f} ± {err[3]:.2f} h")
            ax.legend(fontsize=9)
            fits[key] = dict(tau=float(p[3]), tau_err=float(err[3]),
                             lag=float(p[2]), a0=float(p[1]),
                             ainf=float(p[0]))
        ax.set(ylabel=lab)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set(xlabel="hours since gas switch (20:45 ± 20 min)")
    fig.suptitle("run_60 — det4 observables through the CO2->CF4 flush "
                 "(fixed HV 700.5/649.75 V, 25.64 deg)")
    fig.tight_layout()
    png = os.path.join(args.out, "flush_run60.png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    print(f"\nwrote {png}")
    if fits:
        taus = [v["tau"] for v in fits.values() if v["tau_err"] < v["tau"]]
        print("\nFLUSH CONSTANT per observable (lagged exponential):")
        for k, v in fits.items():
            print(f"  {k:7s} lag {v['lag']:4.1f} h  tau = {v['tau']:5.2f} "
                  f"± {v['tau_err']:4.2f} h   A0 {v['a0']:8.1f} -> "
                  f"Ainf {v['ainf']:8.1f}")
        if taus:
            print(f"\n  ideal-mixing V/Q = 4.6 L / 2 l/h = 2.3 h; "
                  f"measured {np.mean(taus):.2f} h "
                  f"=> effective mixing factor {np.mean(taus) / 2.3:.2f}")
    import json
    with open(os.path.join(args.out, "flush_run60.json"), "w") as jf:
        json.dump(dict(rows=[{k: v for k, v in r.items() if k != "hv"}
                             for r in rows], fits=fits), jf, indent=1)


if __name__ == "__main__":
    main()
