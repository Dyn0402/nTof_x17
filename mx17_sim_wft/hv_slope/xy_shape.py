#!/usr/bin/env python3
"""
Re-derive the per-view DATA shape observables on an unbiased sample.

Why. The T14 data leg turned out to be a biased subsample of the detector: its
per-view reconstruction-quality cut removes saturated events, and removes three
times more of them in Y (rail fraction 0.33 -> 0.11) than in X (0.33 -> 0.26).
The amplitude consequence is already reported (iso_ve/report.html §3). But the
SHAPE observables — undershoot, FWHM, rise — were measured on those same legs
and are used as fitting targets downstream (notably the Y undershoot of
-12.0 %, which the beta scan cannot reach). If those observables depend on
amplitude, the per-view difference in what was thrown away puts a per-view
difference into the targets that the detector does not have.

So: same subrun, same FeuReader path, same observable definitions as
`t14_compare.extract_view`, but a voltage- and view-independent selection
(M3 track, chi2 < 1, NClus = 4, inside the fiducial) and no reconstruction cut.
Nothing in t14_compare/ is read except the frozen tables, and nothing is
rewritten there.

Definitions, matched to T14:
  peak strip   the strongest strip among those at >= 5 sigma
  rise_ns      10 -> 90 % of peak, linear interpolation on the 60 ns samples
  fwhm_ns      50 % crossings either side of the peak
  undershoot   min(peak-strip samples AFTER the peak) / peak

    python3 mx17_sim_wft/hv_slope/xy_shape.py
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
SIGMA = 5.0
DT_NS = 60.0
FID_X, FID_Y = (-190.0, 115.0), (-190.0, 165.0)
RAIL = 3500.0          # the same threshold T14's sat_frac_3500 uses
# T14's rail fractions on its own data legs — the target when emulating what
# its selection did, without needing its reco table
T14_SAT = dict(x=0.2596, y=0.1104)
RNG = np.random.default_rng(20260809)


def _cross(w, ipk, a, frac, side):
    lvl = frac * a
    if side < 0:
        below = np.flatnonzero(w[:ipk + 1] < lvl)
        if len(below) == 0:
            return None
        i = below[-1]
        j = i + 1
    else:
        below = np.flatnonzero(w[ipk:] < lvl)
        if len(below) == 0:
            return None
        j = ipk + below[0]
        i = j - 1
    if w[j] == w[i]:
        return float(i)
    return i + (lvl - w[i]) / (w[j] - w[i])


def extract(cfg, feu, pos, refs, view):
    from wft.io import FeuReader
    files = sorted(glob.glob(os.path.join(
        BASE, RUN, SUB, "decoded_root", f"*_{feu:02d}.root")))
    valid = ~np.isnan(pos)
    rows = []
    for path in files:
        rdr = FeuReader(path)
        noise = np.where(rdr.noise > 0, rdr.noise, np.inf)
        want = set(int(e) for e in rdr.event_ids) & set(refs)
        if not want:
            continue
        for eid, _ftst, wfm in rdr.iter_events(want):
            amp = wfm.max(axis=1)
            over = valid & (amp / noise >= SIGMA)
            if not over.any():
                continue
            idx = np.flatnonzero(over)
            pk = int(idx[np.argmax(amp[idx])])
            w = wfm[pk]
            ipk = int(np.argmax(w))
            a = float(w[ipk])
            r10, r90 = _cross(w, ipk, a, 0.10, -1), _cross(w, ipk, a, 0.90, -1)
            hl, hr = _cross(w, ipk, a, 0.50, -1), _cross(w, ipk, a, 0.50, +1)
            tail = w[ipk + 1:]
            rx, ry = refs[eid]
            rows.append(dict(
                view=view, event_id=int(eid), peak_amp=a, peak_sample=ipk,
                n_tail=int(len(tail)),
                # a clipped waveform has a flat top: count samples within 1 %
                # of the maximum, which is 1 for a real peak and >1 at the rail
                n_flat=int((w >= a - 0.01 * abs(a)).sum()),
                undershoot=float(tail.min() / a) if len(tail) else np.nan,
                rise_ns=(r90 - r10) * DT_NS if r10 is not None and r90 is not None else np.nan,
                fwhm_ns=(hr - hl) * DT_NS if hl is not None and hr is not None else np.nan,
                n_over=int(over.sum()), ref_x=rx, ref_y=ry))
    return rows


def m3_angles():
    """M3 reference angles for the long run — reference side, so allowed as a
    selection variable (RECONSTRUCTION_BASIS.md)."""
    from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
    setup_paths()
    from M3RefTracking import M3RefTracking, get_xy_angles
    cfg = get_config("sat_det3")
    cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = BASE, RUN, SUB
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    ax_, ay_, evn = get_xy_angles(rays.ray_data)
    return pd.DataFrame(dict(
        event_id=np.asarray(evn).astype("int64"),
        thx=np.degrees(np.arctan(np.asarray(ax_, float))),
        thy=np.degrees(np.arctan(np.asarray(ay_, float)))))


def summarize(g, label):
    u = g[g.u_ok]
    return dict(
        label=label, n=int(len(g)), n_undershoot=int(len(u)),
        sat_frac=float((g.peak_amp >= RAIL).mean()),
        peak_p50=float(g.peak_amp.median()),
        undershoot_p50=float(u.undershoot.median()),
        undershoot_p25=float(u.undershoot.quantile(0.25)),
        undershoot_p75=float(u.undershoot.quantile(0.75)),
        frac_below_m005=float((u.undershoot < -0.05).mean()),
        nan_rise=float(g.rise_ns.isna().mean()),
        nan_fwhm=float(g.fwhm_ns.isna().mean()),
        fwhm_p50=float(g.fwhm_ns.median()),
        rise_p50=float(g.rise_ns.median()),
        rise_p5=float(g.rise_ns.quantile(0.05)),
        frac_rise_lt240=float((g.rise_ns < 240).mean()))


def emulate_t14(g, target_sat, rng=RNG):
    """Drop railed events at random until the rail fraction matches what T14's
    reconstruction cut left in this view. Isolates the effect of WHICH events
    the cut removed from any other property of the cut."""
    sat = g.peak_amp >= RAIL
    n_keep_sat = int(round(target_sat * (~sat).sum() / max(1e-9, 1 - target_sat)))
    n_keep_sat = min(n_keep_sat, int(sat.sum()))
    keep = np.concatenate([
        g.index[~sat].values,
        rng.choice(g.index[sat].values, n_keep_sat, replace=False)])
    return g.loc[np.sort(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser(
        "~/x17/response_sim/hv_slope/xy_shape"))
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = a.cache or os.path.join(a.out_dir, "shape.parquet")

    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        print(f"loaded {cache} ({len(df):,} rows)")
    else:
        from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
        setup_paths()
        from wft.io import strip_position_map
        from M3RefTracking import M3RefTracking
        cfg = get_config("sat_det3")
        cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = BASE, RUN, SUB
        pos_maps = strip_position_map(cfg)
        rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                             min_nclus=M3_MIN_NCLUS)
        x, y, evn = rays.get_xy_positions(DET_Z)
        x, y, evn = np.asarray(x), np.asarray(y), np.asarray(evn)
        ok = (np.isfinite(x) & np.isfinite(y)
              & (x > FID_X[0]) & (x < FID_X[1])
              & (y > FID_Y[0]) & (y < FID_Y[1]))
        refs = {int(e): (float(u), float(v))
                for e, u, v in zip(evn[ok], x[ok], y[ok])}
        print(f"{len(refs):,} M3 tracks inside the fiducial")
        rows = []
        for view, feu in (("x", cfg.MX17_FEU_X), ("y", cfg.MX17_FEU_Y)):
            r = extract(cfg, feu, pos_maps[feu], refs, view)
            print(f"  {view}: {len(r):,} events")
            rows += r
        df = pd.DataFrame(rows)
        df.to_parquet(cache)

    # the undershoot needs tail to look into: a peak at the very end of the
    # 32-sample window has none. This filter applies to the UNDERSHOOT only —
    # rise and FWHM are measured on everything, as T14 measures them.
    df["u_ok"] = df.peak_sample <= 20
    df = df.merge(m3_angles(), on="event_id", how="left")

    # T14's per-view theta windows, from its own frozen summary
    tw = json.load(open(os.path.expanduser(
        "~/x17/response_sim/stageB_w2/t14_compare/t14_summary.json")
    ))["theta_windows_deg"]

    # Which M3 angle belongs to which view is settled by the data, not by
    # assumption: the in-plane angle is the one the peak-strip WIDTH responds
    # to (an inclined track spreads its charge over more strips, so each strip
    # sees a shorter pulse). Pick the pairing with the stronger FWHM gradient.
    grad = {}
    for view in ("x", "y"):
        g = df[df.view == view]
        for col in ("thx", "thy"):
            lo = g.fwhm_ns[np.abs(g[col]) < 2].median()
            hi = g.fwhm_ns[np.abs(g[col]) > 8].median()
            grad[(view, col)] = float(lo - hi)
    ANG = {v: max(("thx", "thy"), key=lambda c: grad[(v, c)]) for v in ("x", "y")}
    out = {"t14_sat_target": T14_SAT, "rail": RAIL, "theta_windows": tw,
           "angle_pairing": ANG,
           "fwhm_gradient_ns": {f"{v}/{c}": grad[(v, c)]
                                for v in ("x", "y") for c in ("thx", "thy")},
           "views": {}}

    for view in ("x", "y"):
        g = df[df.view == view].reset_index(drop=True)
        col, w = ANG[view], tw[view]
        inwin = g[(g[col] >= w[0]) & (g[col] <= w[1])].reset_index(drop=True)
        rows = [summarize(g, "unselected (M3 fiducial, no reco cut)"),
                summarize(g[g.peak_amp < RAIL], "unsaturated only (peak < 3500)"),
                summarize(emulate_t14(g, T14_SAT[view]),
                          f"rail forced to T14's {T14_SAT[view]:.3f}"),
                summarize(inwin,
                          f"T14 theta window on M3 ({w[0]:.1f}..{w[1]:.1f} deg)"),
                summarize(emulate_t14(inwin, T14_SAT[view]),
                          "theta window AND rail forced to T14's")]
        out["views"][view] = rows

    # observable vs the in-plane M3 angle — the second mechanism
    abins = [(0, 1), (1, 2), (2, 4), (4, 7), (7, 12), (12, 90)]
    out["angle_profile"] = {}
    for view in ("x", "y"):
        g = df[df.view == view]
        col = ANG[view]
        out["angle_profile"][view] = [
            dict(lo=lo, hi=hi, n=int(m.sum()),
                 fwhm=float(g.fwhm_ns[m].median()),
                 rise=float(g.rise_ns[m].median()),
                 peak=float(g.peak_amp[m].median()),
                 undershoot=float(g.undershoot[m & g.u_ok].median()),
                 nan_rise=float(g.rise_ns[m].isna().mean()))
            for lo, hi in abins
            for m in [np.abs(g[col]).between(lo, hi)]]

    # observable vs amplitude — the mechanism, per view
    bins = [(0, 800), (800, 1500), (1500, 2200), (2200, 3000), (3000, 3500),
            (3500, 1e9)]
    prof = {}
    for view in ("x", "y"):
        g = df[df.view == view]
        prof[view] = [dict(lo=lo, hi=hi, n=int(m.sum()),
                           undershoot=float(g.undershoot[m].median()),
                           undershoot_adc=float(
                               (g.undershoot[m] * g.peak_amp[m]).median()),
                           fwhm=float(g.fwhm_ns[m].median()),
                           rise=float(g.rise_ns[m].median()),
                           n_flat=float(g.n_flat[m].median()),
                           nan_rise=float(g.rise_ns[m].isna().mean()))
                      for lo, hi in bins
                      for m in [(g.peak_amp >= lo) & (g.peak_amp < hi)]]
    out["profile"] = prof

    # the frozen T14 legs, for the side-by-side
    t14 = {}
    for view in ("x", "y"):
        t = pd.read_parquet(os.path.expanduser(
            f"~/x17/response_sim/stageB_w2/t14_compare/wf_data_{view}.parquet"))
        s = pd.read_parquet(os.path.expanduser(
            f"~/x17/response_sim/stageB_w2/t14_compare/wf_sim_{view}.parquet"))
        t14[view] = dict(
            data=dict(n=int(len(t)), peak_p50=float(t.peak_amp.median()),
                      fwhm_p50=float(t.fwhm_ns.median()),
                      rise_p50=float(t.rise_ns.median()),
                      rise_p5=float(t.rise_ns.quantile(0.05)),
                      frac_rise_lt240=float((t.rise_ns < 240).mean()),
                      nan_rise=float(t.rise_ns.isna().mean()),
                      sat_frac=float((t.peak_amp >= RAIL).mean())),
            sim=dict(n=int(len(s)), peak_p50=float(s.peak_amp.median()),
                     fwhm_p50=float(s.fwhm_ns.median()),
                     rise_p50=float(s.rise_ns.median()),
                     rise_p5=float(s.rise_ns.quantile(0.05)),
                     frac_rise_lt240=float((s.rise_ns < 240).mean()),
                     nan_rise=float(s.rise_ns.isna().mean()),
                     sat_frac=float((s.peak_amp >= RAIL).mean())))
        # the NaN-rise population: S3_rerun flagged 9.0 % (X) / 3.0 % (Y) of
        # the frozen legs as un-measurable. They are not a rise-time effect —
        # they are events with essentially no signal in that view.
        nan = t.rise_ns.isna()
        t14[view]["data"]["nan_pop"] = dict(
            frac=float(nan.mean()),
            peak_p50_nan=float(t.peak_amp[nan].median()) if nan.any() else None,
            peak_p50_ok=float(t.peak_amp[~nan].median()),
            peak_p90_nan=float(t.peak_amp[nan].quantile(0.9)) if nan.any() else None,
            sat_frac_nan=float((t.peak_amp[nan] >= RAIL).mean()) if nan.any() else None)
        t14[view]["sim"]["nan_frac"] = float(s.rise_ns.isna().mean())

    ub = json.load(open(os.path.expanduser(
        "~/x17/response_sim/stageB_w2/t14_compare/bump_undershoot.json")))
    for view in ("x", "y"):
        t14[view]["data"]["undershoot_p50"] = ub[f"data_{view}"]["undershoot"]["median"]
        t14[view]["sim"]["undershoot_p50"] = ub[f"sim_{view}"]["undershoot"]["median"]
    out["t14"] = t14

    with open(os.path.join(a.out_dir, "xy_shape.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("angle pairing chosen from the FWHM gradient:", ANG,
          out["fwhm_gradient_ns"])
    hdr = (f"{'sample':46s} {'n':>6} {'sat':>6} {'peak':>7} {'under':>7} "
           f"{'fwhm':>7} {'rise50':>7} {'rise5':>7} {'<240':>6} {'nanR':>6}")
    for view in ("x", "y"):
        print(f"\n===== {view.upper()} view")
        print(hdr)
        for r in out["views"][view]:
            print(f"{r['label']:46s} {r['n']:6d} {r['sat_frac']:6.3f} "
                  f"{r['peak_p50']:7.0f} {100 * r['undershoot_p50']:6.1f}% "
                  f"{r['fwhm_p50']:7.1f} {r['rise_p50']:7.1f} "
                  f"{r['rise_p5']:7.1f} {r['frac_rise_lt240']:6.3f} "
                  f"{r['nan_rise']:6.3f}")
        d = t14[view]["data"]
        print(f"{'T14 data leg (frozen)':44s} {d['n']:6d} {d['sat_frac']:6.3f} "
              f"{d['peak_p50']:7.0f} {100 * d['undershoot_p50']:6.1f}% "
              f"{d['fwhm_p50']:7.1f} {d['rise_p50']:7.1f} {d['rise_p5']:7.1f} "
              f"{d['frac_rise_lt240']:6.3f}")
        print(f"  vs peak amplitude   (NaN rise fraction in the last column):")
        for p in prof[view]:
            print(f"    {p['lo']:5.0f}-{min(p['hi'], 4200):5.0f} ADC n={p['n']:5d} "
                  f"under {100 * p['undershoot']:6.1f}% ({p['undershoot_adc']:6.0f} ADC)  "
                  f"fwhm {p['fwhm']:6.1f}  "
                  f"rise {p['rise']:6.1f}  flat {p['n_flat']:.0f}  "
                  f"nan {p['nan_rise']:.3f}")
        print(f"  vs |M3 {ANG[view]}| (the in-plane angle for this view):")
        for p in out["angle_profile"][view]:
            print(f"    {p['lo']:3.0f}-{min(p['hi'], 25):3.0f} deg n={p['n']:5d} "
                  f"peak {p['peak']:6.0f}  under {100 * p['undershoot']:6.1f}%  "
                  f"fwhm {p['fwhm']:6.1f}  rise {p['rise']:6.1f}  "
                  f"nan {p['nan_rise']:.3f}")
    print("\nwrote", os.path.join(a.out_dir, "xy_shape.json"))


if __name__ == "__main__":
    main()
