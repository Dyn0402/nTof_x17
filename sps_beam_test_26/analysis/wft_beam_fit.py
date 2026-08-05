#!/usr/bin/env python3
"""wft forward fit on inclined beam data — the drift-ladder lever.

The flat-geometry kernel measurements (run_56/63/71) could not separate the
gas parameters (sigma_p0, Dp, v) from the layer kernel (c1, c2, tau_s): at
normal incidence every depth bin lands on the same strip. The 25.64 deg
ladders CAN — the track sweeps ~13.8 mm across the strips over the 28.8 mm
gap, so the depth-position correlation is in the data and the wft forward
model applies with w != 0. This is the "full wft forward fit with w != 0"
that datasets.py's run63_rot25 note asks for.

Per drift plateau, with the resistive kernel PINNED (share_lp, from the
bench+beam calibration) and the impulse template taken from the bench bundle
(electronics property, gas-independent), fit the gas parameters

    (v_drift, sigma_p0, Dp)

by minimising the total ref-pinned chi2 over events: for each event the
uRWELL track fixes (p0, w = tan * v), only t0 and the charge profile are
fitted — exactly wft.calibrate's ref-pinned configuration, transplanted to
the beam.

Expected physics: v rises with drift field (wet-gas curve, cf. the run_71
end-lobe 14 um/ns at 233 V/cm); Dp falls with field; sigma_p0 stays put.
If sigma_p0 comes out field-dependent, the kernel is absorbing gas physics
and the share_lp shape needs another look — that is the point of the test.

  ../../.venv/bin/python wft_beam_fit.py run63_rot25 [--view x]
      [--bundle <lp bundle>] [--events-per-plateau 150]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

REPO = "/home/dylan/PycharmProjects/nTof_x17"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sps_beam_test_26", "det4_sps_assessment"))

import datasets                                            # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM       # noqa: E402
from wft.calib import CalibrationBundle                    # noqa: E402
from wft import model as wm                                # noqa: E402

BUNDLE_LP = ("/media/dylan/data/x17/cosmic_bench/Analysis/"
             "mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle_lp")
GAP_MM = 28.8
HALF_WIN = 16           # strips either side of the (mid-sweep) prediction:
                        # a 25.64 deg track sweeps tan*28.8 = 13.8 mm
NOISE_ADC = 10.0        # post-CNS noise, RAW_RUN71_PHYSICS §1; ZS similar


def load_events(D, wf_path, view, mount_deg, q_lo=300.0):
    """Per plateau: list of P dicts + (p0_pred, tan) per event."""
    Z = np.load(wf_path, allow_pickle=True)
    ev, ch, samp, amp = Z["ev"], Z["ch"].astype(int), Z["samp"], Z["amp"]
    plat = Z["ev_plateau"]
    pX, pY = Z["ev_pX"], Z["ev_pY"]         # uRWELL prediction, det-local mm
    nsmp = int(D["n_samples"])
    tanv = np.tan(np.radians(mount_deg))

    vmask = VIEW[ch] == view
    pred = pX if view == "x" else pY
    out = {}
    order = np.lexsort((ch, ev))
    ev_s, ch_s, sm_s, am_s = ev[order], ch[order], samp[order], amp[order]
    starts = np.r_[0, np.flatnonzero(ev_s[1:] != ev_s[:-1]) + 1, len(ev_s)]
    for a, b in zip(starts[:-1], starts[1:]):
        e = int(ev_s[a])
        lab = str(plat[e])
        if not lab:
            continue
        p0 = float(pred[e])
        if not np.isfinite(p0):
            continue
        c, sm, am = ch_s[a:b], sm_s[a:b], am_s[a:b]
        keep = (VIEW[c] == view) & (np.abs(POSITION_MM[c] - p0)
                                    <= HALF_WIN * PITCH_MM)
        if keep.sum() == 0:
            continue
        c, sm, am = c[keep], sm[keep], am[keep]
        chs = np.unique(c)
        W = np.zeros((len(chs), nsmp), np.float64)
        ci = np.searchsorted(chs, c)
        ok = (sm >= 0) & (sm < nsmp)
        W[ci[ok], sm[ok].astype(int)] = am[ok]
        if W.max() < q_lo:
            continue
        P = dict(W=W, pos=POSITION_MM[chs], noise=np.full(len(chs), NOISE_ADC),
                 ch=chs)
        out.setdefault(lab, []).append((P, p0, tanv))
    return out


def plateau_chi2(events, view, v, sigma_p0, Dp, hyper0, k_bins, sgn,
                 warm=None):
    """Total ref-pinned chi2. The uRWELL prediction is the charge centroid
    (mid-sweep); the model's p0 is the position at the MESH, so the pin is
    p0_mesh = pred - sgn*tan*GAP/2, w = sgn*tan*v. ``sgn`` (+1/-1) is the
    rotation direction in detector-local coordinates, auto-detected.

    ``warm`` caches each event's best t0 across objective evaluations (the
    same trick wft.calibrate uses): after the first pass only a local +-20 ns
    refinement runs, which is what makes the hyper fit tractable."""
    h = dict(hyper0)
    h["sigma_p0"] = sigma_p0
    h["Dp"] = Dp
    wm.set_depth_bins(k_bins)
    tot, n = 0.0, 0
    for j, (P, p0, tanv) in enumerate(events):
        t = sgn * tanv
        p0m = p0 - t * GAP_MM / 2.0
        W, noise, pos, sat = wm.prep_plane(P, view)
        # ZS data: an exactly-zero sample was never recorded, not measured as
        # zero -- censor it (chi2_plane's sat path excludes it from the fit;
        # its below-clip penalty is zero at W=0 since the model is >= 0)
        sat = sat | (W == 0)
        t0w = None if warm is None else warm.get(j)
        if t0w is None:
            g = wm.init_guess(P, view, t, p0m, v)
            grid = np.arange(g[2] - 240, g[2] + 241, 30.0)
        else:
            grid = np.arange(t0w - 20, t0w + 21, 10.0)
        best = (np.inf, grid[0])
        for t0 in grid:
            c, _ = wm.chi2_plane(view, W, noise, pos, sat, p0m,
                                 t * v * 1e-3, t0, h)
            if c < best[0]:
                best = (c, t0)
        dof_ev = int((~sat).sum())
        # an occasional NNLS blowup (heavily censored event + wide diffusion)
        # must not poison the plateau sum
        if np.isfinite(best[0]) and best[0] < 1e3 * max(dof_ev, 1):
            tot += best[0]
            n += dof_ev
            if warm is not None:
                warm[j] = best[1]
    return tot, n


def detect_sign(events, view, v0, hyper0, k_bins):
    sub = events[:25]
    c_p, _ = plateau_chi2(sub, view, v0, hyper0["sigma_p0"], hyper0["Dp"],
                          hyper0, k_bins, +1, warm=None)
    c_m, _ = plateau_chi2(sub, view, v0, hyper0["sigma_p0"], hyper0["Dp"],
                          hyper0, k_bins, -1, warm=None)
    sgn = +1 if c_p <= c_m else -1
    print(f"  rotation sign in local {view}: {sgn:+d} "
          f"(chi2 {c_p:.3e} vs {c_m:.3e})")
    return sgn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--view", default="x", choices=("x", "y"),
                    help="the rotation is about the vertical axis, so the "
                         "inclination is in one view; check both if unsure")
    ap.add_argument("--bundle", default=BUNDLE_LP)
    ap.add_argument("--wf", default="")
    ap.add_argument("--mount-deg", type=float, default=25.64)
    ap.add_argument("--events-per-plateau", type=int, default=120)
    ap.add_argument("--v0", type=float, default=14.0,
                    help="seed v [um/ns]; run_71 end-lobe value at 233 V/cm")
    ap.add_argument("--span-json", default="",
                    help="ladder_span_<ds>.json: seed v per plateau from the "
                         "span estimate and bound it to +-35 % -- turns the "
                         "multimodal (v, kernel) landscape into a local one")
    args = ap.parse_args()
    D = datasets.get(args.dataset)
    wf = args.wf or D["stage"] + f"wf_{args.dataset}.npz"

    cal = CalibrationBundle.load(args.bundle)
    wm.use_calibration(cal)
    wm.set_nsamp(int(D["n_samples"]))
    hyper0 = dict(wm.HYPER)
    print(f"bundle: {cal.summary()}")
    print(f"dataset {args.dataset}, view {args.view.upper()}, "
          f"mount {args.mount_deg} deg")

    per = load_events(D, wf, args.view, args.mount_deg)
    vspan = {}
    if args.span_json:
        with open(args.span_json) as f:
            sj = json.load(f)
        # convert spans to v with the run_71 end-lobe anchor (233 V/cm : 14)
        c0 = None
        for l, x in sj.items():
            if abs(x["field_Vcm"] - 233.0) < 10:
                c0 = x["span"] - GAP_MM * 1e3 / 14.0
        if c0 is not None:
            for l, x in sj.items():
                if x["span"] > c0:
                    vspan[l] = GAP_MM * 1e3 / (x["span"] - c0)
            print(f"span-seeded v per plateau: "
                  f"{ {k: round(v, 1) for k, v in vspan.items()} }")
    results = {}
    for lab in sorted(per):
        evs = per[lab][:args.events_per_plateau]
        drift_v = {p[0]: p[3] for p in
                   [(l, lo, hi, dv) for l, lo, hi, dv, _r in D["plateaus"]]
                   }.get(lab, np.nan)
        field = drift_v / (GAP_MM / 10.0) if np.isfinite(drift_v) else np.nan
        # depth bins: cover the gap at a pessimistically slow v, but never
        # beyond the DAQ window itself (64 samples)
        k_bins = int(np.clip(np.ceil(GAP_MM * 1e3 / max(args.v0 * 0.5, 3.0)
                                     / wm.DT), 18, int(D["n_samples"])))
        print(f"\n=== {lab}: {len(evs)} events, drift {drift_v} V "
              f"({field:.0f} V/cm), K={k_bins}")
        v_seed = vspan.get(lab, args.v0)
        v_lo, v_hi = (0.65 * v_seed, 1.35 * v_seed) if lab in vspan \
            else (2.0, 45.0)
        sgn = detect_sign(evs, args.view, v_seed, hyper0, k_bins)
        warm = {}

        def obj(x):
            v, sp, dp = x
            if not (v_lo < v < v_hi) or not (0.02 < sp < 1.2) or \
                    not (0.001 < dp < 0.2):
                return 1e12
            c, n = plateau_chi2(evs, args.view, v, sp, dp, hyper0, k_bins,
                                sgn, warm)
            print(f"    v={v:6.2f} sp={sp:.3f} Dp={dp:.4f} -> "
                  f"chi2/dof {c / max(n, 1):.3f}", flush=True)
            return c

        x0 = np.array([v_seed, hyper0.get("sigma_p0", 0.1),
                       hyper0.get("Dp", 0.02)])
        r = minimize(obj, x0, method="Nelder-Mead",
                     options=dict(xatol=5e-3, fatol=1.0, maxiter=120,
                                  initial_simplex=x0 + np.array(
                                      [[0, 0, 0], [3.0, 0, 0],
                                       [0, 0.08, 0], [0, 0, 0.01]])))
        v, sp, dp = r.x
        c, n = plateau_chi2(evs, args.view, v, sp, dp, hyper0, k_bins, sgn)
        results[lab] = dict(drift_V=float(drift_v), field_Vcm=float(field),
                            v=float(v), sigma_p0=float(sp), Dp=float(dp),
                            chi2_dof=float(c / max(n, 1)), n_events=len(evs))
        print(f"  => {lab}: v = {v:.2f} um/ns, sigma_p0 = {sp:.3f} mm, "
              f"Dp = {dp:.4f} mm/sqrt(ns), chi2/dof {c / max(n, 1):.3f}")

    out = D["stage"] + f"wft_beam_fit_{args.dataset}_{args.view}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")
    if len(results) >= 2:
        print("\nLADDER SUMMARY (v should rise with field, Dp fall, "
              "sigma_p0 stay put):")
        for lab, r in sorted(results.items(), key=lambda kv: -kv[1]["drift_V"]):
            print(f"  {lab:>6} {r['field_Vcm']:6.0f} V/cm  v {r['v']:6.2f}  "
                  f"sigma_p0 {r['sigma_p0']:.3f}  Dp {r['Dp']:.4f}  "
                  f"chi2/dof {r['chi2_dof']:.3f}")


if __name__ == "__main__":
    main()
