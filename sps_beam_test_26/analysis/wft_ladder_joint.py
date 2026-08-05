#!/usr/bin/env python3
"""Joint drift-ladder fit with sigma_p0 SHARED across plateaus.

`wft_beam_fit.py` fits (v, sigma_p0, Dp) independently on each drift plateau and
finds what EXTRACTION_2026-08-05 §4 records: within one plateau sigma_p0 and Dp
are degenerate (sigma_p0 0.05 / Dp 0.063 and sigma_p0 0.57 / Dp 0.067 fit
equally well), so the per-plateau sigma_p0 column is scatter, not measurement.

The lever the ladder actually provides is that the two parameters depend on the
drift field DIFFERENTLY:

    sigma_p0  initial cloud at the mesh          field-INdependent
    Dp        diffusion per sqrt(drift length)   falls with field

So sigma_p0 is fitted ONCE for the whole ladder while (v, Dp) stay per plateau.
That is this script.  It profiles rather than minimising in 15 dimensions:

    for each sigma_p0 on a grid:
        for each plateau (in parallel):
            minimise chi2 over (v, Dp) at that fixed sigma_p0
    joint profile = sum over plateaus

which is both more robust than a 15-parameter Nelder-Mead and gives the
profile curve, i.e. an actual uncertainty on sigma_p0 instead of a point.

Two honesty notes carried into the output:

  * chi2/dof is 120-220 (noise model + the thin rot25 alignment, §4), so a
    Delta chi2 = 1 interval would be ~14x too tight.  The quoted interval is
    rescaled by chi2/dof -- the standard treatment when the fit is good in
    shape but the errors are underestimated in scale.
  * the mount tilt (tan theta_X = -0.0157) is 3.3 % of tan(25.64 deg) and v
    scales as 1/tan, so every v carries a +-3.3 % scale systematic on top of
    its fit error.  Reported, not applied: the sign in detector-local
    coordinates is what `detect_sign` resolves per plateau.

    ../../.venv/bin/python wft_ladder_joint.py run63_rot25 \
        [--events-per-plateau 150] [--jobs 7]
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
import wft_beam_fit as wbf                                 # noqa: E402
from wft.calib import CalibrationBundle                    # noqa: E402
from wft import model as wm                                # noqa: E402

#: arm C -- the promoted canonical det4 bundle (lp kernel, sigma_p0 free).
#: The ladder REfits sigma_p0, so the bundle contributes the kernel
#: (c1, c2, tau_s, kY, sigma_s) and the impulse template, not the gas terms.
BUNDLE = ("/media/dylan/data/x17/cosmic_bench/Analysis/"
          "mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle")
GAP_MM = wbf.GAP_MM
TAN_TILT = 0.0157       # RERUN_2026-08-04 §4: the drift-invariant mount walk


def span_seeds(span_json):
    """v per plateau from the hit-time span, anchored on the run_71 end lobe
    (233 V/cm : 14 um/ns) -- same anchor wft_beam_fit uses."""
    if not span_json or not os.path.exists(span_json):
        return {}
    with open(span_json) as f:
        sj = json.load(f)
    c0 = None
    for x in sj.values():
        if abs(x["field_Vcm"] - 233.0) < 10:
            c0 = x["span"] - GAP_MM * 1e3 / 14.0
    if c0 is None:
        return {}
    return {l: GAP_MM * 1e3 / (x["span"] - c0)
            for l, x in sj.items() if x["span"] > c0}


def plateau_profile(job):
    """One plateau, one process: chi2 vs the shared sigma_p0 grid, with
    (v, Dp) minimised at each grid point.  Continuation seeding (each grid
    point starts from the previous point's solution) keeps the inner fits to
    ~30 evaluations."""
    (lab, evs, view, sgn_seed, grid, v_seed, v_lo, v_hi, hyper0, k_bins,
     nsamp, bundle, quiet) = job
    cal = CalibrationBundle.load(bundle)
    wm.use_calibration(cal)
    wm.set_nsamp(nsamp)

    sgn = sgn_seed
    if sgn is None:
        sgn = wbf.detect_sign(evs, view, v_seed, hyper0, k_bins)
    warm = {}
    out = []
    x = np.array([v_seed, hyper0.get("Dp", 0.02)])
    for sp in grid:
        def obj(z):
            v, dp = z
            if not (v_lo < v < v_hi) or not (0.001 < dp < 0.25):
                return 1e12
            c, _n = wbf.plateau_chi2(evs, view, v, sp, dp, hyper0, k_bins,
                                     sgn, warm)
            return c

        r = minimize(obj, x, method="Nelder-Mead",
                     options=dict(xatol=5e-3, fatol=1.0, maxiter=60,
                                  initial_simplex=x + np.array(
                                      [[0, 0], [1.5, 0], [0, 0.008]])))
        v, dp = r.x
        c, n = wbf.plateau_chi2(evs, view, v, sp, dp, hyper0, k_bins, sgn)
        out.append(dict(sigma_p0=float(sp), v=float(v), Dp=float(dp),
                        chi2=float(c), ndof=int(n)))
        x = r.x                                  # continuation
        if not quiet:
            print(f"  [{lab}] sigma_p0={sp:.3f} -> v={v:6.2f} Dp={dp:.4f} "
                  f"chi2/dof={c / max(n, 1):.2f}", flush=True)
    return lab, sgn, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--view", default="x", choices=("x", "y"))
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--wf", default="")
    ap.add_argument("--mount-deg", type=float, default=25.64)
    ap.add_argument("--events-per-plateau", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--v0", type=float, default=14.0)
    ap.add_argument("--span-json", default="")
    ap.add_argument("--grid", default="0.03,0.08,0.15,0.25,0.35,0.45,0.60,0.80",
                    help="shared sigma_p0 grid [mm]")
    ap.add_argument("--sign", type=int, default=0, choices=(-1, 0, 1),
                    help="force the rotation sign in detector-local "
                         "coordinates instead of auto-detecting per plateau. "
                         "The mount does not turn round between drift "
                         "plateaus, so a per-plateau sign is a symptom, not a "
                         "measurement: the first joint run auto-detected -1 "
                         "on five plateaus and +1 on two (both with margins "
                         "under 15 %, against up to 20 % for the -1 votes), "
                         "which is the fit finding different local minima "
                         "rather than different geometry. 0 = auto (the old "
                         "behaviour), kept only to reproduce that diagnosis.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    D = datasets.get(args.dataset)
    wf = args.wf or D["stage"] + f"wf_{args.dataset}.npz"
    span_json = args.span_json or D["stage"] + f"ladder_span_{args.dataset}.json"
    grid = np.array([float(s) for s in args.grid.split(",")])

    cal = CalibrationBundle.load(args.bundle)
    wm.use_calibration(cal)
    wm.set_nsamp(int(D["n_samples"]))
    hyper0 = dict(wm.HYPER)
    print(f"bundle: {cal.summary()}")
    print(f"{args.dataset}, view {args.view.upper()}, mount {args.mount_deg} deg,"
          f" shared-sigma_p0 grid {list(grid)}")

    per = wbf.load_events(D, wf, args.view, args.mount_deg)
    vspan = span_seeds(span_json)
    if vspan:
        print(f"span-seeded v: { {k: round(v, 1) for k, v in vspan.items()} }")
    drift_of = {p[0]: p[3] for p in D["plateaus"]}
    k_bins = int(np.clip(np.ceil(GAP_MM * 1e3 / max(args.v0 * 0.5, 3.0) / wm.DT),
                         18, int(D["n_samples"])))

    jobs = []
    for lab in sorted(per):
        evs = per[lab][:args.events_per_plateau]
        v_seed = vspan.get(lab, args.v0)
        v_lo, v_hi = (0.65 * v_seed, 1.35 * v_seed) if lab in vspan else (2.0, 45.0)
        jobs.append((lab, evs, args.view, args.sign or None, grid, v_seed,
                     v_lo, v_hi,
                     hyper0, k_bins, int(D["n_samples"]), args.bundle,
                     args.jobs > 1))
        print(f"  {lab}: {len(evs)} events, drift {drift_of.get(lab)} V, "
              f"v seed {v_seed:.1f} [{v_lo:.1f}, {v_hi:.1f}]")

    if args.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.jobs, len(jobs))) as pool:
            res = pool.map(plateau_profile, jobs)
    else:
        res = [plateau_profile(j) for j in jobs]

    prof = {lab: dict(sign=int(sgn), points=pts) for lab, sgn, pts in res}
    labs = sorted(prof)

    # ---- joint profile -----------------------------------------------------
    chi2 = np.zeros(len(grid))
    ndof = np.zeros(len(grid))
    for lab in labs:
        chi2 += np.array([p["chi2"] for p in prof[lab]["points"]])
        ndof += np.array([p["ndof"] for p in prof[lab]["points"]])
    i0 = int(np.argmin(chi2))
    d = chi2 - chi2[i0]
    scale = chi2[i0] / max(ndof[i0], 1)          # errors underestimated by ~this
    # parabolic interpolation of the minimum on the (log-spaced-ish) grid
    sp_hat = grid[i0]
    if 0 < i0 < len(grid) - 1:
        x1, x2, x3 = grid[i0 - 1:i0 + 2]
        y1, y2, y3 = chi2[i0 - 1:i0 + 2]
        den = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if den != 0:
            A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / den
            B = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1)
                 + x1 ** 2 * (y2 - y3)) / den
            if A > 0:
                sp_hat = float(-B / (2 * A))
    inside = grid[d <= scale]                    # Delta chi2 <= 1, rescaled
    lo = float(inside.min()) if len(inside) else float(grid[i0])
    hi = float(inside.max()) if len(inside) else float(grid[i0])

    print("\n=== JOINT PROFILE (shared sigma_p0) ===")
    print(f"{'sigma_p0':>9} {'chi2':>14} {'d chi2':>10} {'chi2/dof':>9}")
    for i, sp in enumerate(grid):
        mark = "  <-- min" if i == i0 else ""
        print(f"{sp:9.3f} {chi2[i]:14.6e} {d[i]:10.1f} "
              f"{chi2[i] / max(ndof[i], 1):9.2f}{mark}")
    print(f"\nsigma_p0 (shared) = {sp_hat:.3f} mm, grid interval "
          f"[{lo:.3f}, {hi:.3f}] at rescaled Delta chi2 = chi2/dof = {scale:.0f}")

    print("\n=== LADDER AT THE JOINT sigma_p0 ===")
    print(f"{'plateau':>8} {'V/cm':>7} {'v':>7} {'Dp':>8} {'chi2/dof':>9} "
          f"{'v span':>7}")
    ladder = {}
    for lab in labs:
        pt = prof[lab]["points"][i0]
        dv = drift_of.get(lab, np.nan)
        field = dv / (GAP_MM / 10.0) if dv == dv else np.nan
        # spread of v across the whole sigma_p0 grid = the residual sensitivity
        vs = np.array([p["v"] for p in prof[lab]["points"]])
        ladder[lab] = dict(drift_V=float(dv), field_Vcm=float(field),
                           v=pt["v"], Dp=pt["Dp"],
                           chi2_dof=pt["chi2"] / max(pt["ndof"], 1),
                           v_grid_spread=float(vs.max() - vs.min()),
                           v_tilt_syst=float(pt["v"] * TAN_TILT
                                             / np.tan(np.radians(args.mount_deg))),
                           sign=prof[lab]["sign"],
                           n_events=len(per[lab][:args.events_per_plateau]))
        print(f"{lab:>8} {field:7.0f} {pt['v']:7.2f} {pt['Dp']:8.4f} "
              f"{ladder[lab]['chi2_dof']:9.2f} "
              f"{ladder[lab]['v_grid_spread']:7.2f}")
    print(f"\nv also carries a +-{TAN_TILT / np.tan(np.radians(args.mount_deg)) * 100:.1f} % "
          f"scale systematic from the mount tilt (tan {TAN_TILT}).")

    out = args.out or D["stage"] + f"wft_ladder_joint_{args.dataset}_{args.view}.json"
    with open(out, "w") as f:
        json.dump(dict(dataset=args.dataset, view=args.view,
                       bundle=args.bundle, mount_deg=args.mount_deg,
                       events_per_plateau=args.events_per_plateau,
                       sigma_p0_shared=sp_hat,
                       sigma_p0_interval=[lo, hi],
                       chi2_dof_at_min=float(scale),
                       grid=[float(g) for g in grid],
                       joint_chi2=[float(c) for c in chi2],
                       ladder=ladder, profiles=prof), f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
