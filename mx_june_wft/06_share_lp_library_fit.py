#!/usr/bin/env python3
"""Fit the wft share_lp kernel to the run_71 RAW clean library.

This is the model-side companion of
``sps_beam_test_26/analysis/kernel_refit_clean.py``. That script measures the
RC cascade with the *measured central strip* as basis; here the basis is the
wft forward model itself: the central trace is reconstructed as

    W0_model(t) = sum_k q_k * tmpl(t - u_k),   q_k >= 0 (NNLS)

with the calibration bundle's measured impulse template — i.e. the drift
ladder is profiled out exactly the way ``wft.model`` profiles it out in a
track fit. The neighbours are then modelled per the ``share_lp`` branch:

    W_d(t) = alpha_d * W0_model(t) + beta_d * RC_tau^{|d|}(W0_model)(t)

and (tau, alpha_d, beta_d) fitted per plateau. Because basis and copies are
both template-integrable objects, the fitted (tau, c1=beta_1, c2=beta_2)
transplant *directly* into a CalibrationBundle with share_mode='lp' — that is
the point of this script. Quote the Y view.

  ../.venv/bin/python 06_share_lp_library_fit.py [--view y] [--bundle <path>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares, nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from wft.calib import CalibrationBundle                       # noqa: E402

LIB = ("/media/dylan/data/x17/sps_run53_det4_check/staging/run_71/"
       "reanalysis_clean_cmmasked/robust_library_run71_raw.npz")
BUNDLE = ("/media/dylan/data/x17/cosmic_bench/Analysis/mx17_det4_day_6-24-26/"
          "long_run/mx17_4/wft/calib_bundle")
SNS = 60.0


def rc1(w, tau, dt):
    a = np.exp(-dt / max(tau, 1.0))
    out = np.empty_like(w)
    acc = 0.0
    for i in range(len(w)):
        acc = acc * a + w[i] * (1.0 - a)
        out[i] = acc
    return out


def central_model(t, w0, grid, tmpl, k_bins=60):
    """NNLS charge ladder through the impulse template, on the library grid."""
    ok = np.isfinite(w0)
    uk = (np.arange(k_bins) + 0.5) * SNS - 10 * SNS   # allow pre-peak arrivals
    A = np.stack([np.interp(t, grid + u, tmpl, left=0, right=0)
                  for u in uk], axis=1)
    q, rn = nnls(A[ok], w0[ok], maxiter=50 * k_bins)
    return A @ q, q, rn


def fit_plateau(t, W, w0m):
    offs = [d for d in (1, -1, 2, -2) if d in W]
    msk = np.isfinite(w0m)
    for d in offs:
        msk &= np.isfinite(W[d])

    def resid(p):
        tau = p[0]
        r = []
        for i, d in enumerate(offs):
            al, be = p[1 + 2 * i], p[2 + 2 * i]
            mdl = al * w0m + be * rc1(np.nan_to_num(w0m), tau, SNS) if abs(d) == 1 \
                else al * w0m + be * rc1(rc1(np.nan_to_num(w0m), tau, SNS), tau, SNS)
            r.append((mdl - W[d])[msk])
        return np.concatenate(r)

    p0 = [400.0] + [0.15, 0.3] * len(offs)
    lo = [5.0] + [0.0, 0.0] * len(offs)
    hi = [5000.0] + [3.0, 3.0] * len(offs)
    res = least_squares(resid, p0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
    par = {d: (res.x[1 + 2 * i], res.x[2 + 2 * i]) for i, d in enumerate(offs)}
    return res.x[0], par, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=LIB)
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--view", default="y", choices=("x", "y"))
    ap.add_argument("--out", default=os.path.dirname(LIB) + "/")
    args = ap.parse_args()

    Z = np.load(args.lib)
    cal = CalibrationBundle.load(args.bundle)
    grid = np.asarray(cal.grid, float)
    tmpl = np.asarray(cal.tmpl[args.view], float)
    t = Z["t_rel"]
    v = args.view

    plateaus = sorted({k.split("_")[1] for k in Z.files if k.startswith("med_")})
    print(f"library {os.path.basename(args.lib)}, view {v.upper()}, "
          f"template from {args.bundle.split('/')[-1]}, plateaus {plateaus}")

    fig, axes = plt.subplots(1, len(plateaus), figsize=(6.5 * len(plateaus), 5),
                             squeeze=False)
    out = {}
    for j, lab in enumerate(plateaus):
        W = {}
        for d in (0, 1, -1, 2, -2):
            k = f"altr_{lab}_{v}_{d:+d}"
            if k in Z.files:
                W[d] = Z[k]
        if 0 not in W or 1 not in W:
            continue
        w0m, q, rn = central_model(t, W[0], grid, tmpl)
        cfrac = 1 - rn / np.sqrt(np.nansum(W[0] ** 2))
        tau, par, res = fit_plateau(t, W, w0m)
        c1 = np.mean([par[d][1] for d in (1, -1) if d in par])
        c2 = np.mean([par[d][1] for d in (2, -2) if d in par])
        a1 = np.mean([par[d][0] for d in (1, -1) if d in par])
        out[lab] = dict(tau=float(tau), c1=float(c1), c2=float(c2),
                        alpha1=float(a1), basis_fit=float(cfrac),
                        per_side={str(d): [float(par[d][0]), float(par[d][1])]
                                  for d in par})
        print(f"\n=== {lab}: tau = {tau:5.0f} ns  c1 = {c1:.3f}  c2 = {c2:.3f}"
              f"  alpha(+-1) = {a1:.3f}   (central basis fit {cfrac:.3f},"
              f" cost {res.cost:.2e})")

        ax = axes[0, j]
        ax.plot(t, W[0], color="0.15", lw=1.5, label="d=0 measured")
        ax.plot(t, w0m, color="0.15", lw=1.0, ls=":", label="d=0 template fit")
        for d, c in ((1, "#1f4e79"), (-1, "#5b9bd5"), (2, "#c1440e"),
                     (-2, "#e8a87c")):
            if d not in W:
                continue
            ax.plot(t, W[d], color=c, lw=1.2, label=f"d={d:+d}")
            al, be = par[d]
            base = rc1(np.nan_to_num(w0m), tau, SNS)
            if abs(d) == 2:
                base = rc1(base, tau, SNS)
            ax.plot(t, al * w0m + be * base, color=c, lw=0.9, ls="--")
        ax.set(yscale="log", ylim=(1e-4, 1.5), title=f"{lab}  tau={tau:.0f} ns",
               xlabel="t - central peak [ns]", ylabel="amp / central peak")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle(f"share_lp fitted in the wft representation — run_71 RAW "
                 f"clean library, view {v.upper()}", y=1.02)
    fig.tight_layout()
    png = args.out + f"share_lp_library_fit_{v}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")

    if len(out) >= 2:
        labs = list(out)
        for k in ("tau", "c1", "c2"):
            vals = [out[l][k] for l in labs]
            print(f"  {k}: {['%.0f' % x if k == 'tau' else '%.3f' % x for x in vals]}"
                  f"  spread ±{100 * np.ptp(vals) / 2 / np.mean(vals):.1f} %")
    with open(args.out + f"share_lp_library_fit_{v}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
