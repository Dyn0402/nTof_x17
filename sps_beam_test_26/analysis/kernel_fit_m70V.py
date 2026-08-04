#!/usr/bin/env python3
"""Fit det4's resistive sharing kernel from flat-track mean waveforms.

What the mean waveforms show (see the figure this writes): a neighbour strip's
signal peaks essentially WITH the central strip, but carries a long tail that
the central strip does not have.  So the sharing is not "a copy delayed by
tau_s" -- it is an RC-*dispersed* copy.  That is the ``share_lp`` branch of
``wft/model.py`` (``_lp_copies``), not the plain-delay branch, and this fit
measures it directly instead of inferring it from a track fit.

The model fitted, per view, per offset d:

    W_d(t) = alpha_d * W_0(t)  +  beta_d * (W_0 (*) K_tau^|d|)(t)

  W_0        the measured central-strip mean waveform.  Using the data's own
             shape as the basis means no shaper template, no drift-ladder
             model and no v_drift enter the kernel measurement at all -- the
             ionisation column is common to every strip in the event because
             the track is normal to the plane.
  alpha_d    prompt fraction: charge that diffused directly onto strip d.
  beta_d     dispersed fraction: charge that arrived through the resistive
             layer.  This is what c1 (d=+-1) and c2 (d=+-2) are.
  K_tau      one-pole RC kernel, cascaded |d| times -- the ladder picture.
  tau        fitted globally per view across d = +-1, +-2.

Acceptance.  Zero suppression records a strip only where it is above ~5 sigma,
so the far tail of a weak neighbour is missing and a naive mean biases beta
LOW.  The fit therefore (a) uses only events with a strong central strip
(default 900-3000 ADC), where the +-2 neighbour is present ~99.5% of the time,
and (b) reports the per-sample acceptance so the bias is visible rather than
hidden.  Quoted beta values are lower bounds where acceptance < 1.

  python kernel_fit_m70V.py [--plateau 625V] [--q0 900,3000]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/analysis")
sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from charge_spreading_m70V import per_strip            # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM    # noqa: E402

STAGE = "/media/dylan/data/x17/sps_run53_det4_check/staging/run_56_m70V/"
SNS = 60.0
SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)
NREL = 30                                   # +-30 samples about the central peak


def mean_waveforms(D, plateau, q0lo, q0hi, raw=False):
    """Per view: mean waveform and acceptance vs offset, aligned per event.

    ``raw`` selects how the missing samples are treated, and the two cases need
    OPPOSITE handling:

    * **RAW runs** lose samples to FEU packet drops.  The loss is uniform and
      independent of the pulse amplitude (measured: acceptance 0.759-0.766,
      flat across the window), so a sample is missing for reasons unrelated to
      what it contained.  The unbiased mean is therefore the mean over the
      samples that DID arrive -- i.e. divide by the acceptance.  Skipping this
      makes every dispersed-copy amplitude come out ~24 % low.
    * **ZS runs** lose samples *because they were small* (below threshold).
      There the mean over recorded samples would badly overestimate, and
      treating a missing sample as ~0 is the closer approximation.  So no
      division -- which is what this function has always done.
    """
    ev, ch, samp, amp = D["ev"], D["ch"], D["samp"], D["amp"]
    m = D["ev_plateau"][ev] == plateau
    ev, ch, samp, amp = ev[m], ch[m], samp[m], amp[m]
    P = per_strip(ev, ch, samp, amp)
    sview, sidx = VIEW[P["ch"]], SIDX[P["ch"]]

    out = {}
    for view in ("x", "y"):
        v = sview == view
        e_, i_, pk_ = P["ev"][v], sidx[v], P["peak"][v]
        o = np.lexsort((-pk_, e_))
        es, is_, pks = e_[o], i_[o], pk_[o]
        f = np.r_[True, es[1:] != es[:-1]]
        lev, lidx, lpk = es[f], is_[f], pks[f]
        ok = (lpk >= q0lo) & (lpk <= q0hi)
        lev, lidx, lpk = lev[ok], lidx[ok], lpk[ok]

        s = VIEW[ch] == view
        kev, kch, ksm, kam = ev[s], ch[s], samp[s], amp[s]
        ksi = SIDX[kch]
        j = np.clip(np.searchsorted(lev, kev), 0, max(len(lev) - 1, 0))
        inref = (len(lev) > 0) & (lev[j] == kev)
        d = np.where(inref, ksi - lidx[j], 999)
        q0 = np.where(inref, lpk[j], np.nan)

        cm = inref & (d == 0)
        o2 = np.lexsort((-kam[cm], kev[cm]))
        ce, cs = kev[cm][o2], ksm[cm][o2]
        ff = np.r_[True, ce[1:] != ce[:-1]]
        pev, psamp = ce[ff], cs[ff]
        jj = np.clip(np.searchsorted(pev, kev), 0, max(len(pev) - 1, 0))
        haspk = (len(pev) > 0) & (pev[jj] == kev)
        rel = ksm.astype(int) - psamp[jj]

        W, ACC = {}, {}
        for dd in (0, 1, -1, 2, -2, 3, -3):
            k = inref & haspk & (d == dd) & (np.abs(rel) <= NREL)
            if k.sum() < 100:
                continue
            w = np.zeros(2 * NREL + 1)
            a = np.zeros(2 * NREL + 1)
            np.add.at(w, rel[k] + NREL, kam[k] / q0[k])
            np.add.at(a, rel[k] + NREL, 1.0)
            acc = a / len(lev)
            if raw:
                # divide by acceptance where it is meaningfully sampled; the
                # far edges of the +-30 sample window are not, and 0/0 there
                # would manufacture spikes
                w = np.divide(w / len(lev), acc,
                              out=np.zeros_like(w), where=acc > 0.05)
                W[dd] = w
            else:
                W[dd] = w / len(lev)
            ACC[dd] = acc
        out[view] = dict(W=W, ACC=ACC, n=len(lev),
                         t=(np.arange(2 * NREL + 1) - NREL) * SNS)
    return out


def rc_cascade(w0, tau, n, dt=SNS):
    """Convolve w0 with n cascaded one-pole RC kernels of time constant tau."""
    if tau <= 1e-3 or n == 0:
        return w0.copy()
    # discrete one-pole: y[i] = y[i-1]*a + x[i]*(1-a)
    a = np.exp(-dt / tau)
    y = w0.copy()
    for _ in range(int(n)):
        z = np.zeros_like(y)
        acc = 0.0
        for i in range(len(y)):
            acc = acc * a + y[i] * (1 - a)
            z[i] = acc
        y = z
    return y


def fit_view(t, W, ACC, acc_min=0.35, verbose=True):
    """Global tau + per-offset (alpha, beta) for one view."""
    w0 = W[0]
    offs = [dd for dd in (1, -1, 2, -2) if dd in W]
    # fit only where the CENTRAL strip is well sampled and after the rise
    base = ACC[0] > acc_min

    def resid(p):
        tau = p[0]
        r = []
        for i, dd in enumerate(offs):
            al, be = p[1 + 2 * i], p[2 + 2 * i]
            mdl = al * w0 + be * rc_cascade(w0, tau, abs(dd))
            msk = base & (ACC[dd] > acc_min * 0.3)
            r.append((mdl - W[dd])[msk])
        return np.concatenate(r)

    p0 = [150.0] + [0.2, 0.1] * len(offs)
    lo = [5.0] + [0.0, 0.0] * len(offs)
    hi = [3000.0] + [2.0, 2.0] * len(offs)
    res = least_squares(resid, p0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
    tau = res.x[0]
    par = {dd: (res.x[1 + 2 * i], res.x[2 + 2 * i]) for i, dd in enumerate(offs)}
    if verbose:
        print(f"    tau = {tau:.0f} ns   (cost {res.cost:.3e})")
        print(f"    {'d':>4} {'alpha (prompt)':>16} {'beta (dispersed)':>18} "
              f"{'beta/(a+b)':>12}")
        for dd in offs:
            al, be = par[dd]
            print(f"    {dd:>4} {al:16.4f} {be:18.4f} {be/(al+be):12.2f}")
    return tau, par, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", default=STAGE + "wf_m70V.npz")
    ap.add_argument("--plateau", default="625V")
    ap.add_argument("--q0", default="900,3000")
    ap.add_argument("--out", default=STAGE)
    ap.add_argument("--raw", action="store_true",
                    help="input is a RAW run: correct for the uniform "
                         "FEU packet loss by dividing by the acceptance")
    ap.add_argument("--label", default="run 56 m70V, Ar/CO2/iso 95/3/2, drift 700 V",
                    help="dataset description for figure/print headers")
    args = ap.parse_args()
    q0lo, q0hi = (float(x) for x in args.q0.split(","))

    D = np.load(args.wf)
    raw = bool(args.raw)
    if raw:
        print("RAW mode: dividing mean waveforms by the per-sample "
              "acceptance (packet loss is amplitude-independent)")
    M = mean_waveforms(D, args.plateau, q0lo, q0hi, raw=raw)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    summary = {}
    for col, view in enumerate(("x", "y")):
        R = M[view]
        t, W, ACC = R["t"], R["W"], R["ACC"]
        print(f"\n=== view {view.upper()}  ({R['n']} events, central strip "
              f"{q0lo:.0f}-{q0hi:.0f} ADC) ===")
        tau, par, res = fit_view(t, W, ACC)
        summary[view] = dict(tau=tau, par=par)

        ax = axes[0, col]
        w0 = W[0]
        ax.plot(t, w0, color="0.15", lw=1.6, label="d=0 (basis)")
        for dd, c in ((1, "#1f4e79"), (-1, "#5b9bd5"),
                      (2, "#c1440e"), (-2, "#e8a87c")):
            if dd not in W:
                continue
            ax.plot(t, W[dd], color=c, lw=1.3, label=f"d={dd:+d}")
            al, be = par[dd]
            ax.plot(t, al * w0 + be * rc_cascade(w0, tau, abs(dd)),
                    color=c, lw=1.0, ls="--")
        ax.set(yscale="log", ylim=(1e-4, 1.5), xlabel="t - central peak [ns]",
               ylabel="mean amp / central peak",
               title=f"{view.upper()} view — data (solid) vs "
                     f"prompt+RC fit (dashed), tau={tau:.0f} ns")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        for dd, c in ((0, "0.15"), (1, "#1f4e79"), (-1, "#5b9bd5"),
                      (2, "#c1440e"), (-2, "#e8a87c")):
            if dd in ACC:
                ax.plot(t, ACC[dd], color=c, lw=1.2, label=f"d={dd:+d}")
        ax.axhline(0.35, ls=":", color="0.5")
        ax.set(xlabel="t - central peak [ns]",
               ylabel="fraction of events with this sample recorded",
               title=f"{view.upper()} view — ZS acceptance "
                     "(below the dotted line the tail is censored)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle(f"det4 sharing kernel — flat mount, {args.plateau} — "
                 f"{args.label}", y=1.0)
    fig.tight_layout()
    fig.savefig(args.out + f"kernel_fit_{args.plateau}.png", dpi=120,
                bbox_inches="tight")
    print(f"\nwrote {args.out}kernel_fit_{args.plateau}.png")

    # ------------------------------------------------------- what to quote
    print("\n" + "=" * 72)
    print("KERNEL PARAMETERS (wft HYPER names), det4, flat, "
          f"{args.plateau} -- {args.label}")
    for view in ("x", "y"):
        S = summary[view]
        p = S["par"]
        c1 = np.mean([p[d][1] for d in (1, -1) if d in p])
        c2 = np.mean([p[d][1] for d in (2, -2) if d in p])
        a1 = np.mean([p[d][0] for d in (1, -1) if d in p])
        print(f"  view {view.upper()}:  tau_s = {S['tau']:6.0f} ns    "
              f"c1 = {c1:.4f}    c2 = {c2:.4f}    "
              f"(prompt diffusion onto +-1: {a1:.4f})")
    sx, sy = summary["x"], summary["y"]
    c1x = np.mean([sx["par"][d][1] for d in (1, -1) if d in sx["par"]])
    c1y = np.mean([sy["par"][d][1] for d in (1, -1) if d in sy["par"]])
    print(f"  kY    = c1_Y / c1_X       = {c1y / c1x:.3f}")
    print(f"  kTauY = tau_Y / tau_X     = {sy['tau'] / sx['tau']:.3f}")

    np.savez(args.out + f"kernel_fit_{args.plateau}.npz",
             **{f"{v}__{k}": np.array(vv) for v, R in M.items()
                for k, vv in (("t", R["t"]),)},
             **{f"{v}__W{d}": R["W"][d] for v, R in M.items() for d in R["W"]},
             **{f"{v}__ACC{d}": R["ACC"][d] for v, R in M.items() for d in R["ACC"]},
             **{f"{v}__tau": summary[v]["tau"] for v in M},
             **{f"{v}__par": np.array([[d, *summary[v]["par"][d]]
                                       for d in summary[v]["par"]]) for v in M})


if __name__ == "__main__":
    main()
