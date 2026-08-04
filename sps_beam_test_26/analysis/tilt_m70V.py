#!/usr/bin/env python3
"""Is det4 actually normal to the beam?  Measure its tilt from the data.

The uRWELL track slopes only say where the BEAM points in the uRWELL frame
(mean 0.18 deg, rms 0.61 deg here).  They say nothing about how det4's own
plane is hung.  If det4 is tilted, the ionisation column is inclined to its
drift axis -- ``w != 0`` in the wft forward model -- and the charge centroid
walks transversely as the column drifts in.  That walk is a direct, absolute
measurement of the tilt, and it is exactly the thing that has to be zero for
the flat-geometry argument behind the kernel measurement to hold.

Two independent estimators, both from this sub-run:

  1. CENTROID WALK.  Per event, the charge-weighted strip position in each
     60 ns sample; averaged over events, its slope vs time is
     ``w = v_drift * tan(theta)``.  With v_drift known this gives theta.
  2. TIME ANTISYMMETRY.  The mean arrival time vs strip offset splits into a
     symmetric part (the resistive sharing, which is left/right symmetric by
     construction) and an antisymmetric part (the tilt, which is not).  The
     antisymmetric slope is ``dt/dx = 1 / (v_drift * tan(theta))``.

Estimator 2 is the useful one for the kernel work, because the same
decomposition CLEANS the kernel measurement: the symmetric part is what the
sharing fit should be run on.

  python tilt_m70V.py [--plateau 625V] [--vdrift 0.034]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", default=STAGE + "wf_m70V.npz")
    ap.add_argument("--plateau", default="625V")
    ap.add_argument("--vdrift", type=float, default=0.034,
                    help="mm/ns; June geometry estimator gave 34 +- 1.5 um/ns "
                         "at 1000 V in Ar/iso -- a stand-in only, this gas and "
                         "field are different")
    ap.add_argument("--q0", default="400,3000")
    ap.add_argument("--out", default=STAGE)
    args = ap.parse_args()
    q0lo, q0hi = (float(x) for x in args.q0.split(","))

    D = np.load(args.wf)
    ev, ch, samp, amp = D["ev"], D["ch"], D["samp"], D["amp"]
    m = D["ev_plateau"][ev] == args.plateau
    ev, ch, samp, amp = ev[m], ch[m], samp[m], amp[m]
    P = per_strip(ev, ch, samp, amp)
    sview, sidx = VIEW[P["ch"]], SIDX[P["ch"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    out = {}
    for ax, view in zip(axes, ("x", "y")):
        v = sview == view
        e_, i_, pk_ = P["ev"][v], sidx[v], P["peak"][v]
        o = np.lexsort((-pk_, e_))
        es, is_, pks = e_[o], i_[o], pk_[o]
        f = np.r_[True, es[1:] != es[:-1]]
        lev, lidx, lpk = es[f], is_[f], pks[f]
        ok = (lpk >= q0lo) & (lpk <= q0hi)
        lev, lidx, lpk = lev[ok], lidx[ok], lpk[ok]

        s = VIEW[ch] == view
        kev, ksm, kam, ksi = ev[s], samp[s], amp[s], SIDX[ch[s]]
        j = np.clip(np.searchsorted(lev, kev), 0, max(len(lev) - 1, 0))
        inref = (len(lev) > 0) & (lev[j] == kev)
        d = np.where(inref, ksi - lidx[j], 999)

        # central-strip peak sample per event -> common time origin
        cm = inref & (d == 0)
        o2 = np.lexsort((-kam[cm], kev[cm]))
        ce, cs = kev[cm][o2], ksm[cm][o2]
        ff = np.r_[True, ce[1:] != ce[:-1]]
        pev, psamp = ce[ff], cs[ff]
        jj = np.clip(np.searchsorted(pev, kev), 0, max(len(pev) - 1, 0))
        haspk = (len(pev) > 0) & (pev[jj] == kev)
        rel = ksm.astype(int) - psamp[jj]

        # ---- 1. centroid walk: <d> weighted by amplitude, per relative sample
        k = inref & haspk & (np.abs(d) <= 4) & (np.abs(rel) <= 12) & (kam > 0)
        num = np.zeros(25)
        den = np.zeros(25)
        np.add.at(num, rel[k] + 12, kam[k] * d[k])
        np.add.at(den, rel[k] + 12, kam[k])
        cen = np.divide(num, den, out=np.full(25, np.nan), where=den > 0)
        t = (np.arange(25) - 12) * SNS
        good = np.isfinite(cen) & (den > 0.02 * np.nanmax(den))
        sl, ic = np.polyfit(t[good], cen[good] * PITCH_MM, 1)   # mm per ns
        theta = np.degrees(np.arctan(abs(sl) / args.vdrift))
        out[view] = dict(t=t, cen=cen, slope=sl, theta=theta, good=good)
        print(f"\nview {view.upper()}  ({len(lev)} events, q0 "
              f"{q0lo:.0f}-{q0hi:.0f} ADC)")
        print(f"  centroid walk   dx/dt = {sl*1000:+.4f} um/ns")
        print(f"                  w/v   = tan(theta) = {sl/args.vdrift:+.4f}"
              f"  ->  theta = {theta:.2f} deg   "
              f"(at v_drift = {args.vdrift*1000:.0f} um/ns)")

        # ---- 2. symmetric / antisymmetric split of the arrival time
        e2, i2, tp2, ed2 = (P["ev"][v], sidx[v], P["tpeak"][v], P["edge"][v])
        j2 = np.clip(np.searchsorted(lev, e2), 0, max(len(lev) - 1, 0))
        in2 = (len(lev) > 0) & (lev[j2] == e2)
        d2 = np.where(in2, i2 - lidx[j2], 999)
        lt = np.full(len(lev), np.nan)
        isl = in2 & (d2 == 0) & ~ed2
        lt[np.searchsorted(lev, e2[isl])] = tp2[isl]
        t0 = np.where(in2, lt[j2], np.nan)
        dtm = {}
        for dd in (-2, -1, 1, 2):
            kk = in2 & (d2 == dd) & ~ed2 & ~np.isnan(t0)
            dt = tp2[kk] - t0[kk]
            core = np.abs(dt) < 150                      # accidental-free core
            if core.sum() > 30:
                dtm[dd] = float(np.median(dt[core]))
        if 1 in dtm and -1 in dtm:
            sym1 = 0.5 * (dtm[1] + dtm[-1])
            asym1 = 0.5 * (dtm[1] - dtm[-1])
            print(f"  arrival time    dt(+1)={dtm[1]:+.1f}  dt(-1)={dtm[-1]:+.1f} ns")
            print(f"                  symmetric  {sym1:+.1f} ns  <- sharing")
            print(f"                  antisym    {asym1:+.1f} ns  <- tilt")
            if abs(asym1) > 1e-6:
                th2 = np.degrees(np.arctan(PITCH_MM /
                                           (args.vdrift * abs(asym1))))
                print(f"                  antisym -> theta = {th2:.2f} deg")
                out[view]["theta_asym"] = th2
            out[view].update(sym1=sym1, asym1=asym1, dtm=dtm)

        ax.plot(t[good], cen[good] * PITCH_MM, "o-", color="#1f4e79", ms=4)
        ax.plot(t[good], np.polyval([sl, ic], t[good]), "--", color="tab:red",
                label=f"slope {sl*1000:+.3f} um/ns\n-> {theta:.2f} deg")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set(xlabel="time relative to the central strip's peak [ns]",
               ylabel="charge centroid offset [mm]",
               title=f"{view.upper()} view — does the column walk?")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"det4 tilt from the centroid walk — run 56 m70V, "
                 f"'flat' mount, {args.plateau}", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out + f"tilt_{args.plateau}.png", dpi=120,
                bbox_inches="tight")
    print(f"\nwrote {args.out}tilt_{args.plateau}.png")


if __name__ == "__main__":
    main()
