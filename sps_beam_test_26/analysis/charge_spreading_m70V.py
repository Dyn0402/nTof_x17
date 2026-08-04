#!/usr/bin/env python3
"""Measure det4's charge-spreading kernel from flat (perpendicular) beam tracks.

The parameters wanted are the wft forward model's sharing hyper-parameters
(`wft/calib.py`, HYPER_NAMES): the +-1-strip copy amplitude ``c1``, the
+-2-strip copy ``c2``, the sharing delay ``tau_s``, the Y-plane scale ``kY``,
and the +-2 delay factor ``tau2_fac`` that distinguishes a linear ladder
(2 tau) from RC diffusion (4 tau).

Why perpendicular tracks make this clean.  In the model each 60 ns slice of
drift charge lands at ``p0 + w * u_k``; at normal incidence ``w = 0``, so every
slice lands at the SAME transverse position.  Any charge on a neighbouring
strip is therefore either (a) direct, from transverse diffusion of that same
column, arriving prompt, or (b) a resistive copy, arriving late.  Two
components, separated in time.  On an inclined cosmic the two are entangled
with the track angle, which is exactly the degeneracy that forced the forward
fit in the first place.

Three measurements, in increasing order of what they constrain:

  1. amplitude profile vs offset          -> c1+diffusion combined (degenerate)
  2. peak-time shift vs offset            -> tau_s and tau2_fac (NOT degenerate)
  3. prompt/delayed two-component fit     -> c1, c2 separated from diffusion

and one control that has to pass before any of it means anything:

  * time walk.  A weak pulse can reconstruct late for purely instrumental
    reasons.  If dt depended only on amplitude and not on offset, measurement 2
    would be an artefact.  The control compares strips of the SAME amplitude at
    different offsets.

  python charge_spreading_m70V.py [--plateau 625V] [--view x]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM      # noqa: E402

STAGE = "/media/dylan/data/x17/sps_run53_det4_check/staging/run_56_m70V/"
SNS = 60.0                                  # ns per sample
SIDX = np.round(POSITION_MM / PITCH_MM).astype(int)
ZS_THR_ADC = 5 * 8.20                       # 5 sigma on FEU3's 8.20 ADC noise


# --------------------------------------------------------------------- pieces
def per_strip(ev, ch, samp, amp):
    """Collapse (event, channel) sample windows into per-strip quantities.

    Returns a dict of arrays, one entry per (event, channel):
      ev, ch, peak, tpeak [ns], nsamp, edge (peak sits at a window edge),
      area, first, last
    """
    key = ev.astype(np.int64) * 1024 + ch.astype(np.int64)
    o = np.lexsort((samp, key))
    ev, ch, samp, amp, key = ev[o], ch[o], samp[o], amp[o], key[o]
    b = np.flatnonzero(np.diff(key) != 0) + 1
    starts, ends = np.r_[0, b], np.r_[b, len(key)]

    n = len(starts)
    out = dict(ev=np.empty(n, np.int64), ch=np.empty(n, np.int16),
               peak=np.empty(n), tpeak=np.empty(n), nsamp=np.empty(n, np.int16),
               edge=np.zeros(n, bool), area=np.empty(n),
               first=np.empty(n, np.int16), last=np.empty(n, np.int16))
    for i, (s, e) in enumerate(zip(starts, ends)):
        A, S = amp[s:e], samp[s:e]
        k = int(np.argmax(A))
        out["ev"][i], out["ch"][i] = ev[s], ch[s]
        out["peak"][i] = A[k]
        out["nsamp"][i] = e - s
        out["area"][i] = A.sum()
        out["first"][i], out["last"][i] = S[0], S[-1]
        # parabolic peak time; needs both neighbours present AND contiguous
        if (k == 0 or k == len(A) - 1 or S[k] - S[k - 1] != 1
                or S[k + 1] - S[k] != 1):
            out["edge"][i] = True
            out["tpeak"][i] = S[k] * SNS
        else:
            y0, y1, y2 = A[k - 1], A[k], A[k + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den < -1e-9 else 0.0
            d = float(np.clip(d, -1.0, 1.0))
            out["tpeak"][i] = (S[k] + d) * SNS
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", default=STAGE + "wf_m70V.npz")
    ap.add_argument("--plateau", default="625V", choices=["590V", "625V", "both"])
    ap.add_argument("--min-peak", type=float, default=150.0,
                    help="central-strip peak ADC floor")
    ap.add_argument("--max-peak", type=float, default=3000.0,
                    help="central-strip ceiling, below the ~3550 ADC saturation")
    ap.add_argument("--out", default=STAGE)
    args = ap.parse_args()

    D = np.load(args.wf, allow_pickle=False)
    ev, ch, samp, amp = D["ev"], D["ch"], D["samp"], D["amp"]
    plateau, pX, pY = D["ev_plateau"], D["ev_pX"], D["ev_pY"]

    if args.plateau != "both":
        m = plateau[ev] == args.plateau
        ev, ch, samp, amp = ev[m], ch[m], samp[m], amp[m]
    print(f"plateau {args.plateau}: {len(np.unique(ev))} events, "
          f"{len(ev)} sample records")

    P = per_strip(ev, ch, samp, amp)
    sview = VIEW[P["ch"]]
    spos = POSITION_MM[P["ch"]]
    sidx = SIDX[P["ch"]]
    ppos = np.where(sview == "x", pX[P["ev"]], pY[P["ev"]])

    results = {}
    for view in ("x", "y"):
        v = sview == view
        print(f"\n{'='*72}\nVIEW {view.upper()}  ({v.sum()} strip records)")
        e_, i_, pk_, tp_, ed_, po_, pp_ = (P["ev"][v], sidx[v], P["peak"][v],
                                           P["tpeak"][v], P["edge"][v],
                                           spos[v], ppos[v])
        # ---- per event: the peak strip -----------------------------------
        order = np.lexsort((-pk_, e_))
        e_s, i_s, pk_s = e_[order], i_[order], pk_[order]
        first = np.r_[True, e_s[1:] != e_s[:-1]]
        lead_ev, lead_idx, lead_pk = e_s[first], i_s[first], pk_s[first]
        ok = (lead_pk >= args.min_peak) & (lead_pk <= args.max_peak)
        lead_ev, lead_idx, lead_pk = lead_ev[ok], lead_idx[ok], lead_pk[ok]
        pos_in = np.searchsorted(lead_ev, e_)
        pos_in = np.clip(pos_in, 0, len(lead_ev) - 1)
        has = (len(lead_ev) > 0) & (lead_ev[pos_in] == e_)
        d = np.where(has, i_ - lead_idx[pos_in], 999)
        q0 = np.where(has, lead_pk[pos_in], np.nan)
        # the peak strip's own time, per record
        lead_t = np.full(len(lead_ev), np.nan)
        isl = has & (d == 0) & ~ed_
        lead_t[np.searchsorted(lead_ev, e_[isl])] = tp_[isl]
        t0 = np.where(has, lead_t[pos_in], np.nan)
        print(f"  {len(lead_ev)} events with a peak strip in "
              f"[{args.min_peak:.0f}, {args.max_peak:.0f}] ADC; "
              f"median peak {np.median(lead_pk):.0f} ADC")

        # ---- 1. amplitude ratio and detection fraction vs offset ---------
        print(f"\n  amplitude ratio to the peak strip, and how often the strip "
              f"is present at all:")
        print(f"  {'d':>4} {'n_present':>10} {'present':>9} {'median A/A0':>12} "
              f"{'p16':>7} {'p84':>7}")
        n_lead = len(lead_ev)
        ratio_tab = {}
        for dd in range(-5, 6):
            k = has & (d == dd) & ~np.isnan(q0)
            if k.sum() < 20:
                continue
            r = pk_[k] / q0[k]
            ratio_tab[dd] = (k.sum() / n_lead, np.median(r))
            print(f"  {dd:>4} {k.sum():10d} {k.sum()/n_lead:8.1%} "
                  f"{np.median(r):12.4f} {np.percentile(r,16):7.4f} "
                  f"{np.percentile(r,84):7.4f}")

        # ---- censoring check: detection fraction vs central amplitude ----
        print(f"\n  censoring check -- fraction of events where |d|=1 and |d|=2 "
              f"are present, vs the peak strip's amplitude\n"
              f"  (if the tail were uncensored these would saturate at 1)")
        print(f"  {'q0 ADC':>12} {'n_ev':>7} {'|d|=1':>8} {'|d|=2':>8} "
              f"{'|d|=3':>8}   {'expected |d|=2 ADC':>19}")
        edges = [150, 250, 400, 600, 900, 1400, 3000]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel_ev = lead_ev[(lead_pk >= lo) & (lead_pk < hi)]
            if len(sel_ev) < 50:
                continue
            inb = np.isin(e_, sel_ev)
            fr = []
            for dm in (1, 2, 3):
                kk = inb & has & (np.abs(d) == dm)
                fr.append(len(np.unique(e_[kk])) / len(sel_ev))
            mid = 0.5 * (lo + hi)
            exp2 = ratio_tab.get(2, (0, 0))[1] * mid
            print(f"  {lo:5d}-{hi:<6d} {len(sel_ev):7d} {fr[0]:8.1%} "
                  f"{fr[1]:8.1%} {fr[2]:8.1%}   {exp2:19.0f}")

        # ---- 2. peak-time shift vs offset --------------------------------
        print(f"\n  peak-time shift relative to the peak strip "
              f"[ns] (parabolic, contiguous windows only):")
        print(f"  {'d':>4} {'n':>8} {'median dt':>11} {'mean dt':>9} "
              f"{'sem':>6} {'p16':>7} {'p84':>7}")
        dt_tab = {}
        for dd in range(-4, 5):
            k = has & (d == dd) & ~ed_ & ~np.isnan(t0)
            if k.sum() < 20:
                continue
            dt = tp_[k] - t0[k]
            dt_tab[dd] = (np.median(dt), np.std(dt) / np.sqrt(len(dt)), k.sum())
            print(f"  {dd:>4} {k.sum():8d} {np.median(dt):11.1f} "
                  f"{np.mean(dt):9.1f} {np.std(dt)/np.sqrt(len(dt)):6.1f} "
                  f"{np.percentile(dt,16):7.1f} {np.percentile(dt,84):7.1f}")

        # ---- the control: time walk --------------------------------------
        print(f"\n  TIME-WALK CONTROL -- median dt [ns] in bins of the strip's "
              f"OWN amplitude.\n  A pure instrumental walk would give the same "
              f"dt at the same amplitude,\n  independent of offset:")
        abins = [(40, 70), (70, 120), (120, 220), (220, 450), (450, 1200)]
        hdr = "".join(f"{f'{a}-{b}':>12}" for a, b in abins)
        print(f"  {'d':>4}{hdr}")
        walk = {}
        for dd in (0, 1, -1, 2, -2):
            row, cells = f"  {dd:>4}", []
            for lo, hi in abins:
                k = (has & (d == dd) & ~ed_ & ~np.isnan(t0)
                     & (pk_ >= lo) & (pk_ < hi))
                if k.sum() < 30:
                    row += f"{'-':>12}"
                    cells.append(np.nan)
                    continue
                mdt = np.median(tp_[k] - t0[k])
                row += f"{mdt:12.1f}"
                cells.append(mdt)
            walk[dd] = cells
            print(row)

        results[view] = dict(ratio=ratio_tab, dt=dt_tab, walk=walk,
                             abins=abins, n_lead=n_lead,
                             lead_pk_med=float(np.median(lead_pk)))

        # ---- 3. transverse profile in mm, uRWELL-referenced ---------------
        u = po_ - pp_
        k = has & (np.abs(u) < 6) & ~np.isnan(q0)
        ub = np.arange(-6, 6.01, 0.39)
        idx_b = np.digitize(u[k], ub) - 1
        prof = np.array([np.median(pk_[k][idx_b == i] / q0[k][idx_b == i])
                         if np.sum(idx_b == i) > 20 else np.nan
                         for i in range(len(ub) - 1)])
        results[view]["prof_u"] = 0.5 * (ub[1:] + ub[:-1])
        results[view]["prof"] = prof

    # ------------------------------------------------------------ derived
    print(f"\n{'='*72}\nDERIVED PARAMETERS ({args.plateau})")
    for view in ("x", "y"):
        R = results[view]
        dt, rt = R["dt"], R["ratio"]

        def sym(tab, dd, i=0):
            a = [tab[s][i] for s in (dd, -dd) if s in tab]
            return float(np.mean(a)) if a else np.nan

        t1, t2 = sym(dt, 1), sym(dt, 2)
        r1, r2 = sym(rt, 1, 1), sym(rt, 2, 1)
        print(f"\n  view {view.upper()}:")
        print(f"    dt(+-1)            = {t1:7.1f} ns   <- sets tau_s")
        print(f"    dt(+-2)            = {t2:7.1f} ns")
        print(f"    dt(2)/dt(1)        = {t2/t1:7.2f}      "
              f"(2.0 = linear ladder, 4.0 = RC diffusion)")
        print(f"    A(+-1)/A(0)        = {r1:7.4f}      "
              f"(direct diffusion + delayed share, NOT c1)")
        print(f"    A(+-2)/A(0)        = {r2:7.4f}      (censored, biased high)")
    rx, ry = results["x"], results["y"]

    def sm(R, dd, key, i):
        return float(np.mean([R[key][s][i] for s in (dd, -dd) if s in R[key]]))
    print(f"\n  kY proxies (Y relative to X):")
    print(f"    amplitude ratio  A1_Y/A1_X = "
          f"{sm(ry,1,'ratio',1)/sm(rx,1,'ratio',1):.3f}   <- kY")
    print(f"    delay ratio      dt1_Y/dt1_X = "
          f"{sm(ry,1,'dt',0)/sm(rx,1,'dt',0):.3f}   <- kTauY")

    np.savez(args.out + f"charge_spreading_{args.plateau}.npz",
             **{f"{v}__{k}": np.array(val, dtype=object) if isinstance(val, dict)
                else val for v, R in results.items() for k, val in R.items()})

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for view, c in (("x", "#1f4e79"), ("y", "#c1440e")):
        R = results[view]
        ax.plot(R["prof_u"], R["prof"], "o-", ms=3, color=c,
                label=f"{view.upper()} view")
    ax.axhline(ZS_THR_ADC / R["lead_pk_med"], ls=":", color="0.5",
               label="5 sigma ZS floor")
    ax.set(yscale="log", xlabel="strip position - uRWELL track [mm]",
           ylabel="amplitude / peak-strip amplitude",
           title="transverse profile (reference-smeared)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for view, c in (("x", "#1f4e79"), ("y", "#c1440e")):
        dd = sorted(results[view]["dt"])
        ax.errorbar(dd, [results[view]["dt"][k][0] for k in dd],
                    yerr=[results[view]["dt"][k][1] for k in dd],
                    fmt="o-", ms=4, color=c, label=f"{view.upper()} view")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="strip offset from the peak strip",
           ylabel="peak-time shift [ns]",
           title="delay vs offset — the sharing signature")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    R = results["x"]
    cen = [0.5 * (a + b) for a, b in R["abins"]]
    for dd, c in ((0, "0.3"), (1, "#1f4e79"), (-1, "#3b8ed0"),
                  (2, "#c1440e"), (-2, "#e08a5a")):
        if dd in R["walk"]:
            ax.plot(cen, R["walk"][dd], "o-", ms=4, color=c, label=f"d={dd:+d}")
    ax.set(xscale="log", xlabel="the strip's own peak amplitude [ADC]",
           ylabel="peak-time shift [ns]",
           title="time-walk control, X view\n(separation at fixed amplitude = real)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.suptitle(f"det4 charge spreading — run 56 m70V, flat mount, "
                 f"{args.plateau}, Ar/CO2/iso 95/3/2, drift 700 V", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out + f"charge_spreading_{args.plateau}.png", dpi=115,
                bbox_inches="tight")
    print(f"\nwrote {args.out}charge_spreading_{args.plateau}.png")


if __name__ == "__main__":
    main()
