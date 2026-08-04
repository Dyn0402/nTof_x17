#!/usr/bin/env python3
"""det4 <-> uRWELL alignment and efficiency for run_56 / meshscan_m70V.

Flat mount, Ar/CO2/iso 95/3/2, drift 700 V.  The sub-run straddles two det4
resist plateaus, so everything here is done per plateau and never merged:

    590.0 V   15:47:25 - 15:52:50     (the m60V ladder's last point, running on)
    624.7 V   15:52:57 - 15:59:34     (the highest voltage det4 ever ran flat)

A 7 s guard band around the 15:52:50-15:52:57 ramp is dropped: the wall-clock
axis is rebuilt from per-file timestamp spans and is only good to a few
seconds, which is fine everywhere except right at the step.

  python align_eff_m70V.py [--gate 600,3650] [--grid 3]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW                # noqa: E402

STAGE = "/media/dylan/data/x17/sps_run53_det4_check/staging/run_56_m70V/"
Z_BACK, Z_DET4 = 1370.0, 1120.0

#: (label, t_lo, t_hi) in wall-clock seconds
def _s(h, m, sec):
    return h * 3600 + m * 60 + sec


PLATEAUS = [
    ("590 V", _s(15, 47, 25), _s(15, 52, 50)),
    ("625 V", _s(15, 52, 57), _s(15, 59, 34)),
]


def det4_clusters(D, sel_hits, n_ev):
    """Per-view leading-cluster position, multiplicity, charge, strip count."""
    h_ev, h_ch, h_amp = D["h_ev"], D["h_ch"], D["h_amp"]
    out = {}
    for v in ("x", "y"):
        k = sel_hits & (VIEW[h_ch] == v)
        ev, pos, amp = h_ev[k], POSITION_MM[h_ch[k]], h_amp[k]
        L = np.full(n_ev, np.nan)
        N = np.zeros(n_ev, np.int32)
        Q = np.zeros(n_ev)
        S = np.zeros(n_ev, np.int32)
        if len(ev):
            o = np.lexsort((pos, ev))
            ev, pos, amp = ev[o], pos[o], amp[o]
            new = np.empty(len(ev), bool)
            new[0] = True
            new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > 3.0)
            cid = np.cumsum(new) - 1
            nc = cid[-1] + 1
            cq = np.bincount(cid, weights=amp, minlength=nc)
            cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
            cn = np.bincount(cid, minlength=nc)
            cev = np.zeros(nc, np.int64)
            cev[cid] = ev
            np.add.at(N, cev, 1)
            np.add.at(Q, cev, cq)
            o2 = np.argsort(cq, kind="stable")
            L[cev[o2]] = cp[o2]
            S[cev[o2]] = cn[o2]
        out[v] = (L, N, Q, S)
    return out


def fit_affine(tx, ty, dx, dy, keep):
    """Robust affine det4 <- uRWELL, trimmed to the best 80% of residuals."""
    U = np.column_stack([tx[keep], ty[keep], np.ones(keep.sum())])
    V = np.column_stack([dx[keep], dy[keep]])
    A, *_ = np.linalg.lstsq(U, V, rcond=None)
    for _ in range(5):
        r = np.hypot(*(V - U @ A).T)
        g = r < np.percentile(r, 80)
        A, *_ = np.linalg.lstsq(U[g], V[g], rcond=None)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=STAGE + "pair_m70V.npz")
    ap.add_argument("--gate", default="", help="det4 in-time gate lo,hi ns")
    ap.add_argument("--grid", type=float, default=3.0)
    ap.add_argument("--out", default=STAGE)
    args = ap.parse_args()

    D = np.load(args.npz)
    n_ev = len(D["ev_id"])
    t = D["ev_t_wall"]
    print(f"{n_ev} events")

    # ---------------------------------------------------------- uRWELL track
    clean = np.ones(n_ev, bool)
    for k in ("fx", "fy", "bx", "by"):
        clean &= (D[k + "_n"] == 1) & np.isfinite(D[k + "_p"])
    f = Z_DET4 / Z_BACK
    tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * f
    ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * f
    clean &= np.isfinite(tx + ty)
    print(f"clean single-cluster uRWELL tracks: {clean.sum()} "
          f"({clean.sum()/n_ev:.1%})")

    # ------------------------------------------------------------ det4 gate
    h_t = D["h_time"]
    if args.gate:
        g0, g1 = (float(v) for v in args.gate.split(","))
        how = "given"
    else:
        hb, eb = np.histogram(h_t, bins=np.arange(-1000, 6000, 100.0))
        cb = 0.5 * (eb[1:] + eb[:-1])
        pk = int(np.argmax(hb))
        thr = 0.25 * hb[pk]
        lo_i = pk - int(np.argmax(hb[pk::-1] < thr)) + 1
        hi_i = pk + int(np.argmax(hb[pk:] < thr))
        g0, g1 = cb[lo_i] - 50, cb[hi_i] + 50
        how = "auto from the hit-time spectrum"
    print(f"det4 in-time gate: {g0:.0f} - {g1:.0f} ns ({how})")
    GATE = (h_t > g0) & (h_t < g1)

    results = {}
    for label, t0, t1 in PLATEAUS:
        inwin = (t >= t0) & (t < t1)
        cl = clean & inwin
        sel_hits = GATE & cl[D["h_ev"]]
        C = det4_clusters(D, sel_hits, n_ev)
        lead = {v: C[v][0] for v in "xy"}
        ncl = {v: C[v][1] for v in "xy"}
        nst = {v: C[v][3] for v in "xy"}

        fired = np.zeros(n_ev, bool)
        fired[D["h_ev"][GATE & cl[D["h_ev"]]]] = True
        reco = np.isfinite(lead["x"] + lead["y"])
        disch = ((ncl["x"] + ncl["y"] >= 6) | (nst["x"] > 40) | (nst["y"] > 40))

        A = fit_affine(tx, ty, lead["x"], lead["y"], cl & reco & ~disch)
        M = A[:2].T
        pred = np.column_stack([tx, ty, np.ones(n_ev)]) @ A
        pX, pY = pred[:, 0], pred[:, 1]
        resid = np.hypot(lead["x"] - pX, lead["y"] - pY)
        within5 = cl & reco & (resid < 5.0)

        print(f"\n=== {label}  ({inwin.sum()} events, {cl.sum()} clean tracks) ===")
        print(f"  alignment: rot {np.degrees(np.arctan2(M[1,0], M[0,0])):+.2f} deg, "
              f"det(A) {np.linalg.det(M):+.4f}, "
              f"median |res| {np.nanmedian(resid[cl & reco]):.2f} mm")
        nd = cl & ~disch
        for name, m in (("fired at all", fired), ("reconstructed", reco),
                        ("within 5 mm", within5), ("discharge-flagged", disch)):
            print(f"  {name:20} {np.mean(m[cl]):7.1%} all "
                  f"{np.mean(m[nd]):7.1%} excl. discharges")

        results[label] = dict(pX=pX, pY=pY, clean=cl, within5=within5,
                              fired=fired, reco=reco, disch=disch, A=A,
                              resid=resid, lead=lead, n=cl.sum())

    # ------------------------------------------------------------- the maps
    J = np.load("/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                "det4_sps_assessment/stripes_g_det4.npz")
    G = args.grid
    cmap = LinearSegmentedColormap.from_list(
        "eff", ["#2b1b3d", "#3b5b92", "#3f9e9e", "#8ecf6a", "#f2e661"])
    cmap.set_bad("#e8e8e8")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                             gridspec_kw=dict(height_ratios=[1.4, 1]))
    save = {}
    for j, (label, _, _) in enumerate(PLATEAUS):
        R = results[label]
        cl, w5, pX, pY = R["clean"], R["within5"], R["pX"], R["pY"]
        lo_x, hi_x = np.percentile(pX[cl], [0.2, 99.8])
        lo_y, hi_y = np.percentile(pY[cl], [0.2, 99.8])
        ex = np.arange(np.floor(lo_x / G) * G, np.ceil(hi_x / G) * G + G, G)
        ey = np.arange(np.floor(lo_y / G) * G, np.ceil(hi_y / G) * G + G, G)
        den, _, _ = np.histogram2d(pX[cl], pY[cl], bins=[ex, ey])
        num, _, _ = np.histogram2d(pX[w5], pY[w5], bins=[ex, ey])
        eff = np.divide(num, den, out=np.full_like(num, np.nan), where=den >= 25)

        ax = axes[0, j]
        im = ax.pcolormesh(ex, ey, eff.T, cmap=cmap, vmin=0, vmax=1)
        for lo, hi in J["bands"]:
            if hi > ex[0] and lo < ex[-1]:
                ax.axvline(lo, color="w", lw=0.7, alpha=0.45)
                ax.axvline(hi, color="w", lw=0.7, alpha=0.45)
        ax.set(xlabel="det4-local X [mm]", ylabel="det4-local Y [mm]",
               title=f"{label} — {R['n']:,} clean tracks, "
                     f"median {np.nanmedian(eff):.0%}")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.035, label="efficiency")

        px_e = np.arange(np.floor(lo_x), np.ceil(hi_x) + 1, 1.0)
        dX, _ = np.histogram(pX[cl], bins=px_e)
        nX, _ = np.histogram(pX[w5], bins=px_e)
        aX, _ = np.histogram(pX[cl & R["fired"]], bins=px_e)
        prof = np.divide(nX, dX, out=np.full(len(dX), np.nan), where=dX >= 40)
        prof_any = np.divide(aX, dX, out=np.full(len(dX), np.nan), where=dX >= 40)
        pc = 0.5 * (px_e[1:] + px_e[:-1])
        ax = axes[1, j]
        ax.plot(pc, prof_any, color="0.65", lw=1.1, label="fired at all")
        ax.plot(pc, prof, color="#1f4e79", lw=1.6, label="within 5 mm")
        for lo, hi in J["bands"]:
            ax.axvspan(lo, hi, color="tab:green", alpha=0.16)
        ax.set(xlim=(pc[0], pc[-1]), ylim=(0, 1), xlabel="det4-local X [mm]",
               ylabel="efficiency", title=f"{label} X profile "
                                          "(green = June live bands)")
        ax.legend(loc="upper left", fontsize=8)
        save[label] = dict(ex=ex, ey=ey, eff=eff, den=den, num=num,
                           prof=prof, prof_any=prof_any, pc=pc, A=R["A"])

    fig.suptitle("det4 (mx17_E) at H4 — run 56 meshscan_m70V, flat mount, "
                 "Ar/CO2/iso 95/3/2, drift 700 V", y=0.98, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out + "m70V_efficiency.png", dpi=115, bbox_inches="tight")
    print("\nwrote " + args.out + "m70V_efficiency.png")

    np.savez(args.out + "m70V_efficiency.npz",
             **{f"{k}__{kk}": vv for k, v in save.items() for kk, vv in v.items()})

    # band-by-band numbers
    for label, _, _ in PLATEAUS:
        R = results[label]
        print(f"\n{label} — efficiency in the June live bands the beam covers:")
        inband = np.zeros(n_ev, bool)
        for lo, hi in J["bands"]:
            m = R["clean"] & (R["pX"] > lo) & (R["pX"] < hi)
            if m.sum() < 200:
                continue
            inband |= (R["pX"] > lo) & (R["pX"] < hi)
            print(f"  X {lo:6.1f}-{hi:6.1f}  n={m.sum():7d}  "
                  f"within 5 mm {np.mean(R['within5'][m]):6.1%}  "
                  f"fired {np.mean(R['fired'][m]):6.1%}  "
                  f"discharge {np.mean(R['disch'][m]):5.1%}")
        c = R["clean"]
        print(f"  inside live bands {np.mean(R['within5'][c & inband]):6.1%}"
              f"   between them {np.mean(R['within5'][c & ~inband]):6.1%}"
              f"   whole spot {np.mean(R['within5'][c]):6.1%}")


if __name__ == "__main__":
    main()
