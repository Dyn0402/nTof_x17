#!/usr/bin/env python3
"""Alignment, efficiency and waveform extraction for a flat-mount det4 dataset.

Generalises `align_eff_m70V.py` + `extract_waveforms_m70V.py` over
`datasets.py`.  Emits a `wf_<dataset>.npz` with the same field names those
scripts used, so `kernel_fit_m70V.py --wf ... --plateau ...` and
`tilt_m70V.py` run on it unchanged.

Two things it does that the run_56 version did not:

  * **fits z_det4** instead of trusting it.  `run_config.json` says 1155 mm and
    the run_56 work used 1120 mm; the detector was re-hung at the 08-02 ~21:00
    access, so neither is safe.  z is scanned for the minimum median track
    residual.
  * **one global alignment** across plateaus.  Within a dataset the mount does
    not move, so fitting it once is both correct and better determined; the
    per-plateau residual is then a check that it really did not move.

  python flat_align_eff.py run63_operating [--max-events 90000]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import uproot

import datasets

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM      # noqa: E402

Z_BACK = 1370.0
ZS_BASELINE = 256.0
BAND_MARGIN_MM = 2.0
STRIPES = ("/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
           "det4_sps_assessment/stripes_g_det4.npz")


def det4_clusters(h_ev, h_ch, h_amp, sel, n_ev):
    lead, ncl, nst = {}, {}, {}
    for v in ("x", "y"):
        k = sel & (VIEW[h_ch] == v)
        ev, pos, amp = h_ev[k], POSITION_MM[h_ch[k]], h_amp[k]
        L = np.full(n_ev, np.nan)
        N = np.zeros(n_ev, np.int32)
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
            o2 = np.argsort(cq, kind="stable")
            L[cev[o2]] = cp[o2]
            S[cev[o2]] = cn[o2]
        lead[v], ncl[v], nst[v] = L, N, S
    return lead, ncl, nst


def fit_affine(tx, ty, dx, dy, keep):
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
    ap.add_argument("dataset", choices=sorted(datasets.DATASETS))
    ap.add_argument("--gate", default="600,3600")
    ap.add_argument("--grid", type=float, default=3.0)
    ap.add_argument("--max-events", type=int, default=90000)
    ap.add_argument("--align-from", default="",
                    help="wf_<dataset>.npz whose alignment (A, z) to borrow "
                         "instead of fitting -- same-mount-epoch datasets only")
    args = ap.parse_args()
    D = datasets.get(args.dataset)
    stage = D["stage"]

    P = np.load(stage + f"pair_{args.dataset}.npz")
    n_ev = len(P["ev_id"])
    plateau = P["plateau"]
    print(f"{args.dataset}: {n_ev} events, {D['mount']}, {D['gas']}")

    clean = np.ones(n_ev, bool)
    for k in ("fx", "fy", "bx", "by"):
        clean &= (P[k + "_n"] == 1) & np.isfinite(P[k + "_p"])
    fxp, fyp, bxp, byp = (P["fx_p"], P["fy_p"], P["bx_p"], P["by_p"])

    g0, g1 = (float(v) for v in args.gate.split(","))
    h_ev, h_ch, h_amp = P["h_ev"], P["h_ch"], P["h_amp"]
    GATE = (P["h_time"] > g0) & (P["h_time"] < g1)
    sel = GATE & clean[h_ev]
    lead, ncl, nst = det4_clusters(h_ev, h_ch, h_amp, sel, n_ev)
    reco = np.isfinite(lead["x"] + lead["y"])
    disch = (ncl["x"] + ncl["y"] >= 6) | (nst["x"] > 40) | (nst["y"] > 40)

    # ------------------------------------------------------------- fit z
    if args.align_from:
        # borrow the alignment of a better-conditioned dataset in the SAME
        # mount epoch (e.g. run61_op25's 454k tracks for run63_rot25's thin
        # low-field ladder -- no access separates them, TAX record)
        Zsrc = np.load(args.align_from)
        A = np.asarray(Zsrc["A"], float)
        z = float(np.asarray(Zsrc["z_det4"]).ravel()[0])
        print(f"\nalignment borrowed from {args.align_from}: z = {z:.0f} mm")
    else:
        print("\nscanning z_det4 for the minimum track residual:")
        best = None
        for z in np.arange(1000, 1301, 10.0):
            f = z / Z_BACK
            tx = fxp + (bxp - fxp) * f
            ty = fyp + (byp - fyp) * f
            keep = clean & reco & ~disch & np.isfinite(tx + ty)
            A = fit_affine(tx, ty, lead["x"], lead["y"], keep)
            pred = np.column_stack([tx, ty, np.ones(n_ev)]) @ A
            r = np.nanmedian(np.hypot(lead["x"][keep] - pred[keep, 0],
                                      lead["y"][keep] - pred[keep, 1]))
            if best is None or r < best[1]:
                best = (z, r, A)
            if int(z) % 50 == 0:
                print(f"   z={z:6.0f} mm  median |res| {r:.3f} mm")
        z, res_med, A = best
        print(f"  best z = {z:.0f} mm, median |residual| {res_med:.3f} mm "
              f"(config says {D['z_det4']:.0f})")

    f = z / Z_BACK
    tx = fxp + (bxp - fxp) * f
    ty = fyp + (byp - fyp) * f
    clean &= np.isfinite(tx + ty)
    pred = np.column_stack([tx, ty, np.ones(n_ev)]) @ A
    pX, pY = pred[:, 0], pred[:, 1]
    resid = np.hypot(lead["x"] - pX, lead["y"] - pY)
    M = A[:2].T
    print(f"  alignment: roll {np.degrees(np.arctan2(M[1,0], M[0,0])):+.2f} deg, "
          f"det(A) {np.linalg.det(M):+.4f}, "
          f"row scales {np.linalg.norm(M[0]):.4f} / {np.linalg.norm(M[1]):.4f}")

    fired = np.zeros(n_ev, bool)
    fired[h_ev[GATE & clean[h_ev]]] = True
    within5 = clean & reco & (resid < 5.0)

    J = np.load(STRIPES)
    inband = np.zeros(n_ev, bool)
    for lo, hi in J["bands"]:
        inband |= (pX > lo + BAND_MARGIN_MM) & (pX < hi - BAND_MARGIN_MM)

    print(f"\n{'plateau':>8} {'drift':>7} {'resist':>7} {'clean':>8} "
          f"{'|res|':>7} {'fired':>7} {'within5':>8} {'in-band':>8}")
    for lab, lo, hi, dr, re in D["plateaus"]:
        m = clean & (plateau == lab)
        if m.sum() < 100:
            continue
        mb = m & inband
        print(f"{lab:>8} {dr:7.1f} {re:7.1f} {m.sum():8d} "
              f"{np.nanmedian(resid[m & reco]):7.2f} {np.mean(fired[m]):7.1%} "
              f"{np.mean(within5[m]):8.1%} {np.mean(within5[mb]):8.1%}")

    # ------------------------------------------------------------- eff map
    G = args.grid
    cmap = LinearSegmentedColormap.from_list(
        "eff", ["#2b1b3d", "#3b5b92", "#3f9e9e", "#8ecf6a", "#f2e661"])
    cmap.set_bad("#e8e8e8")
    labs = [l for l, *_ in D["plateaus"] if (clean & (plateau == l)).sum() > 500]
    fig, axes = plt.subplots(2, len(labs), figsize=(4.6 * len(labs), 9),
                             squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.4, 1]))
    for j, lab in enumerate(labs):
        m = clean & (plateau == lab)
        w = within5 & (plateau == lab)
        lo_x, hi_x = np.percentile(pX[m], [0.2, 99.8])
        lo_y, hi_y = np.percentile(pY[m], [0.2, 99.8])
        ex = np.arange(np.floor(lo_x / G) * G, np.ceil(hi_x / G) * G + G, G)
        ey = np.arange(np.floor(lo_y / G) * G, np.ceil(hi_y / G) * G + G, G)
        den, _, _ = np.histogram2d(pX[m], pY[m], bins=[ex, ey])
        num, _, _ = np.histogram2d(pX[w], pY[w], bins=[ex, ey])
        eff = np.divide(num, den, out=np.full_like(num, np.nan), where=den >= 25)
        ax = axes[0, j]
        im = ax.pcolormesh(ex, ey, eff.T, cmap=cmap, vmin=0, vmax=1)
        ax.set(title=f"{lab} ({m.sum():,} tracks)", xlabel="det4 X [mm]",
               ylabel="det4 Y [mm]" if j == 0 else "")
        ax.set_aspect("equal")
        if j == len(labs) - 1:
            fig.colorbar(im, ax=ax, fraction=0.04, label="efficiency")
        px_e = np.arange(np.floor(lo_x), np.ceil(hi_x) + 1, 1.0)
        dX, _ = np.histogram(pX[m], bins=px_e)
        nX, _ = np.histogram(pX[w], bins=px_e)
        prof = np.divide(nX, dX, out=np.full(len(dX), np.nan), where=dX >= 40)
        pc = 0.5 * (px_e[1:] + px_e[:-1])
        ax = axes[1, j]
        ax.plot(pc, prof, color="#1f4e79", lw=1.4)
        for blo, bhi in J["bands"]:
            ax.axvspan(blo, bhi, color="tab:green", alpha=0.16)
        ax.set(ylim=(0, 1), xlim=(pc[0], pc[-1]), xlabel="det4 X [mm]",
               ylabel="efficiency" if j == 0 else "")
    fig.suptitle(f"det4 {args.dataset} — {D['mount']}, {D['gas']}, "
                 f"z={z:.0f} mm", y=0.99)
    fig.tight_layout()
    fig.savefig(stage + f"eff_{args.dataset}.png", dpi=110, bbox_inches="tight")
    print(f"\nwrote {stage}eff_{args.dataset}.png")

    # ----------------------------------------------------- waveform extract
    good = clean & reco & ~disch & inband & (resid < 5.0) & (plateau != "")
    idx = np.flatnonzero(good)
    print(f"\nwaveform selection: {good.sum()} events")
    per = {}
    for lab, *_ in D["plateaus"]:
        k = idx[plateau[idx] == lab]
        cap = args.max_events // max(len(labs), 1)
        if len(k) > cap:
            k = np.random.default_rng(0).choice(k, cap, replace=False)
        per[lab] = np.sort(k)
        print(f"  {lab:>8} {len(per[lab]):7d} events")
    idx = np.sort(np.concatenate([v for v in per.values() if len(v)]))

    # RAW runs sit on raw per-channel baselines with the common mode still in;
    # correct them the way extract_det4_only.py --cm masked does (pedestal
    # means + per-block per-sample median with the signal strips and the
    # oscillating channels excluded).  This closes the RAW_RUN71_PHYSICS §3b
    # open item: the uRWELL-referenced waveform selection now works on RAW.
    raw = bool(D.get("raw"))
    NCH, NSMP = 512, int(D["n_samples"])
    CM_MASK_HALF = 10
    CM_BAD_CH = (510, 372)
    SIDX_ALL = np.round(POSITION_MM / PITCH_MM).astype(int)
    ped_mean = np.zeros(NCH, np.float32)
    if raw:
        # per-channel raw baselines from any hits file's 'pedestals' tree
        # (same source extract_det4_only.py uses)
        import os
        pf = None
        for sub, _s, _t, idxs in D["subruns"]:
            dd = os.path.join(stage, sub) if os.path.isdir(
                os.path.join(stage, sub)) else stage
            for i in idxs:
                cand = os.path.join(dd, f"hits_{sub}_{i}_03.root")
                if os.path.exists(cand):
                    pf = cand
                    break
            if pf:
                break
        Pp = uproot.open(pf + ":pedestals").arrays(["channel", "mean"],
                                                   library="np")
        ped_mean[Pp["channel"].astype(int)] = Pp["mean"]
        print(f"RAW baseline correction on: pedestal medians "
              f"{np.median(ped_mean):.0f} ADC + signal-masked block CM")

    def raw_correct(iev, c, sm, am):
        """pedestal + masked-CM correct one RAW event's sparse samples."""
        a = am - ped_mean[c]
        cmok = ~np.isin(c, CM_BAD_CH)
        for v in ("x", "y"):
            lv = lead[v][iev]
            if np.isfinite(lv):
                s0 = int(round(lv / PITCH_MM))
                sig = (VIEW[c] == v) & (np.abs(SIDX_ALL[c] - s0) <= CM_MASK_HALF)
                cmok &= ~sig
        cm = np.zeros_like(a)
        blk = c.astype(int) // 64
        for b in np.unique(blk):
            ib = blk == b
            for s in np.unique(sm[ib]):
                iss = ib & (sm == s)
                src = iss & cmok
                if src.sum() >= 8:
                    cm[iss] = np.median(a[src])
        return a - cm

    recs = {k: [] for k in ("ev", "ch", "samp", "amp")}
    sub_arr, fg_arr, evid = P["subrun"], P["fgroup"], P["ev_id"]
    for sub, stem, _t0, idxs in D["subruns"]:
        import os
        d = os.path.join(stage, sub) if os.path.isdir(os.path.join(stage, sub)) \
            else stage
        for i in idxs:
            m = idx[(sub_arr[idx] == sub) & (fg_arr[idx] == i)]
            if not len(m):
                continue
            fn = os.path.join(d, f"dec_{sub}_{i}_03.root")
            if not os.path.exists(fn):
                fn = os.path.join(d, f"dec_{i}_03.root")
            T = uproot.open(fn)["nt"]
            nt_ev = T["eventId"].array(library="np").astype(np.int64)
            order = np.argsort(nt_ev)
            jj = np.clip(np.searchsorted(nt_ev, evid[m], sorter=order),
                         0, len(nt_ev) - 1)
            entry = order[jj]
            ok = nt_ev[entry] == evid[m]
            entry, mm = entry[ok], m[ok]
            want = np.zeros(T.num_entries, bool)
            want[entry] = True
            pos_of = np.full(T.num_entries, -1, np.int64)
            pos_of[entry] = mm
            print(f"  {sub}/{i}: {len(entry)} events")
            step = 50000
            for s in range(0, T.num_entries, step):
                e = min(s + step, T.num_entries)
                if not want[s:e].any():
                    continue
                a = T.arrays(["sample", "channel", "amplitude"],
                             entry_start=s, entry_stop=e, library="np")
                for q in np.flatnonzero(want[s:e]):
                    c = a["channel"][q]
                    if len(c) == 0:
                        continue
                    iev = pos_of[s + q]
                    sm_ = a["sample"][q].astype(np.int16)
                    am_ = a["amplitude"][q].astype(np.float32)
                    if raw:
                        am_ = raw_correct(iev, c.astype(int), sm_, am_)
                    recs["ev"].append(np.full(len(c), iev, np.int64))
                    recs["ch"].append(c.astype(np.int16))
                    recs["samp"].append(sm_)
                    recs["amp"].append(am_)

    out = {k: np.concatenate(v) for k, v in recs.items()}
    if not raw:
        out["amp"] = out["amp"] - ZS_BASELINE
    # (RAW waveforms were pedestal- and masked-CM-corrected per event above;
    # ZS waveforms sit on the flat on-FEU 256 baseline)
    out["ev_pX"], out["ev_pY"], out["ev_resid"] = pX, pY, resid
    out["ev_plateau"] = plateau
    out["ev_t_wall"] = P["ev_t_wall"]
    out["A"] = A
    out["z_det4"] = np.array([z])
    np.savez_compressed(stage + f"wf_{args.dataset}.npz", **out)
    print(f"wrote {stage}wf_{args.dataset}.npz: {len(out['ev'])} sample "
          f"records over {len(np.unique(out['ev']))} events")


if __name__ == "__main__":
    main()
