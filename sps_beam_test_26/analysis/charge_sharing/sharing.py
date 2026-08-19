#!/usr/bin/env python3
"""det4's lateral charge sharing at H4, measured separately in X and Y.

Companion to `../spatial_resolution/`: same run, same 44 k head-on beam tracks,
same reference.  The question here is not *where* the track was but *how far the
charge spread across the strips*, in each view.

Why this run.  At normal incidence every drift slice lands at the same transverse
position, so the lateral profile is diffusion + resistive spread and nothing else.
And the uRWELL track gives an **external, sub-strip** impact point (0.24 mm against
a 0.78 mm pitch), so the profile is built against the true track rather than against
det4's own centroid, which would be circular.

Three things have to be controlled or the answer is an artefact:

1. **Dead neighbours.**  det4 only amplifies in ~12 bands.  In the X view a track
   near a band edge has DEAD neighbours, which truncates the kernel and fakes a
   narrow one.  Every track is required to sit `MARGIN_STRIPS` clear of any dead
   strip.  In Y the live run is 95 mm wide and this is free; in X it is the
   whole measurement.
2. **The zero-suppression threshold.**  Strip multiplicity is not a property of the
   detector alone -- it is a property of the detector *at a threshold*.  X clusters
   carry ~1.4x the charge of Y clusters here, so comparing raw multiplicities
   compares thresholds, not sharing.  The headline comparison is therefore made in
   a **matched cluster-charge window**.
3. **The chamber's own X tilt.**  tan(theta_X) = -0.015 walks the drift column
   0.45 mm across the 30 mm gap (RERUN_2026-08-04_NEW_MACHINE.md), which broadens
   and skews X but not Y.  Subtracted from the width, and its skew is reported.

Run:  ../../../.venv/bin/python sharing.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "det4_sps_assessment"))
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM        # noqa: E402

DATA = Path("/media/dylan/data/x17/sps_run53_det4_check/"
            "flat_ArCO2iso_95-3-2__run53-56/mapping_check/det4_run_53_v2.npz")
Z_DET4, Z_BACK = 1120.0, 1370.0
GATE = (600.0, 1850.0)          # det4 drift window, run 53, off the hit-time spectrum
AMP_MIN = 60.0                  # above the 5-sigma ZS floor
SAT = 3000.0                    # repeated-constant pathology above this
BAD_CH = (372, 510)             # the two oscillating channels (RAW_RUN71_REANALYSIS)
MARGIN_STRIPS = 3.5
POINTING_MM = 0.242             # uRWELL pointing at det4, from ../spatial_resolution
TILT_TAN = 0.015                # det4's X tilt; drift gap 30 mm
GAP_MM = 30.0
QLO, QHI = 400.0, 1600.0        # matched cluster-charge window [ADC]
QBINS = [0, 300, 600, 1000, 1600, 2600, 1e9]


def robust_lstsq(M, target, niter=6):
    keep = np.ones(len(target), bool)
    for _ in range(niter):
        coef, *_ = np.linalg.lstsq(M[keep], target[keep], rcond=None)
        r = target - M @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = np.abs(r - np.median(r)) < 3 * s
    return coef


def live_mask(ch, amp, t, view):
    """Per-strip liveness from occupancy, in the view's own position order."""
    g = (t > GATE[0]) & (t < GATE[1]) & (amp > AMP_MIN)
    cnt = np.bincount(ch[g], minlength=512).astype(float)
    idx = np.flatnonzero(VIEW == view)
    pos = POSITION_MM[idx]
    o = np.argsort(pos)
    idx, pos = idx[o], pos[o]
    live = cnt[idx] > 0.15 * np.percentile(cnt[idx], 95)
    live &= ~np.isin(idx, BAD_CH)
    return idx, pos, live


def interior_ok(pred, pos, live, margin):
    """True where every strip within `margin` pitches of `pred` is live."""
    ok = np.zeros(len(pred), bool)
    lo = np.searchsorted(pos, pred - margin * PITCH_MM)
    hi = np.searchsorted(pos, pred + margin * PITCH_MM, side="right")
    dead_cum = np.concatenate([[0], np.cumsum(~live)])
    inside = (lo > 0) & (hi < len(pos)) & (hi > lo)
    ok[inside] = (dead_cum[hi[inside]] - dead_cum[lo[inside]]) == 0
    return ok


def bands(pos, live, min_strips=3):
    out, s = [], None
    for i, L in enumerate(live):
        if L and s is None:
            s = i
        if (not L or i == len(live) - 1) and s is not None:
            e = i if not L else i + 1
            if e - s >= min_strips:
                out.append((float(pos[s]), float(pos[e - 1]), int(e - s)))
            s = None
    return out


def clusters(e, p, a, nev):
    lead = np.full(nev, np.nan)
    ncl = np.zeros(nev, np.int16)
    o = np.lexsort((p, e))
    e, p, a = e[o], p[o], a[o]
    new = np.empty(len(e), bool)
    new[0] = True
    new[1:] = (e[1:] != e[:-1]) | (np.diff(p) > 3.0)
    cid = np.cumsum(new) - 1
    nc = cid[-1] + 1
    cq = np.bincount(cid, weights=a, minlength=nc)
    cp = np.bincount(cid, weights=p * a, minlength=nc) / np.maximum(cq, 1e-9)
    cev = np.zeros(nc, np.int64)
    cev[cid] = e
    np.add.at(ncl, cev, 1)
    b = np.argsort(cq, kind="stable")
    lead[cev[b]] = cp[b]
    return lead, ncl


def pooled_width(d, fr):
    w = fr.sum()
    mu = float((fr * d).sum() / w)
    return mu, float(np.sqrt((fr * (d - mu) ** 2).sum() / w))


def main():
    D = np.load(DATA)
    nev = len(D["fx_p"])
    ev, ch, amp, t = D["h_ev"], D["h_ch"], D["h_amp"], D["h_time"]
    liv = {"x": live_mask(ch, amp, t, "x"), "y": live_mask(ch, amp, t, "y")}
    out = {"meta": dict(pitch_mm=PITCH_MM, pointing_mm=POINTING_MM,
                        margin_strips=MARGIN_STRIPS, q_window=[QLO, QHI],
                        gate_ns=list(GATE), amp_min=AMP_MIN,
                        tilt_walk_mm=TILT_TAN * GAP_MM)}
    for v in "xy":
        out[f"bands_{v}"] = bands(liv[v][1], liv[v][2])
        print(f"{v.upper()} view: {liv[v][2].sum()}/256 strips live, "
              f"{len(out[f'bands_{v}'])} bands >= 3 strips")

    sel = (t > GATE[0]) & (t < GATE[1]) & (amp > AMP_MIN)
    ev, ch, amp = ev[sel], ch[sel], amp[sel]
    pos, is_x = POSITION_MM[ch], VIEW[ch] == "x"
    dx, nx = clusters(ev[is_x], pos[is_x], amp[is_x], nev)
    dy, ny = clusters(ev[~is_x], pos[~is_x], amp[~is_x], nev)
    clean = ((D["fx_n"] == 1) & (D["fy_n"] == 1) & (D["bx_n"] == 1) & (D["by_n"] == 1)
             & (nx == 1) & (ny == 1) & np.isfinite(dx) & np.isfinite(dy))
    tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * Z_DET4 / Z_BACK
    ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * Z_DET4 / Z_BACK
    Mall = np.column_stack([tx, ty, np.ones(nev)])
    pred = {}
    for v, dd in (("x", dx), ("y", dy)):
        c = robust_lstsq(Mall[clean], dd[clean])
        pred[v] = np.full(nev, np.nan)
        pred[v][clean] = Mall[clean] @ c
    out["n_clean"] = int(clean.sum())
    print(f"alignment fitted on {clean.sum()} clean events\n")

    for v in "xy":
        other = "y" if v == "x" else "x"
        idx, spos, live = liv[v]
        oidx, opos, olive = liv[other]
        good = clean & np.isfinite(pred[v]) & np.isfinite(pred[other])
        good[good] &= interior_ok(pred[v][good], spos, live, MARGIN_STRIPS)
        good[good] &= interior_ok(pred[other][good], opos, olive, 0.5)
        gi = np.flatnonzero(good)

        m = np.isin(ev, gi) & (VIEW[ch] == v)
        e_s, p_s, a_s = ev[m], pos[m], amp[m]
        satev = np.unique(e_s[a_s > SAT])
        k = ~np.isin(e_s, satev)
        e_s, p_s, a_s = e_s[k], p_s[k], a_s[k]
        d_all = p_s - pred[v][e_s]
        near = np.abs(d_all) < 4.0 * PITCH_MM
        e_n, d_n, a_n = e_s[near], d_all[near], a_s[near]
        o = np.argsort(e_n, kind="stable")
        e_n, d_n, a_n = e_n[o], d_n[o], a_n[o]
        uev, start = np.unique(e_n, return_index=True)
        tot = np.add.reduceat(a_n, start, dtype=float)
        nst = np.diff(np.append(start, len(a_n)))
        rep = np.repeat(np.arange(len(uev)), nst)
        fr = a_n / tot[rep]
        print(f"=== {v.upper()} view: {len(uev)} tracks clear of dead strips ===")
        print(f"  cluster charge p25/med/p75 = {np.percentile(tot,25):.0f}/"
              f"{np.median(tot):.0f}/{np.percentile(tot,75):.0f} ADC")

        # --- multiplicity: raw, vs charge, and in the matched window
        qm = (tot >= QLO) & (tot < QHI)
        mult_q = []
        for i in range(len(QBINS) - 1):
            s = (tot >= QBINS[i]) & (tot < QBINS[i + 1])
            if s.sum() > 200:
                mult_q.append(dict(lo=QBINS[i], hi=None if QBINS[i + 1] > 1e8 else QBINS[i + 1],
                                   mean=float(nst[s].mean()), n=int(s.sum())))
        hist = np.bincount(nst, minlength=9)[:9]
        out[f"mult_{v}"] = dict(
            raw=float(nst.mean()), matched=float(nst[qm].mean()),
            n=int(len(uev)), n_matched=int(qm.sum()),
            median_charge=float(np.median(tot)),
            hist=[int(x) for x in hist], vs_charge=mult_q)
        print(f"  strips/cluster: raw {nst.mean():.2f}   "
              f"matched ({QLO:.0f}-{QHI:.0f} ADC) {nst[qm].mean():.2f}")
        print("  multiplicity: " + " ".join(f"{i}:{100*x/len(uev):.1f}%"
                                            for i, x in enumerate(hist) if x))

        # --- kernel profile (matched window), charge fraction vs signed offset.
        # This MUST include the strips that did not fire.  A strip only enters the
        # hit list if it passed zero-suppression, so averaging over fired strips
        # only is survivorship bias: it reports "given that this strip fired, how
        # much did it carry", which cannot fall off with distance and does not.
        # Build the full 8-strip window around every track and fill absent strips
        # with zero.
        sm = qm[rep]
        # each hit -> its index in the view's position-ordered strip list
        hit_si = np.clip(np.searchsorted(spos, d_n + pred[v][e_n] - 1e-6),
                         0, len(spos) - 1)
        key_hit = e_n.astype(np.int64) * 256 + hit_si
        srt = np.argsort(key_hit)
        key_hit_s, fr_s = key_hit[srt], fr[srt]

        ev_m = uev[qm]
        i0 = np.searchsorted(spos, pred[v][ev_m])
        jj = np.arange(-4, 4)
        J = i0[:, None] + jj[None, :]
        DD = spos.take(np.clip(J, 0, len(spos) - 1)) - pred[v][ev_m][:, None]
        valid = (J >= 0) & (J < len(spos))
        KEY = ev_m[:, None].astype(np.int64) * 256 + np.clip(J, 0, 255)
        p = np.searchsorted(key_hit_s, KEY)
        p_c = np.clip(p, 0, len(key_hit_s) - 1)
        found = (key_hit_s[p_c] == KEY) & valid
        FR = np.where(found, fr_s[p_c], 0.0)
        DDf, FRf = DD[valid].ravel(), FR[valid].ravel()

        edges = np.arange(-3.25, 3.30, 0.25) * PITCH_MM
        ctr = 0.5 * (edges[1:] + edges[:-1])
        prof, perr, cnts = [], [], []
        for i in range(len(ctr)):
            s = (DDf >= edges[i]) & (DDf < edges[i + 1])
            cnts.append(int(s.sum()))
            if s.sum() > 40:
                prof.append(float(FRf[s].mean()))
                perr.append(float(FRf[s].std() / np.sqrt(s.sum())))
            else:
                prof.append(None)
                perr.append(None)
        out[f"kernel_{v}"] = dict(centre_mm=[float(x) for x in ctr],
                                  frac=prof, err=perr, n=cnts,
                                  note="mean charge fraction, absent strips counted as 0")

        # --- charge budget by strip rank, matched
        kk = np.round(d_n / PITCH_MM).astype(int)
        budget = {}
        for j in range(4):
            s = sm & (np.abs(kk) <= j)
            budget[f"within_{j}"] = float(
                np.bincount(rep[s], weights=fr[s], minlength=len(uev))[qm].mean())
        out[f"budget_{v}"] = budget
        print("  charge within +-k strips: "
              + "  ".join(f"{j}:{budget[f'within_{j}']:.3f}" for j in range(4)))

        # --- pooled width, raw and matched, with pointing (and X tilt) removed
        tilt = TILT_TAN * GAP_MM / np.sqrt(12) if v == "x" else 0.0
        wid = {}
        for tag, mask in (("raw", np.ones(len(d_n), bool)), ("matched", sm)):
            mu, w = pooled_width(d_n[mask], fr[mask])
            wid[tag] = dict(mean_mm=mu, rms_mm=w,
                            rms_deconv_mm=float(np.sqrt(max(
                                w ** 2 - POINTING_MM ** 2 - tilt ** 2, 0))))
        wid["tilt_term_mm"] = float(tilt)
        out[f"width_{v}"] = wid
        print(f"  pooled lateral rms (matched): {wid['matched']['rms_mm']*1e3:.0f} um "
              f"-> {wid['matched']['rms_deconv_mm']*1e3:.0f} um deconvolved"
              + (f" (tilt term {tilt*1e3:.0f} um)" if tilt else ""))

        # --- controls
        L = fr[sm & (d_n > -1.7 * PITCH_MM) & (d_n < -0.5 * PITCH_MM)].sum()
        Rg = fr[sm & (d_n > 0.5 * PITCH_MM) & (d_n < 1.7 * PITCH_MM)].sum()
        inband = []
        for lo, hi, _ in out[f"bands_{v}"]:
            if hi - lo < 6:
                continue
            s = (pred[v][uev] >= lo) & (pred[v][uev] <= hi)
            if s.sum() < 300:
                continue
            e2 = np.linspace(lo, hi, 6)
            pts = []
            for i in range(5):
                q = s & (pred[v][uev] >= e2[i]) & (pred[v][uev] < e2[i + 1])
                if q.sum() > 100:
                    pts.append(dict(pos=float(0.5 * (e2[i] + e2[i + 1])),
                                    mult=float(nst[q].mean()), n=int(q.sum())))
            inband.append(dict(band=[lo, hi], points=pts))
        out[f"controls_{v}"] = dict(asym_minus_over_plus=float(L / Rg), inband=inband)
        print(f"  control - kernel asymmetry (-1/+1 strip): {L/Rg:.3f}\n")

    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {HERE/'results.json'}")


if __name__ == "__main__":
    main()
