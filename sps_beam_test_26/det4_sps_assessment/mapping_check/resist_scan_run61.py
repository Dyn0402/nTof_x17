#!/usr/bin/env python3
"""det4 resist-voltage scan on run_61, **one curve per measurement condition**.

Conditions come from `run61_conditions.py`; each writes `resist_scan.{csv,png}`
into its own directory on the data disk. There is deliberately no combined
curve any more.

Why: the earlier version of this script produced a single
`resist_scan_combined.png` labelled "15.465 deg" spanning 580-790 V. It was
wrong. The mount was rotated back to 25.64 deg during the 14:00-16:06 gap, so
the 580-720 V half of that curve was taken at a different angle from the
725-790 V half, and the two were plotted as one scan with the angle change
buried at the ~720 V seam. The same seam also carries a pedestal-set change.
Two things move across it, neither of them voltage.

Each condition is fit and gated entirely on its own -- alignment, drift gate,
clustering. Fitting the det4<->uRWELL transform across both at once gave
det(A) 1.05 and 6.6 mm median residual (vs 0.9-1.8 mm fit separately) and
quietly gutted the high-voltage points, because a bad global transform was
being applied to good data. Independently the fits are self-consistent
(condition 2's three sub-runs agree on det(A) to 4 decimal places), and the
8 % scale difference between the conditions is just the rotation:
cos(15.465)/cos(25.64) = 1.069 against an observed 1.081.

Extending this: decode FEU3 for the new sub-run, pair it against uRWELL
(`extract_pair.py`), drop the npz in `paired_npz/`, and add its sub-run start
and plateau windows to the right condition in `run61_conditions.py`. If the
mount or the gas moved, it is a **new** condition, not more points on an old
one.
"""
import csv
import datetime as dt
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment/mapping_check")
from det4_sps_map import POSITION_MM, VIEW              # noqa: E402
from run61_conditions import CONDITIONS, Z, outdir, sources, parse_t  # noqa: E402


def process_session(sources, tag):
    """Load + concat one session's sources, fit ITS OWN alignment, return a
    dict of per-voltage rows (voltages merged across a session's own subrun
    boundaries where a point straddles one)."""
    keys_ev = ("fx_p", "fx_n", "fy_p", "fy_n", "bx_p", "bx_n", "by_p", "by_n", "ev_ts")
    acc = {k: [] for k in keys_ev}
    acc_hit = {k: [] for k in ("h_ev", "h_ch", "h_amp", "h_sig", "h_time")}
    wall, point_list = [], []

    off = 0
    for path, t0, points in sources:
        D = np.load(path)
        n_ev = len(D["fx_p"])
        for k in keys_ev:
            acc[k].append(D[k])
        for k in acc_hit:
            v = D[k]
            if k == "h_ev":
                v = v + off
            acc_hit[k].append(v)
        wall.append(np.array([t0 + dt.timedelta(seconds=float(s) / 1e9) for s in D["ev_ts"]]))
        point_list += [(v, parse_t(t0s, t0), parse_t(t1s, t0)) for v, t0s, t1s in points]
        print(f"  {path}: {n_ev} events")
        off += n_ev

    D = {k: np.concatenate(v) for k, v in acc.items()}
    H = {k: np.concatenate(v) for k, v in acc_hit.items()}
    wall = np.concatenate(wall)
    n_ev = len(D["fx_p"])
    print(f"  [{tag}] total: {n_ev} events, {len(H['h_ev'])} det4 hits, {len(point_list)} resist points")

    clean = np.ones(n_ev, bool)
    for k in ("fx", "fy", "bx", "by"):
        clean &= (D[k + "_n"] == 1) & np.isfinite(D[k + "_p"])
    tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * Z / 1370.0
    ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * Z / 1370.0
    clean &= np.isfinite(tx + ty)
    print(f"  [{tag}] clean single-cluster uRWELL tracks: {clean.sum()}")

    h_ev, h_ch, h_amp, h_t = H["h_ev"], H["h_ch"], H["h_amp"], H["h_time"]
    hb, eb = np.histogram(h_t, bins=np.arange(-1000, 6000, 100.0))
    cb = 0.5 * (eb[1:] + eb[:-1])
    pk = np.argmax(hb)
    thr = 0.25 * hb[pk]
    lo_i = pk - np.argmax(hb[pk::-1] < thr) + 1
    hi_i = pk + np.argmax(hb[pk:] < thr)
    g0, g1 = cb[lo_i] - 50, cb[hi_i] + 50
    print(f"  [{tag}] det4 drift gate: {g0:.0f} - {g1:.0f} ns (auto)")
    GATE = (h_t > g0) & (h_t < g1)
    sel = GATE & clean[h_ev]

    lead, ncl, nstrip = {}, {}, {}
    for v in ("x", "y"):
        k = sel & (VIEW[h_ch] == v)
        ev, pos, amp = h_ev[k], POSITION_MM[h_ch[k]], h_amp[k]
        o = np.lexsort((pos, ev))
        ev, pos, amp = ev[o], pos[o], amp[o]
        new = np.empty(len(ev), bool); new[0] = True
        new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > 3.0)
        cid = np.cumsum(new) - 1
        nc = cid[-1] + 1 if len(cid) else 0
        cq = np.bincount(cid, weights=amp, minlength=nc)
        cn = np.bincount(cid, minlength=nc)
        cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
        cev = np.zeros(nc, np.int64); cev[cid] = ev
        L = np.full(n_ev, np.nan); N = np.zeros(n_ev, np.int32); S = np.zeros(n_ev, np.int32)
        np.add.at(N, cev, 1)
        o2 = np.argsort(cq, kind="stable")
        L[cev[o2]] = cp[o2]; S[cev[o2]] = cn[o2]
        lead[v], ncl[v], nstrip[v] = L, N, S

    fired = np.zeros(n_ev, bool)
    fired[h_ev[GATE & clean[h_ev]]] = True
    reco = np.isfinite(lead["x"] + lead["y"])
    disch = (ncl["x"] + ncl["y"] >= 6) | (nstrip["x"] > 40) | (nstrip["y"] > 40)

    fit = clean & reco & ~disch
    U = np.column_stack([tx[fit], ty[fit], np.ones(fit.sum())])
    V = np.column_stack([lead["x"][fit], lead["y"][fit]])
    A, *_ = np.linalg.lstsq(U, V, rcond=None)
    for _ in range(4):
        r = np.hypot(*(V - U @ A).T)
        g = r < np.percentile(r, 80)
        A, *_ = np.linalg.lstsq(U[g], V[g], rcond=None)
    M = A[:2].T
    print(f"  [{tag}] alignment: rot {np.degrees(np.arctan2(M[1,0], M[0,0])):+.2f} deg, "
          f"det(A) {np.linalg.det(M):+.4f}, median|res| {np.median(np.hypot(*(V - U @ A).T)):.2f} mm")

    pred = np.column_stack([tx, ty, np.ones(n_ev)]) @ A
    pX, pY = pred[:, 0], pred[:, 1]
    resid = np.hypot(lead["x"] - pX, lead["y"] - pY)
    within5 = clean & reco & (resid < 5.0)

    by_v = {}
    for v, t0, t1 in point_list:
        by_v.setdefault(v, []).append((t0, t1))

    session_rows = []
    for v in sorted(by_v, reverse=True):
        m = clean & np.zeros(n_ev, bool)
        for t0, t1 in by_v[v]:
            m |= clean & (wall >= t0) & (wall < t1)
        n = m.sum()
        if n == 0:
            continue
        f_any, f_reco, f_w5, f_disch = np.mean(fired[m]), np.mean(reco[m]), np.mean(within5[m]), np.mean(disch[m])
        se = np.sqrt(f_w5 * (1 - f_w5) / n)
        session_rows.append(dict(resist_v=v, n_clean=int(n), fired=f_any, reco=f_reco,
                                  within5mm=f_w5, within5mm_err=se, discharge=f_disch))
    return session_rows



for cond in CONDITIONS:
    print("\n=== %s ===" % cond["key"])
    print("    mount %.3f deg | gas %s" % (cond["mount_deg"], cond["gas"]))
    rows = process_session(sources(cond), cond["key"])
    if not rows:
        print("    no points -- skipped")
        continue
    rows.sort(key=lambda r: -r["resist_v"])

    print(f"\n{'resist V':>9} {'n clean':>9} {'fired':>8} {'reco':>8} "
          f"{'within5':>12} {'discharge':>10}")
    for r in rows:
        print(f"{r['resist_v']:9.1f} {r['n_clean']:9d} {r['fired']:8.1%} "
              f"{r['reco']:8.1%} {r['within5mm']:8.1%}+-{r['within5mm_err']:5.1%} "
              f"{r['discharge']:9.1%}")

    out = outdir(cond)
    with open(out + "resist_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", out + "resist_scan.csv")

    rv = np.array([r["resist_v"] for r in rows])
    w5 = np.array([r["within5mm"] for r in rows])
    w5e = np.array([r["within5mm_err"] for r in rows])
    fa = np.array([r["fired"] for r in rows])
    dc = np.array([r["discharge"] for r in rows])
    nn = np.array([r["n_clean"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    ax1.errorbar(rv, w5, yerr=w5e, marker="o", color="#2a78d6",
                 label="within 5 mm of uRWELL track")
    ax1.plot(rv, fa, marker="s", ms=4, color="0.6", ls="--", label="fired at all")
    for x, y, n in zip(rv, w5, nn):
        ax1.annotate(f"n={n}", (x, y), textcoords="offset points",
                     xytext=(0, 6), fontsize=7, ha="center")
    ax1.set_ylabel("det4 efficiency")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title("det4 (mx17_E) resist-voltage scan\n%s" % cond["label"])
    ax2.plot(rv, dc, marker="o", color="#e34948")
    ax2.set_ylabel("discharge-flagged")
    ax2.set_xlabel("resist voltage [V]")
    fig.tight_layout()
    fig.savefig(out + "resist_scan.png", dpi=130)
    plt.close(fig)
    print("wrote", out + "resist_scan.png")
