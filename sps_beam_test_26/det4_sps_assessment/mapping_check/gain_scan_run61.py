#!/usr/bin/env python3
"""det4 ADC amplitude and saturation vs resist voltage, **one curve per
measurement condition** -- the same conditions and the same plateau windows as
`resist_scan_run61.py`, both imported from `run61_conditions.py`.

Amplitude and saturation are computed on in-time hits (drift gate) belonging to
clean uRWELL-tagged, non-discharge events -- i.e. the same population the
efficiency curve calls "good", so gain and efficiency are comparable
point-for-point *within a condition*. Discharge events are excluded on purpose:
a discharge's amplitude is breakdown, not single-track gain, and would bias the
metric high.

**Never compare absolute ADC across conditions.** The two run_61 conditions
were decoded against different pedestal sets (2026-08-01 21:12 vs 2026-08-02
15:04), and the old combined plot showed a step in mean/median ADC exactly at
the boundary while the saturation fraction stepped the *other* way -- the tell
that it is a calibration offset, not physics. That, plus the mount rotation at
the same boundary, is why there is no combined curve any more. Trust the trend
within a condition.

Unlike the efficiency scan this does not need the det4<->uRWELL alignment fit
(amplitude, saturation and discharge clustering are all det4-local), but it is
still split by condition, because the pedestal set is part of the condition.
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
from det4_sps_map import POSITION_MM, VIEW                       # noqa: E402
from run61_conditions import (CONDITIONS, PEDESTAL_SET, outdir,  # noqa: E402
                              sources, parse_t)


for cond in CONDITIONS:
    print("\n=== %s ===" % cond["key"])
    print("    mount %.3f deg | gas %s | pedestals %s"
          % (cond["mount_deg"], cond["gas"], PEDESTAL_SET[cond["key"]]))
    by_v = {}
    for path, t0, points in sources(cond, gain=True):
        D = np.load(path)
        n_ev = len(D["fx_p"])
        wall = np.array([t0 + dt.timedelta(seconds=float(s) / 1e9) for s in D["ev_ts"]])

        clean = np.ones(n_ev, bool)
        for k in ("fx", "fy", "bx", "by"):
            clean &= (D[k + "_n"] == 1) & np.isfinite(D[k + "_p"])
        tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * 1120.0 / 1370.0
        ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * 1120.0 / 1370.0
        clean &= np.isfinite(tx + ty)

        h_ev, h_ch, h_amp, h_t, h_sat = D["h_ev"], D["h_ch"], D["h_amp"], D["h_time"], D["h_sat"]
        hb, eb = np.histogram(h_t, bins=np.arange(-1000, 6000, 100.0))
        cb = 0.5 * (eb[1:] + eb[:-1])
        pk = np.argmax(hb)
        thr = 0.25 * hb[pk]
        lo_i = pk - np.argmax(hb[pk::-1] < thr) + 1
        hi_i = pk + np.argmax(hb[pk:] < thr)
        g0, g1 = cb[lo_i] - 50, cb[hi_i] + 50
        GATE = (h_t > g0) & (h_t < g1)

        lead, ncl, nstrip = {}, {}, {}
        sel = GATE & clean[h_ev]
        for v in ("x", "y"):
            k = sel & (VIEW[h_ch] == v)
            ev, pos, amp = h_ev[k], POSITION_MM[h_ch[k]], h_amp[k]
            o = np.lexsort((pos, ev))
            ev, pos, amp = ev[o], pos[o], amp[o]
            new = np.empty(len(ev), bool); new[0] = True
            new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > 3.0)
            cid = np.cumsum(new) - 1
            nc = cid[-1] + 1 if len(cid) else 0
            cn = np.bincount(cid, minlength=nc)
            cev = np.zeros(nc, np.int64); cev[cid] = ev
            N = np.zeros(n_ev, np.int32); S = np.zeros(n_ev, np.int32)
            np.add.at(N, cev, 1)
            o2 = np.argsort(np.bincount(cid, weights=amp, minlength=nc), kind="stable")
            S[cev[o2]] = cn[o2]
            ncl[v], nstrip[v] = N, S
        disch = (ncl["x"] + ncl["y"] >= 6) | (nstrip["x"] > 40) | (nstrip["y"] > 40)

        good_hit = GATE & clean[h_ev] & ~disch[h_ev]
        print(f"{path}: gate {g0:.0f}-{g1:.0f} ns, gain-population hits {good_hit.sum()} "
              f"of {len(h_ev)} total")

        hit_wall = wall[h_ev]
        for v, t0s, t1s in points:
            tt0, tt1 = parse_t(t0s, t0), parse_t(t1s, t0)
            hm = good_hit & (hit_wall >= tt0) & (hit_wall < tt1)
            by_v.setdefault(v, []).append(hm.nonzero()[0])
            # store amp/sat alongside indices scoped to this source via closures below
            by_v[v][-1] = (h_amp[hm].copy(), h_sat[hm].copy())

    if not by_v:
        print("    no points -- skipped")
        continue
    out = outdir(cond)
    rows = []
    print(f"\n{'resist V':>9} {'n hits':>9} {'mean ADC':>9} {'median ADC':>11} {'p95 ADC':>8} {'sat %':>7}")
    for v in sorted(by_v, reverse=True):
        amp = np.concatenate([a for a, s in by_v[v]])
        sat = np.concatenate([s for a, s in by_v[v]])
        n = len(amp)
        if n == 0:
            continue
        mean_a, med_a, p95_a = amp.mean(), np.median(amp), np.percentile(amp, 95)
        f_sat = sat.mean()
        se_sat = np.sqrt(f_sat * (1 - f_sat) / n)
        print(f"{v:9.1f} {n:9d} {mean_a:9.1f} {med_a:11.1f} {p95_a:8.1f} {f_sat:6.2%}+-{se_sat:5.2%}")
        rows.append(dict(resist_v=v, n_hits=int(n), mean_adc=float(mean_a), median_adc=float(med_a),
                          p95_adc=float(p95_a), sat_frac=float(f_sat), sat_frac_err=float(se_sat)))

    with open(out + "gain_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", out + "gain_scan.csv")

    rv = np.array([r["resist_v"] for r in rows])
    mean_a = np.array([r["mean_adc"] for r in rows])
    med_a = np.array([r["median_adc"] for r in rows])
    p95_a = np.array([r["p95_adc"] for r in rows])
    sat = np.array([r["sat_frac"] for r in rows])
    sate = np.array([r["sat_frac_err"] for r in rows])
    nn = np.array([r["n_hits"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                    gridspec_kw=dict(height_ratios=[2, 1]))
    ax1.plot(rv, mean_a, marker="o", color="#2a78d6", label="mean ADC")
    ax1.plot(rv, med_a, marker="s", color="#4a3aa7", label="median ADC")
    ax1.plot(rv, p95_a, marker="^", color="0.6", ls="--", label="p95 ADC")
    ax1.set_ylabel("ADC amplitude")
    ax1.legend()
    ax1.set_title("det4 (mx17_E) gain vs resist voltage\n%s\npedestals %s; gated, clean-track, non-discharge hits only"
                  % (cond["label"], PEDESTAL_SET[cond["key"]]))
    ax2.errorbar(rv, sat * 100, yerr=sate * 100, marker="o", color="#e34948")
    ax2.set_ylabel("saturated hits [%]")
    ax2.set_xlabel("resist voltage [V]")
    fig.tight_layout()
    fig.savefig(out + "gain_scan.png", dpi=130)
    print("wrote", out + "gain_scan.png")
    plt.close(fig)
