#!/usr/bin/env python3
"""patches.py -- where the noise bands sit on each chamber, how they line up with the readout connectors, the
x/y sign relation, and their relation to discharges (HANDOFF satstudy follow-up 1, items b, c, d).

    python patches.py            # every bench chamber scanned by ../satstudy/bands.py scan -> $BANDS_OUT/patches.json, patches_<key>.npz

Input: the per-trigger channel baselines of ../satstudy/bands.py (bands_scan[_KEY].npz in ~/x17/cosmic_bench/satstudy/).
A channel is "offset" as in bands.flags (|window mean after CM| > 5 sigma, RMS about it < 2 sigma), a band is >= 8
offset channels on one 64-channel chip (bands.band_events), and a PATCH is a run of offset channels contiguous in
strip POSITION (gaps of <= 2 strips bridged).

Discharge per chip: >= 8 channels of the chip above 3000 ADC after CM in the window (as bands.nulls).
Spark trigger: > 150 strips above 6 sigma in x + y (as bands.nulls).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(HERE, '..', 'satstudy'), REPO, os.path.join(REPO, 'mx_june_cosmic_qa')]
import bands as B  # noqa: E402

OUTD = os.environ.get('BANDS_OUT', os.path.expanduser('~/x17/cosmic_bench/satstudy/bands')); os.makedirs(OUTD, exist_ok=True)
KEYS = ('sat_det3', 'g_det3', 'g_det3_wknd', 'g_det2', 'g_det4', 'g_det6_long', 'g_det7_long')
DISCH_ADC, DISCH_N = 3000, 8


def positions(key):
    from qa_config import get_config
    from wft import io as wio
    cfg = get_config(key); P = wio.strip_position_map(cfg)
    return cfg, P[cfg.MX17_FEU_X], P[cfg.MX17_FEU_Y]


def runs_of(mask_sorted, gap=2):
    """[(lo, hi)] index runs of True in a 1-D bool array, bridging gaps of <= gap False."""
    idx = np.flatnonzero(mask_sorted)
    if len(idx) == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    return [(idx[a], idx[b]) for a, b in zip(np.r_[0, br + 1], np.r_[br, len(idx) - 1])]


def analyse(key):
    Z = np.load(B.scan_path(key))
    cfg, px, py = positions(key)
    pos = dict(x=px, y=py)
    # align x and y on eventId
    ex, ey = Z['eid_x'], Z['eid_y']
    com, ix, iy = np.intersect1d(ex, ey, return_indices=True)
    I = dict(x=ix, y=iy)
    ts = Z['ts_x'][ix].astype(np.int64)
    o = np.argsort(com); com, ts = com[o], ts[o]
    for v in 'xy':
        I[v] = I[v][o]
    off, mc, band, disch, nsig = {}, {}, {}, {}, 0
    for v in 'xy':
        f = B.flags(Z, v)[I[v]]
        off[v] = f
        mc[v] = Z[f'mc_{v}'][I[v]].astype(np.float32)
        band[v] = B.band_events(Z, v)[I[v]]
        mx = Z[f'mx_{v}'][I[v]].astype(np.float32)
        disch[v] = (mx > DISCH_ADC).reshape(len(mx), 8, 64).sum(2) >= DISCH_N
        nsig = nsig + (mx > 6 * Z[f'noise_{v}']).sum(1)
        del mx
    spark = nsig > 150
    anyb = band['x'].any(1) | band['y'].any(1)
    n = len(com)
    t = (ts - ts[0]) * 1e-8                       # s (DREAM timestamp, 10 ns)
    res = dict(key=key, n=int(n), duration_s=float(t[-1]), trig_rate_hz=float(n / t[-1]),
               feu_x=int(cfg.MX17_FEU_X), feu_y=int(cfg.MX17_FEU_Y),
               p_band=float(anyb.mean()), p_band_x=float(band['x'].any(1).mean()), p_band_y=float(band['y'].any(1).mean()),
               p_spark=float(spark.mean()),
               p_disch_any=float((disch['x'].any(1) | disch['y'].any(1)).mean()))
    # ---- (b) where: per-channel offset frequency over band events, in position, and patch edges vs connectors
    occ, prof, edges_conn, edge_any, patch_w, sign_coh = {}, {}, {}, {}, {}, {}
    rect = []                                        # (x_lo, x_hi, y_lo, y_hi, sign_x, sign_y, amp_x, amp_y)
    for v in 'xy':
        p = pos[v]; order = np.argsort(np.where(np.isfinite(p), p, 1e9)); order = order[np.isfinite(p[order])]
        occ[v] = off[v][anyb].mean(0)                # per channel
        prof[v] = dict(pos=p[order].round(2).tolist(), occ=occ[v][order].round(4).tolist())
        ne, nc, ws, coh = 0, 0, [], []
        for e in np.flatnonzero(band[v].any(1)):
            fo = off[v][e][order]
            for a, b in runs_of(fo):
                if b - a + 1 < 8:
                    continue
                ch = order[a:b + 1][fo[a:b + 1]]
                ws.append(p[order[b]] - p[order[a]])
                s = np.sign(mc[v][e][ch]); coh.append(max((s > 0).mean(), (s < 0).mean()))
                for r in (a, b):                   # edge channels: on a 64-channel boundary (0 or 63 mod 64)?
                    if r == 0 or r == len(order) - 1:
                        continue                   # the plane's own end is not a test
                    ne += 1; nc += (order[r] % 64 in (0, 63))
        edges_conn[v] = float(nc / max(ne, 1)); edge_any[v] = int(ne)
        patch_w[v] = {q: float(np.percentile(ws, q)) for q in (10, 50, 90)} if ws else {}
        sign_coh[v] = {q: float(np.percentile(coh, q)) for q in (10, 50)} if coh else {}
    res.update(occ_profile=prof, edge_on_connector_boundary=edges_conn, n_edges=edge_any,
               edge_expect_random=2 / 64, patch_width_mm=patch_w, patch_sign_coherence=sign_coh)
    # x/y: sign relation and amplitude correlation in events with a band in both views
    both = np.flatnonzero(band['x'].any(1) & band['y'].any(1))
    sx, sy, ax_, ay_ = [], [], [], []
    for e in both:
        mx_ = mc['x'][e][off['x'][e]]; my_ = mc['y'][e][off['y'][e]]
        sx.append(np.median(mx_)); sy.append(np.median(my_))
        xs = pos['x'][off['x'][e]]; ys = pos['y'][off['y'][e]]
        xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
        if len(xs) and len(ys):
            rect.append((np.percentile(xs, 5), np.percentile(xs, 95), np.percentile(ys, 5), np.percentile(ys, 95), sx[-1], sy[-1]))
    sx, sy = np.array(sx), np.array(sy)
    res['xy'] = dict(n_both=int(len(both)), same_sign=float((np.sign(sx) == np.sign(sy)).mean()) if len(sx) else None,
                     corr_amp=float(np.corrcoef(sx, sy)[0, 1]) if len(sx) > 2 else None,
                     corr_logabs=float(np.corrcoef(np.log(np.abs(sx)), np.log(np.abs(sy)))[0, 1]) if len(sx) > 2 else None)
    # ---- (c) bands vs sparks / discharges in the same trigger, per chamber
    res['p_band_given_spark'] = float(anyb[spark].mean()) if spark.any() else None
    res['p_band_given_nospark'] = float(anyb[~spark].mean())
    res['p_spark_given_band'] = float(spark[anyb].mean())
    # ---- (d) time since the last discharge on the SAME chip (any trigger, spark-vetoed included)
    dts_band, dts_ctrl, amp_band = [], [], []
    for v in 'xy':
        last = np.full(8, -np.inf)
        dt_e = np.full((n, 8), np.inf)
        for e in range(n):
            dt_e[e] = t[e] - last                    # time since the last discharge on chip (before this trigger)
            last[disch[v][e]] = t[e]
        bv = band[v]
        amp = np.zeros((n, 8))
        a = np.abs(mc[v]) * off[v]
        amp = np.where(off[v].reshape(n, 8, 64).sum(2) > 0, a.reshape(n, 8, 64).sum(2) / np.maximum(off[v].reshape(n, 8, 64).sum(2), 1), 0)
        for c in range(8):
            if bv[:, c].sum() < 20:
                continue
            dts_band.append(dt_e[bv[:, c], c]); amp_band.append(amp[bv[:, c], c])
            dts_ctrl.append(dt_e[~bv[:, c] & ~disch[v][:, c], c])
    db = np.concatenate(dts_band) if dts_band else np.array([]); dc = np.concatenate(dts_ctrl) if dts_ctrl else np.array([])
    ab = np.concatenate(amp_band) if amp_band else np.array([])
    edges = np.array([0, 0.2, 1, 5, 30, 120, 600, np.inf])
    if len(db):
        hb = np.histogram(db, edges)[0] / len(db); hc = np.histogram(dc, edges)[0] / len(dc)
        fin = np.isfinite(db)
        med_amp = [float(np.median(ab[(db >= lo) & (db < hi)])) if ((db >= lo) & (db < hi)).sum() > 10 else None
                   for lo, hi in zip(edges[:-1], edges[1:])]
        res['since_discharge'] = dict(edges_s=edges[:-1].tolist(), frac_band=hb.round(4).tolist(), frac_ctrl=hc.round(4).tolist(),
                                      median_amp_band=med_amp,
                                      corr_logdt_logamp=float(np.corrcoef(np.log(db[fin] + 1e-3), np.log(ab[fin] + 1))[0, 1]) if fin.sum() > 10 else None,
                                      median_dt_band=float(np.median(db)), median_dt_ctrl=float(np.median(dc)))
    # ---- lifetime: does a band on a chip persist into the NEXT trigger, as a function of the trigger spacing?
    dt = np.diff(t); bb = np.concatenate([band['x'], band['y']], 1)   # (n, 16)
    now, nxt = bb[:-1], bb[1:]
    pers = []
    for lo, hi in zip([0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0], [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, np.inf]):
        k = (dt >= lo) & (dt < hi)
        a = now[k]; b2 = nxt[k]
        pers.append(dict(lo=lo, hi=hi if np.isfinite(hi) else None, n_pairs=int(k.sum()), n_band_chips=int(a.sum()),
                         p_same_chip_next=float(b2[a].mean()) if a.sum() else None,
                         p_chip_base=float((a * bb.mean(0)).sum() / max(a.sum(), 1))))
    res['persistence'] = pers
    res['p_chip_band_given_band_anywhere_prev'] = float(nxt[now.any(1)].mean())
    # persistent pairs (dt < 3 ms, same chip banded in both): sign kept? amplitude ratio (later / earlier)?
    cm = np.concatenate([np.where(off[v], mc[v], 0).reshape(n, 8, 64).sum(2) /
                         np.maximum(off[v].reshape(n, 8, 64).sum(2), 1) for v in 'xy'], 1)        # (n, 16) signed mean
    i0, c0 = np.nonzero(now & nxt & (dt < 3e-3)[:, None])
    res['persist_pairs'] = dict(n=int(len(i0)), same_sign=float((np.sign(cm[i0, c0]) == np.sign(cm[i0 + 1, c0])).mean()) if len(i0) else None,
                                ratio_q={q: float(np.percentile(np.abs(cm[i0 + 1, c0] / cm[i0, c0]), q)) for q in (25, 50, 75)} if len(i0) else None,
                                dt_ms=(dt[i0] * 1e3).round(3).tolist(), ratio=np.abs(cm[i0 + 1, c0] / cm[i0, c0]).round(3).tolist())
    # onset: a discharge on a chip in trigger n, band on the same chip in trigger n+1, by spacing
    dd = np.concatenate([disch['x'], disch['y']], 1)[:-1]
    ons = []
    for lo, hi in ((0, 3e-3), (3e-3, 3e-2), (3e-2, np.inf)):
        k = (dt >= lo) & (dt < hi)
        a = dd[k]; b2 = nxt[k]
        ons.append(dict(lo=lo, hi=hi if np.isfinite(hi) else None, n_disch_chips=int(a.sum()),
                        p_band_next=float(b2[a].mean()) if a.sum() else None))
    res['after_discharge_next_trigger'] = ons
    # ---- chips: band rate vs discharge rate per chip
    res['chips'] = {v: dict(band=band[v].mean(0).round(4).tolist(), disch=disch[v].mean(0).round(4).tolist()) for v in 'xy'}
    np.savez_compressed(os.path.join(OUTD, f'patches_{key}.npz'), rect=np.array(rect), occ_x=occ['x'], occ_y=occ['y'],
                        pos_x=pos['x'], pos_y=pos['y'], eid=com, t=t, anyb=anyb, spark=spark)
    return res


if __name__ == '__main__':
    keys = sys.argv[1:] or KEYS
    p = os.path.join(OUTD, 'patches.json')
    out = json.load(open(p)) if os.path.exists(p) else {}
    for k in keys:
        if not os.path.exists(B.scan_path(k)):
            continue
        r = analyse(k); out[k] = r
        print(k, {a: r[a] for a in ("persistence", "trig_rate_hz", 'p_band', 'p_spark', 'p_band_given_spark', 'p_band_given_nospark',
                                    'edge_on_connector_boundary', 'patch_width_mm', 'patch_sign_coherence', 'xy')}, flush=True)
        print('  since', r.get('since_discharge'), flush=True)
        json.dump(out, open(p, 'w'), indent=1)
