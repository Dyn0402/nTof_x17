#!/usr/bin/env python3
"""gc_bench.py -- GC, bench: column duration T (FA colsum.py, every trigger) vs wall-clock time for every non-excluded
bench row of the pass registry (det2/3/4/6/7), and what a v fitted once per v group does to the angle scale.

Inputs: <row>/gc/colsum.npz (GC condor pass, local pull under ~/x17/cosmic_bench/model_audit/followups/gc/) or, for the
nine rows FA ran, <row>/fa/colsum.npz.  Clock: acquisition start from the file stem (minute) + DREAM timestamp x 10 ns.

Per row: clean triggers (both views >= 5 pulse strips, one cluster), T = mean of the views; saturated and unsaturated
tracks read different absolute T (FA), so every comparison is made per class and combined with the class weights.
  - T_row (trimmed mean per class), duration, 10-min block series v_rel(t) = T_row / T_block;
  - drift within the row: block range, event-weighted rms of v_rel, slope [%/h];
  - attachment proxy g = b/a of the box fit (the column's charge gradient: q(cathode)/q(mesh) - 1), unsaturated only;
  - eps_group = T_calib / T_row - 1 against the row's calibration row (registry calib_key; the v group whose kw/w0 the
    row inherits), i.e. the tan-scale error of the row if v is the group's: tan_reco = (1 + eps) tan_true.
    Only rows at the calibration row's drift voltage are compared (otherwise the field, not the gas, moves T).
Writes results/bench_rows.json, results/bench_blocks.json.
"""
import csv
import json
import os

import numpy as np

from gc_common import FA_LOCAL, GC_LOCAL, REPO, RES, TICK_S, load_colsum, robust_blocks, stem_time, tmean

REG = os.path.join(REPO, 'rerun', 'pass_registry.csv')
BLOCK_S = 600.0
MIN_EVT = 400
GOLD = {'o22_long_det2': ('mx17_det2_det3_overnight_6-22-26', 'longer_run', 'mx17_2'),
        'sat_det3': ('mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V', 'mx17_3'),
        'g_det4': ('mx17_det4_day_6-24-26', 'long_run', 'mx17_4'),
        'g_det6_long': ('mx17_det6_det7_overnight_6-26-26', 'long_run', 'mx17_6'),
        'g_det7_long': ('mx17_det6_det7_overnight_6-26-26', 'long_run', 'mx17_7')}


def colsum_path(run, sub, det):
    for p in (os.path.join(GC_LOCAL, run, sub, det, 'gc', 'colsum.npz'),
              os.path.join(FA_LOCAL, run, sub, det, 'fa', 'colsum.npz')):
        if os.path.exists(p):
            return p
    return None


def load_row(path):
    df, S, raw = load_colsum(path)
    tags = [str(s) for s in raw['acq_tags']]
    t0 = np.array([stem_time(t).timestamp() for t in tags])
    ts = np.where(df.x_ts.to_numpy() >= 0, df.x_ts.to_numpy(), df.y_ts.to_numpy()).astype(float)
    df['t'] = t0[df.acq.to_numpy()] + ts * TICK_S
    m = (df.x_has & df.y_has & (df.x_nsig >= 5) & (df.y_nsig >= 5) & (df.x_ncl == 1) & (df.y_ncl == 1)
         & np.isfinite(df.x_T) & np.isfinite(df.y_T)).to_numpy()
    c = df[m].sort_values('t').reset_index(drop=True)
    c['T'] = 0.5 * (c.x_T + c.y_T)
    c['sat'] = (c.x_sat | c.y_sat)
    g = []
    for p in 'xy':
        g.append(np.where(c[f'{p}_a'] > 0, c[f'{p}_b'] / c[f'{p}_a'], np.nan))
    c['g'] = np.nanmean(np.vstack(g), axis=0)
    return c


def classes(c):
    out = {}
    for name, k in (('unsat', ~c.sat.to_numpy()), ('sat', c.sat.to_numpy())):
        if k.sum() >= 100:
            out[name] = dict(n=int(k.sum()), T=tmean(c['T'].to_numpy()[k]))
    return out


def vrel_events(c, cls):
    """per-event T normalised to its class's row mean: v_rel_evt = T_class / T_evt (so blocks average to v(t)/v_row)."""
    T = c['T'].to_numpy(); sat = c.sat.to_numpy()
    r = np.full(len(c), np.nan)
    for name, k in (('unsat', ~sat), ('sat', sat)):
        if name in cls:
            r[k] = T[k] / cls[name]['T']
    return r


def main():
    rows = [r for r in csv.DictReader(open(REG)) if r['cls'] != 'excluded' and r['det'] != 'mx17_1']
    loaded, out, blocks = {}, [], {}
    for r in rows:
        run, sub, det = r['run'], r['subrun'], r['det']
        p = colsum_path(run, sub, det)
        if p is None:
            out.append(dict(key=r['key'], det=det, run=run, subrun=sub, missing=True)); continue
        c = load_row(p)
        if len(c) < 100:
            out.append(dict(key=r['key'], det=det, run=run, subrun=sub, missing=True, n_clean=int(len(c)))); continue
        loaded[r['key']] = c
    for r in rows:
        if r['key'] not in loaded:
            continue
        c = loaded[r['key']]; det = r['det']
        cls = classes(c)
        rel = vrel_events(c, cls)          # T_evt / T_class: blocks of it are T_block / T_row
        ok = np.isfinite(rel)
        t = c.t.to_numpy()
        rate = ok.sum() / max(t.max() - t.min(), 1.0)
        width = max(BLOCK_S, MIN_EVT / rate) if rate > 0 else BLOCK_S   # >= 10 min and >= MIN_EVT triggers per block
        tc, Tb, eb, nb = robust_blocks(t[ok], rel[ok], width)
        vb = 1.0 / Tb                                    # v_block / v_row
        ev = 1.0 / Tb ** 2 * eb
        dur_h = (t.max() - t.min()) / 3600.0 if len(t) else 0.0
        slope = np.nan
        if len(tc) >= 3:
            slope = float(np.polyfit((tc - tc[0]) / 3600.0, vb, 1, w=1 / np.maximum(ev, 1e-4))[0] * 100)
        rms = float(np.sqrt(np.average((vb - np.average(vb, weights=nb)) ** 2, weights=nb))) if len(vb) > 1 else 0.0
        # block noise removed: the rms of v(t) the data resolve
        rms_int = float(np.sqrt(max(rms ** 2 - np.average(ev ** 2, weights=nb), 0.0))) if len(vb) > 1 else 0.0
        q = 0.5 * (c.x_q + c.y_q).to_numpy()
        un0 = ~c.sat.to_numpy()
        qc, qb, qe, qn = robust_blocks(t[un0], q[un0], width)
        un = ~c.sat.to_numpy() & np.isfinite(c.g.to_numpy())
        gc_, gb, ge, gn = robust_blocks(t[un], c.g.to_numpy()[un], width)
        d = dict(key=r['key'], det=det, run=r['run'], subrun=r['subrun'], cls=r['cls'], calib_key=r['calib_key'],
                 refit=r['refit'], drift_V=float(r['drift_V']), resist_V=float(r['resist_V']), n_clean=int(len(c)),
                 classes=cls, sat_frac=float(c.sat.mean()) if len(c) else np.nan,
                 t_start=float(t.min()), t_end=float(t.max()), dur_h=dur_h,
                 vt_range_pct=float((vb.max() - vb.min()) * 100) if len(vb) else np.nan,
                 vt_rms_pct=rms * 100, vt_rms_int_pct=rms_int * 100, vt_slope_pct_per_h=slope, block_min=width / 60,
                 q_unsat=float(np.median(q[un0])) if un0.sum() > 50 else np.nan,
                 q_range_pct=float((qb.max() / qb.min() - 1) * 100) if len(qb) > 1 else np.nan,
                 g_row=tmean(c.g.to_numpy()[un]) if un.sum() > 50 else np.nan,
                 g_range=float(gb.max() - gb.min()) if len(gb) else np.nan)
        # against the calibration row of the v group
        ck = r['calib_key']
        cal = loaded.get(ck)
        calrow = next((x for x in rows if x['key'] == ck), None)
        if cal is not None and calrow is not None and abs(float(calrow['drift_V']) - float(r['drift_V'])) < 1:
            ccls = classes(cal)
            num = den = 0.0
            for name in ('unsat', 'sat'):
                if name in cls and name in ccls:
                    w = cls[name]['n']
                    num += w * (ccls[name]['T'] / cls[name]['T'] - 1); den += w
            d['eps_group_pct'] = float(num / den * 100) if den else np.nan
            d['dt_from_calib_h'] = float((np.median(c.t) - np.median(cal.t)) / 3600.0)
        else:
            d['eps_group_pct'] = np.nan
        out.append(d)
        blocks[r['key']] = dict(det=det, t=tc.tolist(), v_rel=vb.tolist(), err=ev.tolist(), n=nb.tolist(),
                                g_t=gc_.tolist(), g=gb.tolist(), g_err=ge.tolist(), q_t=qc.tolist(), q=qb.tolist(),
                                T_unsat=cls.get('unsat', {}).get('T'), T_sat=cls.get('sat', {}).get('T'),
                                drift_V=float(r['drift_V']), resist_V=float(r['resist_V']), cls=r['cls'],
                                calib_key=ck, refit=r['refit'])
        print(f"{r['key'][:90]:90s} n={len(c):6d} dur={dur_h:5.2f}h blk={width/60:4.0f}m rng={d['vt_range_pct']:5.2f}% rmsI={rms_int*100:4.2f}% q{d['q_range_pct']:5.0f}% "
              f"slope={slope:6.2f}%/h eps_grp={d['eps_group_pct']:6.2f}%")
    json.dump(out, open(os.path.join(RES, 'bench_rows.json'), 'w'), indent=1)
    json.dump(blocks, open(os.path.join(RES, 'bench_blocks.json'), 'w'))
    print(f'{sum(1 for d in out if not d.get("missing"))} rows, {sum(1 for d in out if d.get("missing"))} missing')


if __name__ == '__main__':
    main()
