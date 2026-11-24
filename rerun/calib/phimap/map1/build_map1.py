#!/usr/bin/env python
"""build_map1.py -- bench re-finalization step 4: install the per-mounting maps (interior only) in a bundle per row and
recompute the angle columns of the FULL pass-1 events table of every one of the 156 rows.

Reads   $JOB/pass1_all/<run>/<subrun>/<det>/{events.parquet, bow_gate_table.parquet, calib_bundle/}   (read-only)
        $JOB/phimap/{row_constants.json, blocks/*.json, final/<row>.parquet, mountings.csv}           (step 3)
Writes  $JOB/map1/<run>/<subrun>/<det>/{calib_bundle/, events.parquet, map1_row.json}
        $JOB/map1/build_summary.json

Decisions (Dylan, 2026-11-24):
  1. interior only: the block gets outside = 'block', the row's PASS-1 w0/kw as ``fallback`` (the constants stored in the row
     bundle are the ones measured WITH the map), and for chamber D a ``domain_x`` with U >= 0 (x >= 200 mm);
  2. no map (columns copied from pass 1): E, A 6-22, A 6-23, B 6-17, the A 6-27 rows at 100 V and 300 V;
  3. C: railed tracks (q_uend at the end of the depth grid) are rescaled per row and view by step 3's factor (x_resc, y_resc);
     D and E railed tracks are flagged (x_railed, y_railed), not changed;
  4. C rows without a map of their own use the borrowed map (map_borrowed kept).

Columns changed in events.parquet, for a view p where the map is used (map_used_p): p_tan_theta, p_theta_deg, p_slope_reliable,
p_bow_h (and bow_h from x).  Everything else unchanged.  Columns added: map_unit, map_variant, map_borrowed, map_used_x/y,
x/y_railed, x/y_resc (factor applied: 1 unless a railed track of a C row), x/y_tan_pass1 (the old value).
Where the map is not used the pass-1 value is kept bit for bit; its agreement with the bundle code path (bow_pairs with the
fallback constants) is reported per row (``max_diff_bundle_vs_pass1``).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import rows as R  # noqa: E402
from wft import reco  # noqa: E402
from wft.calib import CalibrationBundle  # noqa: E402

JOB = R.JOB
STEP3 = f'{JOB}/phimap'
OUT = os.environ.get('MAP1_OUT', f'{JOB}/map1')
NOMAP_DRIFT_V = (100, 300)


def nomap_reason(m, rc):
    if not rc.get('unit'):
        return 'step 3: no map possible (B 6-17: 104 gate events; A 6-23: pass-1 angles uncorrelated with the reference)'
    if m.chamber == 'E':
        return 'chamber E stays on the bow block (Dylan 11-24)'
    if m.mounting == 'A_0622':
        return 'A 6-22: 80 % of the tracks are railed, the map gains nothing on all events'
    if m.unit == 'A_0627' and int(m.drift_V) in NOMAP_DRIFT_V:
        return 'A 6-27 drift scan at 100 / 300 V: no angle is usable with or without a map'
    return None


def installed_block(rc, cal_pass, chamber):
    blk = json.load(open(f"{STEP3}/{rc['block']}"))
    blk['outside'] = 'block'
    blk['fallback'] = {'w0': {p: float(cal_pass.w0[p]) for p in 'xy'}, 'kw': {p: float(cal_pass.kw[p]) for p in 'xy'}}
    if chamber == 'D':     # D's X view only for x >= 200 mm (U >= 0); the Y view keeps the whole domain
        blk['domain_x'] = {'U': [0.0, float(blk['U_domain'][1])], 'W': [float(blk['W_domain'][0]), float(blk['W_domain'][1])]}
    blk['provenance'] = dict(blk.get('provenance') if isinstance(blk.get('provenance'), dict) else {'text': blk.get('provenance')},
                             step4='interior only (outside = block), fallback = pass-1 w0/kw', unit=rc['unit'])
    return blk


def main():
    mt = pd.read_csv(f'{STEP3}/mountings.csv').set_index('key', drop=False)
    rcs = json.load(open(f'{STEP3}/row_constants.json'))
    reg = R.reg()
    # C railed-track factors per row and view (step 3): the unique x_resc among railed events; pooled value for rows without
    resc, pooled = {}, {'x': [], 'y': []}
    for k in reg.key:
        if mt.loc[k, 'chamber'] != 'C':
            continue
        f = pd.read_parquet(f'{STEP3}/final/{k}.parquet')
        for v in 'xy':
            u = np.unique(np.round(f[f'{v}_resc'].values[f[f'cens_{v}'].values], 12)) if f[f'cens_{v}'].any() else np.array([])
            resc[(k, v)] = float(u[0]) if len(u) == 1 else None
            if len(u) > 1:
                raise RuntimeError(f'{k} {v}: several rescale factors {u}')
    # the pooled factor: the value shared by the rows with < 150 railed gate events (step 3 RESULT table)
    for v in 'xy':
        cnt = {}
        for (k, vv), val in resc.items():
            if vv == v and val is not None:
                cnt[round(val, 9)] = cnt.get(round(val, 9), 0) + 1
        pooled[v] = max(cnt, key=cnt.get)
    print('pooled railed factors', pooled, flush=True)
    summ = {}
    for k in reg.key:
        m, rc, r = mt.loc[k], rcs[k], reg.loc[k]
        rd = R.rowdir(k)
        od = f'{OUT}/{r.run}/{r.subrun}/{r.det}'
        os.makedirs(od, exist_ok=True)
        e = pd.read_parquet(rd + '/events.parquet')
        g = pd.read_parquet(rd + '/bow_gate_table.parquet', columns=['x_w', 'x_p0', 'y_w', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'acq_tag', 'timestamp'])
        # the gate table is the reference-matched subset of the events (index = event_id); ge = position of each gate row in e
        # (event_id repeats between acquisitions: the key is event_id, acq_tag, timestamp)
        key = lambda ev, tag, ts: pd.MultiIndex.from_arrays([np.asarray(ev), np.asarray(tag).astype(str), np.asarray(ts).astype('int64')])
        ek = key(e.event_id.values, e.acq_tag.values, e.timestamp.values)
        assert ek.is_unique, k
        ge = ek.get_indexer(key(g.index.values, g.acq_tag.values, g.timestamp.values))
        assert (ge >= 0).all(), f'{k}: gate events missing in events.parquet'
        for c in ('x_w', 'x_p0', 'y_w', 'y_p0', 'x_tan_theta', 'y_tan_theta'):
            assert np.array_equal(e[c].values[ge], g[c].values, equal_nan=True), f'{k}: {c} differs between events and gate table'
        cal = CalibrationBundle.load(os.path.dirname(R.bundle_path(k)))
        K = float(r.K) * R.SAMPLE_NS
        why = nomap_reason(m, rc)
        info = dict(key=k, chamber=m.chamber, unit=(rc.get('unit') or ''), nomap=why, n_events=int(len(e)))
        # pass-1 reproduction by the bundle code path (diagnostic)
        rp = reco.bow_pairs(cal, e.x_w.values, e.x_p0.values, e.y_w.values, e.y_p0.values)
        for v in 'xy':
            a, b = rp[f'tan_{v}'], e[f'{v}_tan_theta'].values
            mm = np.isfinite(a) & np.isfinite(b)
            d = np.abs(a - b)[mm]
            info[f'pass1_repro_{v}'] = dict(n_diff_gt_1e12=int((d > 1e-12).sum()), n=int(mm.sum()), max=float(d.max()) if len(d) else 0.0,
                                            n_nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()))
        out = e.copy()
        for v in 'xy':
            out[f'{v}_tan_pass1'] = e[f'{v}_tan_theta'].values
            out[f'{v}_railed'] = (e[f'{v}_q_uend'].values >= K - 1)
            out[f'{v}_resc'] = 1.0
            out[f'map_used_{v}'] = False
        out['map_unit'] = ''
        out['map_variant'] = 'none'
        out['map_borrowed'] = False
        if why is None:
            blk = installed_block(rc, cal, m.chamber)
            assert abs(rc['v_drift'] - cal.v_drift) < 1e-9 * max(1.0, abs(cal.v_drift)), (k, rc['v_drift'], cal.v_drift)
            cal.bow = dict(cal.bow or {}, phi=blk)
            cal.w0 = {p: float(rc['w0'][p]) for p in 'xy'}
            cal.kw = {p: float(rc['kw'][p]) for p in 'xy'}
            bdir = f'{od}/calib_bundle'
            cal.save(bdir, note=f"bench re-finalization step 4: map {rc['unit']} ({rc['block']}), interior only; w0/kw with the map")
            cal2 = CalibrationBundle.load(bdir)     # what is used below is the reloaded bundle
            q = reco.bow_pairs(cal2, e.x_w.values, e.x_p0.values, e.y_w.values, e.y_p0.values)
            f3 = pd.read_parquet(f'{STEP3}/final/{k}.parquet')
            assert len(f3) == len(g)
            for v in 'xy':
                used = q[f'phi_{v}']
                t = q[f'tan_{v}'].copy()
                fac = np.ones(len(e))
                if m.chamber == 'C':
                    fv = resc[(k, v)] if resc[(k, v)] is not None else pooled[v]
                    fac = np.where(out[f'{v}_railed'].values, fv, 1.0)
                tnew = t * fac
                # not used: the pass-1 value itself (bit for bit); used: the map (x railed-track factor on C)
                out[f'{v}_tan_theta'] = np.where(used, tnew, e[f'{v}_tan_theta'].values)
                out[f'{v}_resc'] = np.where(used, fac, 1.0)
                out[f'map_used_{v}'] = used
                tt = out[f'{v}_tan_theta'].values
                with np.errstate(invalid='ignore'):
                    out[f'{v}_theta_deg'] = np.where(used, np.degrees(np.arctan(tt)), e[f'{v}_theta_deg'].values)
                    out[f'{v}_slope_reliable'] = np.where(used, np.abs(tt) >= reco.TAN_MIN_SLOPE, e[f'{v}_slope_reliable'].values)
                out[f'{v}_bow_h'] = np.where(used, q[f'h_{v}'], e[f'{v}_bow_h'].values)
                # checks: the fallback of the map bundle is the pass bundle's bow_pairs, bit for bit
                same = np.array_equal(q[f'tan_{v}'][~used], rp[f'tan_{v}'][~used], equal_nan=True) \
                    and np.array_equal(q[f'h_{v}'][~used], rp[f'h_{v}'][~used], equal_nan=True)
                info[f'fallback_equals_pass_bundle_{v}'] = bool(same)
                nu =~used & np.isfinite(e[f'{v}_tan_theta'].values)
                dd = np.abs(q[f'tan_{v}'] - e[f'{v}_tan_theta'].values)[nu & np.isfinite(q[f'tan_{v}'])]
                info[f'max_diff_bundle_vs_pass1_{v}'] = float(dd.max()) if len(dd) else 0.0
                info[f'n_bundle_ne_pass1_{v}'] = int((dd > 1e-12).sum())
                info[f'n_not_used_{v}'] = int(nu.sum())
                info[f'max_diff_unused_table_vs_pass1_{v}'] = float(np.nanmax(np.abs(out[f'{v}_tan_theta'].values - e[f'{v}_tan_theta'].values)[~used]) if (~used).any() else 0.0)
                # step 3: the production map where the map is used (no rescale)
                s3 = np.full(len(e), np.nan)
                s3[ge] = f3[f'{v}_tan_map'].values
                mm = used & np.isfinite(q[f'tan_{v}']) & np.isfinite(s3)
                info[f'max_diff_vs_step3_{v}'] = float(np.abs(q[f'tan_{v}'] - s3)[mm].max()) if mm.any() else 0.0
                info[f'n_used_{v}'] = int(used.sum())
                info[f'n_step3_compared_{v}'] = int(mm.sum())
            out['bow_h'] = np.where(out.map_used_x.values, out.x_bow_h.values, e['bow_h'].values)
            out['map_unit'] = rc['unit']
            out['map_variant'] = rc['map_variant']
            out['map_borrowed'] = bool(rc['map_borrowed'])
            info['block'] = rc['block']
            info['map_borrowed'] = bool(rc['map_borrowed'])
        else:
            # no map: the pass-1 bundle unchanged, all columns copied
            cal.save(f'{od}/calib_bundle', note='bench re-finalization step 4: no map, pass-1 bundle unchanged')
        for c in ('x_tan_theta', 'y_tan_theta', 'x_theta_deg', 'y_theta_deg'):
            out[c] = out[c].astype(float)
        out.to_parquet(f'{od}/events.parquet')
        json.dump(info, open(f'{od}/map1_row.json', 'w'), indent=1, default=float)
        summ[k] = info
        print(k[-60:], info['unit'] or '-', 'used', info.get('n_used_x', 0), info.get('n_used_y', 0),
              'dstep3', f"{info.get('max_diff_vs_step3_x', 0):.1e} {info.get('max_diff_vs_step3_y', 0):.1e}",
              'dp1', f"{info.get('max_diff_bundle_vs_pass1_x', 0):.1e} {info.get('max_diff_bundle_vs_pass1_y', 0):.1e}", flush=True)
    json.dump(summ, open(f'{OUT}/build_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
