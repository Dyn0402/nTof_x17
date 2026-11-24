#!/usr/bin/env python
"""driver.py <unit> [<unit> ...] | all  -- fit the per-mounting angle map(s), 5-time-block cross-validation, outputs.

Per unit (a mounting, or one big C row; see mountings.csv):
  1. training events: the unit's `train` rows (main drift field, >= 60 usable events), railed tracks excluded per view
     (q_uend at the end of the depth grid), D X only for reference x >= 200 mm;
  2. a full fit (production map) and 5 fits each without time block b of every row (blocks of the full gate table of the row,
     rank in (acq_tag, timestamp)): K, KS, lam by inner CV on that training part, row constants refit on the training part;
  3. for every row applied to (train, tilt_only, borrowed, tiny): constants with the map applied (full data) and per fold
     (events outside the block), predictions for ALL gate-table events by wft.reco.bow_pairs: x_tan_map / y_tan_map from the
     production map, *_cv from the fold map and fold constants of the event's block;
  4. outputs under $PHIMAP_JOB/phimap/: blocks/<unit>[_tilt].json, rows/<key>.parquet, units/<unit>.json (selection,
     constants, reproduction check).
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib as L  # noqa: E402
import rows as R  # noqa: E402
from lib import Ev, Layout  # noqa: E402

OUT = R.OUTDIR
MIN_USE = 60       # usable events per view for a row to be trained on / to have its own constants
MIN_C = 30
G = {}


def domain(det):
    ix, iy = R.active(det)
    f = lambda v, d: (np.floor(v * 100) / 100 if d < 0 else np.ceil(v * 100) / 100)
    return ((f((ix[0] - 200) / 100, -1), f((ix[1] - 200) / 100, 1)), (f((iy[0] - 200) / 100, -1), f((iy[1] - 200) / 100, 1)))


def ev_from(d, ri, det):
    fx = ~d.cens_x.values.astype(bool)
    fy = ~d.cens_y.values.astype(bool)
    if det == 'mx17_7':
        fx = fx & (d.xm.values >= 200.0)
    return dict(row=np.full(len(d), ri), wx=d.x_w.values * 1e3, wy=d.y_w.values * 1e3, p0x=d.x_p0.values, p0y=d.y_p0.values,
                trx=d.tan_ref_x.values, try_=d.tan_ref_y.values, fx=fx, fy=fy, blk=d.blk.values, xm=d.xm.values, ym=d.ym.values)


def concat_ev(parts, dom):
    kw = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return Ev(**kw, dom=dom)


def row_consts(e, phi, w0i, c1i, blk_out=-1, nit=4):
    """Constants (w0, c1) [2] of one row with the map ``phi`` applied: robust line of w against (1 + S)(tan_ref + A), on events
    with fx / fy outside block blk_out; U, W from the constants themselves (iterated).  n [2] events used."""
    w0 = np.array(w0i, float).reshape(2, 1).copy()
    c1 = np.array(c1i, float).reshape(2, 1).copy()
    n = np.zeros(2, int)
    e = Ev(**e.__dict__)
    e.row = np.zeros(e.n, int)
    for it in range(nit):
        U, W, zx, zy = L.uw(e, w0, c1)
        ax, ay, sx, sy = L.bow_phi.evaluate(phi, U, W)
        for v, (w, tr, f, a, s) in enumerate(((e.wx, e.trx, e.fx, ax, sx), (e.wy, e.try_, e.fy, ay, sy))):
            m = f & (e.blk != blk_out)
            n[v] = int(m.sum())
            if m.sum() >= MIN_C:
                c = L.robust_line((1 + s[m]) * (tr[m] + a[m]), w[m])
                w0[v, 0], c1[v, 0] = c
    return w0[:, 0], c1[:, 0], n


def run_task(b):
    """fit on the training events outside block b (b = -1: all): selection + production-form fit"""
    ev, nrow, dom = G['ev'], G['nrow'], G['dom']
    t0 = time.time()
    trn = ev.sub(ev.blk != b)
    K, KS, lam, sc, info = L.select_model(trn, nrow)
    lay = Layout(K, KS, dom)
    w0, c1 = G['ladder_fn'](trn)
    fit = L.fit_gn(trn, nrow, lay, lam, w0, c1)
    return b, dict(K=K, KS=KS, lam=lam, score=sc, info=info, n_train=int(trn.n), n_fx=int(trn.fx.sum()), n_fy=int(trn.fy.sum()),
                   n_it=fit['n_it'], secs=round(time.time() - t0, 1)), fit


def ladder_pooled(ev, nrow):
    """ladder constants; rows with < 30 usable events in a view take the median of the others"""
    w0 = np.full((2, nrow), np.nan)
    c1 = np.full((2, nrow), np.nan)
    for r in range(nrow):
        for v, (w, tr, f) in enumerate(((ev.wx, ev.trx, ev.fx), (ev.wy, ev.try_, ev.fy))):
            m = (ev.row == r) & f
            if m.sum() >= MIN_C:
                w0[v, r], c1[v, r] = L.robust_line(tr[m], w[m])
    for v in range(2):
        bad = ~np.isfinite(c1[v])
        if bad.any():
            w0[v, bad] = np.nanmedian(w0[v])
            c1[v, bad] = np.nanmedian(c1[v])
    return w0, c1


def process(unit, df, workers=6, quick=False):
    t00 = time.time()
    U_ = df[(df.unit == unit)]
    mnt = U_.mounting.iloc[0]
    det = R.full().loc[U_.key.iloc[0]].det
    dom = domain(det)
    print(f'[{unit}] domain {dom}', flush=True)
    # ---- rows
    train_keys = []
    loaded = {}
    for k in U_[U_.role == 'train'].key:
        t = R.load_full(k)
        d = t[t.sel].sort_values('trank').reset_index(drop=True)
        e = ev_from(d, 0, det)
        if min(e['fx'].sum(), e['fy'].sum()) >= MIN_USE:
            train_keys.append(k)
    # applied rows: the unit's train rows, tilt_only / tiny rows of the mounting, borrowed rows with this donor
    apply = list(U_.key)
    if not unit.startswith('C_'):
        M = df[df.mounting == mnt]
        apply += [k for k in M.key if k not in apply and M.set_index('key').loc[k, 'role'] in ('tilt_only', 'tiny')]
    else:
        apply += list(df[(df.donor == unit) & (df.role == 'borrowed')].key)
    apply = list(dict.fromkeys(apply))
    parts = []
    for ri, k in enumerate(train_keys):
        t = R.load_full(k)
        d = t[t.sel].sort_values('trank').reset_index(drop=True)
        parts.append(ev_from(d, ri, det))
    ev = concat_ev(parts, dom)
    nrow = len(train_keys)
    print(f'[{unit}] train rows {nrow}, events {ev.n}, fx {ev.fx.sum()}, fy {ev.fy.sum()}; apply rows {len(apply)}', flush=True)
    G.update(ev=ev, nrow=nrow, dom=dom, ladder_fn=lambda e: ladder_pooled(e, nrow))
    blocks_list = [-1] + list(range(R.NB))
    with ProcessPoolExecutor(max_workers=min(workers, len(blocks_list))) as ex:
        res = list(ex.map(run_task, blocks_list))
    fits = {b: (meta, fit) for b, meta, fit in res}
    for b in blocks_list:
        m = fits[b][0]
        print(f'[{unit}] fold {b}: K={m["K"]} KS={m["KS"]} lam={m["lam"]} score={m["score"]:.4f} edge={m["info"]["lam_edge"]} it={m["n_it"]} {m["secs"]}s', flush=True)
    # ---- blocks
    blk = {}
    for b in blocks_list:
        meta, fit = fits[b]
        prov = dict(unit=unit, mounting=mnt, fold=b, K=meta['K'], KS=meta['KS'], lam=meta['lam'], train_rows=train_keys, n_train=meta['n_train'],
                    fitter='rerun/calib/phimap/lib.py fit_gn', domain=[list(dom[0]), list(dom[1])])
        blk[b] = (L.block_from_fit(fit, prov), L.block_from_fit(fit, prov, tilt_only=True))
    os.makedirs(f'{OUT}/blocks', exist_ok=True)
    os.makedirs(f'{OUT}/rows', exist_ok=True)
    os.makedirs(f'{OUT}/units', exist_ok=True)
    json.dump(blk[-1][0], open(f'{OUT}/blocks/{unit}.json', 'w'))
    if (U_.role == 'tilt_only').any() or (df[(df.mounting == mnt)].role == 'tilt_only').any():
        json.dump(blk[-1][1], open(f'{OUT}/blocks/{unit}_tilt.json', 'w'))
    # ---- apply rows
    info_rows = {}
    ladder_all = {}
    perrow = {}
    for k in apply:
        t = R.load_full(k)
        bund = R.bundle(k)
        d = t[t.sel].sort_values('trank').reset_index(drop=True)
        e = Ev(**ev_from(d, 0, det), dom=dom)
        ok = min(e.fx.sum(), e.fy.sum()) >= MIN_USE
        perrow[k] = dict(t=t, bund=bund, e=e, ok=ok)
    # constants per row and variant: variant 'full' (all rows), 'tilt' (rows at another drift field)
    role = df.set_index('key').role
    out_cols = {}
    for k in apply:
        pr = perrow[k]
        e, bund, t = pr['e'], pr['bund'], pr['t']
        variants = ['full', 'tilt'] if role[k] == 'tilt_only' else ['full']
        nt = len(t)
        vd = float(bund['v_drift'])
        cols = {}
        info_r = dict(role=role[k], n_sel=int(e.n), n_fx=int(e.fx.sum()), n_fy=int(e.fy.sum()), v_drift=vd, variants={})
        for var in variants:
            sel_ = 0 if var == 'full' else 1
            const = {}
            for b in blocks_list:
                phi = blk[b][sel_]
                if pr['ok']:
                    w0i, c1i = ladder_pooled(Ev(**{**e.__dict__, 'row': np.zeros(e.n, int)}), 1)
                    w0, c1, nn = row_consts(Ev(**e.__dict__), phi, w0i[:, 0], c1i[:, 0], blk_out=b)
                    if nn.min() < MIN_C:
                        w0, c1, nn = row_consts(Ev(**e.__dict__), phi, w0i[:, 0], c1i[:, 0], blk_out=-1)
                        nn = nn * 0
                else:
                    w0, c1, nn = None, None, np.zeros(2, int)
                const[b] = (w0, c1, nn)
            pr[f'const_{var}'] = const
        perrow[k] = pr
    # rows without own constants take those of the nearest-in-time row of the unit that has them (same map variant)
    haveconst = [k for k in apply if perrow[k]['ok']]
    for k in apply:
        pr = perrow[k]
        if pr['ok']:
            pr['const_src'] = 'fit'
            continue
        if not haveconst:
            pr['const_src'] = 'none'
            continue
        dk = lambda j: abs(float(df.set_index('key').loc[j, 't_min']) - float(df.set_index('key').loc[k, 't_min']))
        j = min(haveconst, key=dk)
        pr['const_src'] = f'borrowed:{j}'
        for var in [v for v in ('full', 'tilt') if f'const_{v}' in pr]:
            pr[f'const_{var}'] = {b: ((perrow[j].get(f'const_{var}') or perrow[j]['const_full'])[b][0],
                                      (perrow[j].get(f'const_{var}') or perrow[j]['const_full'])[b][1], np.zeros(2, int)) for b in blocks_list}
    # ---- predictions on all gate-table events
    for k in apply:
        pr = perrow[k]
        t, bund = pr['t'], pr['bund']
        if pr['const_src'] == 'none':
            continue
        wx, p0x, wy, p0y = (t[c].values.astype(float) for c in ('x_w', 'x_p0', 'y_w', 'y_p0'))
        bow0 = {kk: vv for kk, vv in (bund.get('bow') or {}).items() if kk != 'phi'}
        out = pd.DataFrame({'ridx': t.ridx.values, 'sel': t.sel.values, 'blk': t.blk.values, 'cens_x': t.cens_x.values, 'cens_y': t.cens_y.values,
                            'x_tan_pass1': t.x_tan_theta.values, 'y_tan_pass1': t.y_tan_theta.values, 'tan_ref_x': t.tan_ref_x.values,
                            'tan_ref_y': t.tan_ref_y.values, 'xm': t.xm.values, 'ym': t.ym.values})
        for var in [v for v in ('full', 'tilt') if f'const_{v}' in pr]:
            sel_ = 0 if var == 'full' else 1
            const = pr[f'const_{var}']
            w0, c1, _ = const[-1]
            tx, ty = L.predict(blk[-1][sel_], float(bund['v_drift']), w0, c1, wx, p0x, wy, p0y, bow0)
            cvx, cvy = np.full(len(t), np.nan), np.full(len(t), np.nan)
            for b in range(R.NB):
                m = t.blk.values == b
                w0b, c1b, _ = const[b]
                a, c = L.predict(blk[b][sel_], float(bund['v_drift']), w0b, c1b, wx[m], p0x[m], wy[m], p0y[m], bow0)
                cvx[m], cvy[m] = a, c
            nm = {'full': ('x_tan_map', 'y_tan_map', 'x_tan_map_cv', 'y_tan_map_cv'), 'tilt': ('x_tan_map_tilt', 'y_tan_map_tilt', 'x_tan_map_cv_tilt', 'y_tan_map_cv_tilt')}[var]
            out[nm[0]], out[nm[1]], out[nm[2]], out[nm[3]] = tx, ty, cvx, cvy
        out['unit'] = unit
        out['map_variant'] = 'full'
        out['const_src'] = pr['const_src']
        out.to_parquet(f'{OUT}/rows/{k}.parquet')
        w0, c1, nn = pr['const_full'][-1]
        info_rows[k] = dict(unit=unit, role=role[k], const_src=pr['const_src'], v_drift=float(bund['v_drift']), n_sel=pr['e'].n,
                            n_const={'x': int(nn[0]), 'y': int(nn[1])},
                            w0={'x': float(w0[0]), 'y': float(w0[1])}, c1={'x': float(c1[0]), 'y': float(c1[1])},
                            kw={'x': float(c1[0] / bund['v_drift']), 'y': float(c1[1] / bund['v_drift'])},
                            old_w0=bund['w0'], old_kw=bund['kw'])
        if 'const_tilt' in pr:
            w0t, c1t, nnt = pr['const_tilt'][-1]
            info_rows[k]['tilt'] = dict(w0={'x': float(w0t[0]), 'y': float(w0t[1])}, kw={'x': float(c1t[0] / bund['v_drift']), 'y': float(c1t[1] / bund['v_drift'])})
    # ---- reproduction check: largest train row, full map, vs the fitter's own spline algebra
    krep = max(train_keys, key=lambda k: perrow[k]['e'].n) if train_keys else None
    repro = None
    if krep is not None:
        pr = perrow[krep]
        e = pr['e']
        meta, fit = fits[-1]
        lay = fit['lay']
        w0, c1, _ = pr['const_full'][-1]
        ee = Ev(**e.__dict__)
        ee.row = np.zeros(ee.n, int)
        U, W, zx, zy = L.uw(ee, np.array(w0).reshape(2, 1), np.array(c1).reshape(2, 1))
        (Ax, Ay), (Sx, Sy) = L.eval_AS(lay, fit['c'], U, W)
        px = (ee.wx - w0[0]) / (c1[0] * (1 + Sx)) - Ax
        py = (ee.wy - w0[1]) / (c1[1] * (1 + Sy)) - Ay
        t = pr['t']
        bund = pr['bund']
        cal = L.make_cal(bund, w0, (c1 / bund['v_drift']), phi=blk[-1][0])
        s_ = t[t.sel].sort_values('trank')
        r = L.reco.bow_pairs(cal, s_.x_w.values, s_.x_p0.values, s_.y_w.values, s_.y_p0.values)
        dx = np.abs(r['tan_x'] - px)
        dy = np.abs(r['tan_y'] - py)
        repro = dict(row=krep, n=int(len(px)), max_abs_diff_x=float(dx.max()), max_abs_diff_y=float(dy.max()),
                     note='fitter spline algebra (Layout.fields @ c) vs wft.reco.bow_pairs with the block installed in the row bundle')
        print(f'[{unit}] reproduction {krep[-40:]}: max |dtan| x {dx.max():.2e} y {dy.max():.2e}', flush=True)
    summ = dict(unit=unit, mounting=mnt, det=det, domain=[list(dom[0]), list(dom[1])], train_rows=train_keys, apply_rows=apply, n_train=int(ev.n),
                folds={str(b): fits[b][0] for b in blocks_list}, rows=info_rows, reproduction=repro, secs=round(time.time() - t00))
    json.dump(summ, open(f'{OUT}/units/{unit}.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f'[{unit}] done in {time.time() - t00:.0f}s', flush=True)
    return summ


if __name__ == '__main__':
    df = pd.read_csv(os.environ.get('PHIMAP_CSV', os.path.join(HERE, 'mountings.csv')))
    # the t_min / t_max columns are in mountings.build() only; recompute from the table
    import mountings as MT
    df['t_min'] = [MT.acq_minutes(a) for a in df.acq0]
    df['t_max'] = df['t_min']
    df['mounting'] = df['mounting'].astype(str)
    df['unit'] = df['unit'].fillna('').astype(str)
    df['donor'] = df['donor'].fillna('').astype(str)
    units = sys.argv[1:]
    allu = [u for u in df.unit.unique() if u]
    if units == ['all'] or not units:
        units = allu
    for u in units:
        process(u, df)
