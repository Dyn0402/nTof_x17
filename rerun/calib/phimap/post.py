#!/usr/bin/env python
"""post.py -- after driver.py: tilt-only decision, censored-track treatment, final columns, summary tables (RESULT.md numbers).

Reads  $JOB/phimap/{mountings.csv, units/*.json, rows/<key>.parquet}  (written by driver.py),
writes rows/<key>.parquet again with the final columns, and  $JOB/phimap/summary.json  (+ copies of the small files into the
repo next to this script).

Final columns per gate-table event (all events of the row, NaN where not computable):
  x_tan_pass1, y_tan_pass1            pass 1 (as in the gate table)
  x_tan_map, y_tan_map                production map (bow_pairs, full-data constants, row's map variant)
  x_tan_map_cv, y_tan_map_cv          5-time-block CV: map AND constants trained without the event's block
  cens_x, cens_y                      the view's q_uend reaches the end of the depth grid (railed)
  x_resc, y_resc, *_resc_cv           C only: rescale factor of railed tracks (1 otherwise)
  x_excl, y_excl                      railed tracks of D and E (and D X at x < 200 mm): exclude from angle statistics
  x_tan_final(_cv), y_tan_final(_cv)  map x resc, NaN where excluded
  map_variant                         'full' | 'tilt' (tilt part only, S = 0) | 'none'
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib as L  # noqa: E402
import rows as R  # noqa: E402

OUT = R.OUTDIR
NBOOT = 400
FAR_FACTOR = 3.0


def boot_s68(res, nb=NBOOT, seed=3):
    """res: dict name -> residual array (same events).  -> {name: (s68, err)}, {name: (diff to first, err)}"""
    rng = np.random.default_rng(seed)
    names = list(res)
    M = np.vstack([res[k] for k in names])
    n = M.shape[1]
    S = np.empty((nb, len(names)))
    for b in range(nb):
        i = rng.integers(0, n, n)
        q = np.percentile(M[:, i], [16, 84], axis=1)
        S[b] = (q[1] - q[0]) / 2
    full = {k: L.s68(res[k]) for k in names}
    return ({k: (full[k], float(S[:, j].std())) for j, k in enumerate(names)},
            {k: (full[k] - full[names[0]], float((S[:, j] - S[:, 0]).std())) for j, k in enumerate(names)})


def masks(d, det):
    ux = d.sel.values & ~d.cens_x.values
    uy = d.sel.values & ~d.cens_y.values
    if det == 'mx17_7':
        ux = ux & (d.xm.values >= 200.0)
    return ux, uy


def load_all():
    df = pd.read_csv(f'{OUT}/mountings.csv')
    df['unit'] = df.unit.fillna('').astype(str)
    df['donor'] = df.donor.fillna('').astype(str)
    reg = R.full()
    frames = {}
    for k in df.key:
        p = f'{OUT}/rows/{k}.parquet'
        if os.path.exists(p):
            frames[k] = pd.read_parquet(p)
    return df.set_index('key', drop=False), frames, reg


def tilt_decision(df, frames):
    """Per unit with tilt_only rows: use the scale part (full block) only if it beats the tilt-only block out of sample on those rows,
    both views, by > 2 sigma pooled; else tilt only.  -> {unit: dict}"""
    dec = {}
    for unit, g in df[df.role == 'tilt_only'].groupby('unit'):
        res = {'x': {}, 'y': {}}
        for k in g.key:
            d = frames.get(k)
            if d is None or 'x_tan_map_cv_tilt' not in d:
                continue
            ux, uy = masks(d, 'mx17_3')
            for v, u in (('x', ux), ('y', uy)):
                tr = d[f'tan_ref_{v}'].values[u]
                for nm, c in (('p1', f'{v}_tan_pass1'), ('full', f'{v}_tan_map_cv'), ('tilt', f'{v}_tan_map_cv_tilt')):
                    res[v].setdefault(nm, []).append(L.dth(d[c].values[u], tr))
        out = {}
        use_full = True
        for v in 'xy':
            r = {nm: np.concatenate(a) for nm, a in res[v].items()}
            if not r:
                continue
            s, dif = boot_s68({'tilt': r['tilt'], 'full': r['full'], 'p1': r['p1']})
            out[v] = dict(n=int(len(r['p1'])), p1=s['p1'], tilt=s['tilt'], full=s['full'], full_minus_tilt=dif['full'][0], err=dif['full'][1])
            use_full &= (dif['full'][0] < -2 * dif['full'][1])
        out['use_full'] = bool(use_full)
        dec[unit] = out
    return dec


def finalize(df, frames, dec):
    """add resc / excl / final columns; write back."""
    reg = R.full()
    # ---- C: rescale of railed tracks
    crows = [k for k in df.key if df.loc[k, 'chamber'] == 'C' and k in frames]
    slopes = {}
    pool = {}
    for v in 'xy':
        for kind in ('prod', 'cv'):
            pool[(v, kind)] = []
    for k in crows:
        d = frames[k]
        for v in 'xy':
            m = d.sel.values & d[f'cens_{v}'].values
            tr = d[f'tan_ref_{v}'].values
            pool[(v, 'prod')].append((tr[m], d[f'{v}_tan_map'].values[m], d.blk.values[m]))
            pool[(v, 'cv')].append((tr[m], d[f'{v}_tan_map_cv'].values[m], d.blk.values[m]))

    def slope_of(tr, t):
        return float((tr * t).sum() / (tr * tr).sum()) if len(tr) > 20 else np.nan

    def slope_ic(tr, t):
        if len(tr) < 20:
            return (np.nan, np.nan)
        A = np.vstack([np.ones(len(tr)), tr]).T
        c = np.linalg.lstsq(A, t, rcond=None)[0]
        return (float(c[1]), float(c[0]))

    cinfo = {}
    for k in crows:
        d = frames[k]
        for v in 'xy':
            m = d.sel.values & d[f'cens_{v}'].values
            n = int(m.sum())
            tr = d[f'tan_ref_{v}'].values
            use_pool = n < 150
            tp = np.concatenate([a[0] for a in pool[(v, 'prod')]])
            tm = np.concatenate([a[1] for a in pool[(v, 'prod')]])
            if use_pool:
                s_prod = slope_of(tp, tm)
            else:
                s_prod = slope_of(tr[m], d[f'{v}_tan_map'].values[m])
            f_prod = 1.0 / s_prod
            f_cv = np.ones(len(d))
            for b in range(R.NB):
                if use_pool:
                    sel_ = [(a[0][a[2] != b], a[1][a[2] != b]) for a in pool[(v, 'cv')]]
                    s_b = slope_of(np.concatenate([a[0] for a in sel_]), np.concatenate([a[1] for a in sel_]))
                else:
                    mm = m & (d.blk.values != b)
                    s_b = slope_of(tr[mm], d[f'{v}_tan_map_cv'].values[mm])
                f_cv[d.blk.values == b] = 1.0 / s_b
            ic = slope_ic(tr[m], d[f'{v}_tan_map'].values[m]) if n >= 150 else (np.nan, np.nan)
            cens = d[f'cens_{v}'].values
            d[f'{v}_resc'] = np.where(cens, f_prod, 1.0)
            d[f'{v}_resc_cv'] = np.where(cens, f_cv, 1.0)
            cinfo[(k, v)] = dict(n_railed=n, pooled=bool(use_pool), slope=s_prod, factor=f_prod, slope_with_intercept=ic[0], intercept=ic[1])
    # ---- all rows: excl, final
    for k, d in frames.items():
        det = reg.loc[k, 'det']
        ch = df.loc[k, 'chamber']
        for v in 'xy':
            if f'{v}_resc' not in d:
                d[f'{v}_resc'] = 1.0
                d[f'{v}_resc_cv'] = 1.0
            cens = d[f'cens_{v}'].values
            excl = cens.copy() if ch in ('D', 'E') else np.zeros(len(d), bool)
            if det == 'mx17_7' and v == 'x':
                excl = excl | ~(d.xm.values >= 200.0)
            d[f'{v}_excl'] = excl
        # the row's map variant
        role = df.loc[k, 'role']
        unit = df.loc[k, 'unit']
        var = 'full'
        if role == 'tilt_only' and unit in dec and not dec[unit]['use_full']:
            var = 'tilt'
        if role == 'tilt_only':
            for v in 'xy':
                if var == 'tilt':
                    d[f'{v}_tan_map_full_alt'] = d[f'{v}_tan_map']
                    d[f'{v}_tan_map_cv_full_alt'] = d[f'{v}_tan_map_cv']
                    d[f'{v}_tan_map'] = d[f'{v}_tan_map_tilt']
                    d[f'{v}_tan_map_cv'] = d[f'{v}_tan_map_cv_tilt']
        d['map_variant'] = var
        ix, iy = R.active(det)
        d['interior'] = ((d.xm > ix[0]) & (d.xm < ix[1]) & (d.ym > iy[0]) & (d.ym < iy[1])).values
        for v in 'xy':
            d[f'{v}_tan_final'] = np.where(d[f'{v}_excl'], np.nan, d[f'{v}_tan_map'] * d[f'{v}_resc'])
            d[f'{v}_tan_final_cv'] = np.where(d[f'{v}_excl'], np.nan, d[f'{v}_tan_map_cv'] * d[f'{v}_resc_cv'])
        d.to_parquet(f'{OUT}/final/{k}.parquet')
    return cinfo


def agg(frames, keys, df, det, nb=NBOOT, what=('p1', 'map', 'cv')):
    """pooled residuals over rows: per view: pass 1 / production / cv on the unrailed (and D x >= 200) selected events"""
    out = {}
    for v in 'xy':
        res = {'p1': [], 'map': [], 'cv': [], 'final_cv': []}
        far = {'p1': 0, 'cv': 0}
        for k in keys:
            d = frames.get(k)
            if d is None:
                continue
            u = masks(d, det)[0 if v == 'x' else 1]
            tr = d[f'tan_ref_{v}'].values[u]
            res['p1'].append(L.dth(d[f'{v}_tan_pass1'].values[u], tr))
            res['map'].append(L.dth(d[f'{v}_tan_map'].values[u], tr))
            res['cv'].append(L.dth(d[f'{v}_tan_map_cv'].values[u], tr))
        if not res['p1']:
            continue
        r = {a: np.concatenate(b) for a, b in res.items() if b}
        ok = np.isfinite(r['cv']) & np.isfinite(r['p1'])
        r = {a: b[ok] for a, b in r.items()}
        if len(r['p1']) < 30:
            continue
        s, dif = boot_s68({'p1': r['p1'], 'cv': r['cv'], 'map': r['map']}, nb=nb)
        lim = FAR_FACTOR * s['p1'][0]
        out[v] = dict(n=int(len(r['p1'])), p1=s['p1'][0], p1_err=s['p1'][1], cv=s['cv'][0], cv_err=s['cv'][1], map=s['map'][0], map_err=s['map'][1],
                      diff_cv=dif['cv'][0], diff_cv_err=dif['cv'][1], gain_cv=-dif['cv'][0] / s['p1'][0],
                      far_p1=float((np.abs(r['p1']) > lim).mean()), far_cv=float((np.abs(r['cv']) > lim).mean()))
    return out


def constants_and_nomap(df, frames, dec):
    """row_constants.json (per row: unit, block file, variant, w0 / kw per view with the map applied) + parquet of the nomap rows."""
    units = {}
    for u in df.unit.unique():
        p = f'{OUT}/units/{u}.json'
        if u and os.path.exists(p):
            units[u] = json.load(open(p))
    out = {}
    for k in df.key:
        r = df.loc[k]
        if k in frames:
            d = frames[k]
            u = d['unit'].iloc[0]
            var = d['map_variant'].iloc[0] if 'map_variant' in d else 'full'
            info = units[u]['rows'][k]
            c = info if var == 'full' else info['tilt']
            out[k] = dict(unit=u, role=r.role, chamber=r.chamber, mounting=r.mounting, map_variant=var, block=f'blocks/{u}' + ('_tilt' if var == 'tilt' else '') + '.json',
                          w0=c['w0'], kw=c['kw'], v_drift=info['v_drift'], n_const=info['n_const'], const_src=info['const_src'],
                          map_borrowed=bool(r.role == 'borrowed'), map_trained_here=bool(k in units[u]['train_rows']), n_gate=int(r.n_gate),
                          cens_x=float(r.cens_x), cens_y=float(r.cens_y), ntracks_unavailable=bool(r.ntracks_unavailable),
                          x_view_valid_from_mm=(200.0 if r.chamber == 'D' else None),
                          map_recommended=bool(not (r.mounting == 'A_0622')), note=str(r.note) if isinstance(r.note, str) else '')
        else:
            t = R.load_full(k)
            d = pd.DataFrame({'ridx': t.ridx.values, 'sel': t.sel.values, 'blk': t.blk.values, 'cens_x': t.cens_x.values, 'cens_y': t.cens_y.values,
                              'x_tan_pass1': t.x_tan_theta.values, 'y_tan_pass1': t.y_tan_theta.values, 'tan_ref_x': t.tan_ref_x.values,
                              'tan_ref_y': t.tan_ref_y.values, 'xm': t.xm.values, 'ym': t.ym.values})
            for c_ in ('x_tan_map', 'y_tan_map', 'x_tan_map_cv', 'y_tan_map_cv', 'x_tan_final', 'y_tan_final', 'x_tan_final_cv', 'y_tan_final_cv'):
                d[c_] = np.nan
            d['unit'] = ''
            d['map_variant'] = 'none'
            d.to_parquet(f'{OUT}/final/{k}.parquet')
            out[k] = dict(unit='', role=r.role, chamber=r.chamber, mounting=r.mounting, map_variant='none', block=None, n_gate=int(r.n_gate),
                          note=str(r.note) if isinstance(r.note, str) and r.note else 'row has too few gate events for its own constants: stays on the bow block',
                          ntracks_unavailable=bool(r.ntracks_unavailable))
    json.dump(out, open(f'{OUT}/row_constants.json', 'w'), indent=1, default=float)
    return out


def main():
    df, frames, reg = load_all()
    dec = tilt_decision(df, frames)
    os.makedirs(f'{OUT}/final', exist_ok=True)
    cinfo = finalize(df, frames, dec)
    constants_and_nomap(df, frames, dec)
    summ = dict(tilt=dec, censor={f'{k[0]}|{k[1]}': v for k, v in cinfo.items()})
    json.dump(summ, open(f'{OUT}/summary_post.json', 'w'), indent=1, default=float)
    print(json.dumps(dec, indent=1, default=float))
    print('C rescale:')
    for (k, v), c in cinfo.items():
        print(' ', k[-50:], v, c)


if __name__ == '__main__':
    main()
