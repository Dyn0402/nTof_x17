#!/usr/bin/env python
"""report.py -- RESULT.md and result.json from driver.py + post.py outputs (numbers only; the discussion is written by hand)."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib as L  # noqa: E402
import rows as R  # noqa: E402
import post as P  # noqa: E402

OUT = R.OUTDIR
# r3 (time-block CV, all events incl. railed, constants refit on the test block) targets: (s68, err) per view
R3 = {
    ('A_0627', 'long'): {'x': (0.688, 0.005), 'y': (0.599, 0.005)},
    ('A_0625', 'all'): {'x': (0.681, 0.010), 'y': (0.666, 0.008)},
    ('B_0622', 'long'): {'x': (0.675, 0.006), 'y': (0.735, 0.006)},
    ('D_0626', 'long'): {'x': (0.802, 0.019), 'y': (0.829, 0.013)},
    ('E', '6-23'): {'x': (2.354, 0.077), 'y': (1.377, 0.051)},
    ('E', '6-24'): {'x': (2.120, 0.048), 'y': (1.324, 0.042)},
    ('C_long', 'all'): {'x': (0.932, 0.010), 'y': (1.228, 0.014)},   # r3: the three C rows together, time5
}
R3_P1 = {('A_0627', 'long'): {'x': 0.923, 'y': 0.805}, ('A_0625', 'all'): {'x': 0.876, 'y': 1.030}, ('B_0622', 'long'): {'x': 0.961, 'y': 0.966},
         ('D_0626', 'long'): {'x': 0.968, 'y': 1.028}, ('E', '6-23'): {'x': 2.405, 'y': 1.627}, ('E', '6-24'): {'x': 2.141, 'y': 1.436}}


def load():
    df = pd.read_csv(f'{OUT}/mountings.csv')
    df['unit'] = df.unit.fillna('').astype(str)
    df['donor'] = df.donor.fillna('').astype(str)
    df['note'] = df.note.fillna('').astype(str)
    df = df.set_index('key', drop=False)
    fr = {}
    for k in df.key:
        p = f'{OUT}/final/{k}.parquet'
        if os.path.exists(p):
            fr[k] = pd.read_parquet(p)
    return df, fr


def pooled(fr, keys, det, censor='excl', which=('p1', 'cv', 'map', 'final_cv'), nb=400):
    """per view dict of s68 +- err for pass 1, CV, production; censor 'excl' (unrailed events only) or 'incl' (all selected events,
    CV rescaled for C: final_cv)"""
    out = {}
    for v in 'xy':
        rs = {'p1': [], 'cv': [], 'map': [], 'final_cv': []}
        for k in keys:
            d = fr.get(k)
            if d is None:
                continue
            if censor == 'excl':
                u = P.masks(d, det)[0 if v == 'x' else 1]
            else:
                u = d.sel.values.copy()
                if det == 'mx17_7' and v == 'x':
                    u &= d.xm.values >= 200.0
            tr = d[f'tan_ref_{v}'].values[u]
            rs['p1'].append(L.dth(d[f'{v}_tan_pass1'].values[u], tr))
            rs['cv'].append(L.dth(d[f'{v}_tan_map_cv'].values[u], tr))
            rs['map'].append(L.dth(d[f'{v}_tan_map'].values[u], tr))
            resc = d[f'{v}_resc_cv'].values if f'{v}_resc_cv' in d else 1.0
            rs['final_cv'].append(L.dth((d[f'{v}_tan_map_cv'].values * resc)[u], tr))
        if not rs['p1']:
            continue
        r = {a: np.concatenate(b) for a, b in rs.items()}
        ok = np.isfinite(r['cv']) & np.isfinite(r['p1'])
        r = {a: b[ok] for a, b in r.items()}
        if len(r['p1']) < 30:
            continue
        s, dif = P.boot_s68({'p1': r['p1'], 'cv': r['cv'], 'map': r['map'], 'final_cv': r['final_cv']}, nb=nb)
        lim = P.FAR_FACTOR * s['p1'][0]
        out[v] = dict(n=int(len(r['p1'])), p1=s['p1'][0], p1_err=s['p1'][1], cv=s['cv'][0], cv_err=s['cv'][1], map=s['map'][0], map_err=s['map'][1],
                      final_cv=s['final_cv'][0], final_cv_err=s['final_cv'][1],
                      diff=dif['cv'][0], diff_err=dif['cv'][1], gain=-dif['cv'][0] / s['p1'][0], gain_final=(s['p1'][0] - s['final_cv'][0]) / s['p1'][0],
                      far_p1=float((np.abs(r['p1']) > lim).mean()), far_cv=float((np.abs(r['cv']) > lim).mean()))
    return out


def position_check(df, fr, keys, det):
    """mid-gap position (p0 + 14 tan) minus the reference (ref_mesh + 14 tan_ref), s68 [mm] pass 1 vs CV map (unrailed events)."""
    res = {}
    for v in 'xy':
        a, b = [], []
        for k in keys:
            d = fr.get(k)
            if d is None:
                continue
            t = R.load_full(k)
            u = P.masks(d, det)[0 if v == 'x' else 1]
            p0 = t[f'{v}_p0'].values[u]
            ref = t[f'ref_mesh_{v}'].values[u] + 14 * t[f'tan_ref_{v}'].values[u]
            p1 = p0 + 14 * d[f'{v}_tan_pass1'].values[u] - ref
            cv = p0 + 14 * d[f'{v}_tan_map_cv'].values[u] - ref
            rowoff = np.median(p1)
            a.append(p1 - rowoff)
            b.append(cv - np.median(cv))
        if a:
            a, b = np.concatenate(a), np.concatenate(b)
            ok = np.isfinite(a) & np.isfinite(b)
            res[v] = dict(p1=L.s68(a[ok]), cv=L.s68(b[ok]), n=int(ok.sum()))
    return res


def edge_check(df, fr, keys, det, edge=30.0):
    """events inside the active area but within `edge` mm of its border (not in any fit): production map vs pass 1 (unrailed)."""
    res = {}
    for v in 'xy':
        a, b = [], []
        for k in keys:
            d = fr.get(k)
            if d is None:
                continue
            t = R.load_full(k)
            ix, iy = R.active(R.full().loc[k].det, margin=0.0)
            inside = (t.xm > ix[0]) & (t.xm < ix[1]) & (t.ym > iy[0]) & (t.ym < iy[1])
            m = t.gate.values & ~t.sel.values & inside.values & ~d[f'cens_{v}'].values
            if det == 'mx17_7' and v == 'x':
                m &= t.xm.values >= 200.0
            tr = d[f'tan_ref_{v}'].values[m]
            a.append(L.dth(d[f'{v}_tan_pass1'].values[m], tr))
            b.append(L.dth(d[f'{v}_tan_map'].values[m], tr))
        if a:
            a, b = np.concatenate(a), np.concatenate(b)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 100:
                s, dif = P.boot_s68({'p1': a[ok], 'map': b[ok]}, nb=200)
                res[v] = dict(n=int(ok.sum()), p1=s['p1'][0], p1_err=s['p1'][1], map=s['map'][0], map_err=s['map'][1], diff=dif['map'][0], diff_err=dif['map'][1])
    return res


def tilt_fields(df, fr):
    """per drift field of the tilt_only rows: pass 1, tilt-only CV and full-map CV (both variants are in the final parquet when the
    decision was 'tilt', otherwise the full is x_tan_map_cv and the tilt is x_tan_map_cv_tilt)."""
    out = []
    g = df[df.role == 'tilt_only']
    for V, gg in g.groupby('drift_V'):
        rec = dict(drift_V=int(V), rows=len(gg), n=0)
        for v in 'xy':
            a = {'p1': [], 'full': [], 'tilt': []}
            for k in gg.key:
                d = fr.get(k)
                if d is None:
                    continue
                u = P.masks(d, 'mx17_3')[0 if v == 'x' else 1]
                tr = d[f'tan_ref_{v}'].values[u]
                full_col = f'{v}_tan_map_cv_full_alt' if f'{v}_tan_map_cv_full_alt' in d else f'{v}_tan_map_cv'
                tilt_col = f'{v}_tan_map_cv_tilt'
                a['p1'].append(L.dth(d[f'{v}_tan_pass1'].values[u], tr))
                a['full'].append(L.dth(d[full_col].values[u], tr))
                a['tilt'].append(L.dth(d[tilt_col].values[u], tr))
            if a['p1']:
                a = {q: np.concatenate(w) for q, w in a.items()}
                rec['n'] = int(len(a['p1']))
                rec[f'{v}_p1'], rec[f'{v}_tilt'], rec[f'{v}_full'] = L.s68(a['p1']), L.s68(a['tilt']), L.s68(a['full'])
        out.append(rec)
    return out


def bias_C(df, fr):
    """C railed tracks: mean of (angle residual x sign(tan_ref)) [deg] and s68 for pass 1, map CV, rescaled map CV (|tan_ref| > 0.05)."""
    out = {}
    for v in 'xy':
        r = {'p1': [], 'cv': [], 'resc': [], 'n': 0, 'cv_u': [], 'p1_u': []}
        for k in df[df.chamber == 'C'].key:
            d = fr.get(k)
            if d is None:
                continue
            m = d.sel.values & d[f'cens_{v}'].values & (np.abs(d[f'tan_ref_{v}'].values) > 0.05)
            tr = d[f'tan_ref_{v}'].values[m]
            sg = np.sign(tr)
            r['p1'].append(L.dth(d[f'{v}_tan_pass1'].values[m], tr) * sg)
            r['cv'].append(L.dth(d[f'{v}_tan_map_cv'].values[m], tr) * sg)
            r['resc'].append(L.dth((d[f'{v}_tan_map_cv'].values * d[f'{v}_resc_cv'].values)[m], tr) * sg)
        r = {q: np.concatenate(w) for q, w in r.items() if isinstance(w, list) and w}
        out[v] = dict(n=int(len(r['p1'])), **{f'{q}_mean': float(np.nanmean(w)) for q, w in r.items()}, **{f'{q}_s68': L.s68(w) for q, w in r.items()})
    return out


def diag_E():
    """E with one map per day (phimap_diag/, driver.py with make_split_csv.py), unrailed events of each day's train rows."""
    base = os.path.join(os.path.dirname(OUT), 'phimap_diag')
    out = {}
    for u in ('E_0623', 'E_0624'):
        p = f'{base}/units/{u}.json'
        if not os.path.exists(p):
            continue
        uj = json.load(open(p))
        fr = {k: pd.read_parquet(f'{base}/rows/{k}.parquet') for k in uj['train_rows']}
        out[u] = dict(pooled(fr, uj['train_rows'], 'mx17_4'), n_train=uj['n_train'],
                      sel={b: (m['K'], m['KS'], m['lam']) for b, m in uj['folds'].items()})
    return out


def main():
    df, fr = load()
    reg = R.full()
    units = {}
    for u in sorted(df.unit.unique()):
        if u and os.path.exists(f'{OUT}/units/{u}.json'):
            units[u] = json.load(open(f'{OUT}/units/{u}.json'))
    post = json.load(open(f'{OUT}/summary_post.json'))
    res = dict(units={}, rows={}, tilt=post['tilt'], censor=post['censor'], tilt_fields=tilt_fields(df, fr), bias_C=bias_C(df, fr), diag_E=diag_E())
    # ---------------- per unit
    for u, uj in units.items():
        det = uj['det']
        keys_train = uj['train_rows']
        r = dict(mounting=uj['mounting'], n_train=uj['n_train'], train_rows=len(keys_train), folds={b: {k: v for k, v in m.items() if k != 'info'} | m['info'] for b, m in uj['folds'].items()})
        r['train'] = pooled(fr, keys_train, det)
        r['train_incl_railed'] = pooled(fr, keys_train, det, censor='incl')
        # subsets used by r3
        sub = df.loc[keys_train]
        long = [k for k in keys_train if df.loc[k, 'subrun'].startswith(('long_run', 'longer_run', 'short_run'))]
        if long and len(long) != len(keys_train):
            r['long_rows'] = pooled(fr, long, det)
            r['long_rows_incl_railed'] = pooled(fr, long, det, censor='incl')
        if u == 'E':
            for tag, pat in (('6-23', '6-23'), ('6-24', '6-24')):
                ks = [k for k in keys_train if pat in df.loc[k, 'run']]
                r[f'E_{tag}'] = pooled(fr, ks, det)
                r[f'E_{tag}_incl_railed'] = pooled(fr, ks, det, censor='incl')
        main_keys = long if long else keys_train
        r['position'] = position_check(df, fr, main_keys, det)
        r['edge'] = edge_check(df, fr, main_keys, det)
        # other rows of the unit
        for role in ('tilt_only', 'borrowed', 'tiny'):
            ks = [k for k in uj['apply_rows'] if df.loc[k, 'role'] == role]
            if ks:
                r[role] = pooled(fr, ks, det)
        res['units'][u] = r
    # ---------------- per row
    for k in df.key:
        if k not in fr:
            continue
        det = reg.loc[k, 'det']
        res['rows'][k] = dict(unit=df.loc[k, 'unit'], role=df.loc[k, 'role'], n_gate=int(df.loc[k, 'n_gate']), stats=pooled(fr, [k], det, nb=100))
    # ---------------- C: the three r3 rows together (own map or borrowed), railed tracks included, with and without the rescale
    c3 = [k for k in df.key if df.loc[k, 'chamber'] == 'C' and df.loc[k, 'subrun'] in ('long_run', 'longer_run', 'short_run')]
    res['C3'] = dict(rows=c3, **{'incl': pooled(fr, c3, 'mx17_6', censor='incl'), 'excl': pooled(fr, c3, 'mx17_6')})
    # ---------------- recommendations per unit and view, edge policy; merged into row_constants.json
    rec = {}
    for u, r in res['units'].items():
        rec[u] = {}
        for v in 'xy':
            t = r['train'].get(v)
            e = r['edge'].get(v)
            rec[u][v] = dict(map_recommended=bool(t and t['diff'] < -2 * t['diff_err']),
                             edge_worse=bool(e and e['diff'] > 2 * e['diff_err']))
        if u == 'A_0622':
            for v in 'xy':
                rec[u][v]['map_recommended'] = False
    res['recommend'] = rec
    json.dump(res, open(f'{OUT}/result.json', 'w'), indent=1, default=float)
    rc = json.load(open(f'{OUT}/row_constants.json'))
    for k, c in rc.items():
        u = c.get('unit')
        if u in rec:
            for v in 'xy':
                ok = rec[u][v]['map_recommended']
                if df.loc[k, 'role'] == 'tilt_only' and df.loc[k, 'drift_V'] <= 300:
                    ok = False
                c[f'{v}_map_recommended'] = bool(ok)
                c[f'{v}_edge_use_pass1'] = bool(rec[u][v]['edge_worse'])
        else:
            c['x_map_recommended'] = c['y_map_recommended'] = False
    json.dump(rc, open(f'{OUT}/row_constants.json', 'w'), indent=1, default=float)
    import mdreport
    open(f'{OUT}/RESULT_tables.md', 'w').write(mdreport.write_md(df, fr, res))
    return df, fr, res


if __name__ == '__main__':
    main()
