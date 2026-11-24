#!/usr/bin/env python
"""gates.py -- bench re-finalization step 4: the gates of the interior-only maps against pass 1.

Reads  $JOB/map1/<row>/events.parquet (build_map1.py), $JOB/phimap/final/<row>.parquet (step 3: the cross-validated columns),
       the pass-1 gate tables.     Writes  $JOB/map1/gates.json  (every number of the report).

Population per view: the gate selection of step 3 (``sel``: single track, both views ok, M3 chi2 < 1, >= 4 clusters per view, reference
position in the 30 mm interior, |tan_ref| <= 0.6), without the railed tracks of D and E (flagged, not changed; as step 3).  C railed
tracks are in, with the per-row rescale.  Three angle values per event:
  pass 1      x_tan_pass1
  map (prod)  the shipped table column x_tan_theta (production map on its own training events, in sample; pass 1 where the map is
              not used)
  map (CV)    step 3's cross-validated value (x_tan_final_cv: map and constants trained without the event's time block, railed
              rescale of C) where the map is used, pass 1 elsewhere.  THE QUOTED NUMBERS.
Statistics: s68 = half the 16-84 % range of atan(tan) - atan(tan_ref) [deg]; far fraction = share with |residual| > 3 s68(pass 1);
mid-gap position residual = (p0 + 14 mm tan) - (ref_mesh + 14 mm tan_ref) [mm]; the mesh position p0 is not touched by the map.
Errors: event bootstrap (400), the paired difference (new - pass 1) is resampled together.
Edge zone: gate events (no interior condition) inside the active area but within 30 mm of its border.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import rows as R  # noqa: E402
from common import mx17_active_area as AA  # noqa: E402

JOB = R.JOB
OUT = os.environ.get('MAP1_OUT', f'{JOB}/map1')
STEP3 = f'{JOB}/phimap'
NBOOT = 400
MID = R.MID
FAR = 3.0
MIN_ROW = 2000          # rows with at least this many gate events get their own line
MIN_FLAG = 300          # rows with fewer events are only pooled


def s68(a):
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return np.nan
    q = np.percentile(a, [16, 84])
    return float((q[1] - q[0]) / 2)


def dth(t, tr):
    return np.degrees(np.arctan(t) - np.arctan(tr))


def group_of(m, rc):
    """pooling group of a row: its map unit, or the mounting when it has none (drift-scan rows of A 6-27 at 100/300 V apart)."""
    if rc.get('unit') and rc.get('nomap') is None:
        return rc['unit']
    if m.unit == 'A_0627':
        return 'A_0627 (100/300 V, no map)'
    return m.mounting


def load_row(k):
    mt = pd.read_csv(f'{STEP3}/mountings.csv').set_index('key', drop=False) if not hasattr(load_row, 'mt') else load_row.mt
    load_row.mt = mt
    r = R.full().loc[k]
    m = mt.loc[k]
    t = R.load_full(k)                       # gate table (index event_id), COLS, sel, gate, xm, ym, blk, cens_x/y
    f3 = pd.read_parquet(f'{STEP3}/final/{k}.parquet')
    assert (f3.ridx.values == t.ridx.values).all()
    od = f'{OUT}/{r.run}/{r.subrun}/{r.det}'
    e = pd.read_parquet(f'{od}/events.parquet')
    key = lambda ev, tag, ts: pd.MultiIndex.from_arrays([np.asarray(ev), np.asarray(tag).astype(str), np.asarray(ts).astype('int64')])
    ek = key(e.event_id.values, e.acq_tag.values, e.timestamp.values)
    ge = ek.get_indexer(key(t.index.values, t.acq_tag.values, t.timestamp.values))
    assert (ge >= 0).all()
    info = json.load(open(f'{od}/map1_row.json'))
    ix, iy = R.active(r.det)
    ax, ay = R.active(r.det, margin=0.0)
    edge = (t.gate.values & ~((t.xm > ix[0]) & (t.xm < ix[1]) & (t.ym > iy[0]) & (t.ym < iy[1])).values
            & ((t.xm > ax[0]) & (t.xm < ax[1]) & (t.ym > ay[0]) & (t.ym < ay[1])).values)
    D = dict(key=k, chamber=m.chamber, group=None, role=m.role, mounting=m.mounting, n_sel=int(t.sel.sum()), n_gate=int(m.n_gate))
    for v in 'xy':
        used = e[f'map_used_{v}'].values[ge]
        p1 = t[f'{v}_tan_theta'].values
        assert np.array_equal(p1, e[f'{v}_tan_pass1'].values[ge], equal_nan=True)
        new = e[f'{v}_tan_theta'].values[ge]
        cvv = np.where(used, f3[f'{v}_tan_final_cv'].values, p1)
        # D X for x < 200 mm and D/E railed: step 3 flags them; here D X x < 200 is simply outside the map's region (pass 1)
        railed = e[f'{v}_railed'].values[ge]
        exclude = railed if m.chamber in ('D', 'E') else np.zeros(len(t), bool)
        tr = t[f'tan_ref_{v}'].values
        D[v] = dict(sel=t.sel.values & ~exclude, edge=edge & ~exclude, used=used, railed=railed, p1=p1, new=new, cv=cvv, tr=tr,
                    p0=t[f'{v}_p0'].values, ref=t[f'ref_mesh_{v}'].values, resc=e[f'{v}_resc'].values[ge], blk=t.blk.values)
    D['info'] = info
    return D


def cat(parts, v, field):
    return np.concatenate([p[v][field] for p in parts])


def boot(old, new, cv, far_lim, nb=NBOOT, seed=11):
    """s68 of three residual vectors with bootstrap errors, paired differences, far fractions."""
    n = len(old)
    rng = np.random.default_rng(seed)
    st = lambda a: s68(a)
    base = dict(n=int(n), p1=st(old), new=st(new), cv=st(cv),
                far_p1=float((np.abs(old) > far_lim).mean()), far_new=float((np.abs(new) > far_lim).mean()), far_cv=float((np.abs(cv) > far_lim).mean()))
    if n < 30:
        return base
    B = {k: [] for k in ('p1', 'new', 'cv', 'dnew', 'dcv', 'fdcv')}
    for _ in range(nb):
        i = rng.integers(0, n, n)
        a, b, c = old[i], new[i], cv[i]
        sa, sb, sc = st(a), st(b), st(c)
        B['p1'].append(sa)
        B['new'].append(sb)
        B['cv'].append(sc)
        B['dnew'].append(sb - sa)
        B['dcv'].append(sc - sa)
        B['fdcv'].append((np.abs(c) > far_lim).mean() - (np.abs(a) > far_lim).mean())
    for k, v in B.items():
        base[k + '_err'] = float(np.std(v))
    base['diff_new'] = base['new'] - base['p1']
    base['diff_cv'] = base['cv'] - base['p1']
    base['far_diff_cv'] = base['far_cv'] - base['far_p1']
    return base


def view_stats(parts, v, which='sel', subset=None):
    """pooled angle / position statistics of ``which`` events (sel | edge) of the parts, view v; subset: only map-used events."""
    def g(f):
        return np.concatenate([p[v][f][p[v][which] & (p[v]['used'] if subset == 'used' else True)] for p in parts])
    tr, p1, new, cv, p0, ref = g('tr'), g('p1'), g('new'), g('cv'), g('p0'), g('ref')
    ok = np.isfinite(tr) & np.isfinite(p1) & np.isfinite(new) & np.isfinite(cv)
    tr, p1, new, cv, p0, ref = (a[ok] for a in (tr, p1, new, cv, p0, ref))
    if len(tr) < 5:
        return dict(n=int(len(tr)))
    old_r, new_r, cv_r = dth(p1, tr), dth(new, tr), dth(cv, tr)
    lim = FAR * s68(old_r)
    a = boot(old_r, new_r, cv_r, lim)
    pos = lambda t: (p0 + MID * t) - (ref + MID * tr)
    pp1, pnew, pcv = pos(p1), pos(new), pos(cv)
    b = boot(pp1, pnew, pcv, FAR * s68(pp1), seed=12)
    a['pos'] = {k: b[k] for k in b if not k.startswith('far')}
    a['mesh_s68'] = s68(p0 - ref)
    a['frac_used'] = float(np.mean(np.concatenate([p[v]['used'][p[v][which]] for p in parts]))) if which else None
    # largest |new - pass 1| where the map is not used (must be 0)
    nu = np.concatenate([(~p[v]['used'] & p[v][which]) for p in parts])
    allp1, allnew = np.concatenate([p[v]['p1'] for p in parts]), np.concatenate([p[v]['new'] for p in parts])
    dd = np.abs(allnew - allp1)[nu & np.isfinite(allp1) & np.isfinite(allnew)]
    a['max_diff_unused'] = float(dd.max()) if len(dd) else 0.0
    return a


def row_task(k):
    D = load_row(k)
    out = dict(key=k, chamber=D['chamber'], mounting=D['mounting'], role=D['role'], n_sel=D['n_sel'], info=D['info'])
    for v in 'xy':
        out[v] = view_stats([D], v, 'sel')
        out['edge_' + v] = view_stats([D], v, 'edge')
    return out


def group_task(args):
    name, keys = args
    parts = []
    for k in keys:
        D = load_row(k)
        parts.append(D)
    out = dict(group=name, rows=len(keys), keys=keys)
    for v in 'xy':
        out[v] = view_stats(parts, v, 'sel')
        out[v + '_used'] = view_stats(parts, v, 'sel', subset='used')
        out['edge_' + v] = view_stats(parts, v, 'edge')
        # railed tracks of C: the rescale
        if parts[0]['chamber'] == 'C':
            rail = []
            for p in parts:
                q = p[v]
                mm = q['sel'] & q['railed'] & np.isfinite(q['p1']) & np.isfinite(q['cv'])
                sg = np.sign(q['tr'][mm])
                rail.append((dth(q['p1'][mm], q['tr'][mm]) * sg, dth(q['cv'][mm], q['tr'][mm]) * sg))
            a, b = np.concatenate([x[0] for x in rail]), np.concatenate([x[1] for x in rail])
            out['railed_' + v] = dict(n=int(len(a)), mean_p1=float(a.mean()), mean_cv=float(b.mean()), s68_p1=s68(a), s68_cv=s68(b))
        # large-angle subsets: |tan_ref| bins
        # (angle bins for the profile figure)
        tr = np.concatenate([p[v]['tr'][p[v]['sel']] for p in parts])
        p1 = np.concatenate([p[v]['p1'][p[v]['sel']] for p in parts])
        cv = np.concatenate([p[v]['cv'][p[v]['sel']] for p in parts])
        ok = np.isfinite(tr) & np.isfinite(p1) & np.isfinite(cv)
        bins = np.linspace(-0.5, 0.5, 11)
        prof = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = ok & (tr >= lo) & (tr < hi)
            prof.append([float(lo), float(hi), int(mm.sum()), float(np.mean(dth(p1[mm], tr[mm]))) if mm.sum() > 20 else None,
                         float(np.mean(dth(cv[mm], tr[mm]))) if mm.sum() > 20 else None])
        out['profile_' + v] = prof
    return out


def main():
    mt = pd.read_csv(f'{STEP3}/mountings.csv').set_index('key', drop=False)
    rcs = json.load(open(f'{STEP3}/row_constants.json'))
    reg = R.reg()
    groups = {}
    nm = {k: json.load(open(f"{OUT}/{reg.loc[k].run}/{reg.loc[k].subrun}/{reg.loc[k].det}/map1_row.json"))['nomap'] for k in reg.key}
    for k in reg.key:
        m = mt.loc[k]
        rc = dict(rcs[k], nomap=nm[k])
        g = group_of(m, rc)
        if m.chamber == 'C':
            g = {'C_long': 'C_long (own map)', 'C_longer': 'C_longer (own map)'}.get(rc.get('unit'), g) if rc.get('role') == 'train' else 'C borrowed (' + str(rc.get('unit')) + ')'
        groups.setdefault(g, []).append(k)
    big = [k for k in reg.key if int(mt.loc[k, 'n_gate']) >= MIN_FLAG]
    print('rows with >= %d gate events: %d; groups: %s' % (MIN_FLAG, len(big), {g: len(v) for g, v in groups.items()}), flush=True)
    with Pool(14) as pool:
        rows = pool.map(row_task, big, chunksize=1)
        grp = pool.map(group_task, list(groups.items()), chunksize=1)
    allrows = {k: dict(chamber=mt.loc[k, 'chamber'], n_gate=int(mt.loc[k, 'n_gate']), mounting=mt.loc[k, 'mounting'], role=mt.loc[k, 'role'],
                       group=[g for g, v in groups.items() if k in v][0], nomap=nm[k],
                       info=json.load(open(f"{OUT}/{reg.loc[k].run}/{reg.loc[k].subrun}/{reg.loc[k].det}/map1_row.json"))) for k in reg.key}
    json.dump(dict(rows=rows, groups=grp, allrows=allrows, min_row=MIN_ROW, min_flag=MIN_FLAG, nboot=NBOOT), open(f'{OUT}/gates.json', 'w'), indent=1, default=float)
    print('written', f'{OUT}/gates.json')


if __name__ == '__main__':
    main()
