#!/usr/bin/env python3
"""multitrack.py -- det3 golden row: which events hold more than one particle, and the efficiency counted with
multi-track credit (Dylan 10-26: "if there are two tracks and we reconstruct both, we should get credit if one matches
M3"; quote single-track first, then add the multi-track events).

    ../../.venv/bin/python multitrack.py feat     # one waveform pass: time extent of every production candidate
    ../../.venv/bin/python multitrack.py count    # the accounting -> multitrack.json

Every production candidate (events.candidates.parquet: one forward fit per spatial cluster, up to 3 per view) is tagged
from its own waveforms (strips within max(4, |15 tan| + 3) mm of the cluster's mid-gap position):
  particle  the cluster's summed signal stays above 20 % of its peak for <= EXT_MAX samples. A drift column cannot
            last longer: 30 mm / 36 um/ns = 840 ns = 14 samples, plus the ~6-sample pulse.
  band      it lasts longer, typically the whole 32-sample window: a low-amplitude coherent band on a group of edge
            strips (x < 6 mm, y > 355 mm), not a particle. These alone make half of the n_dropped > 2 rejections.
Per interior M3 ray (rerun/effbudget, R = 5 mm in the raw strip frame, as budget.py):
  class    S: <= 1 particle cluster in each view; M: >= 2 in a view; V: spark-vetoed (no fit at all)
  prod     the budget's 'detected' (winner within R, n_dropped <= 2 in both views)
  noband   as prod, n_dropped counted without the band clusters' strips
  any      some x candidate AND some y candidate within R (no pairing, no n_dropped cut)
  paired   one gated x-y track (select_tracks' track_id) within R in both views. The production gate (|dt0 - offset|
           <= 120 ns and the plausibility flag) is beam-tuned and misses 35 % of clean bench singles, so it is shown
           but not used for the credit.
x-y pairing without the reference (pairing()): in M events with exactly 2 particle clusters per view and the M3 track
on one of each, choose the assignment with the lower summed cost, cost = sum of z^2 (capped at 25) of the x-minus-y
log charge, t0, q_u50, q_u90, with medians and robust sigmas from the clean singles (wft.reco.xy_pair_cost's form);
right = it puts the two M3-matched candidates together.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402  (paths)

CAND = os.path.expanduser('~/x17/cosmic_bench/outliers/final/mx17_det3_saturday_scan_6-27-26/'
                          'long_run_resist_490V_drift_1000V/mx17_3/wft/final_phys/events.candidates.parquet')
RAYS = os.path.expanduser('~/x17/cosmic_bench/effbudget/rays_det3.parquet')
FEAT = os.path.join(S.OUT, 'cand_feat_det3.parquet')
EXT_MAX, R_MM = 24, 5.0


def rays():
    R = pd.read_parquet(RAYS)
    return R[(R.region == 'interior') & ~R.not_measured].set_index('event_id')


def feat():
    from qa_config import get_config
    from wft import io as wio
    cfg = get_config(S.KEY); pm = wio.strip_position_map(cfg)
    C = pd.read_parquet(CAND); C = C[C['rank'] >= 0]
    want = set(C.event_id.astype(int)) & set(rays().index)
    C = C[C.event_id.isin(want)]
    by = {k: g for k, g in C.groupby(['event_id', 'plane'])}
    rows = []
    for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        p = pm[feu]
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            r = wio.FeuReader(f)
            for eid, _ft, W in r.iter_events(want):
                g = by.get((int(eid), v))
                if g is None:
                    continue
                for _, c in g.iterrows():
                    mid = c.p0 + 15 * c.tan_theta
                    sel = np.isfinite(p) & (np.abs(p - mid) < max(4.0, abs(15 * c.tan_theta) + 3))
                    prof = W[sel].astype(np.float32).sum(0)
                    pk = float(prof.max()) if len(prof) else 0.0
                    rows.append(dict(event_id=int(eid), plane=v, rank=int(c['rank']),
                                     ext=int((prof > 0.2 * pk).sum()) if pk > 0 else 0,
                                     peak=float(W[sel].max()) if sel.any() else 0.0))
            print(f, len(rows), flush=True)
    pd.DataFrame(rows).to_parquet(FEAT)


def count():
    R = rays(); C = pd.read_parquet(CAND); C = C[C['rank'] >= 0]
    F = pd.read_parquet(FEAT)
    C = C.merge(F, on=['event_id', 'plane', 'rank'], how='left')
    C['particle'] = C.ext.fillna(0) <= EXT_MAX
    C = C[C.event_id.isin(R.index)]
    C['mesh'] = np.where(C.plane == 'x', R.mesh_x.reindex(C.event_id).to_numpy(), R.mesh_y.reindex(C.event_id).to_numpy())
    C['near'] = (C.p0 - C.mesh).abs() <= R_MM
    npart = C[C.particle].groupby(['event_id', 'plane']).size().unstack(fill_value=0).reindex(R.index, fill_value=0)
    nband = C[~C.particle].groupby(['event_id', 'plane']).n_strips.sum().unstack(fill_value=0).reindex(R.index, fill_value=0)
    D = pd.DataFrame(index=R.index)
    D['cat'] = R.cat_xy_5
    D['cls'] = np.where(R.cat_xy_5 == 'spark', 'V', np.where((npart.max(axis=1) >= 2), 'M', 'S'))
    D['prod'] = R.cat_xy_5 == 'detected'
    nd = {v: (R[f'{v}_n_dropped'] - nband.get(v, 0)).fillna(99) for v in 'xy'}
    D['noband'] = D['prod'] | ((R.cat_xy_5 == 'rejected_cut') & (nd['x'] <= 2) & (nd['y'] <= 2))
    nx = C[C.near & (C.plane == 'x')].event_id.unique(); ny = C[C.near & (C.plane == 'y')].event_id.unique()
    D['any'] = D.index.isin(nx) & D.index.isin(ny)
    T = C[C.near & C.track_gated & (C.track_id >= 0)].groupby(['event_id', 'track_id']).plane.nunique()
    D['paired'] = D.index.isin(T[T == 2].index.get_level_values(0))
    D['n_tracks'] = C[C.track_gated & (C.track_id >= 0)].groupby('event_id').track_id.nunique().reindex(R.index).fillna(0)
    D.to_parquet(os.path.join(S.OUT, 'multitrack_det3.parquet'))
    N = len(D); out = dict(n=N, ext_max=EXT_MAX, r_mm=R_MM, classes={})
    for k, g in D.groupby('cls'):
        out['classes'][k] = dict(n=int(len(g)), frac=len(g) / N,
                                 **{m: float(g[m].mean()) for m in ('prod', 'noband', 'any', 'paired')},
                                 cats=g.cat.value_counts().to_dict(),
                                 n_tracks=g.n_tracks.value_counts().sort_index().to_dict())
    out['total'] = {m: float(D[m].mean()) for m in ('prod', 'noband', 'any', 'paired')}
    # the n_dropped > 2 rejections: band-only vs a second particle
    rc = D[D.cat == 'rejected_cut']
    out['rejected_cut'] = dict(n=int(len(rc)), frac=len(rc) / N, second_particle=int((rc.cls == 'M').sum()),
                               band_only=int((rc.cls == 'S').sum()), recovered_noband=int(rc.noband.sum()))
    out['pairing'] = pairing(C, D, R)
    json.dump(out, open(os.path.join(S.OUT, 'multitrack.json'), 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


FEATS = dict(q=['lq'], t0=['t0'], q_t0=['lq', 't0'], all=['lq', 't0', 'u50', 'u90'])


def _d(a, b):
    return dict(lq=np.log(a.q_sum / b.q_sum), t0=a.t0 - b.t0, u50=a.q_u50 - b.q_u50, u90=a.q_u90 - b.q_u90)


def pairing(C, D, R):
    P = C[C.particle]
    w = P[P.event_id.isin(D.index[(D.cls == 'S') & D['prod']]) & (P['rank'] == 0)].pivot(index='event_id', columns='plane')
    f = dict(lq=np.log(w.q_sum.x / w.q_sum.y), t0=w.t0.x - w.t0.y, u50=w.q_u50.x - w.q_u50.y, u90=w.q_u90.x - w.q_u90.y)
    cal = {k: (float(np.nanmedian(v)), float(1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))))) for k, v in f.items()}
    cost = lambda a, b, ks: sum(min(((_d(a, b)[k] - cal[k][0]) / cal[k][1]) ** 2, 25.0) for k in ks)
    right = {k: [] for k in FEATS}
    for e in D.index[D.cls == 'M']:
        g = P[P.event_id == e]; X, Y = g[g.plane == 'x'], g[g.plane == 'y']
        if len(X) != 2 or len(Y) != 2:
            continue
        ix = np.flatnonzero(X.near.to_numpy()); iy = np.flatnonzero(Y.near.to_numpy())
        if len(ix) != 1 or len(iy) != 1:
            continue
        for k, ks in FEATS.items():
            straight = (cost(X.iloc[0], Y.iloc[0], ks) + cost(X.iloc[1], Y.iloc[1], ks)
                        <= cost(X.iloc[0], Y.iloc[1], ks) + cost(X.iloc[1], Y.iloc[0], ks))
            right[k].append(bool(straight == (ix[0] == iy[0])))
    n = C[C.particle & C.event_id.isin(D.index[D.cls == 'M'])].groupby(['event_id', 'plane']).size().unstack(fill_value=0)
    return dict(cal=cal, n=len(right['q']), right={k: float(np.mean(v)) for k, v in right.items()},
                topology={f'{a}x{b}': int(((n.x == a) & (n.y == b)).sum()) for a in (1, 2, 3) for b in (1, 2, 3)
                          if ((n.x == a) & (n.y == b)).any()},
                ext_hist=np.histogram(C.ext.fillna(0), bins=np.arange(0, 34, 2))[0].tolist())


if __name__ == '__main__':
    {'feat': feat, 'count': count}[sys.argv[1]]()
