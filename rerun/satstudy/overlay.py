#!/usr/bin/env python3
"""overlay.py -- bench overlay with truth on both tracks: two clean det3 single tracks summed in the waveforms, the
production seeder + fit re-run, and the x<->y pairing scored against each donor's own fit (Dylan 10-26: charge pairing
for the reconstruction paper). The bench version of sept26_prelim_analysis/intra_bench.py (run_145, beam seeder).

    WFT_PIVOT_START=1 WFT_COL_GUARD=0.01 X17_BENCH_ROOT=~/x17/cosmic_bench/outliers/stage \\
        ../../.venv/bin/python overlay.py build [--per-bin 50] [--jobs 3]      -> overlay/overlays.parquet, candidates
    ../../.venv/bin/python overlay.py score                                    -> overlay.json

Donors: golden-row interior events that det3 sees as one particle and detects (multitrack.py class S, production
'detected'), EVEN event ids (the pairing calibration uses the odd ones). A pair is two donors of the same file tag and
the same ftst in both views (so the 60 ns sample grids coincide) whose t0 agree within DT0 ns in both views (two
prompt tracks reach the mesh together; the bench t0 jitter is 72 ns RMS). Overlay: donor a's full planes, plus donor
b's waveforms on b's footprint +- MARGIN mm ([p0, p0 + 30 tan] at its fit), and a's hits plus b's hits there (the
larger kept per strip). b's noise is added too on those strips (noise x sqrt 2 there), as intra_bench's 'add' mode.
Truth per view: each donor's production fit. A track is found in a view when a candidate's p0 is within MATCH mm of it.
Pairs are drawn evenly over min(sep_x, sep_y) bins, sep = |mid-gap positions|.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402
import multitrack as MT  # noqa: E402
import pairing as P  # noqa: E402

OD = os.path.join(S.OUT, 'overlay')
EV = MT.CAND.replace('.candidates', '')
SEP_BINS = [0, 6, 12, 18, 24, 40, 80, 400]
DT0, MARGIN, MATCH = 30.0, 5.0, 2.0


def donors():
    D = pd.read_parquet(os.path.join(S.OUT, 'multitrack_det3.parquet'))
    E = pd.read_parquet(EV).drop_duplicates('event_id').set_index('event_id')
    ids = D.index[(D.cls == 'S') & D['prod']]
    E = E.loc[ids]
    return E[E.index % 2 == 0]


def pairs(E, tags, per_bin, rng):
    E = E.assign(tag=[tags.get(e) for e in E.index]).dropna(subset=['tag'])
    E['mx'] = E.x_p0 + 15 * E.x_tan_theta; E['my'] = E.y_p0 + 15 * E.y_tan_theta
    rows = []
    for (tag, fx, fy), g in E.groupby(['tag', 'x_ftst', 'y_ftst']):
        if len(g) < 2:
            continue
        a, b = np.triu_indices(len(g), 1)
        A, B = g.iloc[a], g.iloc[b]
        ok = ((A.x_t0.to_numpy() - B.x_t0.to_numpy()).__abs__() <= DT0) & \
             ((A.y_t0.to_numpy() - B.y_t0.to_numpy()).__abs__() <= DT0)
        rows.append(pd.DataFrame(dict(tag=tag, a=A.index[ok], b=B.index[ok],
                                      sep_x=np.abs(A.mx.to_numpy() - B.mx.to_numpy())[ok],
                                      sep_y=np.abs(A.my.to_numpy() - B.my.to_numpy())[ok])))
    Pq = pd.concat(rows, ignore_index=True)
    Pq['sep'] = Pq[['sep_x', 'sep_y']].min(axis=1)
    Pq['sbin'] = np.digitize(Pq.sep, SEP_BINS) - 1
    out = []
    for k, g in Pq.groupby('sbin'):
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 30)))
        used, pick = {}, []
        for r in g.itertuples():     # each donor at most twice per bin
            if used.get(r.a, 0) < 2 and used.get(r.b, 0) < 2:
                pick.append(r.Index); used[r.a] = used.get(r.a, 0) + 1; used[r.b] = used.get(r.b, 0) + 1
            if len(pick) >= per_bin:
                break
        out.append(Pq.loc[pick])
    return pd.concat(out, ignore_index=True)


def build(per_bin, jobs):
    from concurrent.futures import ProcessPoolExecutor
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import io as wio, seed as wseed, reco as wr
    from wft.calib import CalibrationBundle
    os.makedirs(OD, exist_ok=True)
    cfg = get_config(S.KEY); cal = CalibrationBundle.load(S.BUNDLE)
    pm = wio.strip_position_map(cfg); feu = dict(x=cfg.MX17_FEU_X, y=cfg.MX17_FEU_Y)
    E = donors()
    files = {p: {wio.file_tag(f): f for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu[p])} for p in 'xy'}
    rdr = {p: {t: wio.FeuReader(f) for t, f in files[p].items()} for p in 'xy'}
    tags = {}
    for t, r in rdr['x'].items():
        for e in r.event_ids.tolist():
            tags[int(e)] = t
    Pq = pairs(E, tags, per_bin, np.random.default_rng(7))
    print('pairs per bin', Pq.groupby('sbin').size().to_dict(), flush=True)
    need = set(Pq.a) | set(Pq.b)
    H = wr._load_hits(cfg); H = H[H.eventId.isin(need)]
    Hb = {int(e): g for e, g in H.groupby('eventId')}
    WF = {p: {} for p in 'xy'}
    for p in 'xy':
        for t, r in rdr[p].items():
            for eid, ft, w in r.iter_events(need & set(r.event_ids.tolist())):
                WF[p][eid] = w
    payloads, meta = [], []
    todo = [(None, int(a)) for a in sorted(set(Pq.a))[:30]] + [(r, None) for r in Pq.itertuples()]
    for oid, (r, single) in enumerate(todo):
        a = single if r is None else int(r.a); b = None if r is None else int(r.b)
        tag = tags[a]; W, hh = {}, [Hb[a]]
        for p in 'xy':
            W[p] = WF[p][a].copy()
            if b is None:
                continue
            p0, tn = E.loc[b, f'{p}_p0'], E.loc[b, f'{p}_tan_theta']
            lo, hi = min(p0, p0 + 30 * tn) - MARGIN, max(p0, p0 + 30 * tn) + MARGIN
            reg = np.flatnonzero(np.isfinite(pm[feu[p]]) & (pm[feu[p]] >= lo) & (pm[feu[p]] <= hi))
            W[p][reg] += WF[p][b][reg]
            g = Hb[b]; hh.append(g[(g.feu == feu[p]) & g.channel.isin(reg)])
        h = pd.concat(hh, ignore_index=True).assign(eventId=oid)
        h = h.sort_values('amplitude').drop_duplicates(['feu', 'channel'], keep='last')
        sd = wseed.seeds_from_hits(h, pm, feu['x'], feu['y'], hot=cal.hot).get(oid)
        m = dict(oid=oid, a=a, b=-1 if b is None else b, mode='single' if b is None else 'overlay', seeded=sd is not None)
        if r is not None:
            m.update(sep_x=r.sep_x, sep_y=r.sep_y, sep=r.sep, sbin=int(r.sbin))
        meta.append(m)
        if sd is None:
            continue
        wins, used = {}, {}
        for p in 'xy':
            ws, us = [], []
            for s in sd[p]:
                win = wio.extract_window(W[p], rdr[p][tag].noise, pm[feu[p]], s.channels, wr.PAD_STRIPS
                                         if hasattr(wr, 'PAD_STRIPS') else 3)
                if win is None:
                    continue
                ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch)); us.append(s)
            if ws:
                wins[p], used[p] = ws, us
        payloads.append((oid, wins, used, sd['n_hits'], sd['spark'],
                         dict(x=int(E.loc[a, 'x_ftst']), y=int(E.loc[a, 'y_ftst']))))
    print(len(payloads), 'payloads', flush=True)
    C = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=wr._worker_init, initargs=(S.BUNDLE,)) as pool:
        for row in pool.map(wr._worker_fit, payloads, chunksize=2):
            for c in row.pop('_cand', []):
                c['oid'] = row['event_id']; C.append(c)
    pd.DataFrame(meta).to_parquet(os.path.join(OD, 'overlays.parquet'))
    pd.DataFrame(C).to_parquet(os.path.join(OD, 'candidates.parquet'))
    print('wrote', OD)


def score():
    M = pd.read_parquet(os.path.join(OD, 'overlays.parquet')); C = pd.read_parquet(os.path.join(OD, 'candidates.parquet'))
    C = C[C['rank'] >= 0]
    E = pd.read_parquet(EV).drop_duplicates('event_id').set_index('event_id')
    w = P.singles(); cal = P.calib(w[w.index % 2 == 1])
    rows = []
    for m in M.itertuples():
        g = C[C.oid == m.oid]; r = dict(oid=m.oid, mode=m.mode, sbin=getattr(m, 'sbin', -1), sep=getattr(m, 'sep', np.nan))
        for p in 'xy':
            gp = g[g.plane == p]
            for k, d in (('a', m.a), ('b', m.b)):
                if d < 0:
                    continue
                dist = (gp.p0 - E.loc[d, f'{p}_p0']).abs()
                r[f'{k}{p}'] = int(dist.idxmin()) if len(gp) and dist.min() <= MATCH else -1
        if m.mode == 'single':
            r['ok'] = r['ax'] >= 0 and r['ay'] >= 0
        else:
            found = all(r[k] >= 0 for k in ('ax', 'ay', 'bx', 'by')) and r['ax'] != r['bx'] and r['ay'] != r['by']
            r['both_found'] = found
            if found:
                X = {k: C.loc[r[f'{k}x']] for k in 'ab'}; Y = {k: C.loc[r[f'{k}y']] for k in 'ab'}
                for nm, ks in (('q', ['lq']), ('q_u', ['lq', 'u50', 'u90']), ('q_u_t0', ['lq', 'u50', 'u90', 't0'])):
                    c = lambda x, y: float(P.cost(P.diffs(x.q_sum, y.q_sum, x.q_u50, y.q_u50, x.q_u90, y.q_u90, x.t0, y.t0), cal, ks))
                    r[f'right_{nm}'] = (c(X['a'], Y['a']) + c(X['b'], Y['b'])) < (c(X['a'], Y['b']) + c(X['b'], Y['a']))
                ta, tb = X['a'].track_id, X['b'].track_id
                r['prod_right'] = bool(ta >= 0 and tb >= 0 and ta != tb and Y['a'].track_id == ta and Y['b'].track_id == tb)
                r['prod_paired_both'] = bool(ta >= 0 and tb >= 0 and ta != tb)
        rows.append(r)
    T = pd.DataFrame(rows); T.to_parquet(os.path.join(OD, 'scored.parquet'))
    sg = T[T['mode'] == 'single']; ov = T[T['mode'] == 'overlay']
    out = dict(n_single=int(len(sg)), single_ok=float(sg.ok.mean()), n_overlay=int(len(ov)), sep_bins=SEP_BINS, bins=[])
    for k, g in ov.groupby('sbin'):
        f = g[g.both_found.astype(bool)]
        out['bins'].append(dict(lo=SEP_BINS[k], hi=SEP_BINS[k + 1], n=int(len(g)), both_found=float(g.both_found.mean()),
                                **{c: float(f[c].mean()) if len(f) else None
                                   for c in ('right_q', 'right_q_u', 'right_q_u_t0', 'prod_right', 'prod_paired_both')}))
    f = ov[ov.both_found.astype(bool) & (ov.sep >= 24)]
    out['ge24'] = {c: float(f[c].mean()) for c in ('right_q', 'right_q_u', 'right_q_u_t0', 'prod_right')} | dict(n=int(len(f)))
    json.dump(out, open(os.path.join(S.OUT, 'overlay.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--per-bin', type=int, default=50)
    ap.add_argument('--jobs', type=int, default=3); a = ap.parse_args()
    build(a.per_bin, a.jobs) if a.cmd == 'build' else score()
