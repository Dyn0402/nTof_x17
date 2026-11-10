#!/usr/bin/env python3
"""make_two_track_separation.py -- two-track separation of chamber A (det3) on the final reconstruction, by overlay.

    .venv/bin/python papers/paper1_v2/figs/make_two_track_separation.py run   [--per-bin 150] [--n-val 100] [--jobs 2]
    .venv/bin/python papers/paper1_v2/figs/make_two_track_separation.py smoke [--n 6]      (timing, a few overlays)
    .venv/bin/python papers/paper1_v2/figs/make_two_track_separation.py plot                 (score + figure + json)

Method (the 10-26 bench overlay, rerun/satstudy/overlay.py + rerun/multiparticle/bench.py, redone on the final chain):
  donors   Run 1 (sat_det3) triggers with an interior reference ray, both projected reference angles within +-20 deg,
           reconstructed by the final tables (Analysis_pass1_twotrack) as exactly one track, one candidate per view,
           each within 5 mm of the ray at the mesh, not spark-vetoed.
  pair     two donors whose fitted t0 agree within 30 ns in both views (two prompt particles of one trigger; the bench
           t0 jitter is ~70 ns). Separation s = min over views of |mid-gap distance between the two reference rays|
           (mid-gap = mesh + 15 mm x tan). Pairs are drawn per s bin, each donor at most twice per bin, so the two
           tracks sit at their real positions (no strip shift: dead strips, gain and edge stay those of the chamber).
  overlay  donor a's full planes plus donor b's pedestal- and common-mode-subtracted waveforms on b's footprint
           +- MARGIN mm; the sum is clipped at a's per-sample ADC top rail (4095 - ped - CM of a: the summed signal
           saturates like a real one); dead strips of the chamber get nothing added. Seeding hits: a's plus b's on
           that footprint (larger amplitude per strip) -- hits only choose the windows. Then the production path:
           seeds_from_hits (spark veto, dead-strip bridging), band re-seed, windows, head-on rescue windows,
           wft.reco._worker_fit with the final bundle (two-track split, charge corroboration split_corrob='charge',
           cost pairing with the fixed repair) -- the same function the pass ran.
  checks   'single': donor a alone through the same path (must reproduce the table);
           'noise' : a plus a far partner's EMPTY strips over a's footprint (noise added twice, no second particle).
Scoring: a muon is found in a view when a distinct candidate (rank >= 0, not a replaced split parent) lies within
R = 5 mm of its reference ray at the mesh (the paper's matching radius); 'resolved' = both muons found with distinct
candidates in both views; 'reconstructed' = resolved AND each muon's X and Y candidates belong to one reconstructed
track (track_id), different for the two. Resolutions are s68 = 68th percentile of |d - median d|.

Writes: work dir $WORK_DATA/claude-jobs/v2_twotrack/ (overlays.parquet, candidates.parquet, timing.json);
        papers/paper1_v2/figs/out/two_track_separation.{pdf,png,json}.
"""
import argparse, json, os, sys, time
os.environ.setdefault('WFT_PIVOT_START', '1')       # the pass's environment (STAGES/*_wft.cli.json)
os.environ.setdefault('WFT_COL_GUARD', '0.01')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'papers')]

WD = os.environ.get('WORK_DATA') or os.path.expanduser('~/work_data')
CJ = os.path.join(WD, 'claude-jobs')
REL = 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V'
RAWDIR = os.environ.get('TT_RAW') or os.path.join(CJ, '17c92c46-displays/june_tests', REL)
TABLES = os.environ.get('TT_TABLES') or os.path.join(CJ, 'twotrack/Analysis_pass1_twotrack', REL, 'mx17_3/wft')
RAYS = os.environ.get('TT_RAYS') or os.path.join(CJ, '17c92c46/x17rerun/paper1_eff/budget', REL, 'mx17_3/rays_det3.parquet')
RUNCFG = os.environ.get('TT_RUNCFG') or os.path.join(os.path.dirname(RAWDIR), 'run_config.json')
WORK = os.environ.get('TT_WORK') or os.path.join(CJ, 'v2_twotrack')
OUT = os.path.join(HERE, 'out')
FEU = dict(x=7, y=8)
R_MM, DT0, MARGIN, ANG_MAX, MID = 5.0, 30.0, 6.0, 20.0, 15.0
SEP_BINS = [0, 3, 6, 9, 12, 18, 24, 40, 80]
FAR = 100.0


def cfg_ns():
    from types import SimpleNamespace
    return SimpleNamespace(run_config_path=RUNCFG, MAP_CSV_PATH=os.path.join(REPO, 'mx17_m1_map.csv'),
                           DET_NAME='mx17_3', MX17_FEU_X=FEU['x'], MX17_FEU_Y=FEU['y'])


# ------------------------------------------------------------------------------------------------ donors and pairs
def donors():
    E = pd.read_parquet(os.path.join(TABLES, 'events.parquet')).set_index('event_id')
    C = pd.read_parquet(os.path.join(TABLES, 'events.candidates.parquet'))
    C = C[C['rank'] >= 0]
    B = pd.read_parquet(RAYS).drop_duplicates('event_id').set_index('event_id')
    B = B[B.region == 'interior']
    n = C.groupby(['event_id', 'plane']).size().unstack().fillna(0)
    one = n.index[(n.get('x', 0) == 1) & (n.get('y', 0) == 1)]
    D = E.loc[E.index.intersection(one).intersection(B.index)]
    D = D[(D.n_tracks == 1) & D.x_ok & D.y_ok & ~D.spark.astype(bool)]
    D = D.join(B[['mesh_x', 'mesh_y', 'tan_x', 'tan_y']])
    ok = ((D.x_p0 - D.mesh_x).abs() < R_MM) & ((D.y_p0 - D.mesh_y).abs() < R_MM)
    ok &= (np.degrees(np.arctan(D.tan_x.abs())) <= ANG_MAX) & (np.degrees(np.arctan(D.tan_y.abs())) <= ANG_MAX)
    D = D[ok].copy()
    D['mx'] = D.mesh_x + MID * D.tan_x
    D['my'] = D.mesh_y + MID * D.tan_y
    return D


def pairs(D, per_bin, n_val, rng):
    ids = D.index.to_numpy()
    mx, my, tx, ty = (D[c].to_numpy() for c in ('mx', 'my', 'x_t0', 'y_t0'))
    tags = D.acq_tag.to_numpy()
    rows = []
    for i in range(len(ids) - 1):
        j = np.arange(i + 1, len(ids))
        ok = (np.abs(tx[j] - tx[i]) <= DT0) & (np.abs(ty[j] - ty[i]) <= DT0)
        j = j[ok]
        sx, sy = np.abs(mx[j] - mx[i]), np.abs(my[j] - my[i])
        keep = (np.minimum(sx, sy) < 24) | (rng.random(len(j)) < 0.05)     # memory: thin the plentiful wide pairs
        rows.append(np.c_[np.full(keep.sum(), i), j[keep], sx[keep], sy[keep]].astype(np.float32))
    P = np.concatenate(rows)
    P = pd.DataFrame(dict(a=ids[P[:, 0].astype(int)], b=ids[P[:, 1].astype(int)], sep_x=P[:, 2], sep_y=P[:, 3]))
    P['sep'] = P[['sep_x', 'sep_y']].min(axis=1)
    tg = dict(zip(ids, tags))
    P['same_tag'] = P.a.map(tg).to_numpy() == P.b.map(tg).to_numpy()
    P['sbin'] = np.digitize(P.sep, SEP_BINS) - 1
    out = []
    for k in range(len(SEP_BINS) - 1):
        g = P[P.sbin == k].sample(frac=1, random_state=int(rng.integers(1 << 30)))
        used, pick = {}, []
        for r in g.itertuples():
            a, b = (r.a, r.b) if rng.random() < 0.5 else (r.b, r.a)       # which donor carries the full planes
            if used.get(a, 0) < 2 and used.get(b, 0) < 2:
                pick.append((r.Index, a, b)); used[a] = used.get(a, 0) + 1; used[b] = used.get(b, 0) + 1
            if len(pick) >= per_bin:
                break
        s = P.loc[[p[0] for p in pick]].copy(); s['a'] = [p[1] for p in pick]; s['b'] = [p[2] for p in pick]
        out.append(s.assign(mode='overlay'))
    # checks: a alone; a plus a far partner's empty strips over a's footprint
    far = P[(P.sep_x > FAR) & (P.sep_y > FAR)].sample(frac=1, random_state=int(rng.integers(1 << 30)))
    far = far.drop_duplicates('a').head(n_val)
    out.append(far.assign(mode='noise', sbin=-1))
    out.append(far.assign(mode='single', b=-1, sbin=-1))
    info = dict(n_pairs_kept=int(len(P)),
                n_pairs_bin={int(k): int(v) for k, v in P.groupby('sbin').size().items()})
    del P
    return pd.concat(out, ignore_index=True), info


# ------------------------------------------------------------------------------------------------ overlay + reco
def footprint(pos, p0, tn):
    lo, hi = min(p0, p0 + 30 * tn) - MARGIN, max(p0, p0 + 30 * tn) + MARGIN
    return np.flatnonzero(np.isfinite(pos) & (pos >= lo) & (pos <= hi))


def rss_mb():
    try:
        for ln in open('/proc/self/status'):
            if ln.startswith('VmRSS'):
                return int(ln.split()[1]) / 1024
    except OSError:
        return float('nan')


def _timed_fit(payload):
    from wft import reco as wr
    t = time.process_time()
    row = wr._worker_fit(payload)
    row['_fit_cpu_s'] = time.process_time() - t
    row['_rss_mb'] = rss_mb()
    return row


def iter_events_chunked(rdr, wanted, step=200):
    """wft.io.FeuReader.iter_events(wanted, with_raw=True), copied, but reading the file in contiguous blocks of
    ``step`` entries: iter_events reads from the first to the last of up to 400 WANTED entries in one go, which for
    sparse picks is most of a file (2.7 GB resident on the laptop). Same pedestal / common-mode recipe (CM_MASK = 0)."""
    from wft import io as wio
    assert wio.CM_MASK == 0
    idx = np.flatnonzero(np.isin(rdr.event_ids, np.fromiter(wanted, dtype=np.int64)))
    nblk = 512 // wio.CNS_BLOCK
    for lo in range(0, rdr.n_entries, step):
        blk = idx[(idx >= lo) & (idx < lo + step)]
        if not len(blk):
            continue
        arr = rdr.tree.arrays(['eventId', 'amplitude', 'ftst'], entry_start=int(blk[0]), entry_stop=int(blk[-1]) + 1,
                              library='np')
        for i in blk:
            j = i - int(blk[0]); a = arr['amplitude'][j]
            if len(a) != rdr.n_sample * 512:
                continue
            w = a.reshape(-1, 512).astype(np.float32) - rdr.ped
            cms = np.median(w.reshape(w.shape[0], nblk, wio.CNS_BLOCK), axis=2)
            w -= np.repeat(cms, wio.CNS_BLOCK, axis=1)
            yield int(arr['eventId'][j]), int(arr['ftst'][j]), w.T.copy(), np.asarray(a, dtype=np.int16).reshape(-1, 512).T.copy()
        del arr


def build(Pq, D, jobs, tag='run', limit=None):
    from concurrent.futures import ProcessPoolExecutor
    import uproot
    from wft import io as wio, seed as wseed, reco as wr, model as wm
    from wft.calib import CalibrationBundle
    os.makedirs(WORK, exist_ok=True)
    t_start = time.time()
    bundle = os.path.join(TABLES, 'calib_bundle')
    cal = CalibrationBundle.load(bundle)
    assert (cal.xy_pairing or {}).get('split_corrob') == 'charge', 'not the final (charge-corroborated) bundle'
    cfg = cfg_ns()
    pm = wio.strip_position_map(cfg)
    dead = {p: np.asarray((cal.dead or {}).get(p, []), int) for p in 'xy'} if wseed.DEAD_SEED else {}
    if limit:
        Pq = Pq.head(limit)
    need = set(Pq.a.astype(int)) | set(Pq.b[Pq.b >= 0].astype(int))
    # seeding hits (as wft.reco._load_hits: DROP_FLAT_HITS='both')
    hd = os.path.join(RAWDIR, 'combined_hits_root_rr1_cm0')
    fs = sorted(f for f in os.listdir(hd) if f.endswith('.root') and '_datrun_' in f)
    hs = []
    for f in fs:                                   # chunked: keep only the needed events' X/Y hits
        for h in uproot.iterate(f'{hd}/{f}:hits', ['eventId', 'feu', 'channel', 'amplitude', 'significance',
                                                   'trunc_left', 'trunc_right'], step_size='50 MB', library='pd'):
            h = h[h.feu.isin([FEU['x'], FEU['y']]) & h.eventId.isin(need) & ~(h.trunc_left & h.trunc_right)]
            hs.append(h[['eventId', 'feu', 'channel', 'amplitude', 'significance']])
    H = pd.concat(hs, ignore_index=True)
    Hb = {int(e): g for e, g in H.groupby('eventId')}
    print(f'[mem] after hits {rss_mb():.0f} MB', flush=True)
    # waveforms (+ raw ADC for the top rail) of every needed event
    dd = os.path.join(RAWDIR, 'decoded_root_fa80c06')
    WF, RAW, FT, NOISE = {p: {} for p in 'xy'}, {p: {} for p in 'xy'}, {p: {} for p in 'xy'}, {p: {} for p in 'xy'}
    for p in 'xy':
        for f in sorted(os.listdir(dd)):
            if '_datrun_' not in f or not f.endswith(f'_{FEU[p]:02d}.root'):
                continue
            r = wio.FeuReader(os.path.join(dd, f)); t = wio.file_tag(f)
            NOISE[p][t] = r.noise
            for eid, ft, w, raw in iter_events_chunked(r, need):
                WF[p][eid], RAW[p][eid], FT[p][eid] = w, raw, ft
            del r
    t_read = time.time()
    print(f'[mem] after waveforms {rss_mb():.0f} MB', flush=True)
    payloads, meta = [], []
    for r in Pq.itertuples():
        oid, a, b = int(r.Index), int(r.a), int(r.b)
        tg = D.at[a, 'acq_tag']
        m = dict(oid=oid, a=a, b=b, mode=r.mode, sep=r.sep, sep_x=r.sep_x, sep_y=r.sep_y, sbin=int(r.sbin),
                 same_tag=bool(r.same_tag))
        if a not in WF['x'] or a not in WF['y'] or (r.mode != 'single' and (b not in WF['x'] or b not in WF['y'])):
            meta.append(dict(m, status='no_waveform')); continue
        W, hh, nsat_add = {}, [Hb.get(a, H.iloc[:0])], 0
        for p in 'xy':
            pos = pm[FEU[p]]
            W[p] = WF[p][a].copy()
            if r.mode == 'single':
                continue
            src = b                                    # 'noise': b's strips over a's footprint carry no particle
            fp_ev = b if r.mode == 'overlay' else a
            reg = footprint(pos, D.at[fp_ev, f'{p}_p0'], D.at[fp_ev, f'{p}_tan_theta'])
            reg = reg[~np.isin(reg, dead.get(p, []))]
            rail = W[p][reg] + (4095.0 - RAW[p][a][reg].astype(np.float32))
            s = W[p][reg] + WF[p][src][reg]
            nsat_add += int((s > rail).sum())
            W[p][reg] = np.minimum(s, rail)
            g = Hb.get(src)
            if g is not None:
                hh.append(g[(g.feu == FEU[p]) & g.channel.isin(reg)])
        h = pd.concat(hh, ignore_index=True).assign(eventId=oid)
        h = h.sort_values('amplitude').drop_duplicates(['feu', 'channel'], keep='last')
        sd = wseed.seeds_from_hits(h, pm, FEU['x'], FEU['y'], hot=cal.hot, dead=dead).get(oid)
        if sd is not None and wm.BAND_NSIG > 0:
            band = {FEU[p]: set(np.flatnonzero(wm.band_rows(W[p], NOISE[p][tg])).tolist()) for p in 'xy'}
            rb = wr.band_reseed(oid, h, band, pm, FEU['x'], FEU['y'], cal.hot, dead)
            sd = rb if rb is not None else sd
        m.update(seeded=sd is not None, spark=bool(sd and sd['spark']), n_hits=int(sd['n_hits']) if sd else 0,
                 n_seed_x=len(sd['x']) if sd else 0, n_seed_y=len(sd['y']) if sd else 0, n_sat_added=nsat_add)
        if sd is None or sd['spark'] or not (sd['x'] or sd['y']):
            meta.append(dict(m, status='vetoed' if sd and sd['spark'] else 'no_seed')); continue
        wins, used = {}, {}
        for p in 'xy':
            pos = pm[FEU[p]]
            if not sd[p]:
                if wr.HEADON_RESCUE:
                    wins[f'_rescue_{p}'] = dict(W=W[p], noise=NOISE[p][tg], pos=pos)
                continue
            ws, us = [], []
            for s_ in sd[p]:
                win = wio.extract_window(W[p], NOISE[p][tg], pos, s_.channels, 3, dead=dead.get(p))
                if win is None:
                    continue
                ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch)); us.append(s_)
            if ws:
                wins[p], used[p] = ws, us
        if not any(k in wins for k in 'xy'):
            meta.append(dict(m, status='no_window')); continue
        payloads.append((oid, wins, used, sd['n_hits'], sd['spark'], dict(x=FT['x'][a], y=FT['y'][a])))
        meta.append(dict(m, status='fitted'))
    del WF, RAW, r
    import gc; gc.collect()
    t_pay = time.time()
    print(f'[mem] payloads built {rss_mb():.0f} MB', flush=True)
    rows, cands = [], []
    with ProcessPoolExecutor(max_workers=jobs, initializer=wr._worker_init, initargs=(bundle,)) as pool:
        for row in pool.map(_timed_fit, payloads, chunksize=1):
            for c in row.pop('_cand', []):
                cands.append(c)
            row.pop('_splits', None); row.pop('_ladder', None)
            rows.append(row)
    t_fit = time.time()
    M = pd.DataFrame(meta)
    M.to_parquet(os.path.join(WORK, f'{tag}_overlays.parquet'))
    pd.DataFrame(cands).to_parquet(os.path.join(WORK, f'{tag}_candidates.parquet'))
    pd.DataFrame(rows).to_parquet(os.path.join(WORK, f'{tag}_events.parquet'))
    tm = dict(n_overlays=len(M), n_fitted=len(payloads), jobs=jobs, read_s=t_read - t_start,
              build_s=t_pay - t_read, fit_wall_s=t_fit - t_pay,
              fit_core_s_per_event=(t_fit - t_pay) * jobs / max(len(payloads), 1),
              by_mode={k: int(v) for k, v in M['mode'].value_counts().items()})
    json.dump(tm, open(os.path.join(WORK, f'{tag}_timing.json'), 'w'), indent=1)
    print(json.dumps(tm, indent=1), flush=True)
    return M


# ------------------------------------------------------------------------------------------------ scoring
def _assign(pos, ra, rb):
    """Best distinct assignment of candidates (positions pos) to the two reference positions -> (i_a, i_b) or None
    for a muon not matched within R_MM. With one muon only (rb None) -> (i_a, None)."""
    if not len(pos):
        return None, None
    da = np.abs(pos - ra)
    if rb is None:
        i = int(np.argmin(da)); return (i if da[i] < R_MM else None), None
    db = np.abs(pos - rb)
    best, bi = np.inf, (None, None)
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j and da[i] < R_MM and db[j] < R_MM and da[i] + db[j] < best:
                best, bi = da[i] + db[j], (i, j)
    if bi[0] is not None:
        return bi
    ia, ib = int(np.argmin(da)), int(np.argmin(db))      # only one can be matched
    return (ia if da[ia] < R_MM else None), (ib if db[ib] < R_MM and not da[ia] < R_MM else None)


def _vec(tx, ty):
    v = np.array([tx, ty, 1.0]); return v / np.linalg.norm(v)


def _open(ax, ay, bx, by):
    return float(np.degrees(np.arccos(np.clip(_vec(ax, ay) @ _vec(bx, by), -1, 1))))


def score_tables(M, C):
    D = donors()
    C = C[C['rank'] >= 0]
    if 'split_replaced' in C:
        C = C[~C.split_replaced.fillna(False).astype(bool)]
    G = {k: g for k, g in C.groupby('event_id')}
    out = []
    for m in M.itertuples():
        r = dict(oid=m.oid, mode=m.mode, sep=m.sep, sep_x=m.sep_x, sep_y=m.sep_y, sbin=m.sbin, status=m.status)
        g = G.get(m.oid, C.iloc[:0])
        two = m.mode == 'overlay'
        ref = {'a': D.loc[m.a]}
        if two:
            ref['b'] = D.loc[m.b]
        pick = {}
        for p in 'xy':
            gp = g[g.plane == p].reset_index(drop=True)
            ia, ib = _assign(gp.p0.to_numpy(), ref['a'][f'mesh_{p}'], ref['b'][f'mesh_{p}'] if two else None)
            pick[p] = (gp, ia, ib)
        fa = all(pick[p][1] is not None for p in 'xy')
        r['a_found'] = fa
        if not two:
            r['a_paired'] = False
            if fa:
                cx, cy = pick['x'][0].loc[pick['x'][1]], pick['y'][0].loc[pick['y'][1]]
                r['a_paired'] = bool(cx.track_id >= 0 and cx.track_id == cy.track_id)
                for p, c in (('x', cx), ('y', cy)):
                    r[f'res_{p}'] = c.p0 - ref['a'][f'mesh_{p}']
                    r[f'dth_{p}'] = np.degrees(np.arctan(c.tan_theta) - np.arctan(ref['a'][f'tan_{p}']))
            out.append(r); continue
        fb = all(pick[p][2] is not None for p in 'xy')
        r['b_found'] = fb
        r['resolved'] = fa and fb
        # 10-26 definition, for comparison: distinct candidates within 2 mm of each donor's own single-track fit
        ok2 = True
        for p in 'xy':
            gp = pick[p][0]
            if not len(gp):
                ok2 = False; continue
            o = _assign(gp.p0.to_numpy(), ref['a'][f'{p}_p0'], ref['b'][f'{p}_p0'])
            pa = gp.p0.to_numpy()
            ok2 &= (o[0] is not None and o[1] is not None and abs(pa[o[0]] - ref['a'][f'{p}_p0']) <= 2.0
                    and abs(pa[o[1]] - ref['b'][f'{p}_p0']) <= 2.0)
        r['resolved_2mm_own'] = bool(ok2)
        r['reconstructed'] = False
        if r['resolved']:
            c = {k: {p: pick[p][0].loc[pick[p][1 if k == 'a' else 2]] for p in 'xy'} for k in 'ab'}
            ta, tb = c['a']['x'].track_id, c['b']['x'].track_id
            r['reconstructed'] = bool(ta >= 0 and tb >= 0 and ta != tb and c['a']['y'].track_id == ta
                                      and c['b']['y'].track_id == tb)
            # geometry of the pair: reco against reference
            rt = {k: {p: (c[k][p].p0, c[k][p].tan_theta) for p in 'xy'} for k in 'ab'}
            tt = {k: {p: (ref[k][f'mesh_{p}'], ref[k][f'tan_{p}']) for p in 'xy'} for k in 'ab'}
            for nm, S in (('reco', rt), ('ref', tt)):
                r[f'{nm}_open'] = _open(S['a']['x'][1], S['a']['y'][1], S['b']['x'][1], S['b']['y'][1])
                for p in 'xy':
                    r[f'{nm}_dth_{p}'] = float(np.degrees(np.arctan(S['a'][p][1]) - np.arctan(S['b'][p][1])))
                    r[f'{nm}_sep_{p}'] = float((S['a'][p][0] + MID * S['a'][p][1]) - (S['b'][p][0] + MID * S['b'][p][1]))
            cv = 'x' if m.sep_x <= m.sep_y else 'y'
            r['dsep_close'] = r[f'reco_sep_{cv}'] - r[f'ref_sep_{cv}']
            for k in 'ab':
                for p in 'xy':
                    r[f'dth1_{k}{p}'] = float(np.degrees(np.arctan(rt[k][p][1]) - np.arctan(tt[k][p][1])))
                    r[f'res1_{k}{p}'] = float(rt[k][p][0] - tt[k][p][0])
        out.append(r)
    return pd.DataFrame(out)


def s68(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(np.abs(x - np.median(x)), 68)) if len(x) >= 5 else float('nan')


def s68_err(x, nboot=200, seed=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5:
        return float('nan')
    rng = np.random.default_rng(seed)
    return float(np.std([s68(rng.choice(x, len(x))) for _ in range(nboot)]))


def binom(k, n):
    from scipy.stats import beta
    if n == 0:
        return np.nan, np.nan, np.nan
    lo = beta.ppf(0.1587, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.8413, k + 1, n - k) if k < n else 1.0
    return k / n, k / n - lo, hi - k / n


def summarise(T, tm):
    res = dict(method='overlay of two real single-muon triggers of chamber A Run 1 on the final reconstruction',
               match_radius_mm=R_MM, sep_bins_mm=SEP_BINS, sep_definition='min over views of |mid-gap distance| of '
               'the two reference rays (mid-gap = mesh + 15 mm x tan)', s68='68th percentile of |d - median d|',
               timing=tm)
    for mode in ('single', 'noise'):
        g = T[T['mode'] == mode]
        k = int(g.a_found.sum()); n = len(g)
        e = binom(k, n)
        res[f'check_{mode}'] = dict(n=n, found=k, eff=e[0], err_lo=e[1], err_hi=e[2],
                                    paired=float(g.a_paired.mean()) if n else None,
                                    res_s68_mm={p: s68(g[f'res_{p}']) for p in 'xy' if f'res_{p}' in g},
                                    angle_s68_deg={p: s68(g[f'dth_{p}']) for p in 'xy' if f'dth_{p}' in g},
                                    status={k_: int(v) for k_, v in g.status.value_counts().items()})
    ov = T[T['mode'] == 'overlay']
    bins = []
    for k in range(len(SEP_BINS) - 1):
        g = ov[ov.sbin == k]
        n = len(g)
        b = dict(lo=SEP_BINS[k], hi=SEP_BINS[k + 1], n=n, status={s: int(v) for s, v in g.status.value_counts().items()})
        for key in ('reconstructed', 'resolved', 'resolved_2mm_own'):
            kk = int(g[key].fillna(False).astype(bool).sum()); e = binom(kk, n)
            b[key] = dict(k=kk, p=e[0], err_lo=e[1], err_hi=e[2])
        b['one_found'] = float((g.a_found.fillna(False) | g.b_found.fillna(False)).mean()) if n else None
        f = g[g.reconstructed.fillna(False).astype(bool)]
        b['n_reco'] = int(len(f))
        if len(f) >= 5:
            do = f.reco_open - f.ref_open
            b['open_s68_deg'] = s68(do); b['open_s68_err'] = s68_err(do); b['open_bias_deg'] = float(np.median(do))
            for p in 'xy':
                dp = f[f'reco_dth_{p}'] - f[f'ref_dth_{p}']
                b[f'dth_{p}_s68_deg'] = s68(dp); b[f'dth_{p}_s68_err'] = s68_err(dp)
            b['sep_s68_mm'] = s68(f.dsep_close); b['sep_s68_err'] = s68_err(f.dsep_close)
            b['sep_bias_mm'] = float(np.median(f.dsep_close))
            b['track_angle_s68_deg'] = s68(np.r_[[f[f'dth1_{k_}{p}'] for k_ in 'ab' for p in 'xy']].ravel())
            b['track_pos_s68_mm'] = s68(np.r_[[f[f'res1_{k_}{p}'] for k_ in 'ab' for p in 'xy']].ravel())
            b['ref_open_median_deg'] = float(np.median(f.ref_open))
        bins.append(b)
    res['bins'] = bins
    return res


# ------------------------------------------------------------------------------------------------ figure
def plot(res, name='two_track_separation'):
    import matplotlib.pyplot as plt
    import paper_style as ps
    ps.use()
    col = ps.color('det3')
    fig, ax = ps.fig_for('full', height=2.45, ncols=3)
    B = res['bins']
    lo = np.array([b['lo'] for b in B], float); hi = np.array([b['hi'] for b in B], float)
    xc = 0.5 * (lo + hi); xe = 0.5 * (hi - lo)
    def P(key):
        return (np.array([b[key]['p'] for b in B]), np.array([b[key]['err_lo'] for b in B]),
                np.array([b[key]['err_hi'] for b in B]))
    p, el, eh = P('reconstructed')
    ax[0].errorbar(xc, p, yerr=[el, eh], xerr=xe, fmt='o', color=col, ms=4, lw=1.0, label='both reco.')
    p2, el2, eh2 = P('resolved')
    ax[0].errorbar(xc, p2, yerr=[el2, eh2], xerr=xe, fmt='s', color=col, mfc='white', ms=4, lw=0.8,
                   label='both found')
    ch = res['check_noise']['eff']
    ax[0].axhline(ch ** 2, color=ps.MUTED, lw=0.9, ls=':', label='check$^2$')
    fwd, inv = (lambda v: np.sqrt(np.clip(v, 0, None))), (lambda v: np.square(v))
    for a in ax:
        a.set_xscale('function', functions=(fwd, inv))
        a.set_xlim(0, 82); a.set_xticks([0, 6, 12, 24, 40, 80])
    fig.supxlabel('separation of the two tracks in the closer view (mm)', fontsize=ps.BASE_PT)
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel('probability')
    ax[0].legend(loc='lower right', fontsize=7, handlelength=1.2, bbox_to_anchor=(1.0, 0.02))
    def arr(key):
        return np.array([b.get(key, np.nan) for b in B], float)
    ok = arr('open_s68_deg')
    ax[1].errorbar(xc, ok, yerr=arr('open_s68_err'), xerr=xe, fmt='o', color=col, ms=4, lw=1.0, label='space')
    ax[1].errorbar(xc * 1.03, arr('dth_x_s68_deg'), yerr=arr('dth_x_s68_err'), fmt='^', color=ps.ACCENT, ms=3.5,
                   lw=0.8, label='X projection')
    ax[1].errorbar(xc * 0.97, arr('dth_y_s68_deg'), yerr=arr('dth_y_s68_err'), fmt='v', color=ps.ACCENT, mfc='white',
                   ms=3.5, lw=0.8, label='Y projection')
    ax[1].set_ylabel('opening-angle resolution $s_{68}$ (deg)'); ax[1].set_ylim(0, None)
    ax[1].legend(loc='upper right', fontsize=7)
    ax[2].errorbar(xc, arr('sep_s68_mm'), yerr=arr('sep_s68_err'), xerr=xe, fmt='o', color=col, ms=4, lw=1.0,
                   label='$s_{68}$')
    ax[2].plot(xc, arr('sep_bias_mm'), 's', color=col, mfc='white', ms=4, label='median (bias)')
    ax[2].axhline(0, color=ps.MUTED, lw=0.6)
    ax[2].set_ylabel('separation error (mm)')
    ax[2].legend(loc='upper right', fontsize=7)
    ps.panel_labels(ax)
    return ps.save(fig, os.path.join(OUT, name))


def merged(prefix):
    import glob
    fs = {k: sorted(glob.glob(os.path.join(WORK, f'{prefix}*_{k}.parquet'))) for k in ('overlays', 'candidates')}
    return [pd.concat([pd.read_parquet(f) for f in fs[k]], ignore_index=True) for k in ('overlays', 'candidates')]


def main():
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('run', 'smoke', 'plot', 'pairs'))
    ap.add_argument('--per-bin', type=int, default=150)
    ap.add_argument('--n-val', type=int, default=100)
    ap.add_argument('--n', type=int, default=6, help='smoke: overlays per separation bin (+ n singles, n noise)')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--work', default=None, help='work dir (default $WORK_DATA/claude-jobs/v2_twotrack)')
    ap.add_argument('--prefix', default='run', help='plot: score every <work>/<prefix>*_overlays.parquet')
    a = ap.parse_args()
    if a.work:
        WORK = a.work
    if a.cmd in ('run', 'smoke', 'pairs'):
        D = donors()
        Pq, info = pairs(D, a.per_bin, a.n_val, np.random.default_rng(a.seed))
        print(f'{len(D)} donors;', info, Pq.groupby(['mode', 'sbin']).size().to_dict(), flush=True)
        if a.cmd == 'pairs':
            return
        if a.cmd == 'smoke':
            sm = pd.concat([Pq[Pq['mode'] == 'single'].head(a.n), Pq[Pq['mode'] == 'noise'].head(a.n)] +
                           [Pq[(Pq['mode'] == 'overlay') & (Pq.sbin == k)].head(a.n) for k in range(len(SEP_BINS) - 1)])
            build(sm, D, a.jobs, tag='smoke')
            a.prefix = 'smoke'
        else:
            tag = 'run' if a.nshard == 1 else f'run_{a.shard:03d}'
            build(Pq[Pq.index % a.nshard == a.shard], D, a.jobs, tag=tag)
            if a.nshard > 1:
                return
    import glob
    tms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(WORK, f'{a.prefix}*_timing.json')))]
    E = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(WORK, f'{a.prefix}*_events.parquet')))])
    tm = dict(n_shards=len(tms), n_fitted=int(sum(t['n_fitted'] for t in tms)),
              fit_cpu_s_total=float(E['_fit_cpu_s'].sum()) if '_fit_cpu_s' in E else None)
    M, C = merged(a.prefix)
    M = M.merge(E[['event_id', '_fit_cpu_s']].rename(columns={'event_id': 'oid'}), on='oid', how='left') \
        if '_fit_cpu_s' in E else M
    tm['fit_cpu_s_mean_by_mode'] = {f'{k[0]}:{k[1]}': float(v) for k, v in
                                    M.groupby(['mode', 'sbin'])['_fit_cpu_s'].mean().items()} if '_fit_cpu_s' in M else None
    T = score_tables(M, C)
    res = summarise(T, tm)
    res['sample'] = a.prefix
    os.makedirs(OUT, exist_ok=True)
    name = 'two_track_separation' if a.prefix == 'run' else f'two_track_separation_{a.prefix}'
    json.dump(res, open(os.path.join(OUT, name + '.json'), 'w'), indent=1, default=float)
    T.to_parquet(os.path.join(WORK, f'{a.prefix}_scored.parquet'))
    print(json.dumps({k: v for k, v in res.items() if k != 'bins'}, indent=1, default=float))
    for b in res['bins']:
        print(b['lo'], b['hi'], b['n'], 'reco', b['reconstructed']['p'], 'resolved', b['resolved']['p'],
              '2mm_own', b['resolved_2mm_own']['p'], 'open', b.get('open_s68_deg'), 'sep', b.get('sep_s68_mm'),
              b.get('sep_bias_mm'), b['status'])
    print(plot(res, name))


if __name__ == '__main__':
    main()
