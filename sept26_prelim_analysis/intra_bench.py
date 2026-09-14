#!/usr/bin/env python3
"""
intra_bench -- waveform-overlay truth bench for two tracks in one chamber.

HANDOFF_INTRA_TWO_TRACK_RECO.md §5. Two clean single-track triggers of one
chamber, file tag and trigger phase are summed on the second donor's signal
strips, their hits merged, and the production seeder and fit re-run on the
result. Truth for each track is its donor's frozen single-track fit.

    python -m sept26_prelim_analysis.intra_bench build --jobs 14   # ~15 min
    python -m sept26_prelim_analysis.intra_bench floor             # hits only
    python -m sept26_prelim_analysis.intra_bench derive
    python -m sept26_prelim_analysis.make_intra_bench_report
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths  # noqa: E402

RUN, SUBRUN = 'run_145', 'stat090_0000'
ARMS = ('A', 'C')
PAD = 3
#: strips beyond the second donor's seed span that still carry its charge
REGION_MARGIN = 6
SEP_BINS = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 40.0, 80.0, 400.0])
COARSE_SEP = [0.0, 6.0, 12.0, 24.0, 80.0, 400.0]
CLASSES = ('coincident', 'between', 'offset')
COINC_NS = 30.0
#: above DT_XY_TOL_NS = 120, so an offset pair is separable by time alone
OFFSET_NS = 150.0
PER_CELL = 60
MAX_REUSE = 3
MATCH_MM = 3.0
N_SANITY = 8
N_NOISE_PER_CELL = 20
SCHEMA = 'sept26_prelim/intra_bench/2'

DONOR_COLS = ['tag', 'event_id', 'x_p0', 'y_p0', 'x_t0', 'y_t0', 'x_ftst', 'y_ftst',
              'x_tan_theta', 'y_tan_theta', 'x_dchi2', 'y_dchi2', 'x_n_strips',
              'y_n_strips', 'chi2dof_x', 'chi2dof_y', 'x_q_sum', 'y_q_sum']


def out_dir(variant: str = '') -> Path:
    """The production baseline, or ``<baseline>/<variant>`` for an A/B run."""
    return paths.out('intra_bench', variant) if variant else paths.out('intra_bench')


def reco_dir(arm: str) -> Path:
    return paths.spell('out', 'reco_fullpass', RUN, SUBRUN, f'mx17_{arm}')


# --------------------------------------------------------------------------- #
# donors and pairs
# --------------------------------------------------------------------------- #
def donors(arm: str) -> pd.DataFrame:
    """Clean single-track triggers: the only gated track of its chamber, both
    slopes measured, pointing at the capsule, one candidate per plane."""
    src = paths.require(paths.spell('out', 'stage3_fullpass',
                                    f'tracks_{RUN}_{SUBRUN}.parquet'), 'stage-3 tracks')
    t = pd.read_parquet(src)
    t = t[(t.arm == arm) & t.gated]
    mult = t.groupby(['tag', 'event_id']).event_id.transform('size')
    d = t[(mult == 1) & t.x_slope_reliable & t.y_slope_reliable & t.angle_calibrated
          & (t.dca_axis_mm < 30) & (t.n_cand_x == 1) & (t.n_cand_y == 1)]
    return d[DONOR_COLS].reset_index(drop=True)


def time_class(dt_x, dt_y) -> np.ndarray:
    dt_x, dt_y = np.asarray(dt_x, float), np.asarray(dt_y, float)
    return np.where((dt_x < COINC_NS) & (dt_y < COINC_NS), 'coincident',
                    np.where((dt_x > OFFSET_NS) & (dt_y > OFFSET_NS), 'offset', 'between'))


def sample_pairs(D: pd.DataFrame, seed: int, per_cell: int) -> pd.DataFrame:
    """Donor pairs of one tag and one trigger phase, stratified on the smaller
    of the two per-view separations and on the time difference."""
    frames = []
    for _key, g in D.groupby(['tag', 'x_ftst', 'y_ftst']):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        frames.append(pd.DataFrame(dict(a=idx[i], b=idx[j])))
    P = pd.concat(frames, ignore_index=True)
    A, B = D.loc[P.a].reset_index(drop=True), D.loc[P.b].reset_index(drop=True)
    P['tag'] = A.tag.to_numpy()
    P['a_eid'], P['b_eid'] = A.event_id.to_numpy(), B.event_id.to_numpy()
    for v in 'xy':
        P[f'sep_{v}'] = np.abs(A[f'{v}_p0'] - B[f'{v}_p0']).to_numpy()
        P[f'dt_{v}'] = np.abs(A[f'{v}_t0'] - B[f'{v}_t0']).to_numpy()
    P['sep_min'] = np.minimum(P.sep_x, P.sep_y)
    P['cls'] = time_class(P.dt_x, P.dt_y)
    P['sbin'] = np.digitize(P.sep_min, SEP_BINS) - 1
    P = P[(P.sbin >= 0) & (P.sbin < len(SEP_BINS) - 1)]
    P = P.sample(frac=1.0, random_state=seed)
    need = {(c, k): per_cell for c in CLASSES for k in range(len(SEP_BINS) - 1)}
    use, keep = Counter(), []
    for r in P.itertuples():
        key = (r.cls, r.sbin)
        if need[key] <= 0 or use[r.a] >= MAX_REUSE or use[r.b] >= MAX_REUSE:
            continue
        keep.append(r.Index)
        need[key] -= 1
        use[r.a] += 1
        use[r.b] += 1
    return P.loc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# overlays
# --------------------------------------------------------------------------- #
class TagData:
    """One file tag of one arm: hits, waveforms of the events needed, seeds."""

    def __init__(self, arm, tag, cfg, cal, pos, eids, rng, local_mm: float = 0.0,
                 local_mode: str = 'rescue', split_gap_mm: float = 0.0):
        from ntof_tracking import wft_beam as wb
        from wft import io as wio
        self.arm, self.tag, self.cfg, self.cal, self.pos = arm, tag, cfg, cal, pos
        self.local_mm, self.local_mode = float(local_mm), local_mode
        self.split_gap_mm = float(split_gap_mm)
        self.feu = {'x': cfg.MX17_FEU_X, 'y': cfg.MX17_FEU_Y}
        self.hits = wb.read_hits_tag(wb.hits_file_for_tag(cfg, tag),
                                     tuple(self.feu.values()))
        # donor bookkeeping (signal regions) always on the production floor, so
        # every variant overlays exactly the same strips
        self.seeds = wb.seeds_from_hits_beam(self.hits, pos, self.feu['x'],
                                             self.feu['y'], hot=cal.hot, local_mm=0.0)
        self.hits_by = {int(e): g for e, g in self.hits.groupby('eventId')}
        self.rdr, self.wf, self.ftst = {}, {'x': {}, 'y': {}}, {'x': {}, 'y': {}}
        for p, feu in self.feu.items():
            f = [f for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu)
                 if wio.file_tag(f) == tag]
            self.rdr[p] = wio.FeuReader(f[0])
        ids = set(self.rdr['x'].event_ids.tolist()) & set(self.rdr['y'].event_ids.tolist())
        empties = np.array(sorted(ids - set(self.hits_by)))
        self.empties = rng.choice(empties, size=min(len(empties), 200), replace=False)
        want = set(int(e) for e in eids) | set(int(e) for e in self.empties)
        for p in 'xy':
            for eid, ft, w in self.rdr[p].iter_events(want):
                self.wf[p][eid], self.ftst[p][eid] = w, ft
        self.order = {}
        for p, feu in self.feu.items():
            o = np.argsort(pos[feu])
            self.order[p] = o[np.isfinite(pos[feu][o])]

    def region(self, eid: int, plane: str) -> np.ndarray:
        """Channels carrying donor ``eid``'s charge: its seed span +- margin."""
        return seed_region(self.seeds[eid][plane], self.order[plane])

    def payload(self, oid: int, a: int, b: int | None, mode: str):
        """``mode``: 'single' (a alone), 'overlay' (a + b's signal strips),
        'noise' (a + a charge-free trigger's waveforms on b's strips)."""
        from ntof_tracking import wft_beam as wb
        from wft import io as wio
        W, H = {}, [self.hits_by[a]]
        e = int(self.empties[oid % len(self.empties)]) if mode == 'noise' else None
        for p, feu in self.feu.items():
            W[p] = self.wf[p][a].copy()
            if mode == 'single':
                continue
            reg = self.region(b, p)
            src = self.wf[p][b] if mode == 'overlay' else self.wf[p][e]
            if src.shape != W[p].shape:
                raise ValueError(f'sample count differs: {a} vs {b if e is None else e}')
            W[p][reg] += src[reg]
            if mode == 'overlay':
                hb = self.hits_by[b]
                H.append(hb[(hb.feu == feu) & hb.channel.isin(reg)])
        h = pd.concat(H, ignore_index=True).assign(eventId=oid)
        h = h.sort_values('amplitude').drop_duplicates(['feu', 'channel'], keep='last')
        sd = wb.seeds_from_hits_beam(h, self.pos, self.feu['x'], self.feu['y'],
                                     hot=self.cal.hot, local_mm=self.local_mm,
                                     local_mode=self.local_mode,
                                     split_gap_mm=self.split_gap_mm).get(oid)
        if sd is None:
            return None, {}
        wins, used = {}, {}
        for p, feu in self.feu.items():
            ws, us = [], []
            for s in sd[p]:
                win = wio.extract_window(W[p], self.rdr[p].noise, self.pos[feu],
                                         s.channels, PAD)
                if win is None:
                    continue
                ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch))
                us.append(s)
            if ws:
                wins[p], used[p] = ws, us
        ext = {p: [(float(np.nanmin(self.pos[self.feu[p]][s.channels])),
                    float(np.nanmax(self.pos[self.feu[p]][s.channels])), int(s.n_strips))
                   for s in used.get(p, [])] for p in 'xy'}
        ftst = {p: self.ftst[p][a] for p in 'xy'}
        return (oid, wins, used, sd['n_hits'], False, ftst), dict(seeds=ext, empty=e)


def seed_region(seeds, order: np.ndarray) -> np.ndarray:
    rank = np.flatnonzero(np.isin(order, np.concatenate([c.channels for c in seeds])))
    lo = max(0, rank.min() - REGION_MARGIN)
    hi = min(len(order) - 1, rank.max() + REGION_MARGIN)
    return order[lo:hi + 1]


def _seed_of(ext, p0):
    for k, (lo, hi, _n) in enumerate(ext):
        if lo - MATCH_MM <= p0 <= hi + MATCH_MM:
            return k
    return -1


def build(arms, jobs: int, per_cell: int, seed: int, variant: str = '',
          pairing: bool = False, local_mm: float = 0.0, local_mode: str = 'rescue',
          split_gap_mm: float = 0.0) -> None:
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft import reco as wr
    from wft.calib import CalibrationBundle

    od = out_dir(variant)
    rng = np.random.default_rng(seed)
    M, C, DON = [], [], []
    oid = 0
    t_start = time.time()
    for arm in arms:
        bundle = str(paths.require(reco_dir(arm) / 'calib_bundle_prelim', f'arm {arm} bundle'))
        cal = CalibrationBundle.load(bundle)
        cfg = wb.beam_config(arm, run=RUN, sub_run=SUBRUN)
        pos = wio.strip_position_map(cfg)
        D = donors(arm)
        P = sample_pairs(D, seed, per_cell)
        DON.append(D.assign(arm=arm))
        print(f'[bench] arm {arm}: {len(D):,} donors, {len(P):,} pairs '
              f'({P.groupby("cls").size().to_dict()})', flush=True)
        noise_rows = P.groupby(['cls', 'sbin'], group_keys=False).head(N_NOISE_PER_CELL)
        Dk = D.set_index(['tag', 'event_id'])
        pairing_path = (str(paths.require(out_dir() / f'xy_pairing_{arm}.json',
                                          'x/y pairing calibration (run calib-pairing)'))
                        if pairing else None)
        with ProcessPoolExecutor(max_workers=jobs, initializer=wr._worker_init,
                                 initargs=(bundle, pairing_path)) as pool:
            for tag, pt in P.groupby('tag'):
                san = D[D.tag == tag].event_id.head(N_SANITY).tolist()
                td = TagData(arm, tag, cfg, cal, pos, set(pt.a_eid) | set(pt.b_eid) | set(san),
                             rng, local_mm=local_mm, local_mode=local_mode,
                             split_gap_mm=split_gap_mm)
                todo = [('single', int(e), None, None) for e in san]
                for r in pt.itertuples():
                    todo.append(('overlay', int(r.a_eid), int(r.b_eid), r))
                    if r.Index in noise_rows.index:
                        todo.append(('noise', int(r.a_eid), int(r.b_eid), r))
                payloads, metas = [], []
                for mode, a, b, r in todo:
                    pl, info = td.payload(oid, a, b, mode)
                    meta = dict(oid=oid, arm=arm, tag=tag, mode=mode, a_eid=a,
                                b_eid=-1 if b is None else b,
                                empty_eid=-1 if info.get('empty') is None else info['empty'],
                                seeded=pl is not None)
                    if r is not None:
                        meta.update(cls=r.cls, sbin=int(r.sbin), sep_x=r.sep_x,
                                    sep_y=r.sep_y, dt_x=r.dt_x, dt_y=r.dt_y)
                    ta = Dk.loc[(tag, a)]
                    tb = Dk.loc[(tag, b)] if b is not None else None
                    for p in 'xy':
                        ext = info.get('seeds', {}).get(p, [])
                        meta[f'n_seeds_{p}'] = len(ext)
                        ka = _seed_of(ext, ta[f'{p}_p0'])
                        kb = _seed_of(ext, tb[f'{p}_p0']) if tb is not None else -1
                        meta[f'seed_a_{p}'], meta[f'seed_b_{p}'] = ka, kb
                        meta[f'merged_{p}'] = bool(mode == 'overlay' and ka >= 0 and ka == kb)
                        meta[f'seed_w_a_{p}'] = (ext[ka][1] - ext[ka][0]) if ka >= 0 else np.nan
                        meta[f'seed_w_b_{p}'] = (ext[kb][1] - ext[kb][0]) if kb >= 0 else np.nan
                    metas.append(meta)
                    if pl is not None:
                        payloads.append(pl)
                    oid += 1
                t0 = time.time()
                by_oid = {m['oid']: m for m in metas}
                for row in pool.map(wr._worker_fit, payloads, chunksize=2):
                    for c in row.pop('_cand', []):
                        c['oid'] = row['event_id']
                        C.append(c)
                    by_oid[row['event_id']]['n_tracks'] = row['n_tracks']
                M.extend(metas)
                print(f'[bench]   {arm} {tag}: {len(payloads):,} payloads in '
                      f'{time.time() - t0:.0f} s', flush=True)
    Mdf = pd.DataFrame(M)
    pd.DataFrame(C).to_parquet(od / 'candidates.parquet', index=False)
    Mdf.to_parquet(od / 'overlays.parquet', index=False)
    pd.concat(DON, ignore_index=True).to_parquet(od / 'donors.parquet', index=False)
    (od / 'build.meta.json').write_text(json.dumps(dict(
        schema=SCHEMA, run=RUN, subrun=SUBRUN, arms=list(arms), seed=seed,
        per_cell=per_cell, classes=list(CLASSES), region_margin=REGION_MARGIN, pad=PAD,
        sep_bins=SEP_BINS.tolist(), coinc_ns=COINC_NS, offset_ns=OFFSET_NS,
        match_mm=MATCH_MM, n_overlays=int(len(Mdf)),
        variant=variant or 'production', xy_pairing=bool(pairing), sig_floor_local_mm=float(local_mm),
        sig_floor_local_mode=local_mode, split_gap_mm=float(split_gap_mm),
        minutes=round((time.time() - t_start) / 60, 1),
        built=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1))
    print(f'[bench] wrote {od}')


# --------------------------------------------------------------------------- #
# the significance floor, hits only
# --------------------------------------------------------------------------- #
FLOOR_RATIO_BINS = [0.0, 1.0, 2.0, 4.0, 8.0, np.inf]
FLOOR_SEP_BINS = [0.0, 12.0, 24.0, 80.0, 400.0]


def local_floor(half_mm: float):
    """10 % of the brightest strip within +-half_mm, instead of the plane."""
    from wft import seed as wseed

    def f(df, rel=wseed.SIG_REL_FLOOR):
        if not rel or len(df) == 0:
            return df
        keep = np.zeros(len(df), bool)
        sig, p = df.significance.to_numpy(), df.pos.to_numpy()
        for idx in df.groupby(['eventId', 'feu']).indices.values():
            idx = idx[np.isfinite(p[idx])]
            if not len(idx):
                continue
            ii = idx[np.argsort(p[idx])]
            pp, ss = p[ii], sig[ii]
            lo = np.searchsorted(pp, pp - half_mm, 'left')
            hi = np.searchsorted(pp, pp + half_mm, 'right')
            mx = np.array([ss[l:h].max() for l, h in zip(lo, hi)])
            keep[ii] = ss >= rel * mx
        return df[keep].copy()
    return f


def floor_study(arms, pairs_per_tag: int, seed: int) -> None:
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft import seed as wseed
    from wft.calib import CalibrationBundle

    prod = wseed.apply_significance_floor
    variants = {'production': prod, 'local_16mm': local_floor(16.0),
                'local_40mm': local_floor(40.0), 'no_floor': lambda df, rel=0: df}

    def seeds_with(fn, h, pos, fx, fy, hot):
        wseed.apply_significance_floor = fn
        try:
            return wb.seeds_from_hits_beam(h, pos, fx, fy, hot=hot)
        finally:
            wseed.apply_significance_floor = prod

    def seeded(seeds, pos_feu, p0):
        return any(np.nanmin(pos_feu[s.channels]) - MATCH_MM <= p0
                   <= np.nanmax(pos_feu[s.channels]) + MATCH_MM for s in seeds)

    def key(r):
        return None if r is None else tuple(
            tuple(sorted(int(c) for s in r[p] for c in s.channels)) for p in 'xy')

    rng = np.random.default_rng(seed)
    rows, side = [], []
    for arm in arms:
        cal = CalibrationBundle.load(str(reco_dir(arm) / 'calib_bundle_prelim'))
        cfg = wb.beam_config(arm, run=RUN, sub_run=SUBRUN)
        pos = wio.strip_position_map(cfg)
        fx, fy = cfg.MX17_FEU_X, cfg.MX17_FEU_Y
        feu = {'x': fx, 'y': fy}
        order = {}
        for p, f in feu.items():
            o = np.argsort(pos[f])
            order[p] = o[np.isfinite(pos[f][o])]
        D = donors(arm)
        for tag, Dt in D.groupby('tag'):
            hits = wb.read_hits_tag(wb.hits_file_for_tag(cfg, tag), (fx, fy))
            hits['pos'] = np.where(hits.feu == fx, pos[fx][hits.channel], pos[fy][hits.channel])
            by = {int(e): g for e, g in hits.groupby('eventId')}
            base = wb.seeds_from_hits_beam(hits, pos, fx, fy, hot=cal.hot)
            dn = set(int(e) for e in Dt.event_id)
            for v, fn in variants.items():
                if v == 'production':
                    continue
                alt = seeds_with(fn, hits, pos, fx, fy, cal.hot)
                ev = set(base) | set(alt)
                side.append(dict(arm=arm, tag=tag, variant=v, n_triggers=len(ev),
                                 changed=sum(key(base.get(e)) != key(alt.get(e)) for e in ev),
                                 n_donors=len(dn),
                                 donors_changed=sum(key(base.get(e)) != key(alt.get(e)) for e in dn)))
            ids = Dt.event_id.to_numpy()
            n = min(pairs_per_tag, len(ids) * (len(ids) - 1) // 2)
            i, j = rng.integers(0, len(ids), 3 * n), rng.integers(0, len(ids), 3 * n)
            ok = i != j
            Dx = Dt.set_index('event_id')
            for a, b in zip(ids[i[ok][:n]], ids[j[ok][:n]]):
                H = [by[a]]
                for p, f in feu.items():
                    hb = by[b]
                    H.append(hb[(hb.feu == f) & hb.channel.isin(seed_region(base[b][p], order[p]))])
                h = (pd.concat(H).sort_values('amplitude')
                     .drop_duplicates(['feu', 'channel'], keep='last').assign(eventId=0))
                ratio = {p: by[b][by[b].feu == f].significance.max()
                         / by[a][by[a].feu == f].significance.max() for p, f in feu.items()}
                for v, fn in variants.items():
                    sd = seeds_with(fn, h, pos, fx, fy, cal.hot).get(0, {'x': [], 'y': []})
                    for p, f in feu.items():
                        rows.append(dict(arm=arm, variant=v, plane=p, ratio=ratio[p],
                                         sep=abs(Dx.loc[a, f'{p}_p0'] - Dx.loc[b, f'{p}_p0']),
                                         seeded=seeded(sd[p], pos[f], Dx.loc[a, f'{p}_p0'])))
            print(f'[floor] {arm} {tag}: done', flush=True)
    R = pd.DataFrame(rows)
    R['ratio_lo'] = pd.cut(R.ratio, FLOOR_RATIO_BINS, right=False,
                           labels=FLOOR_RATIO_BINS[:-1]).astype(float)
    R['sep_lo'] = pd.cut(R.sep, FLOOR_SEP_BINS, labels=FLOOR_SEP_BINS[:-1]).astype(float)
    od = out_dir()
    (R.groupby(['arm', 'plane', 'variant', 'ratio_lo']).seeded.agg(['mean', 'size'])
     .rename(columns={'mean': 'seeded', 'size': 'n'}).reset_index()
     .to_csv(od / 'floor_by_ratio.csv', index=False))
    (R.groupby(['arm', 'plane', 'variant', 'sep_lo']).seeded.agg(['mean', 'size'])
     .rename(columns={'mean': 'seeded', 'size': 'n'}).reset_index()
     .to_csv(od / 'floor_by_sep.csv', index=False))
    Sd = pd.DataFrame(side).groupby(['arm', 'variant'])[
        ['n_triggers', 'changed', 'n_donors', 'donors_changed']].sum().reset_index()
    Sd['frac_triggers_changed'] = Sd.changed / Sd.n_triggers
    Sd['frac_donors_changed'] = Sd.donors_changed / Sd.n_donors
    Sd.to_csv(od / 'floor_side_effect.csv', index=False)
    print(f'[floor] wrote {od}')


# --------------------------------------------------------------------------- #
# scoring the overlays
# --------------------------------------------------------------------------- #
def rsig(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    q = np.percentile(v, [16, 84])
    return float(0.5 * (q[1] - q[0]))


def score(M: pd.DataFrame, C: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """One row per (overlay, donor): plane-level and track-level recovery."""
    truth = D.set_index(['arm', 'tag', 'event_id'])
    cand = {k: g for k, g in C.groupby('oid')}
    rows = []
    for m in M.itertuples():
        g = cand.get(m.oid)
        donors_ = [('a', m.a_eid)] + ([('b', m.b_eid)] if m.mode == 'overlay' else [])
        tr = {r: truth.loc[(m.arm, m.tag, e)] for r, e in donors_}
        tracks = {}
        if g is not None:
            gg = g[(g.track_id >= 0) & g.track_gated]
            for tid, h in gg.groupby('track_id'):
                hx, hy = h[h.plane == 'x'], h[h.plane == 'y']
                if len(hx) and len(hy):
                    tracks[tid] = (hx.iloc[0], hy.iloc[0])

        def near(plane, p0):
            if g is None:
                return None
            h = g[g.plane == plane]
            if not len(h):
                return None
            return h.iloc[int(np.argmin(np.abs(h.p0.to_numpy() - p0)))]

        assign = {}
        for tid, (cx, cy) in tracks.items():
            dx = {r: abs(cx.p0 - t['x_p0']) for r, t in tr.items()}
            dy = {r: abs(cy.p0 - t['y_p0']) for r, t in tr.items()}
            rx, ry = min(dx, key=dx.get), min(dy, key=dy.get)
            assign[tid] = (rx if dx[rx] < MATCH_MM else None,
                           ry if dy[ry] < MATCH_MM else None)
        for r, t in tr.items():
            row = dict(oid=m.oid, arm=m.arm, mode=m.mode, donor=r,
                       cls=getattr(m, 'cls', None), sbin=getattr(m, 'sbin', np.nan),
                       sep_x=getattr(m, 'sep_x', np.nan), sep_y=getattr(m, 'sep_y', np.nan),
                       seeded=m.seeded, n_tracks=getattr(m, 'n_tracks', np.nan),
                       merged_x=m.merged_x, merged_y=m.merged_y,
                       seed_lost=bool(getattr(m, f'seed_{r}_x') < 0 or getattr(m, f'seed_{r}_y') < 0))
            for p in 'xy':
                c = near(p, t[f'{p}_p0'])
                ok = c is not None and abs(c.p0 - t[f'{p}_p0']) < MATCH_MM
                row[f'cand_found_{p}'] = bool(ok)
                row[f'dp0_{p}'] = float(c.p0 - t[f'{p}_p0']) if ok else np.nan
                row[f'dtan_{p}'] = float(c.tan_theta - t[f'{p}_tan_theta']) if ok else np.nan
                row[f'strips_{p}'] = float(c.n_strips) if ok else np.nan
                row[f'strips0_{p}'] = float(t[f'{p}_n_strips'])
                row[f'chi2dof_{p}'] = float(c.chi2 / max(c.dof, 1)) if ok else np.nan
                row[f'chi2dof0_{p}'] = float(t[f'chi2dof_{p}'])
            row['track_found'] = any(ax == r and ay == r for ax, ay in assign.values())
            row['in_swapped_track'] = any(ax is not None and ay is not None and ax != ay
                                          for ax, ay in assign.values() if r in (ax, ay))
            if m.mode == 'overlay':
                o = 'b' if r == 'a' else 'a'
                row['rank_disagree'] = bool(np.sign(t['x_dchi2'] - tr[o]['x_dchi2'])
                                            != np.sign(t['y_dchi2'] - tr[o]['y_dchi2']))
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, mode, cls, sbin), g in S.groupby(['arm', 'mode', 'cls', 'sbin'], dropna=False):
        ev = g.groupby('oid')
        both = ev.track_found.all()
        row = dict(arm=arm, mode=mode, cls=cls, sep_lo=SEP_BINS[int(sbin)],
                   sep_hi=SEP_BINS[int(sbin) + 1], n_events=int(ev.ngroups),
                   eff_tracks=float(both.mean()),
                   eff_cand_x=float(g.cand_found_x.mean()), eff_cand_y=float(g.cand_found_y.mean()),
                   swapped=float(g.in_swapped_track.mean()),
                   merged_x=float(g.merged_x.mean()), merged_y=float(g.merged_y.mean()),
                   n_tracks_ge2=float((ev.n_tracks.first() >= 2).mean()))
        for p in 'xy':
            row[f'rsig_dp0_{p}'] = rsig(g[f'dp0_{p}'])
            row[f'rsig_dtan_{p}'] = rsig(g[f'dtan_{p}'])
            row[f'strips_{p}'] = float(g[f'strips_{p}'].median())
            row[f'strips0_{p}'] = float(g[f'strips0_{p}'].median())
            row[f'chi2dof_{p}'] = float(g[f'chi2dof_{p}'].median())
            row[f'chi2dof0_{p}'] = float(g[f'chi2dof0_{p}'].median())
        rows.append(row)
    return pd.DataFrame(rows)


OUTCOMES = ('found', 'swapped', 'merged', 'seed_lost', 'fit_elsewhere', 'unpaired')


def outcomes(S: pd.DataFrame) -> pd.DataFrame:
    """Where each donor track of an overlay ends up, first cause wins."""
    ov = S[S['mode'] == 'overlay'].copy()
    ov['outcome'] = np.select(
        [ov.track_found.to_numpy(bool), ov.seed_lost.to_numpy(bool),
         (ov.merged_x | ov.merged_y).to_numpy(bool),
         ~(ov.cand_found_x & ov.cand_found_y).to_numpy(bool),
         ov.in_swapped_track.to_numpy(bool)],
        ['found', 'seed_lost', 'merged', 'fit_elsewhere', 'swapped'], 'unpaired')
    ov['sep_lo'] = pd.cut(np.minimum(ov.sep_x, ov.sep_y), COARSE_SEP,
                          labels=COARSE_SEP[:-1]).astype(float)
    tab = pd.crosstab([ov.arm, ov.cls, ov.sep_lo], ov.outcome, normalize='index')
    tab = tab.reindex(columns=list(OUTCOMES), fill_value=0.0)
    tab['n_donors'] = ov.groupby(['arm', 'cls', 'sep_lo']).size()
    return tab.reset_index()


# --------------------------------------------------------------------------- #
# x <-> y pairing: what the planes share besides t0
# --------------------------------------------------------------------------- #
FEATS = ('lq', 'u50', 'u90', 't0')
RULES = {'time': ('t0',), 'charge': ('lq',), 'profile': ('u50', 'u90'),
         'charge+profile': ('lq', 'u50', 'u90'), 'all': FEATS}
S3_COLS = ['subrun', 'tag', 'event_id', 'arm', 'track_id', 'gated', 'n_cand_x', 'n_cand_y',
           'x_p0', 'y_p0', 'x_q_sum', 'y_q_sum', 'x_q_u50', 'y_q_u50', 'x_q_u90', 'y_q_u90',
           'x_t0', 'y_t0', 'x_ftst', 'y_ftst']


def dt_xy(arm: str) -> dict:
    b = json.loads((reco_dir(arm) / 'calib_bundle_prelim' / 'bundle.json').read_text())
    return {int(k): float(v) for k, v in b['dt_xy'].items()}


def xy_features(dt, qx, qy, u50x, u50y, u90x, u90y, t0x, t0y, fx, fy) -> dict:
    fd = np.atleast_1d(np.asarray(fx, int) - np.asarray(fy, int))
    off = np.array([dt.get(int(k), -18.8) for k in fd])
    return dict(lq=np.log(np.maximum(np.asarray(qx, float), 1.0)
                          / np.maximum(np.asarray(qy, float), 1.0)),
                u50=np.asarray(u50x, float) - np.asarray(u50y, float),
                u90=np.asarray(u90x, float) - np.asarray(u90y, float),
                t0=np.asarray(t0x, float) - np.asarray(t0y, float) - off)


def stage3_all() -> pd.DataFrame:
    fs = sorted(glob.glob(str(paths.spell('out', 'stage3_fullpass', f'tracks_{RUN}_stat090_*.parquet'))))
    T = pd.concat([pd.read_parquet(f, columns=S3_COLS) for f in fs], ignore_index=True)
    T = T[T.gated].copy()
    T['mult'] = T.groupby(['subrun', 'tag', 'event_id', 'arm']).event_id.transform('size')
    return T


def calibrate_features(T: pd.DataFrame, exclude_subrun: str | None = None):
    """Median and robust width of each x-minus-y feature on clean single tracks."""
    cal, rows = {}, []
    for arm in ARMS:
        s = T[(T.arm == arm) & (T.mult == 1) & (T.n_cand_x == 1) & (T.n_cand_y == 1)
              & (T.subrun != exclude_subrun)]
        F = xy_features(dt_xy(arm), s.x_q_sum, s.y_q_sum, s.x_q_u50, s.y_q_u50,
                        s.x_q_u90, s.y_q_u90, s.x_t0, s.y_t0, s.x_ftst, s.y_ftst)
        cal[arm] = {}
        for k in FEATS:
            v = F[k][np.isfinite(F[k])]
            cal[arm][k] = (float(np.median(v)), rsig(v))
            rows.append(dict(arm=arm, feature=k, median=cal[arm][k][0], rsig=cal[arm][k][1], n=len(v)))
    return cal, pd.DataFrame(rows)


def pair_cost(cal, arm, F, rule) -> np.ndarray:
    tot = 0.0
    for k in RULES[rule]:
        med, sig = cal[arm][k]
        z = (np.asarray(F[k], float) - med) / sig
        tot = tot + np.minimum(np.nan_to_num(z * z, nan=25.0), 25.0)
    return np.atleast_1d(np.asarray(tot, float))


def bench_pairing(M, C, D, S, cal) -> pd.DataFrame:
    """For overlays with both tracks present in both planes: production's
    pairing, and which assignment each rule prefers."""
    Dx = D.set_index(['arm', 'tag', 'event_id'])
    rd = S[(S['mode'] == 'overlay') & (S.donor == 'a')].set_index('oid').rank_disagree
    dts = {a: dt_xy(a) for a in ARMS}
    cand = {k: g for k, g in C.groupby('oid')}
    rows = []
    for m in M[M['mode'] == 'overlay'].itertuples():
        g = cand.get(m.oid)
        if g is None:
            continue
        lab = {}
        for p in 'xy':
            h = g[g.plane == p]
            for who, e in (('a', m.a_eid), ('b', m.b_eid)):
                d = np.abs(h.p0.to_numpy() - Dx.loc[(m.arm, m.tag, e), f'{p}_p0'])
                if len(d) and d.min() < MATCH_MM:
                    lab[(p, who)] = h.iloc[int(np.argmin(d))]
        if len(lab) < 4 or any(lab[(p, 'a')].name == lab[(p, 'b')].name for p in 'xy'):
            continue
        xa, xb, ya, yb = lab[('x', 'a')], lab[('x', 'b')], lab[('y', 'a')], lab[('y', 'b')]
        if xa.track_id >= 0 and xa.track_id == ya.track_id and xb.track_id >= 0 and xb.track_id == yb.track_id:
            prod = 'correct'
        elif xa.track_id >= 0 and xa.track_id == yb.track_id and xb.track_id >= 0 and xb.track_id == ya.track_id:
            prod = 'swapped'
        else:
            prod = 'incomplete'
        r = dict(oid=m.oid, arm=m.arm, cls=m.cls, sep=min(m.sep_x, m.sep_y),
                 rank_disagree=bool(rd.get(m.oid, False)), production=prod)

        def F(x, y):
            return xy_features(dts[m.arm], x.q_sum, y.q_sum, x.q_u50, y.q_u50,
                               x.q_u90, y.q_u90, x.t0, y.t0, x.ftst, y.ftst)
        for rule in RULES:
            st = pair_cost(cal, m.arm, F(xa, ya), rule)[0] + pair_cost(cal, m.arm, F(xb, yb), rule)[0]
            cr = pair_cost(cal, m.arm, F(xa, yb), rule)[0] + pair_cost(cal, m.arm, F(xb, ya), rule)[0]
            r[rule] = 0.5 if st == cr else float(st < cr)
        rows.append(r)
    return pd.DataFrame(rows)


def pairing_summary(R: pd.DataFrame) -> pd.DataFrame:
    R = R.assign(prod_correct=(R.production == 'correct').astype(float),
                 prod_swapped=(R.production == 'swapped').astype(float),
                 prod_incomplete=(R.production == 'incomplete').astype(float))
    agg = dict(n=('oid', 'size'), prod_correct=('prod_correct', 'mean'),
               prod_swapped=('prod_swapped', 'mean'), prod_incomplete=('prod_incomplete', 'mean'),
               **{k: (k, 'mean') for k in RULES})
    a = R.groupby(['arm', 'cls']).agg(**agg).reset_index().assign(rank_disagree='all')
    b = R.groupby(['arm', 'cls', 'rank_disagree']).agg(**agg).reset_index()
    b['rank_disagree'] = b.rank_disagree.map({True: 'yes', False: 'no'})
    return pd.concat([a, b], ignore_index=True)


def data_pairing(T: pd.DataFrame, cal, PS: pd.DataFrame) -> pd.DataFrame:
    """Real two-track chambers: how often each rule prefers the pairing
    production did not choose, and the swap fraction that implies if the
    rule is as accurate on data as on the bench."""
    frames = []
    for arm in ARMS:
        t = T[(T.arm == arm) & (T.mult == 2)].sort_values(['subrun', 'tag', 'event_id', 'track_id'])
        a, b = t.iloc[0::2].reset_index(drop=True), t.iloc[1::2].reset_index(drop=True)
        if not ((a.event_id.to_numpy() == b.event_id.to_numpy()).all()
                and (a.tag.to_numpy() == b.tag.to_numpy()).all()):
            raise AssertionError('two-track chambers did not pair up row by row')
        dt = dt_xy(arm)

        def F(x, y):
            return xy_features(dt, x.x_q_sum, y.y_q_sum, x.x_q_u50, y.y_q_u50, x.x_q_u90,
                               y.y_q_u90, x.x_t0, y.y_t0, x.x_ftst, y.y_ftst)
        d = pd.DataFrame(dict(arm=arm, cls=time_class(np.abs(a.x_t0 - b.x_t0), np.abs(a.y_t0 - b.y_t0))))
        for rule in RULES:
            cur = pair_cost(cal, arm, F(a, a), rule) + pair_cost(cal, arm, F(b, b), rule)
            alt = pair_cost(cal, arm, F(a, b), rule) + pair_cost(cal, arm, F(b, a), rule)
            d[rule] = np.where(cur == alt, 0.5, (alt < cur).astype(float))
        frames.append(d)
    R = pd.concat(frames, ignore_index=True)
    out = R.groupby(['arm', 'cls']).agg(n=('arm', 'size'), **{k: (k, 'mean') for k in RULES}).reset_index()
    acc = PS[PS.rank_disagree == 'all'].set_index(['arm', 'cls'])['charge+profile']
    out['bench_acc'] = [acc.get((r.arm, r.cls), np.nan) for r in out.itertuples()]
    p, A = out['charge+profile'], out.bench_acc
    out['implied_swap_frac'] = (p - (1 - A)) / (2 * A - 1)
    out['implied_swap_err'] = np.sqrt(p * (1 - p) / out.n) / (2 * A - 1)
    return out


def derive(variant: str = '') -> None:
    od = out_dir(variant)
    M = pd.read_parquet(od / 'overlays.parquet')
    C = pd.read_parquet(od / 'candidates.parquet')
    D = pd.read_parquet(od / 'donors.parquet')
    S = score(M, C, D)
    S.to_parquet(od / 'scores.parquet', index=False)
    s1 = S[S['mode'] == 'single']
    n_bad = int((~(s1.cand_found_x & s1.cand_found_y) | (s1.dp0_x.abs() > 1e-6)
                 | (s1.dp0_y.abs() > 1e-6)).sum())
    T = summarise(S[S['mode'] != 'single'])
    T.to_csv(od / 'summary.csv', index=False)
    ov = S[S['mode'] == 'overlay']
    (ov.groupby(['arm', 'cls', 'rank_disagree']).in_swapped_track.agg(['mean', 'size'])
     .rename(columns={'mean': 'swapped', 'size': 'n_donors'}).reset_index()
     .to_csv(od / 'swap_vs_rank.csv', index=False))
    O = outcomes(S)
    O.to_csv(od / 'outcomes.csv', index=False)
    T3 = stage3_all()
    cal, calt = calibrate_features(T3)
    calt.to_csv(od / 'pairing_features.csv', index=False)
    R = bench_pairing(M, C, D, S, cal)
    R.to_csv(od / 'pairing_rules.csv', index=False)
    PS = pairing_summary(R)
    PS.to_csv(od / 'pairing_rules_summary.csv', index=False)
    DP = data_pairing(T3, cal, PS)
    DP.to_csv(od / 'data_pairing.csv', index=False)
    (od / 'derive.meta.json').write_text(json.dumps(dict(
        schema=SCHEMA, n_single_refits=int(len(s1)), n_single_refits_differ=n_bad,
        data_subruns=sorted(T3.subrun.unique().tolist()),
        derived=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(f'sanity: {len(s1)} single-donor re-fits, {n_bad} differ from the frozen fit')
    print(O.round(3).to_string(index=False))
    print(PS.round(3).to_string(index=False))
    print(DP.round(3).to_string(index=False))


# --------------------------------------------------------------------------- #
# the fixes: calibration, single-track A/B, comparison
# --------------------------------------------------------------------------- #
#: Chosen on the coincident overlays of the bench sub-run: charge alone is best
#: in A, charge + arrival profile in C (pairing_rules_summary.csv).
PAIRING_FEATURES = {'A': ['lq'], 'C': ['lq', 'u50', 'u90']}


def calib_pairing(arms) -> None:
    """Per-chamber ``xy_pairing`` from clean single tracks of the run's other
    local sub-runs, so the bench sub-run is not calibrated on itself."""
    from wft.calib import check_xy_pairing
    T3 = stage3_all()
    cal, tab = calibrate_features(T3, exclude_subrun=SUBRUN)
    for arm in arms:
        p = dict(features=PAIRING_FEATURES[arm],
                 median={k: cal[arm][k][0] for k in FEATS},
                 rsig={k: cal[arm][k][1] for k in FEATS},
                 provenance=dict(run=RUN, subruns=sorted(set(T3.subrun) - {SUBRUN}),
                                 n_single_tracks=int(tab[tab.arm == arm].n.max()),
                                 selection='gated, only gated track of its chamber, one candidate per plane',
                                 features_chosen_on=f'intra_bench {RUN}/{SUBRUN} coincident overlays',
                                 built=time.strftime('%Y-%m-%dT%H:%M:%S')))
        check_xy_pairing(p)
        path = out_dir() / f'xy_pairing_{arm}.json'
        path.write_text(json.dumps(p, indent=1))
        print(f'[pairing] {arm}: {p["features"]} from {p["provenance"]["n_single_tracks"]:,} '
              f'single tracks -> {path}')


def _seed_key(r):
    return None if r is None else tuple(
        tuple(sorted(int(c) for s in r[p] for c in s.channels)) for p in 'xy')


def _gated_tracks(c) -> list:
    if c is None or not len(c):
        return []
    g = c[(c.track_id >= 0) & c.track_gated]
    out = []
    for _tid, h in g.groupby('track_id'):
        hx, hy = h[h.plane == 'x'], h[h.plane == 'y']
        if len(hx) and len(hy):
            out.append((hx.iloc[0], hy.iloc[0]))
    return out


def floor_ab(arms, local_mm: float, jobs: int, local_mode: str = 'rescue',
             split_gap_mm: float = 0.0) -> None:
    """Single-track A/B of a local significance floor on the bench sub-run:
    re-fit every production-fitted trigger whose seeds the floor changes and
    compare its gated tracks with the frozen pass. Triggers whose seeds do not
    change fit identically by construction."""
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft import reco as wr
    from wft.calib import CalibrationBundle

    od = out_dir(f'floor_ab_{local_mm:g}mm' + ('_rescue' if local_mode == 'rescue' else '')
                 + (f'_split{split_gap_mm:g}' if split_gap_mm else ''))
    ev_rows, tr_rows, acct = [], [], []
    for arm in arms:
        rd = reco_dir(arm)
        bundle = str(paths.require(rd / 'calib_bundle_prelim', f'arm {arm} bundle'))
        cal = CalibrationBundle.load(bundle)
        cfg = wb.beam_config(arm, run=RUN, sub_run=SUBRUN)
        pos = wio.strip_position_map(cfg)
        fx, fy = cfg.MX17_FEU_X, cfg.MX17_FEU_Y
        clean = set(donors(arm)[['tag', 'event_id']].itertuples(index=False, name=None))
        with ProcessPoolExecutor(max_workers=jobs, initializer=wr._worker_init,
                                 initargs=(bundle,)) as pool:
            for tag in wb.subrun_tags(cfg):
                ev_path = rd / f'events_{tag}.parquet'
                if not ev_path.exists():
                    continue
                prod_n = pd.read_parquet(ev_path, columns=['event_id', 'n_tracks']).set_index('event_id').n_tracks
                prod_c = pd.read_parquet(rd / f'events_{tag}.candidates.parquet')
                hits = wb.read_hits_tag(wb.hits_file_for_tag(cfg, tag), (fx, fy))
                base = wb.seeds_from_hits_beam(hits, pos, fx, fy, hot=cal.hot, local_mm=0.0)
                alt = wb.seeds_from_hits_beam(hits, pos, fx, fy, hot=cal.hot, local_mm=local_mm,
                                              local_mode=local_mode, split_gap_mm=split_gap_mm)
                changed = {e for e in set(base) | set(alt) if _seed_key(base.get(e)) != _seed_key(alt.get(e))}
                fitted = set(int(e) for e in prod_n.index)
                todo = changed & fitted
                acct.append(dict(arm=arm, tag=tag, n_prod_fitted=len(fitted), n_changed=len(changed),
                                 n_changed_fitted=len(todo), n_lost_all_seeds=len(todo - set(alt)),
                                 n_changed_unfitted=len(changed - fitted),
                                 n_prod_gated_tracks=int(((prod_c.track_id >= 0) & prod_c.track_gated
                                                          & (prod_c.plane == 'x')).sum())))
                t0 = time.time()
                new = {}
                for r in pool.map(wr._worker_fit, wb._windows_for_tag(cfg, tag, pos, alt, todo & set(alt), PAD),
                                  chunksize=4):
                    new[r['event_id']] = (r['n_tracks'], pd.DataFrame(r.pop('_cand', [])))
                pc = {k: g for k, g in prod_c.groupby('event_id')}
                for e in sorted(todo):
                    n_new, cn = new.get(e, (0, None))
                    ev_rows.append(dict(arm=arm, tag=tag, event_id=e, n_tracks_prod=int(prod_n.get(e, 0)),
                                        n_tracks_new=int(n_new), clean_single=(tag, e) in clean))
                    tn = _gated_tracks(cn)
                    for px, py in _gated_tracks(pc.get(e)):
                        m = [(nx, ny) for nx, ny in tn
                             if abs(nx.p0 - px.p0) < MATCH_MM and abs(ny.p0 - py.p0) < MATCH_MM]
                        row = dict(arm=arm, tag=tag, event_id=e, clean_single=(tag, e) in clean,
                                   recovered=bool(m), dp0_x=np.nan, dp0_y=np.nan, dtan_x=np.nan, dtan_y=np.nan)
                        if m:
                            nx, ny = m[0]
                            row.update(dp0_x=nx.p0 - px.p0, dp0_y=ny.p0 - py.p0,
                                       dtan_x=nx.tan_theta - px.tan_theta, dtan_y=ny.tan_theta - py.tan_theta)
                        tr_rows.append(row)
                print(f'[floor-ab] {arm} {tag}: {len(todo):,} changed of {len(fitted):,} fitted, '
                      f'{time.time() - t0:.0f} s', flush=True)
    E, Tk, A = pd.DataFrame(ev_rows), pd.DataFrame(tr_rows), pd.DataFrame(acct)
    S = []
    for arm in arms:
        a = A[A.arm == arm].sum(numeric_only=True)
        e, t = E[E.arm == arm], Tk[Tk.arm == arm]
        tc = t[t.clean_single]
        S.append(dict(arm=arm, local_mm=local_mm, local_mode=local_mode, split_gap_mm=split_gap_mm,
                      triggers_fitted=int(a.n_prod_fitted),
                      seeds_changed=int(a.n_changed_fitted), lost_all_seeds=int(a.n_lost_all_seeds),
                      prod_gated_tracks=int(a.n_prod_gated_tracks), affected_gated_tracks=len(t),
                      not_recovered=int((~t.recovered).sum()),
                      frac_gated_tracks_lost=float((~t.recovered).sum() / max(a.n_prod_gated_tracks, 1)),
                      clean_singles_affected=len(tc), clean_singles_not_recovered=int((~tc.recovered).sum()),
                      rsig_dp0_x=rsig(t.dp0_x), rsig_dp0_y=rsig(t.dp0_y),
                      rsig_dtan_x=rsig(t.dtan_x), rsig_dtan_y=rsig(t.dtan_y),
                      frac_bit_identical=float(np.mean((t.dp0_x.abs() < 1e-9) & (t.dp0_y.abs() < 1e-9)))
                      if len(t) else np.nan,
                      events_more_tracks=int((e.n_tracks_new > e.n_tracks_prod).sum()),
                      events_fewer_tracks=int((e.n_tracks_new < e.n_tracks_prod).sum())))
    S = pd.DataFrame(S)
    E.to_parquet(od / 'events.parquet', index=False)
    Tk.to_parquet(od / 'tracks.parquet', index=False)
    A.to_csv(od / 'accounting.csv', index=False)
    S.to_csv(od / 'summary.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(S.round(4).to_string(index=False))


def compare(variants) -> None:
    """Paired comparison of bench variants against the production baseline:
    the same donor pairs, the same strips, a different reconstruction."""
    rows = []
    for v in [''] + list(variants):
        S = pd.read_parquet(out_dir(v) / 'scores.parquet')
        ov = S[S['mode'] == 'overlay'].copy()
        ov['merged'] = ov.merged_x | ov.merged_y
        ov['band'] = pd.cut(np.minimum(ov.sep_x, ov.sep_y), [0.0, 12.0, 24.0, 400.0],
                            labels=['<12 mm', '12-24 mm', '>=24 mm']).astype(str)
        for keys, g in list(ov.groupby(['arm', 'cls', 'band'])) + [
                ((arm, 'all', band), g) for (arm, band), g in ov.groupby(['arm', 'band'])]:
            ev = g.groupby('oid').track_found.all()
            rows.append(dict(variant=v or 'production', arm=keys[0], cls=keys[1], band=keys[2],
                             n_events=len(ev), both_found=float(ev.mean()),
                             track_found=float(g.track_found.mean()),
                             swapped=float(g.in_swapped_track.mean()),
                             seed_lost=float(g.seed_lost.mean()), merged=float(g.merged.mean())))
    C = pd.DataFrame(rows)
    C.to_csv(out_dir() / 'compare.csv', index=False)
    pd.set_option('display.width', 250)
    for val in ('both_found', 'swapped', 'seed_lost'):
        print(f'\n{val}')
        print(C.pivot_table(index=['arm', 'cls', 'band'], columns='variant', values=val).round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--arms', nargs='+', default=list(ARMS))
    b.add_argument('--jobs', type=int, default=14)
    b.add_argument('--per-cell', type=int, default=PER_CELL)
    b.add_argument('--seed', type=int, default=20260913)
    b.add_argument('--variant', default='', help='output subdirectory for an A/B run')
    b.add_argument('--pairing', action='store_true', help='use xy_pairing_<arm>.json')
    b.add_argument('--local-mm', type=float, default=0.0, help='local significance floor [mm]')
    b.add_argument('--local-mode', default='rescue', choices=['replace', 'rescue'])
    b.add_argument('--split-gap', type=float, default=0.0, help='split seed clusters at this gap [mm]')
    f = sub.add_parser('floor')
    f.add_argument('--arms', nargs='+', default=list(ARMS))
    f.add_argument('--pairs-per-tag', type=int, default=1500)
    f.add_argument('--seed', type=int, default=7)
    d = sub.add_parser('derive')
    d.add_argument('--variant', default='')
    c = sub.add_parser('calib-pairing')
    c.add_argument('--arms', nargs='+', default=list(ARMS))
    ab = sub.add_parser('floor-ab')
    ab.add_argument('--arms', nargs='+', default=list(ARMS))
    ab.add_argument('--local-mm', type=float, default=0.0)
    ab.add_argument('--local-mode', default='rescue', choices=['replace', 'rescue'])
    ab.add_argument('--split-gap', type=float, default=0.0)
    ab.add_argument('--jobs', type=int, default=14)
    cm = sub.add_parser('compare')
    cm.add_argument('variants', nargs='+')
    a = ap.parse_args()
    if a.cmd == 'build':
        if (a.pairing or a.local_mm or a.split_gap) and not a.variant:
            ap.error('--pairing / --local-mm / --split-gap need --variant, so the production '
                     'baseline is not overwritten')
        build(a.arms, a.jobs, a.per_cell, a.seed, a.variant, a.pairing, a.local_mm, a.local_mode,
              a.split_gap)
    elif a.cmd == 'floor':
        floor_study(a.arms, a.pairs_per_tag, a.seed)
    elif a.cmd == 'derive':
        derive(a.variant)
    elif a.cmd == 'calib-pairing':
        calib_pairing(a.arms)
    elif a.cmd == 'floor-ab':
        if not (a.local_mm or a.split_gap):
            ap.error('floor-ab needs --local-mm and/or --split-gap: there is nothing to compare')
        floor_ab(a.arms, a.local_mm, a.jobs, a.local_mode, a.split_gap)
    else:
        compare(a.variants)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
