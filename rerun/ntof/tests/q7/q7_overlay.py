#!/usr/bin/env python3
"""q7_overlay.py -- the beam overlay truth bench for Q7 + the tan_max scan, on the ntof_reco path.

    q7_overlay.py build --arm A [--per-cell 40] [--jobs 5]       # donors from $Q7_DATA/real/r145_<arm>.*
    q7_overlay.py score                                          # -> $Q7_DATA/overlay/score_*.parquet

A copy of sept26_prelim_analysis/intra_bench.py's method (that file is not edited), moved to the pass-1 n_TOF chain:
  * donors are clean single tracks of run_145 stat090_0000 reconstructed by q7_real.py with the v1k bundle (timing v,
    in-situ kw, t0 free): one candidate per view, X-Y coincident with both column ends in the u window (gated in the
    'colwin' configuration, which does not test |tan|), no split in any configuration, both views quality_ok, X/Y charge
    balance within 3 sigma of the bundle's ln(qx/qy), max |tan| < STEEP_MAX.  Truth = the donor's own fit;
  * STEEP DONORS ARE INCLUDED ON PURPOSE (NOTE_PAIRING_FIX §4.1): a donor is 'steep' when max(|tan_x|, |tan_y|) >= 0.6
    (the registry raw tan_max; raw = true here, kw ~ 1).  Pairs are stratified on separation (the smaller per-view
    p0 separation) AND on the pair's tan class SS / SX / XX;
  * pairs are taken in one acquisition and one trigger phase (x_ftst, y_ftst), and only time-coincident ones
    (|dt0| < COINC_NS in both views): an IPC pair is one trigger;
  * the overlay is intra_bench's 'replace' mode (b's waveforms replace a's outside a's own region, so no strip carries
    two triggers' noise), hits merged, then ntof_reco's seeding (rr1 hits, min strips 3, dead padding) and windows;
  * every configuration of q7lib (tan_max 0.5 ... 1.0 and colwin, split count / charge, select gate / cost) is fitted
    on the same payload by q7lib.fit_multi.
Plus N_SANITY single donors per acquisition through the overlay path (must reproduce the donor).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q7lib as L  # noqa: E402

RUN, SUBRUN = 145, 'stat090_0000'
PAD = 3
REGION_MARGIN = 6
SEP_BINS = np.array([0.0, 6.0, 12.0, 18.0, 24.0, 40.0, 80.0, 400.0])
TCLS = ('SS', 'SX', 'XX')
STEEP = 0.6
STEEP_MAX = 1.5
COINC_NS = 60.0
MAX_REUSE = 3
MATCH_MM = 3.0
N_SANITY = 6
LQ_NSIG = 3.0
DONOR_CFG = L.cfg_name(L.INF, 'count', 'gate')
OUT = os.path.join(L.DATA, 'overlay')


# --------------------------------------------------------------------------- donors and pairs
def donors(arm, stem=None):
    """``stem``: the real-trigger table (default r145_<arm>, q7_real.py; or r145full_<arm>, the condor tables merged by
    q7_condor_merge.py).  Acquisitions missing in any configuration are not used."""
    E = pd.read_parquet(os.path.join(L.DATA, 'real', f'{stem or "r145"}_{arm}.events.parquet'))
    have = E.groupby('acq_tag').cfg.nunique()
    E = E[E.acq_tag.isin(have[have == E.cfg.nunique()].index)]
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(os.path.join(L.BUNDLES, f'v1k_{arm}'))
    lq = cal.xy_pairing['lq_class']['clean']
    nsp = E.groupby(['acq_tag', 'event_id']).n_splits.max().rename('nsp_any')
    D = E[E.cfg == DONOR_CFG].merge(nsp.reset_index(), on=['acq_tag', 'event_id'])
    t = np.maximum(D.x_tan_theta.abs(), D.y_tan_theta.abs())
    z = (np.log(D.x_q_sum / D.y_q_sum) - lq['median']) / lq['rsig']
    m = ((D.x_n_candidates == 1) & (D.y_n_candidates == 1) & D.x_ok & D.y_ok & (D.n_tracks == 1)
         & (D.nsp_any == 0) & D.x_quality_ok & D.y_quality_ok & (np.abs(z) < LQ_NSIG) & (t < STEEP_MAX))
    D = D[m].copy()
    D['tmax'] = t[m]
    D['steep'] = D.tmax >= STEEP
    D['lq_z'] = z[m]
    D = D.rename(columns={'acq_tag': 'tag'})
    return D.reset_index(drop=True)


def sample_pairs(D, seed, per_cell):
    frames = []
    for _k, g in D.groupby(['tag', 'x_ftst', 'y_ftst']):
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
    P['tcls'] = np.where(A.steep.to_numpy() & B.steep.to_numpy(), 'XX',
                         np.where(A.steep.to_numpy() | B.steep.to_numpy(), 'SX', 'SS'))
    P = P[(P.dt_x < COINC_NS) & (P.dt_y < COINC_NS)]
    P['sbin'] = np.digitize(P.sep_min, SEP_BINS) - 1
    P = P[(P.sbin >= 0) & (P.sbin < len(SEP_BINS) - 1)]
    P = P.sample(frac=1.0, random_state=seed)
    need = {(c, k): per_cell for c in TCLS for k in range(len(SEP_BINS) - 1)}
    use, keep = Counter(), []
    for r in P.itertuples():
        key = (r.tcls, r.sbin)
        if need[key] <= 0 or use[r.a] >= MAX_REUSE or use[r.b] >= MAX_REUSE:
            continue
        keep.append(r.Index)
        need[key] -= 1
        use[r.a] += 1
        use[r.b] += 1
    return P.loc[keep].reset_index(drop=True)


def seed_region(seeds, order):
    rank = np.flatnonzero(np.isin(order, np.concatenate([c.channels for c in seeds])))
    lo = max(0, rank.min() - REGION_MARGIN)
    hi = min(len(order) - 1, rank.max() + REGION_MARGIN)
    return order[lo:hi + 1]


class TagData:
    """intra_bench.TagData, 'replace' overlay, on the ntof_reco seeding (min strips 3, dead padding)."""

    def __init__(self, cfg, tag, cal, eids):
        from ntof_tracking import wft_beam as wb
        from wft import io as wio
        self.cfg, self.tag, self.cal = cfg, tag, cal
        self.pos = wio.strip_position_map(cfg)
        self.feu = {'x': cfg.MX17_FEU_X, 'y': cfg.MX17_FEU_Y}
        self.hits = wb.read_hits_tag(wb.hits_file_for_tag(cfg, tag), tuple(self.feu.values()))
        self.seeds = wb.seeds_from_hits_beam(self.hits, self.pos, self.feu['x'], self.feu['y'], hot=cal.hot,
                                             min_strips=L.MIN_STRIPS, dead=cal.dead)
        self.hits_by = {int(e): g for e, g in self.hits.groupby('eventId') if int(e) in eids}
        self.rdr, self.wf, self.ftst = {}, {'x': {}, 'y': {}}, {'x': {}, 'y': {}}
        for p, feu in self.feu.items():
            f = [f for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu) if wio.file_tag(f) == tag]
            self.rdr[p] = wio.FeuReader(f[0])
            for eid, ft, w in self.rdr[p].iter_events(set(int(e) for e in eids)):
                self.wf[p][eid], self.ftst[p][eid] = w, ft
        self.order = {}
        for p, feu in self.feu.items():
            o = np.argsort(self.pos[feu])
            self.order[p] = o[np.isfinite(self.pos[feu][o])]
        from wft import seed as wseed
        self.dead = (cal.dead or {}) if wseed.DEAD_SEED else {}

    def region(self, eid, plane):
        return seed_region(self.seeds[eid][plane], self.order[plane])

    def payload(self, oid, a, b=None):
        from ntof_tracking import wft_beam as wb
        from wft import io as wio
        W, H = {}, [self.hits_by[a]]
        for p, feu in self.feu.items():
            W[p] = self.wf[p][a].copy()
            if b is None:
                continue
            reg = self.region(b, p)
            src = self.wf[p][b]
            own = np.isin(reg, self.region(a, p))
            W[p][reg[own]] += src[reg[own]]
            W[p][reg[~own]] = src[reg[~own]]
            H[0] = H[0][~((H[0].feu == feu) & H[0].channel.isin(reg[~own]))]
            hb = self.hits_by[b]
            H.append(hb[(hb.feu == feu) & hb.channel.isin(reg)])
        h = pd.concat(H, ignore_index=True).assign(eventId=oid)
        h = h.sort_values('amplitude').drop_duplicates(['feu', 'channel'], keep='last')
        sd = wb.seeds_from_hits_beam(h, self.pos, self.feu['x'], self.feu['y'], hot=self.cal.hot,
                                     min_strips=L.MIN_STRIPS, dead=self.cal.dead).get(oid)
        if sd is None:
            return None, {}
        wins, used = {}, {}
        for p, feu in self.feu.items():
            ws, us = [], []
            for s in sd[p]:
                win = wio.extract_window(W[p], self.rdr[p].noise, self.pos[feu], s.channels, PAD,
                                         dead=self.dead.get(p))
                if win is None:
                    continue
                ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch))
                us.append(s)
            if ws:
                wins[p], used[p] = ws, us
        ext = {p: [(float(np.nanmin(self.pos[self.feu[p]][s.channels])),
                    float(np.nanmax(self.pos[self.feu[p]][s.channels]))) for s in used.get(p, [])] for p in 'xy'}
        ftst = {p: self.ftst[p][a] for p in 'xy'}
        return (oid, wins, used, sd['n_hits'], False, ftst), dict(seeds=ext)


def _seed_of(ext, p0):
    for k, (lo, hi) in enumerate(ext):
        if lo - MATCH_MM <= p0 <= hi + MATCH_MM:
            return k
    return -1


def build(arm, per_cell, seed, jobs, stem=None):
    from wft.calib import CalibrationBundle
    os.makedirs(OUT, exist_ok=True)
    bundle = os.path.join(L.BUNDLES, f'v1k_{arm}')
    cal = CalibrationBundle.load(bundle)
    cfg = L.beam_cfg(RUN, SUBRUN, arm)
    D = donors(arm, stem)
    P = sample_pairs(D, seed, per_cell)
    print(f'[q7ov] {arm}: {len(D)} donors ({int(D.steep.sum())} steep), {len(P)} pairs '
          f'{P.groupby(["tcls", "sbin"]).size().unstack(fill_value=0).to_dict("index")}', flush=True)
    Dk = D.set_index(['tag', 'event_id'])
    configs = L.all_configs()
    allres = {L.cfg_name(*c): ([], [], []) for c in configs}
    metas = []
    oid = (1 if arm == 'A' else 2) * 10_000_000
    t0 = time.time()
    for tag, pt in P.groupby('tag'):
        san = D[D.tag == tag].sort_values('steep', ascending=False).event_id.head(N_SANITY).tolist()
        td = TagData(cfg, tag, cal, set(pt.a_eid) | set(pt.b_eid) | set(san))
        todo = [('single', int(e), None, None) for e in san] + \
               [('overlay', int(r.a_eid), int(r.b_eid), r) for r in pt.itertuples()]
        payloads = []
        for mode, a, b, r in todo:
            pl, info = td.payload(oid, a, b)
            meta = dict(oid=oid, arm=arm, tag=tag, mode=mode, a_eid=a, b_eid=-1 if b is None else b,
                        seeded=pl is not None)
            if r is not None:
                meta.update(tcls=r.tcls, sbin=int(r.sbin), sep_x=r.sep_x, sep_y=r.sep_y, dt_x=r.dt_x, dt_y=r.dt_y)
            ta = Dk.loc[(tag, a)]
            tb = Dk.loc[(tag, b)] if b is not None else None
            for p in 'xy':
                ext = info.get('seeds', {}).get(p, [])
                meta[f'n_seeds_{p}'] = len(ext)
                ka = _seed_of(ext, ta[f'{p}_p0'])
                kb = _seed_of(ext, tb[f'{p}_p0']) if tb is not None else -1
                meta[f'seed_a_{p}'], meta[f'seed_b_{p}'] = ka, kb
                meta[f'merged_{p}'] = bool(b is not None and ka >= 0 and ka == kb)
            metas.append(meta)
            if pl is not None:
                payloads.append(pl)
            oid += 1
        t = time.time()
        res = L.run_payloads(bundle, payloads, configs, jobs)
        for k, (E, C, S) in res.items():
            for lst, src in zip(allres[k], (E, C, S)):
                lst.extend(dict(r_, acq_tag=tag) for r_ in src)
        print(f'[q7ov]   {arm} {tag}: {len(payloads)} payloads in {time.time() - t:.0f} s', flush=True)
        L.write_tables(allres, OUT, f'ov_{arm}')
        pd.DataFrame(metas).to_parquet(os.path.join(OUT, f'ov_{arm}.overlays.parquet'), index=False)
    D.to_parquet(os.path.join(OUT, f'ov_{arm}.donors.parquet'), index=False)
    json.dump(dict(arm=arm, run=RUN, subrun=SUBRUN, per_cell=per_cell, seed=seed, n_overlays=len(metas),
                   n_donors=len(D), n_steep=int(D.steep.sum()), coinc_ns=COINC_NS, steep=STEEP, steep_max=STEEP_MAX,
                   sep_bins=SEP_BINS.tolist(), match_mm=MATCH_MM, overlay='replace', donor_cfg=DONOR_CFG,
                   minutes=(time.time() - t0) / 60, configs=list(allres), donor_stem=stem or 'r145'),
              open(os.path.join(OUT, f'ov_{arm}.meta.json'), 'w'), indent=1)


# --------------------------------------------------------------------------- scoring
ACC = (0.6, 1.0)        # stage-3 true-angle acceptance (raw = true at kw ~ 1)


def score_arm(arm):
    """One row per (cfg, overlay, donor) and one per (cfg, overlay) with the ghosts."""
    M = pd.read_parquet(os.path.join(OUT, f'ov_{arm}.overlays.parquet'))
    C = pd.read_parquet(os.path.join(OUT, f'ov_{arm}.candidates.parquet'))
    D = pd.read_parquet(os.path.join(OUT, f'ov_{arm}.donors.parquet')).set_index(['tag', 'event_id'])
    C = C[C['rank'] >= 0]
    T = C[C.track_id >= 0]
    tx = T[T.plane == 'x'].set_index(['cfg', 'event_id', 'track_id'])
    ty = T[T.plane == 'y'].set_index(['cfg', 'event_id', 'track_id'])
    tr = pd.DataFrame(dict(x_p0=tx.p0, y_p0=ty.p0.reindex(tx.index), x_tan=tx.tan_theta,
                           y_tan=ty.tan_theta.reindex(tx.index),
                           gated=tx.track_gated & ty.track_gated.reindex(tx.index).fillna(False))).reset_index()
    trk = {k: g for k, g in tr.groupby(['cfg', 'event_id'])}
    cand = {k: g for k, g in C.groupby(['cfg', 'event_id'])}
    cfgs = sorted(C.cfg.unique())
    drows, erows = [], []
    for m in M.itertuples():
        dn = [('a', m.a_eid)] + ([('b', m.b_eid)] if m.mode == 'overlay' else [])
        truth = {r: D.loc[(m.tag, e)] for r, e in dn}
        for cf in cfgs:
            g = trk.get((cf, m.oid))
            assign = []
            if g is not None:
                for t in g.itertuples():
                    dx = {r: abs(t.x_p0 - v['x_p0']) for r, v in truth.items()}
                    dy = {r: abs(t.y_p0 - v['y_p0']) for r, v in truth.items()}
                    rx, ry = min(dx, key=dx.get), min(dy, key=dy.get)
                    assign.append((rx if dx[rx] < MATCH_MM else None, ry if dy[ry] < MATCH_MM else None,
                                   bool(t.gated), float(t.x_tan), float(t.y_tan)))
            cg = cand.get((cf, m.oid))
            ev = dict(cfg=cf, oid=m.oid, arm=arm, mode=m.mode, tcls=getattr(m, 'tcls', None),
                      sbin=getattr(m, 'sbin', np.nan), seeded=m.seeded, n_tracks=sum(a_[2] for a_ in assign),
                      n_pairs=len(assign))
            # ghosts: a pair that is not one donor in both views
            for acc in (None,) + ACC:
                ok = [a_ for a_ in assign if acc is None or (a_[2] and abs(a_[3]) < acc and abs(a_[4]) < acc)]
                lab = 'pair' if acc is None else f'acc{acc:g}'
                ev[f'ghost_swap_{lab}'] = sum(1 for a_ in ok if a_[0] and a_[1] and a_[0] != a_[1])
                ev[f'ghost_junk_{lab}'] = sum(1 for a_ in ok if (a_[0] is None) != (a_[1] is None)
                                              or (a_[0] is None and a_[1] is None))
            if m.mode == 'overlay':
                ev['merged'] = bool(m.merged_x or m.merged_y)
            for r, v in truth.items():
                found = [a_ for a_ in assign if a_[0] == r and a_[1] == r]
                row = dict(cfg=cf, oid=m.oid, arm=arm, mode=m.mode, donor=r, tcls=ev['tcls'], sbin=ev['sbin'],
                           sep_x=getattr(m, 'sep_x', np.nan), sep_y=getattr(m, 'sep_y', np.nan),
                           steep=bool(v['steep']), tmax=float(v['tmax']), t_x=float(v['x_tan_theta']),
                           t_y=float(v['y_tan_theta']), merged=bool(m.merged_x or m.merged_y),
                           seed_lost=bool(getattr(m, f'seed_{r}_x') < 0 or getattr(m, f'seed_{r}_y') < 0),
                           paired=bool(found), gated=any(a_[2] for a_ in found),
                           in_ghost=any(a_[0] != a_[1] and r in (a_[0], a_[1]) for a_ in assign))
                for acc in ACC:
                    row[f'found_acc{acc:g}'] = any(a_[2] and abs(a_[3]) < acc and abs(a_[4]) < acc for a_ in found)
                    row[f'truth_in_acc{acc:g}'] = bool(v['tmax'] < acc)
                if found:
                    row['dtan_x'] = found[0][3] - float(v['x_tan_theta'])
                    row['dtan_y'] = found[0][4] - float(v['y_tan_theta'])
                hc = cg[cg.plane == 'x'] if cg is not None else None
                row['cand_x'] = bool(hc is not None and len(hc) and (np.abs(hc.p0 - v['x_p0']) < MATCH_MM).any())
                hc = cg[cg.plane == 'y'] if cg is not None else None
                row['cand_y'] = bool(hc is not None and len(hc) and (np.abs(hc.p0 - v['y_p0']) < MATCH_MM).any())
                drows.append(row)
            erows.append(ev)
    return pd.DataFrame(drows), pd.DataFrame(erows)


def score(arms):
    for arm in arms:
        if not os.path.exists(os.path.join(OUT, f'ov_{arm}.overlays.parquet')):
            continue
        S, E = score_arm(arm)
        S.to_parquet(os.path.join(OUT, f'score_donors_{arm}.parquet'), index=False)
        E.to_parquet(os.path.join(OUT, f'score_events_{arm}.parquet'), index=False)
        print(f'[q7ov] scored {arm}: {len(S)} donor rows, {len(E)} event rows')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--arm', required=True)
    b.add_argument('--per-cell', type=int, default=40)
    b.add_argument('--seed', type=int, default=20261104)
    b.add_argument('--jobs', type=int, default=5)
    b.add_argument('--stem', default=None, help='real-trigger table stem (default r145)')
    s = sub.add_parser('score')
    s.add_argument('--arms', nargs='+', default=['A', 'C'])
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.arm, a.per_cell, a.seed, a.jobs, a.stem)
    else:
        score(a.arms)


if __name__ == '__main__':
    main()
