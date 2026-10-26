#!/usr/bin/env python3
"""sim_overlay.py -- S2 of PLAN.md: the bench overlay (rerun/satstudy/overlay.py) on SIMULATED single muons, with the
same production seeding, fit and pairing rule, so the sim's two-track behaviour can be put next to the data's.

    WFT_PIVOT_START=1 WFT_COL_GUARD=0.01 ../../.venv/bin/python sim_overlay.py build COND [--per-bin 40] [--jobs 2]
    ../../.venv/bin/python sim_overlay.py score COND                     -> sim_overlay_<COND>.json

COND: nom   the sim as produced (det3 bench noise)
      n2    n_TOF noise: every strip carries three more noise realisations (amplitude x 2, as the post-23-July
            DREAM noise floor), and the seeding threshold and the fit's noise are doubled with it
Donors: S1 points (v4c, det3 conditions; SIM below) reconstructed by production wft with the golden-row phys bundle
(t0 prior dropped; S1_EV below, from condor s1_reco.sub). A donor is an event with exactly one candidate per view
(the sim analogue of the data's clean singles), EVEN event id within its point (odd ids calibrate the pairing rule).
Sim muons all cross one 31 mm spot, so b is MOVED: its waveforms are shifted by a whole number of strips in each
view (the response is translation invariant: same kernel everywhere, connector-inverted wiring undone by working in
position order) to a drawn separation. The smaller of the two views' separations is drawn per bin, the other view
uniform between it and 80 mm, view and signs at random. As in the data overlay, b's waveforms are added on b's
footprint +- MARGIN mm only (b's noise there too). Seeds: hits from the summed waveforms at 5 sigma (t13_reco's
rule). Truth per view: each donor's own single-track fit (b's moved by its shift). Opening angle: between the two
donors' gun directions (points th00, thx05/10, thy05/10, txy10 mixed, so 0-14 deg).
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

os.environ.setdefault('WFT_PIVOT_START', '1'); os.environ.setdefault('WFT_COL_GUARD', '0.01')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis'), HERE]

SIM = os.environ.get('PAIRSIM_SIM', '/media/dylan/data/claude_tmp/v3wet')
OUT = os.path.expanduser('~/x17/cosmic_bench/satstudy/pairing_sim')
S1_EV = os.path.join(OUT, 's1')
BUNDLE = os.path.join(OUT, 'sim', 'bundle_det3_phys_not0')
POINTS = {'th00': (0, 0), 'thx05': (5, 0), 'thy05': (0, 5), 'thx10': (10, 0), 'thy10': (0, 10), 'txy10': (10, 10)}
SEP_BINS = [0, 6, 12, 18, 24, 40, 80]
DT0, MARGIN, MATCH, SIGMA = 30.0, 5.0, 2.0, 5.0
PITCH = 0.78


def donors(tag='dphys'):
    rows = []
    for pt in POINTS:
        f = os.path.join(S1_EV, f'events_v4c_{pt}_{tag}.parquet')
        if not os.path.exists(f):
            continue
        E = pd.read_parquet(f)
        E = E[E.x_ok.astype(bool) & E.y_ok.astype(bool) & (E.x_n_candidates == 1) & (E.y_n_candidates == 1)]
        rows.append(E.assign(point=pt))
    return pd.concat(rows, ignore_index=True)


def opening_deg(pa, pb):
    v = [np.array([np.tan(np.radians(POINTS[p][0])), np.tan(np.radians(POINTS[p][1])), 1.0]) for p in (pa, pb)]
    c = v[0] @ v[1] / np.linalg.norm(v[0]) / np.linalg.norm(v[1])
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def draw_pairs(E, per_bin, rng):
    E = E[E.event_id % 2 == 0].reset_index(drop=True)
    out = []
    for k in range(len(SEP_BINS) - 1):
        n, tries = 0, 0
        while n < per_bin and tries < per_bin * 200:
            tries += 1
            i, j = rng.integers(0, len(E), 2)
            A, B = E.iloc[i], E.iloc[j]
            if i == j or A.x_ftst != B.x_ftst or A.y_ftst != B.y_ftst:
                continue
            if abs(A.x_t0 - B.x_t0) > DT0 or abs(A.y_t0 - B.y_t0) > DT0:
                continue
            smin = rng.uniform(SEP_BINS[k], SEP_BINS[k + 1]); soth = rng.uniform(smin, 80.0)
            sep = dict(x=smin, y=soth) if rng.random() < 0.5 else dict(x=soth, y=smin)
            sh = {}
            for p in 'xy':
                ma, mb = A[f'{p}_p0'] + 15 * A[f'{p}_tan_theta'], B[f'{p}_p0'] + 15 * B[f'{p}_tan_theta']
                target = ma + (sep[p] if rng.random() < 0.5 else -sep[p])
                sh[p] = int(np.rint((target - mb) / PITCH))
            out.append(dict(sbin=k, a=int(i), b=int(j), pa=A.point, pb=B.point, ea=int(A.event_id),
                            eb=int(B.event_id), shx=sh['x'], shy=sh['y'], open_deg=opening_deg(A.point, B.point)))
            n += 1
    P = pd.DataFrame(out)
    for p in 'xy':
        ma = E.loc[P.a, f'{p}_p0'].to_numpy() + 15 * E.loc[P.a, f'{p}_tan_theta'].to_numpy()
        mb = E.loc[P.b, f'{p}_p0'].to_numpy() + 15 * E.loc[P.b, f'{p}_tan_theta'].to_numpy() + P[f'sh{p}'] * PITCH
        P[f'sep_{p}'] = np.abs(ma - mb)
    P['sep'] = P[['sep_x', 'sep_y']].min(axis=1)
    return E, P


class Planes:
    """Position-ordered access to a plane: rank r -> channel, so a shift by k strips is a shift in rank."""

    def __init__(self, pos):
        self.pos = pos
        ok = np.flatnonzero(np.isfinite(pos))
        self.order = ok[np.argsort(pos[ok])]
        self.rank = np.full(len(pos), -1); self.rank[self.order] = np.arange(len(self.order))

    def shift(self, W, k, keep=None):
        """W moved by k strips (channels shifted off the plane are dropped). keep: channel mask of W to move."""
        out = np.zeros_like(W)
        src = self.order if keep is None else self.order[keep[self.order]]
        dst_rank = self.rank[src] + k
        m = (dst_rank >= 0) & (dst_rank < len(self.order))
        out[self.order[dst_rank[m]]] = W[src[m]]
        return out


def build(cond, per_bin, jobs):
    from concurrent.futures import ProcessPoolExecutor
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import io as wio, seed as wseed, reco as wr
    from wft.calib import CalibrationBundle
    od = os.path.join(OUT, f'sim_overlay_{cond}'); os.makedirs(od, exist_ok=True)
    cfg = get_config('sat_det3'); cal = CalibrationBundle.load(BUNDLE)
    pm = wio.strip_position_map(cfg); feu = dict(x=cfg.MX17_FEU_X, y=cfg.MX17_FEU_Y)
    PL = {p: Planes(pm[feu[p]]) for p in 'xy'}
    rng = np.random.default_rng(11 if cond == 'nom' else 12)
    E, P = draw_pairs(donors(), per_bin, rng)
    print('pairs per bin', P.groupby('sbin').size().to_dict(), flush=True)
    # waveforms of every donor (and of the noise donors for n2), per point
    need = {}
    for r in P.itertuples():
        need.setdefault(r.pa, set()).add(r.ea); need.setdefault(r.pb, set()).add(r.eb)
    noise_src = []
    if cond == 'n2':
        for _ in range(len(P)):
            q = E.iloc[rng.integers(0, len(E), 3)]
            noise_src.append([(x.point, int(x.event_id)) for x in q.itertuples()])
            for x in q.itertuples():
                need.setdefault(x.point, set()).add(int(x.event_id))
    WF, NO = {}, {}
    for pt, ids in need.items():
        for p in 'xy':
            f = glob.glob(os.path.join(SIM, f'v4c_{pt}', f'*_{feu[p]:02d}.root'))[0]
            r = wio.FeuReader(f); NO[p] = r.noise
            for eid, _ft, w in r.iter_events(ids):
                WF[(pt, int(eid), p)] = w.astype(np.float32)
    nfac = 2.0 if cond == 'n2' else 1.0
    payloads, meta = [], []
    for oid, r in enumerate(P.itertuples()):
        A, B = E.loc[r.a], E.loc[r.b]
        W, rows = {}, []
        for p in 'xy':
            w = WF[(r.pa, r.ea, p)].copy()
            k = r.shx if p == 'x' else r.shy
            p0, tn = B[f'{p}_p0'], B[f'{p}_tan_theta']
            lo, hi = min(p0, p0 + 30 * tn) - MARGIN, max(p0, p0 + 30 * tn) + MARGIN
            keep = np.isfinite(pm[feu[p]]) & (pm[feu[p]] >= lo) & (pm[feu[p]] <= hi)
            w += PL[p].shift(WF[(r.pb, r.eb, p)], k, keep)
            if cond == 'n2':
                for (npt, nid) in noise_src[oid]:
                    # half a plane away, and the noise donor's own track (+- 20 mm of its fit) left out, so only its
                    # noise is added (every sim muon crosses the same spot, so its track would otherwise land on the
                    # plane edges and take candidate slots)
                    src = WF[(npt, nid, p)]
                    Dn = E[(E.point == npt) & (E.event_id == nid)].iloc[0]
                    q0, qt = Dn[f'{p}_p0'], Dn[f'{p}_tan_theta']
                    far = ~(np.isfinite(pm[feu[p]]) & (pm[feu[p]] >= min(q0, q0 + 30 * qt) - 20)
                            & (pm[feu[p]] <= max(q0, q0 + 30 * qt) + 20))
                    w += PL[p].shift(src, 256, far) + PL[p].shift(src, -256, far)
            W[p] = w
            noise = NO[p] * nfac
            amp = w.max(axis=1); sig = amp / np.where(noise > 0, noise, np.inf)
            ch = np.flatnonzero(sig >= SIGMA)
            rows.append(pd.DataFrame(dict(eventId=oid, feu=feu[p], channel=ch, amplitude=amp[ch], significance=sig[ch])))
        h = pd.concat(rows, ignore_index=True)
        sd = wseed.seeds_from_hits(h, pm, feu['x'], feu['y'], hot=cal.hot).get(oid)
        meta.append(dict(oid=oid, **r._asdict(), seeded=sd is not None))
        if sd is None:
            continue
        wins, used = {}, {}
        for p in 'xy':
            ws, us = [], []
            for s in sd[p] if isinstance(sd[p], list) else ([sd[p]] if sd[p] else []):
                win = wio.extract_window(W[p], NO[p] * nfac, pm[feu[p]], s.channels, 3)
                if win is None:
                    continue
                ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch)); us.append(s)
            if ws:
                wins[p], used[p] = ws, us
        payloads.append((oid, wins, used, sd['n_hits'], sd['spark'], dict(x=int(A.x_ftst), y=int(A.y_ftst))))
    print(len(payloads), 'payloads', flush=True)
    C = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=wr._worker_init, initargs=(BUNDLE,)) as pool:
        for row in pool.map(wr._worker_fit, payloads, chunksize=2):
            for c in row.pop('_cand', []):
                c['oid'] = row['event_id']; C.append(c)
    pd.DataFrame(meta).to_parquet(os.path.join(od, 'overlays.parquet'))
    pd.DataFrame(C).to_parquet(os.path.join(od, 'candidates.parquet'))
    E.to_parquet(os.path.join(od, 'donors.parquet'))
    print('wrote', od)


def calib_sim(tag='dphys'):
    import gates_data as G
    E = donors(tag); E = E[E.event_id % 2 == 1]
    f = dict(lq=np.log(E.x_q_sum / E.y_q_sum), u50=E.x_q_u50 - E.y_q_u50, u90=E.x_q_u90 - E.y_q_u90,
             t0=E.x_t0 - E.y_t0)
    return {k: (float(np.median(v)), G.rsig(v)) for k, v in f.items()}


def score(cond):
    od = os.path.join(OUT, f'sim_overlay_{cond}')
    M = pd.read_parquet(os.path.join(od, 'overlays.parquet')); C = pd.read_parquet(os.path.join(od, 'candidates.parquet'))
    E = pd.read_parquet(os.path.join(od, 'donors.parquet'))
    C = C[C['rank'] >= 0]
    cal = calib_sim()
    feats = dict(q=['lq'], q_u=['lq', 'u50', 'u90'], q_u_t0=['lq', 'u50', 'u90', 't0'])

    def cost(x, y, ks):
        d = dict(lq=np.log(x.q_sum / y.q_sum), u50=x.q_u50 - y.q_u50, u90=x.q_u90 - y.q_u90, t0=x.t0 - y.t0)
        return sum(min(((d[k] - cal[k][0]) / cal[k][1]) ** 2, 25.0) for k in ks)
    rows = []
    for m in M.itertuples():
        g = C[C.oid == m.oid]; A, B = E.loc[m.a], E.loc[m.b]
        r = dict(oid=m.oid, sbin=m.sbin, sep=m.sep, open_deg=m.open_deg,
                 lqr=abs(0.5 * np.log(A.x_q_sum * A.y_q_sum) - 0.5 * np.log(B.x_q_sum * B.y_q_sum)))
        for p in 'xy':
            gp = g[g.plane == p]
            sh = (m.shx if p == 'x' else m.shy) * PITCH
            for k, D, s in (('a', A, 0.0), ('b', B, sh)):
                dist = (gp.p0 - (D[f'{p}_p0'] + s)).abs()
                r[f'{k}{p}'] = int(dist.idxmin()) if len(gp) and dist.min() <= MATCH else -1
        found = all(r[k] >= 0 for k in ('ax', 'ay', 'bx', 'by')) and r['ax'] != r['bx'] and r['ay'] != r['by']
        r['both_found'] = found
        if found:
            X = {k: C.loc[r[f'{k}x']] for k in 'ab'}; Y = {k: C.loc[r[f'{k}y']] for k in 'ab'}
            for nm, ks in feats.items():
                # bool(): a numpy bool_ in an object column sums as logical OR (mean = 1/n)
                r[f'right_{nm}'] = bool((cost(X['a'], Y['a'], ks) + cost(X['b'], Y['b'], ks)) <
                                        (cost(X['a'], Y['b'], ks) + cost(X['b'], Y['a'], ks)))
            ta, tb = X['a'].track_id, X['b'].track_id
            r['prod_right'] = bool(ta >= 0 and tb >= 0 and ta != tb and Y['a'].track_id == ta and Y['b'].track_id == tb)
        rows.append(r)
    T = pd.DataFrame(rows)
    for c in ('right_q', 'right_q_u', 'right_q_u_t0', 'prod_right'):
        T[c] = T[c].astype(float)
    T.to_parquet(os.path.join(od, 'scored.parquet'))
    cols = ('right_q', 'right_q_u', 'right_q_u_t0', 'prod_right')
    out = dict(cond=cond, n=int(len(T)), cal=cal, sep_bins=SEP_BINS, bins=[])
    for k, g in T.groupby('sbin'):
        f = g[g.both_found.astype(bool)]
        out['bins'].append(dict(lo=SEP_BINS[k], hi=SEP_BINS[k + 1], n=int(len(g)), both_found=float(g.both_found.mean()),
                                n_found=int(len(f)), **{c: float(f[c].mean()) if len(f) else None for c in cols}))
    f = T[T.both_found.astype(bool) & (T.sep >= 24)]
    out['ge24'] = {c: float(f[c].mean()) for c in cols} | dict(n=int(len(f)))
    out['by_open'] = []
    for lo, hi in ((0, 1), (1, 8), (8, 15)):
        g = f[(f.open_deg >= lo) & (f.open_deg < hi)]
        out['by_open'].append(dict(lo=lo, hi=hi, n=int(len(g)), **{c: float(g[c].mean()) if len(g) else None for c in cols}))
    out['by_ratio'] = []
    for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 10)):
        g = f[(f.lqr >= lo) & (f.lqr < hi)]
        out['by_ratio'].append(dict(lo=lo, hi=hi, n=int(len(g)), right_q=float(g.right_q.mean()) if len(g) else None))
    json.dump(out, open(os.path.join(OUT, f'sim_overlay_{cond}.json'), 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('cond', 'n', 'ge24')}, indent=1))
    for b in out['bins']:
        print(b)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('cond')
    ap.add_argument('--per-bin', type=int, default=40); ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    build(a.cond, a.per_bin, a.jobs) if a.cmd == 'build' else score(a.cond)
