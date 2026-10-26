#!/usr/bin/env python3
"""headon3.py -- the head-on recovery as implemented in wft (WFT_HEADON_SIG = 20 + WFT_HEADON_RESCUE), end to end:
both views of a real head-on track scaled by k and injected into real empty-chamber noise, hits emulated, wft.seed
seeding, the wft worker fit with the cross-view rescue; fakes with the head-on view left empty
(Dylan 10-26: extend head-on recovery to 1 strip / low SNR without the fakes skyrocketing).

    WFT_PIVOT_START=1 WFT_COL_GUARD=0.01 WFT_HEADON_SIG=20 \\
        ../../.venv/bin/python headon3.py extract | run [--n 200 --jobs 3] | score

Donors and noise as headon2.py (its cache gives the noise planes; this one adds each donor's FULL other-view plane).
Per donor, view p (the head-on one) and o: W_p = noise_p + k x donor_p on its footprint +- 4 mm, W_o = noise_o + k x
donor_o on its footprint [mesh, mesh + 30 tan] +- 4 mm; noise x 2 adds a second noise plane x sqrt 3 and hits are
taken against 2 sigma. Fake: the same with nothing injected in p. One arm only: HEADON_SIG and HEADON_RESCUE act
only on a plane production left without a seed, and their fits carry `rescued`, so production's outcome is the same
arm with the rescued fits removed.
"""
import argparse, json, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402
import headon2 as H2  # noqa: E402

CACHE = os.path.join(S.OUT, 'headon3_wf.pkl')
OUTP = os.path.join(S.OUT, 'headon3.parquet')
KS = (1.0, 0.1, 0.05)
FOUND = 2.0


def extract():
    from qa_config import get_config
    from wft import io as wio
    C = pickle.load(open(H2.CACHE, 'rb'))
    R = H2.rays(); cfg = get_config(S.KEY); feu = dict(x=cfg.MX17_FEU_X, y=cfg.MX17_FEU_Y)
    out = dict(other={}, truth_o={})
    for v in 'xy':
        o = 'y' if v == 'x' else 'x'
        ids = set(C['donors'][v]); pm = C['pos'][o]; dd = {}
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu[o]):
            for eid, ft, W in wio.FeuReader(f).iter_events(ids):
                m, tn = R.loc[eid, f'mesh_{o}'], R.loc[eid, f'tan_{o}']
                lo, hi = min(m, m + 30 * tn) - 4, max(m, m + 30 * tn) + 4
                reg = np.flatnonzero(np.isfinite(pm) & (pm >= lo) & (pm <= hi))
                dd[eid] = dict(reg=reg, W=W[reg].astype(np.float32), truth=float(m + 15 * tn))
        out['other'][v] = dd
        print(v, len(dd), flush=True)
    pickle.dump(out, open(CACHE, 'wb'))


def hits_of(W, sigma, feu, eid):
    s = H2.SIG_SCALE * W.max(1) / np.maximum(sigma, 1e-3)
    ch = np.flatnonzero(s >= H2.HIT_SIG)
    return pd.DataFrame(dict(eventId=eid, feu=feu, channel=ch, amplitude=W.max(1)[ch], significance=s[ch]))


def run(n, jobs):
    from concurrent.futures import ProcessPoolExecutor
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import io as wio, seed as wseed, reco as wr
    from wft.calib import CalibrationBundle
    assert wseed.HEADON_SIG == 20.0, 'run with WFT_HEADON_SIG=20'
    C = pickle.load(open(H2.CACHE, 'rb')); O = pickle.load(open(CACHE, 'rb'))
    cfg = get_config(S.KEY); cal = CalibrationBundle.load(S.BUNDLE)
    feu = dict(x=cfg.MX17_FEU_X, y=cfg.MX17_FEU_Y); pm = {v: C['pos'][v] for v in 'xy'}
    noise = {v: C['noise'][v].astype(np.float32) for v in 'xy'}
    rng = np.random.default_rng(5)
    meta, rows, oid = [], [], 0
    pool = ProcessPoolExecutor(max_workers=jobs, initializer=wr._worker_init,
                               initargs=(S.BUNDLE, None, dict(HEADON_RESCUE=True)))
    for v in 'xy':
        o = 'y' if v == 'x' else 'x'
        common = [e for e in C['empty'][v] if e in C['empty'][o]]
        dn = C['donors'][v]; keys = [e for e in dn if e in O['other'][v]]
        keys = list(rng.choice(keys, min(n, len(keys)), replace=False))
        for nf in (1, 2):
            sigma = {q: noise[q] * nf for q in 'xy'}
            # clean noise events: production finds nothing in either view
            clean = []
            for i, e in enumerate(common):
                Wn = {q: C['empty'][q][e].astype(np.float32) + (np.sqrt(3) * C['empty'][q][common[(i + 1) % len(common)]].astype(np.float32)
                                                               if nf == 2 else 0) for q in 'xy'}
                if not any(H2.seed_hits(Wn[q], sigma[q], pm[q], [])[0] for q in 'xy'):
                    clean.append(Wn)
            for k in KS:
                payloads = []          # one (view, noise, gain) group at a time: full-plane waveforms are heavy
                for j, e in enumerate(keys):
                    for mode in ('signal', 'fake'):
                        Wn = clean[(j * 7 + (mode == 'fake')) % len(clean)]
                        W = {q: Wn[q].copy() for q in 'xy'}
                        d, do = dn[e], O['other'][v][e]
                        if mode == 'signal':
                            W[v][d['reg']] += k * d['W']
                        W[o][do['reg']] += k * do['W']
                        h = pd.concat([hits_of(W[q], sigma[q], feu[q], oid) for q in 'xy'], ignore_index=True)
                        sd = wseed.seeds_from_hits(h, {feu[q]: pm[q] for q in 'xy'}, feu['x'], feu['y'], hot=cal.hot).get(oid)
                        wins, used = {}, {}
                        if sd is not None and not sd['spark']:
                            for q in 'xy':
                                ws, us = [], []
                                for s_ in sd[q]:
                                    win = wio.extract_window(W[q], sigma[q], pm[q], s_.channels, 3)
                                    if win is not None:
                                        ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch)); us.append(s_)
                                if ws:
                                    wins[q], used[q] = ws, us
                                else:
                                    wins[f'_rescue_{q}'] = dict(W=W[q], noise=sigma[q], pos=pm[q])
                        meta.append(dict(oid=oid, view=v, nf=nf, k=k, mode=mode, eid=int(e), truth=d['truth'], truth_o=do['truth'],
                                         seeded_v=bool(sd and sd[v]), rescued_seed_v=bool(sd and sd[v] and sd[v][0].rescued)))
                        payloads.append((oid, wins, used, 0 if sd is None else sd['n_hits'], False, {}))
                        oid += 1
                for r in pool.map(wr._worker_fit, payloads, chunksize=4):
                    cand = r.pop('_cand', [])
                    rows.append(dict(oid=r['event_id'], x_ok=bool(r.get('x_ok')), y_ok=bool(r.get('y_ok')),
                                     x_p0=r.get('x_p0'), y_p0=r.get('y_p0'),
                                     x_resc=any(c['plane'] == 'x' and c['rank'] == 0 and c.get('rescued') for c in cand),
                                     y_resc=any(c['plane'] == 'y' and c['rank'] == 0 and c.get('rescued') for c in cand)))
                print(v, nf, k, 'done', len(rows), flush=True)
    pool.shutdown()
    M = pd.DataFrame(meta).merge(pd.DataFrame(rows), on='oid', how='left')
    M.to_parquet(OUTP); print('wrote', OUTP)


def score():
    M = pd.read_parquet(OUTP); out = {}
    for (v, nf, k, mode), g in M.groupby(['view', 'nf', 'k', 'mode']):
        o = 'y' if v == 'x' else 'x'
        ok = g[f'{v}_ok'].fillna(False).astype(bool); resc = g[f'{v}_resc'].fillna(False).astype(bool)
        near = ok & ((g[f'{v}_p0'] - g.truth).abs() <= FOUND)
        oo = g[f'{o}_ok'].fillna(False).astype(bool) & ((g[f'{o}_p0'] - g.truth_o).abs() <= FOUND)
        r = out.setdefault(f'{v}|{nf}|{k:g}', {})
        if mode == 'signal':
            r.update(n=int(len(g)), prod=float((near & ~resc).mean()), new=float(near.mean()),
                     new_both=float((near & oo).mean()), other_found=float(oo.mean()),
                     rescued_wrong=float((resc & ok & ~near).mean()))
        else:
            r.update(fake_prod=float((ok & ~resc).mean()), fake_new=float(ok.mean()), n_fake=int(len(g)))
    json.dump(out, open(os.path.join(S.OUT, 'headon3.json'), 'w'), indent=1)
    for k, r in out.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--jobs', type=int, default=3); a = ap.parse_args()
    {'extract': extract, 'score': score}.get(a.cmd, lambda: run(a.n, a.jobs))()
