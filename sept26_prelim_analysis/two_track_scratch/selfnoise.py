"""Does doubled noise on a single track's own strips move its t0?

single      : donor a alone
selfnoise   : a + an empty trigger's waveforms on a's own region
selfnoise_w : same, but the fit is told the noise there is sqrt(2) larger
"""
import sys, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sept26_prelim_analysis import intra_bench as ib, paths
from ntof_tracking import wft_beam as wb
from wft import io as wio, reco as wr
from wft.calib import CalibrationBundle

N_PER_TAG = int(sys.argv[1]) if len(sys.argv) > 1 else 25
OPTS = dict(TWO_TRACK=True, TWO_TRACK_F=300.0, TWO_TRACK_F_CORROB=120.0,
            TWO_TRACK_T0='tied', TWO_TRACK_RESID_Z=8.0)


def payload(td, oid, a, mode):
    W, H = {}, [td.hits_by[a]]
    e = int(td.empties[oid % len(td.empties)])
    boost = {}
    for p in 'xy':
        W[p] = td.wf[p][a].copy()
        if mode != 'single':
            reg = td.region(a, p)
            W[p][reg] += td.wf[p][e][reg]
            boost[p] = set(reg.tolist())
    h = pd.concat(H).assign(eventId=oid)
    sd = wb.seeds_from_hits_beam(h, td.pos, td.feu['x'], td.feu['y'], hot=td.cal.hot,
                                 local_mm=16.0, local_mode='rescue').get(oid)
    wins, used = {}, {}
    for p, feu in td.feu.items():
        ws, us = [], []
        noise = td.rdr[p].noise.copy()
        if mode == 'selfnoise_w':
            idx = np.array(sorted(boost[p]))
            noise[idx] = noise[idx] * np.sqrt(2.0)
        for s in sd[p]:
            win = wio.extract_window(W[p], noise, td.pos[feu], s.channels, ib.PAD)
            if win is None:
                continue
            ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch))
            us.append(s)
        if ws:
            wins[p], used[p] = ws, us
    return (oid, wins, used, sd['n_hits'], False, {p: td.ftst[p][a] for p in 'xy'})


def main():
    rng = np.random.default_rng(7)
    out = []
    for arm in ('A', 'C'):
        bundle = str(ib.reco_dir(arm) / 'calib_bundle_prelim')
        cal = CalibrationBundle.load(bundle)
        cfg = wb.beam_config(arm, run=ib.RUN, sub_run=ib.SUBRUN)
        pos = wio.strip_position_map(cfg)
        D = ib.donors(arm)
        pairing = str(ib.out_dir() / f'xy_pairing_{arm}.json')
        with ProcessPoolExecutor(14, initializer=wr._worker_init,
                                 initargs=(bundle, pairing, OPTS)) as pool:
            for tag, g in D.groupby('tag'):
                eids = g.event_id.head(N_PER_TAG).tolist()
                td = ib.TagData(arm, tag, cfg, cal, pos, set(eids), rng)
                pls, meta = [], {}
                oid = 0
                for a in eids:
                    for mode in ('single', 'selfnoise', 'selfnoise_w'):
                        pls.append(payload(td, oid, a, mode))
                        meta[oid] = (a, mode)
                        oid += 1
                truth = g.set_index('event_id')
                for row in pool.map(wr._worker_fit, pls, chunksize=2):
                    a, mode = meta[row['event_id']]
                    t = truth.loc[a]
                    cands = pd.DataFrame(row.get('_cand', []))
                    rec = dict(arm=arm, tag=tag, a=a, mode=mode, n_tracks=row['n_tracks'])
                    ok = True
                    for p in 'xy':
                        h = cands[cands.plane == p] if len(cands) else cands
                        if not len(h):
                            rec[f'dt0_{p}'] = np.nan; ok = False; continue
                        c = h.iloc[int(np.argmin(np.abs(h.p0.to_numpy() - t[f'{p}_p0'])))]
                        rec[f'dt0_{p}'] = c.t0 - t[f'{p}_t0']
                        rec[f'tid_{p}'] = c.track_id
                        rec[f'gated_{p}'] = bool(c.track_gated)
                    rec['found'] = ok and rec['tid_x'] >= 0 and rec['tid_x'] == rec['tid_y'] and rec['gated_x']
                    out.append(rec)
                print(arm, tag, len(pls), flush=True)
    R = pd.DataFrame(out)
    R.to_parquet(os.path.join(os.path.dirname(__file__), 'selfnoise.parquet'))
    R['jump'] = (R.dt0_x.abs() > 30) | (R.dt0_y.abs() > 30)
    print(R.groupby(['arm', 'mode']).agg(jump=('jump', 'mean'), found=('found', 'mean'),
                                         n=('a', 'size')).round(3))


if __name__ == '__main__':
    main()
