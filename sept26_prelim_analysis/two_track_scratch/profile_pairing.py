"""Do x and y share a charge profile q(depth) closely enough to pair tracks?

Clean single donors, production windows and production one-track fits; the
profile is the NNLS q at the fitted (p0, w, t0). Then for coincident donor pairs
(same tag and phase), decide keep vs swap of the y partners with:
  lq    : current production cost (log x/y charge ratio, xy_pairing_<arm>.json)
  prof  : chi2-like distance of the two normalised profiles
  both  : sum of the two costs, each in units of its spread on true pairs
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sept26_prelim_analysis import intra_bench as ib

N_PER_ARM = 500
SUB = sys.argv[1] if len(sys.argv) > 1 else 'stat090_0000'
ib.SUBRUN = SUB
SFX = '' if SUB == 'stat090_0000' else '_' + SUB[-4:]
_CAL = None


def _init(bundle):
    global _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(bundle)
    wm.use_calibration(_CAL)


def fit_job(args):
    from wft import model as wm, reco as wr
    key, wins = args
    out = dict(key=key)
    for p in 'xy':
        if p not in wins:
            return None
        P = wins[p][0]
        if P['W'].shape[1] != wm.NSAMP:
            wm.set_nsamp(P['W'].shape[1])
        f = wr.fit_plane(P, p, _CAL)
        if f is None:
            return None
        W, noise, pos, sat = wm.prep_plane(P, p)
        _c, q = wm.chi2_plane(p, W, noise, pos, sat, f.p0, f.w, f.t0, wm.HYPER, snap_t0=False)
        if q is None:
            return None
        out[p] = dict(q=np.asarray(q, float), t0=f.t0, qsum=float(q.sum()))
    return out


def main():
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft.calib import CalibrationBundle
    rng = np.random.default_rng(3)
    res = []
    for arm in ('A', 'C'):
        bundle = str(ib.reco_dir(arm) / 'calib_bundle_prelim')
        cal = CalibrationBundle.load(bundle)
        cfg = wb.beam_config(arm, run=ib.RUN, sub_run=ib.SUBRUN)
        pos = wio.strip_position_map(cfg)
        D = ib.donors(arm)
        D = D.sample(n=min(N_PER_ARM * 3, len(D)), random_state=1)
        pairing = json.loads((ib.out_dir() / f'xy_pairing_{arm}.json').read_text())
        jobs = []
        for tag, g in D.groupby('tag'):
            td = ib.TagData(arm, tag, cfg, cal, pos, set(g.event_id), rng)
            for e in g.event_id:
                pl, _ = td.payload(0, int(e), None, 'single')
                if pl is not None:
                    jobs.append(((tag, int(e)), pl[1]))
        with ProcessPoolExecutor(4, initializer=_init, initargs=(bundle,)) as ex:
            fits = {r['key']: r for r in ex.map(fit_job, jobs, chunksize=4) if r}
        print(arm, len(fits), 'donors fitted', flush=True)
        import pickle
        pickle.dump(dict(fits=fits, D=D, pairing=pairing),
                    open(__file__.replace('.py', f'_{arm}{SFX}.pkl'), 'wb'))
        Dk = D.set_index(['tag', 'event_id'])
        keys = list(fits)
        # donor pairs: same tag and phase, coincident in both views
        rows = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                ta, tb = Dk.loc[a], Dk.loc[b]
                if a[0] != b[0] or ta.x_ftst != tb.x_ftst or ta.y_ftst != tb.y_ftst:
                    continue
                if abs(ta.x_t0 - tb.x_t0) > 30 or abs(ta.y_t0 - tb.y_t0) > 30:
                    continue
                rows.append((a, b))
        rng.shuffle(rows)
        rows = rows[:3000]

        def lq(fx, fy):
            z = (np.log(max(fx['qsum'], 1) / max(fy['qsum'], 1)) - pairing['median']['lq']) \
                / pairing['rsig']['lq']
            return min(z * z, 25.0)

        def prof(fx, fy):
            qx, qy = fx['q'], fy['q']
            px, py = qx / max(qx.sum(), 1e-9), qy / max(qy.sum(), 1e-9)
            return float(np.sum((px - py) ** 2 / (px + py + 0.01)))

        for a, b in rows:
            A, B = fits[a], fits[b]
            rec = dict(arm=arm)
            for name, fn in (('lq', lq), ('prof', prof)):
                rec[f'{name}_keep'] = fn(A['x'], A['y']) + fn(B['x'], B['y'])
                rec[f'{name}_swap'] = fn(A['x'], B['y']) + fn(B['x'], A['y'])
            # same-track and cross-track single costs, for the spreads
            rec['prof_same'] = prof(A['x'], A['y'])
            rec['prof_cross'] = prof(A['x'], B['y'])
            rec['lq_same'] = lq(A['x'], A['y'])
            rec['dq'] = abs(np.log(A['x']['qsum'] / B['x']['qsum']))
            res.append(rec)
    R = pd.DataFrame(res)
    R.to_parquet(__file__.replace('.py', f'{SFX}.parquet'))
    for arm, g in R.groupby('arm'):
        s = g.prof_same.median()
        both_keep = g.lq_keep + g.prof_keep / s
        both_swap = g.lq_swap + g.prof_swap / s
        print(arm, len(g), 'pairs; correct-decision rate:',
              'lq', round((g.lq_keep < g.lq_swap).mean(), 3),
              'prof', round((g.prof_keep < g.prof_swap).mean(), 3),
              'both', round((both_keep < both_swap).mean(), 3))
        print('   prof same/cross medians', round(s, 3), round(g.prof_cross.median(), 3))


if __name__ == '__main__':
    main()
