#!/usr/bin/env python3
"""examples.py -- full-plane waveforms of the det3 golden-row rays in every efficiency-budget category (interior,
R = 5 mm, rerun/effbudget) and every C22 far class, for the category note: ~/x17/cosmic_bench/satstudy/wf_examples.pkl.
Up to N per category (all of fit_failed), random_state fixed."""
import os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import satfit as S  # noqa: E402  (paths)

N = 40
N_ALL = ('spark', 'fit_failed', 'rejected_cut')   # every event of these, for counts


def picks():
    R = pd.read_parquet(os.path.expanduser('~/x17/cosmic_bench/effbudget/rays_det3.parquet'))
    I = R[(R.region == 'interior') & ~R.not_measured]
    out = {}
    for c, g in I.groupby('cat_xy_5'):
        if c == 'far':
            for k, h in g.groupby(g.cls.fillna('unclassified')):
                out['far:' + k] = h.event_id.sample(min(N, len(h)), random_state=2).tolist()
        else:
            n = len(g) if c in N_ALL else min(N, len(g))
            out[c] = g.event_id.sample(n, random_state=2).tolist()
    return out


def main():
    from qa_config import get_config
    from wft import io as wio
    P = picks(); want = set(e for v in P.values() for e in v)
    cfg = get_config(S.KEY); have = {}
    for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            r = wio.FeuReader(f)
            if not np.isin(r.event_ids, list(want)).any():
                continue
            for eid, ft, W in r.iter_events(want):
                e = have.setdefault(int(eid), {})
                e[v] = W.astype(np.float16); e[f'noise_{v}'] = r.noise.astype(np.float16)
    pickle.dump(dict(wf=have, picks=P), open(os.path.join(S.OUT, 'wf_examples.pkl'), 'wb'))
    print({k: len(v) for k, v in P.items()})


if __name__ == '__main__':
    main()
