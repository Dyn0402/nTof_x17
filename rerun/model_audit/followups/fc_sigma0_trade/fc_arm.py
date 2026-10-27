#!/usr/bin/env python3
"""fc_arm.py -- heldout_arm.py with one more study override: the depth (charge) grid width --dt [ns].

Everything else is rerun/model_audit/heldout_arm.py (same fitter, same per-row output, same summary): this wrapper only
installs, after heldout_arm's own worker init, wft.model.DT = dt and K = round(K * 60 / dt), i.e. the same drift span
on a finer charge basis (ws_floor: 20 ns removes the 60 ns grid's steep-track bias in the toy; untested on data).
The override is recorded in wft.diag.ACTIVE-style form in the output JSON ('dt_ns').  No wft default is changed.

    fc_arm.py <heldout_arm.py args...> [--dt 20]
"""
import json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit')); sys.path.insert(0, REPO)
import heldout_arm as ha  # noqa: E402

_DT = None


def _init(dt, *args):
    ha._init(*args)
    if dt and abs(dt - 60.0) > 1e-9:
        from wft import model as wm
        k60 = wm.K
        wm.DT = float(dt)
        wm.set_depth_bins(int(round(k60 * 60.0 / dt)))


def main():
    argv = list(sys.argv[1:]); dt = 60.0
    if '--dt' in argv:
        i = argv.index('--dt'); dt = float(argv[i + 1]); del argv[i:i + 2]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1320); ap.add_argument('--start', type=int, default=180)
    ap.add_argument('--k', type=int, default=24); ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--set', default=None); ap.add_argument('--t0-fix', action='store_true'); ap.add_argument('--two-sided', action='store_true')
    a = ap.parse_args(argv)
    hy = {}
    for kv in (a.set or '').split(','):
        if kv:
            k_, v_ = kv.split('='); hy[k_] = [] if v_ == '' else float(v_)
    C = pickle.load(open(a.cache, 'rb')); ids = sorted(C)[a.start:a.start + a.n]; del C
    init_args = (a.cache, a.bundle, a.k, hy, None, None, a.two_sided, a.t0_fix, False, {}, None)
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(dt,) + init_args) as pool:
        rows = [r for rs in pool.map(ha.one, ids, chunksize=2) for r in rs]
    per = np.array(rows, float)
    env = {k: v for k, v in os.environ.items() if k.startswith('WFT_')}
    summ = ha.summarize(per)
    np.savez(a.out, per=per, bundle=a.bundle, hyper=json.dumps(hy), env=json.dumps(env), args=json.dumps(vars(a)), dt_ns=dt)
    json.dump(dict(args=vars(a), env=env, dt_ns=dt, summary=summ), open(a.out.replace('.npz', '.json'), 'w'), indent=1)
    print(a.out, dt, json.dumps({v: {k: summ[v][k] for k in ('n', 'ladder', 'far_pct', 'chi2dof_p50')} for v in summ if v in ('x', 'y')}))


if __name__ == '__main__':
    main()
