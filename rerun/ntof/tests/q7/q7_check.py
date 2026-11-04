#!/usr/bin/env python3
"""q7_check.py -- fit_multi (one fit pass, every configuration) against wft.reco._worker_fit on bundle copies.

    q7_check.py --arm A --n 60 [--configs tm0.6_count_gate colwin_charge_cost ...]

For each configuration a bundle copy carrying it (q7_bundles.make) is run through the production worker
(wft.reco._worker_init/_worker_fit, as ntof_reco.py does) on the same payloads; the candidate tables and event tables must
be identical (floats exactly).  Writes $Q7_DATA/check_<arm>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q7lib as L  # noqa: E402
import q7_bundles as QB  # noqa: E402


def prod_run(bundle, payloads, jobs):
    from concurrent.futures import ProcessPoolExecutor
    from wft import reco
    E, C = [], []
    with ProcessPoolExecutor(max_workers=jobs, initializer=reco._worker_init, initargs=(bundle,)) as pool:
        for r in pool.map(reco._worker_fit, payloads, chunksize=2):
            C.extend(r.pop('_cand', []))
            r.pop('_splits', None)
            E.append(r)
    return pd.DataFrame(E), pd.DataFrame(C)


def same(a, b, keys):
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b):
        return False, f'rows {len(a)} vs {len(b)}'
    bad = []
    for c in a.columns:
        if c not in b:
            continue
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == 'f':
            ok = np.array_equal(x, y, equal_nan=True)
        else:
            ok = (x == y).all()
        if not ok:
            bad.append(c)
    return not bad, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='A')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--tag', default='260805_14H06_000')
    ap.add_argument('--configs', nargs='+',
                    default=['tm0.6_count_gate', 'tm0.6_count_cost', 'tm1_charge_gate', 'colwin_charge_cost',
                             'colwin_count_gate', 'tm0.8_charge_cost'])
    a = ap.parse_args()
    from wft.calib import CalibrationBundle
    bundle = os.path.join(L.BUNDLES, f'v1k_{a.arm}')
    cal = CalibrationBundle.load(bundle)
    cfg = L.beam_cfg(145, 'stat090_0000', a.arm)
    payloads, _ = L.tag_payloads(cfg, a.tag, cal)
    # the multi-candidate events are where configurations differ: take them first
    ism = [max(len(p[1].get('x') or []), len(p[1].get('y') or [])) >= 2 for p in payloads]
    payloads = ([p for p, m in zip(payloads, ism) if m][:int(0.8 * a.n)]
                + [p for p, m in zip(payloads, ism) if not m][:a.n - int(0.8 * a.n)])
    confs = [QB.parse(c) for c in a.configs]
    t = time.time()
    res = L.run_payloads(bundle, payloads, confs, a.jobs)
    t_multi = time.time() - t
    out = dict(arm=a.arm, n=len(payloads), t_multi_s=t_multi, configs={})
    for name in a.configs:
        tm, sp, se = QB.parse(name)
        key = L.cfg_name(tm, sp, se)
        b = QB.make(a.arm, key, out_root=os.path.join(L.DATA, 'bundles_check'))
        t = time.time()
        Ep, Cp = prod_run(b, payloads, a.jobs)
        dt = time.time() - t
        Em, Cm = pd.DataFrame(res[key][0]), pd.DataFrame(res[key][1])
        okE, badE = same(Em, Ep, ['event_id'])
        okC, badC = same(Cm, Cp, ['event_id', 'plane', 'rank', 'p0'])
        out['configs'][key] = dict(events_identical=bool(okE), events_bad=badE, cand_identical=bool(okC), cand_bad=badC,
                                   t_prod_s=dt, n_tracks_sum=int(Ep.n_tracks.sum()))
        print(key, out['configs'][key], flush=True)
    json.dump(out, open(os.path.join(L.DATA, f'check_{a.arm}.json'), 'w'), indent=1, default=str)
    print(f'multi: {t_multi:.0f} s for {len(confs)} configs')


if __name__ == '__main__':
    main()
