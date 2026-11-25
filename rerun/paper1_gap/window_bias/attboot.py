#!/usr/bin/env python3
"""attboot.py <data root> -- event bootstrap of the attachment fit (attfit.fit_att) per row: is 1/tau significantly
above 0, and how far does T move when tau is free (T_att - T_uni, same resample)?  Writes results/attboot.csv.
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'rerun', 'calib', 's4'), os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')]
import attfit  # noqa: E402
import inventory as I  # noqa: E402
import s4_stacks  # noqa: E402
import x_width as xw  # noqa: E402

RES = os.path.join(HERE, 'results')
NBOOT = 16


def one(c):
    key, h, ev, T0, b = c
    if b > 0:
        ks = list(ev)
        rng = np.random.default_rng(500 + b)
        ev = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
    n, M = xw.stack(ev, 'x', tan_max=xw.TAN)
    try:
        a = attfit.fit_att(h, M, T0, True)
        u = attfit.fit_att(h, M, T0, False)
    except ValueError:
        return dict(key=key, boot=b, k=np.nan, T_att=np.nan, T_uni=np.nan)
    return dict(key=key, boot=b, k=a['k'], T_att=a['T'], T_uni=u['T'])


def main():
    root = sys.argv[1]
    inv = pd.read_csv(os.path.join(RES, 'inventory.csv'))
    inv = inv[(inv.n >= 40) & (inv.margin > 0)]
    F = {}
    for f in glob.glob(os.path.join(root, 'x17rerun/paper1_gas', os.environ.get('FIT_DIR', 'fit_v2'), 'fit_T*.json')):
        F.update(json.load(open(f)))
    cases = []
    for q in inv.itertuples():
        arm = dict(F[q.key + '|in30'])
        if 'replaces' in arm:
            arm['edge'] = None
        ev = I.arm_events(arm, root)
        h = s4_stacks.pass_template(os.path.dirname(I.local(arm['template'], root)))
        cases += [(q.key, h, ev, q.T0, b) for b in range(NBOOT + 1)]
    with ProcessPoolExecutor(14) as pool:
        out = list(pool.map(one, cases, chunksize=1))
    D = pd.DataFrame(out)
    D['dT'] = D.T_att - D.T_uni
    D.to_csv(os.path.join(RES, 'attboot.csv'), index=False)
    c = D[D.boot == 0].set_index('key')
    s = D[D.boot > 0].groupby('key').agg(k_sd=('k', 'std'), dT_sd=('dT', 'std'), k_lo=('k', lambda x: np.quantile(x, 0.16)))
    S = inv.set_index('key')[['chamber', 'run', 'subrun', 'n', 'margin', 'T']].join(c[['k', 'T_att', 'T_uni', 'dT']]).join(s)
    S['tau_us'] = 1e-3 / S.k
    S['k_sig'] = S.k / S.k_sd
    S.to_csv(os.path.join(RES, 'attboot_summary.csv'))
    pd.set_option('display.width', 250)
    print(S[['chamber', 'subrun', 'n', 'T_uni', 'T_att', 'dT', 'dT_sd', 'tau_us', 'k_sig']].round(2).to_string())


if __name__ == '__main__':
    main()
