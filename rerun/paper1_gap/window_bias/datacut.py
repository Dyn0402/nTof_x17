#!/usr/bin/env python3
"""datacut.py <data root> -- the toy on real data: cut the real head-on stacks of rows that have plenty of record after
the drift column at progressively earlier times, and refit.  The shift T(cut) - T(full) is the bias that a short record
would cause on that row, including everything the uniform-column model does not describe (attachment, the tail of the
response, the strip-sum loss at long drift); the noise is the row's own and the same events are used at every cut, so
the shift is mostly systematic.  The bootstrap spread of the shift is computed for the cuts that matter (event bootstrap,
same resamples at every cut).

Rows: every |in30 stack of inventory.csv with margin >= 400 ns and n >= 60.  Cut = column end (dt + 60 + T, the full fit)
+ margin, margin in MARGINS.  Writes results/datacut.csv.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'rerun', 'calib', 's4'), os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')]
import inventory as I  # noqa: E402
import s4_stacks  # noqa: E402
import x_width as xw  # noqa: E402
from rc_diffusion import GRID  # noqa: E402

RES = os.path.join(HERE, 'results')
MARGINS = [None, 500, 400, 300, 250, 200, 150, 116, 100, 75, 50, 25, 0]
NBOOT = 12


def fit_cuts(args):
    h, M, T0, col_end, seed = args
    out = []
    for m in MARGINS:
        Mc = M.copy()
        if m is not None:
            Mc[:, GRID > col_end + m] = np.nan
        try:
            r = xw.fit_x(h, Mc, T0)
            out.append(r['T'])
        except ValueError:
            out.append(np.nan)
    return seed, out


def main():
    root = sys.argv[1]
    inv = pd.read_csv(os.path.join(RES, 'inventory.csv'))
    use = inv[(inv.margin >= 400) & (inv.n >= 60)]
    import glob
    F = {}
    for f in glob.glob(os.path.join(root, 'x17rerun/paper1_gas', os.environ.get('FIT_DIR', 'fit_v2'), 'fit_T*.json')):
        F.update(json.load(open(f)))
    jobs, meta = [], []
    for q in use.itertuples():
        arm = dict(F[q.key + '|in30'])
        if 'replaces' in arm:
            arm['edge'] = None
        ev = I.arm_events(arm, root)
        h = s4_stacks.pass_template(os.path.dirname(I.local(arm['template'], root)))
        col_end = q.col_end
        ks = list(ev)
        rng = np.random.default_rng(11)
        for b in range(NBOOT + 1):
            if b == 0:
                e2 = ev
            else:
                e2 = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
            n, M = xw.stack(e2, 'x', tan_max=xw.TAN)
            jobs.append((h, M, q.T0, col_end, b))
            meta.append(dict(key=q.key, chamber=q.chamber, subrun=q.subrun, run=q.run, n=q.n, T_full_stored=q.T,
                             margin_full=q.margin, boot=b))
    print(len(jobs), 'stacks x', len(MARGINS), 'cuts', flush=True)
    with ProcessPoolExecutor(14) as pool:
        res = list(pool.map(fit_cuts, jobs, chunksize=1))
    rows = []
    for mt, (_, Ts) in zip(meta, res):
        for m, T in zip(MARGINS, Ts):
            rows.append(dict(mt, cut_margin=('full' if m is None else m), T_fit=T))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, 'datacut.csv'), index=False)
    full = D[D.cut_margin == 'full'].set_index(['key', 'boot'])['T_fit']
    D['dT'] = D['T_fit'].to_numpy() - full.reindex(pd.MultiIndex.from_arrays([D.key, D.boot])).to_numpy()
    D.to_csv(os.path.join(RES, 'datacut.csv'), index=False)
    c = D[(D.boot == 0) & (D.cut_margin != 'full')].pivot_table(index=['chamber', 'subrun'], columns='cut_margin', values='dT')
    pd.set_option('display.width', 250)
    print(c.round(0).to_string())
    s = D[(D.boot > 0) & (D.cut_margin != 'full')].groupby(['chamber', 'subrun', 'cut_margin']).dT.std().unstack()
    print('bootstrap sd of dT'); print(s.round(0).to_string())


if __name__ == '__main__':
    main()
