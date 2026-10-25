#!/usr/bin/env python3
"""montage.py <det> <class> [--n 4] [--seed 0] [--out dir] -- displays of n random events of one class in one PNG."""
import argparse, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from olib import BASE  # noqa: E402
import display as D  # noqa: E402


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(); ap.add_argument('det'); ap.add_argument('cls')
    ap.add_argument('--n', type=int, default=4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(BASE, 'an', 'montage')); ap.add_argument('--eids', type=int, nargs='*')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    x = pd.read_parquet(os.path.join(BASE, f'classified_{a.det}.parquet'))
    sel = x[(x.cls == a.cls) & x.outlier] if a.cls != 'control' else x[~x.outlier]
    if a.eids:
        sel = x[x.event_id.isin(a.eids)]
    rng = np.random.default_rng(a.seed)
    rows = sel.iloc[rng.choice(len(sel), min(a.n, len(sel)), replace=False)] if not a.eids else sel
    wf = D.extract(a.det, rows.event_id.tolist())
    n = len(rows); fig = plt.figure(figsize=(26, 8.5 * ((n + 1) // 2)))
    sub = fig.subfigures((n + 1) // 2, 2, squeeze=False)
    for k, r in enumerate(rows.to_dict('records')):
        sf = sub[k // 2][k % 2]; axs = sf.subplots(2, 2, gridspec_kw=dict(width_ratios=[1.3, 1]))
        D.draw(a.det, r, wf.get(int(r['event_id']), {}), axs, title_extra=f"[{r['cls']} {r.get('why', '')}]")
        axs[0][0].figure.suptitle(axs[0][0].figure._suptitle.get_text() if axs[0][0].figure._suptitle else '', fontsize=10)
    tag = 'sel' if a.eids else f'{a.cls}_{a.seed}'
    p = os.path.join(a.out, f'{a.det}_{tag}.png'); fig.savefig(p, dpi=55); print(p)


if __name__ == '__main__':
    main()
