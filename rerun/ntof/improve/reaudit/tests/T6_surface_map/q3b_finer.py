#!/usr/bin/env python3
"""q3b_finer.py -- is a finer or higher-order model still improving?  Same held-out protocol as q34_models.py.
Higher potential orders (8, 10) and scale orders (3, 4); free tilt cells of 50, 33 and 25 mm (ridge to the chamber
offset) with a smooth scale; free 25 mm cells for a and s (angle_map.py style).  -> finer_<arm>.csv, finer_diffs_<arm>.csv
"""
import sys

import numpy as np
import pandas as pd

from t6lib import rows_ac, Poly, Free
from q34_models import evaluate, metrics, diffs

E = lambda s: np.arange(-200.0, 200.1, s)  # noqa: E731


def models():
    m = {'free50': Free(), 'free25': Free(edges=E(25), n_min=40), 'pot7_s3': Poly(n=7, m=3)}
    for n in (8, 10):
        for sm in (3, 4):
            m[f'pot{n}_s{sm}'] = Poly(n=n, m=sm)
            m[f'ind{n}_s{sm}'] = Poly(n=n, pot=False, m=sm)
    m['pot7_s4'] = Poly(n=7, m=4)
    m['cellA_s3'] = Poly(cellA=E(50), m=3)
    m['cellA_s4'] = Poly(cellA=E(50), m=4)
    for step, tag in ((400 / 12, '33'), (25, '25')):
        for rg in (1e-3, 1e-4):
            m[f'cellA{tag}_s3_r{rg:g}'] = Poly(cellA=np.linspace(-200, 200, int(round(400 / step)) + 1), m=3, ridge=rg)
    return m


def main():
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    for arm in sys.argv[1:] or ['A', 'C']:
        D = rows_ac(arm)
        M = models()
        R = evaluate(D, M)
        T, boots = metrics(D, R, M)
        T.to_csv(f'finer_{arm}.csv', index=False)
        best = 'cellA_s3' if arm == 'A' else 'pot7_s3'
        pairs = [(k, 'free50') for k in M if k != 'free50'] + [(k, best) for k in M if k not in (best, 'free50')]
        diffs(D, R, boots, pairs).to_csv(f'finer_diffs_{arm}.csv', index=False)
        P = T.pivot(index='model', columns='view', values=['sig_clean', 'w68', 'w95', 'sig_edge'])
        print(arm); print(P.round(4).sort_values(('sig_clean', 'x')).to_string())


if __name__ == '__main__':
    main()
