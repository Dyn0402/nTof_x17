#!/usr/bin/env python3
"""examples.py RESID.pkl [N] -- displays of clipped DREAM pulses: data (raw-derived W) and the base fit's model on the
clipped strip and its +-1 / +-2 neighbours, for N clipped planes (one long clip that reaches the floor, one long clip
that does not, one short clip, ...).  -> figures/examples_<tag>.png"""
import os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main(path, n=6):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rows = pickle.load(open(path, 'rb'))['rows']
    pick, want = [], [('L>=6 floor', lambda L, f: L >= 6 and f), ('L>=6 no floor', lambda L, f: L >= 6 and not f),
                      ('L 3-5 floor', lambda L, f: 3 <= L <= 5 and f), ('L 3-5 no floor', lambda L, f: 3 <= L <= 5 and not f),
                      ('L 1-2', lambda L, f: L <= 2), ('L 1-2 floor', lambda L, f: L <= 2 and f)]
    rng = np.random.default_rng(2)
    for name, cond in want[:n]:
        cands = []
        for r in rows:
            cl = r['raw'] >= 4095
            for i in np.flatnonzero(cl.any(1)):
                k = np.flatnonzero(cl[i]); L = k[-1] - k[0] + 1; f = bool((r['raw'][i, k[-1] + 1:] <= 0).any())
                if cond(L, f) and cl.any(1).sum() == 1:
                    cands.append((r, i))
        if cands:
            pick.append((name,) + cands[rng.integers(len(cands))])
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5)); ax = ax.ravel()
    t = np.arange(32)
    for a, (name, r, i) in zip(ax, pick):
        W, M, sat = r['W'], r['M'], r['sat']
        for o, c in ((0, 'k'), (-1, 'C0'), (1, 'C1'), (-2, 'C2'), (2, 'C3')):
            j = i + o
            if 0 <= j < len(W):
                a.plot(t[:W.shape[1]], W[j], '-', c=c, lw=1.4, label=f'strip {o:+d}' if o else 'clipped strip')
                a.plot(t[:W.shape[1]], M[j], ':', c=c, lw=1.4)
                s = sat[j]
                a.plot(t[:W.shape[1]][s], W[j][s], 'x', c=c, ms=4)
        a.axhline(0, c='grey', lw=.5)
        a.set(title=f"{name}: event {r['eid']} {'xy'[r['view']]} (tan {r['tan']:+.2f})", xlabel='sample (60 ns)', ylabel='ADC')
        a.legend(fontsize=7)
    fig.suptitle('det3 clipped pulses: data (solid), base fit model (dotted), censored samples of the base fit (x)')
    fig.tight_layout()
    tag = os.path.basename(path).replace('.pkl', '')
    fig.savefig(f'{HERE}/figures/examples_{tag}.png', dpi=105)
    print(f'{HERE}/figures/examples_{tag}.png')


if __name__ == '__main__':
    main(sys.argv[1], *(int(x) for x in sys.argv[2:3]))
