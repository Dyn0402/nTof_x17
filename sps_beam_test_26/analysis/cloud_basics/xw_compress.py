#!/usr/bin/env python3
"""xw_compress.py -- is the X response compressed below the ADC ceiling?  Model-free (FINDINGS §34).

At the peak sample of an event's leading strip, r = (W[c-1] + W[c+1]) / (2 W[c]).  With a linear
response r depends only on the cluster's footprint, so it is flat in the peak amplitude A = W[c] (the
xw_sim toy, same layouts and statistics, shows what flat means here, Landau and Polya included).  If
the front end compresses large pulses before it clips, the peak strip is held back while its smaller
neighbours are not, and r rises as A approaches the ceiling -- which widens the stack of bright,
nominally unsaturated events.  Head-on events (|tan_x| < 0.03), window max < 3500 (the unsat cut) and,
separately, all events.

    xw_compress.py [--sets det3L,det4,det7R,det7H] [--nrep 3]   -> results/xw_compress.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x_width as xw                                         # noqa: E402
import xw_sim as X                                           # noqa: E402

ABINS = np.array([200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4200])


def ratios(ev):
    A, R = [], []
    for e in ev.values():
        if 'x' not in e or abs(e['tan_x']) >= xw.TAN:
            continue
        W = np.asarray(e['x']['W'], float)
        c, k = np.unravel_index(np.argmax(W), W.shape)
        if c < 1 or c > len(W) - 2:
            continue
        A.append(W[c, k]); R.append((W[c - 1, k] + W[c + 1, k]) / (2 * W[c, k]))
    return np.array(A), np.array(R)


def binned(A, R):
    rows = []
    for lo, hi in zip(ABINS[:-1], ABINS[1:]):
        m = (A >= lo) & (A < hi)
        if m.sum() < 15:
            rows.append(dict(lo=int(lo), hi=int(hi), n=int(m.sum()))); continue
        r = R[m]
        bs = [np.median(np.random.default_rng(b).choice(r, len(r))) for b in range(200)]
        rows.append(dict(lo=int(lo), hi=int(hi), n=int(m.sum()), med=float(np.median(r)), err=float(np.std(bs))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='det3L,det4,det7R,det7H')
    ap.add_argument('--nrep', type=int, default=3)
    a = ap.parse_args()
    inp = pickle.load(open(X.INPUTS, 'rb'))
    out = {}
    for name in a.sets.split(','):
        S = inp[name]
        ev = pickle.load(open(f'{xw.CACHES}/headon_{S["key"]}.pkl', 'rb'))
        A, R = ratios(ev)
        TA, TR = [], []
        for k in range(a.nrep):
            ta, tr = ratios(X.Gen(S, 'base', 600000 + k).sample())
            TA.append(ta); TR.append(tr)
        TA, TR = np.concatenate(TA), np.concatenate(TR)
        d, t = binned(A, R), binned(TA, TR)
        out[name] = dict(data=d, toy=t, toy_nrep=a.nrep)
        print(f'{name}: neighbour/peak ratio at the peak sample, by peak ADC')
        for x, y in zip(d, t):
            f = lambda q: f'{q["med"]:.3f}±{q["err"]:.3f} (n {q["n"]:4d})' if 'med' in q else f'   --   (n {q["n"]:4d})'
            print(f'   {x["lo"]:4d}-{x["hi"]:4d}: data {f(x)}   toy {f(y)}')
    json.dump(out, open(f'{HERE}/results/xw_compress.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
