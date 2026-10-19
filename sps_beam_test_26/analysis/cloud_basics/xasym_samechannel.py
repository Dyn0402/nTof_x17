#!/usr/bin/env python3
"""xasym_samechannel.py -- the gain-free X asymmetry: each channel against itself.  (FINDINGS §42)

A cluster-level aQ = (Q[c-1] - Q[c+1]) / (...) mixes any sharing asymmetry with the gains of the two neighbours;
on det4 the per-strip gain swings x20 and aQ follows position (-0.27 ... +0.39 along the H4 beam).  Per channel j:
  L_j = median Q_j / Qtot over clusters where j is the -1 neighbour (track one pitch to its +x side)
  R_j = median Q_j / Qtot over clusters where j is the +1 neighbour (track one pitch to its -x side)
with Qtot the cluster's +-3-strip charge.  j's own gain cancels in L_j / R_j; a symmetric footprint gives 1.
A_j = (L_j - R_j) / (L_j + R_j); the chamber's asymmetry is the median over channels with >= NMIN in each role.

    xasym_samechannel.py  -> results/xasym_samechannel.json   (bench: head-on caches; beam: det4 run_71 RAW)
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NMIN = 15
NMIN_BENCH = 4          # the bench head-on samples are ~500 events over ~400 channels


def per_channel(rows, nmin=NMIN):
    """rows: (channel, role -1/+1, Q_j/Qtot).  Returns per-channel A_j and the summary."""
    from collections import defaultdict
    d = defaultdict(lambda: {-1: [], 1: []})
    for c, r, f in rows:
        d[c][r].append(f)
    A = {}
    for c, v in d.items():
        if len(v[-1]) >= nmin and len(v[1]) >= nmin:
            L, R = np.median(v[-1]), np.median(v[1])
            if L + R > 0:
                A[int(c)] = float((L - R) / (L + R))
    a = np.array(list(A.values()))
    rng = np.random.default_rng(2)
    bs = [np.median(a[rng.integers(0, len(a), len(a))]) for _ in range(400)] if len(a) > 3 else [np.nan]
    return dict(n_channels=len(a), median=float(np.median(a)) if len(a) else np.nan, err=float(np.std(bs)),
                mean=float(a.mean()) if len(a) else np.nan, A=A)


def bench(det):
    import x_width as xw
    from strip_gain_audit import bad_set
    from xasym_tests import SETS, AUD, GAPC
    hk, ik = SETS[det]
    bad = bad_set(AUD[hk], 'x')
    rows = []
    seen = set()
    for path in (f'{xw.CACHES}/headon_{hk}.pkl', f'{GAPC}/incl_{ik}.pkl'):
        for e in pickle.load(open(path, 'rb')).values():
            if 'x' not in e or abs(e['tan_x']) >= 0.03:
                continue
            key = (round(e['tan_x'], 5), round(float(e['x']['pos'][0]), 2))
            if key in seen:
                continue
            seen.add(key)
            W = np.asarray(e['x']['W'], float)
            if W.max() >= 3500:
                continue
            Q = W.sum(1)
            c = int(np.argmax(Q))
            if c < 3 or c > len(W) - 4:
                continue
            chs = [int(q) for q in e['x']['ch']]
            if any(q in bad for q in chs[c - 2:c + 3]):
                continue
            tot = Q[c - 3:c + 4].sum()
            if tot <= 0:
                continue
            rows += [(chs[c - 1], -1, Q[c - 1] / tot), (chs[c + 1], 1, Q[c + 1] / tot)]
    return per_channel(rows, NMIN_BENCH)


def beam():
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'det4_sps_assessment')))
    from det4_sps_map import POSITION_MM
    from xasym_boundary import NPZ, QMIN, SAT
    Z = np.load(NPZ, allow_pickle=False)
    ev, ch, amp = Z['ev'], Z['ch'].astype(np.int64), Z['amp'].astype(np.float64)
    k = ch < 256
    ev, ch, amp = ev[k], ch[k], amp[k]
    ue, ei = np.unique(ev, return_inverse=True)
    idx = ei * 256 + ch
    Q = np.bincount(idx, weights=amp, minlength=len(ue) * 256).reshape(len(ue), 256)
    M = np.zeros(len(ue) * 256); np.maximum.at(M, idx, amp); M = M.reshape(len(ue), 256)
    order = np.argsort(POSITION_MM[:256])
    Qs, Ms, chs = Q[:, order], M[:, order], np.arange(256)[order]
    c = np.argmax(Qs, axis=1); r = np.arange(len(ue))
    cc = np.clip(c, 3, 252)
    tot = sum(Qs[r, cc + o] for o in range(-3, 4))
    ok = (c >= 3) & (c <= 252) & (Qs[r, cc] > QMIN) & (Ms[r, cc] < SAT) & (Qs[r, cc - 1] > 0) & (Qs[r, cc + 1] > 0) & (tot > 0)
    rows = list(zip(chs[cc - 1][ok], [-1] * ok.sum(), (Qs[r, cc - 1] / tot)[ok])) + \
        list(zip(chs[cc + 1][ok], [1] * ok.sum(), (Qs[r, cc + 1] / tot)[ok]))
    out = per_channel(rows)
    out['n_events'] = int(ok.sum())
    return out


def main():
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        out[det] = bench(det)
        print(f'bench {det}: A median {out[det]["median"]:+.3f} ± {out[det]["err"]:.3f} over {out[det]["n_channels"]} channels', flush=True)
    out['det4_beam'] = beam()
    b = out['det4_beam']
    print(f'beam det4 run_71: A median {b["median"]:+.3f} ± {b["err"]:.3f} over {b["n_channels"]} channels ({b["n_events"]} events)')
    json.dump(out, open(os.path.join(HERE, 'results', 'xasym_samechannel.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
