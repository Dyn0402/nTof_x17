#!/usr/bin/env python3
"""xasym_boundary.py -- is the X +-1 asymmetry crosstalk between adjacent electronics channels?  (FINDINGS §42)

On det4 the X view is four 64-channel DREAM chips; within a chip +x runs to the next LOWER channel, and at a chip
boundary the spatial neighbours are on different chips (local 0 of chip d sits next to local 63 of chip d+1).
Adjacent-channel crosstalk follows the channel order, the board follows position, so clusters centred on a
chip-edge strip separate them:
  interior        -1 = ch+1, +1 = ch-1        (both electronic neighbours)
  centre local 0  -1 = ch+1, +1 = other chip  (the +1 side has no electronic neighbour)
  centre local 63 -1 = other chip, +1 = ch-1  (the -1 side has no electronic neighbour)
If the -1 excess is a leak from channel n into n+1, it vanishes (or reverses) for centres at local 63.
SPS run_71 RAW, det4, masked common mode (head-on beam; the spot straddles the chip edge at 149.76 mm).

    xasym_boundary.py [npz]  -> results/xasym_boundary.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment'))
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM          # noqa: E402

NPZ = '/media/dylan/data/x17/sps_run53_det4_check/staging/run_71/wf_run71_raw_det4only_cmmasked.npz'
QMIN, SAT = 2000.0, 3500.0


def boot(v, n=400):
    v = np.asarray(v, float)
    rng = np.random.default_rng(5)
    return float(v.mean()), float(np.std([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else NPZ
    Z = np.load(path, allow_pickle=False)
    ev, ch, amp = Z['ev'], Z['ch'].astype(np.int64), Z['amp'].astype(np.float64)
    k = ch < 256
    ev, ch, amp = ev[k], ch[k], amp[k]
    assert np.all(VIEW[np.unique(ch)] == 'x')
    ue, ei = np.unique(ev, return_inverse=True)
    idx = ei * 256 + ch
    Q = np.bincount(idx, weights=amp, minlength=len(ue) * 256).reshape(len(ue), 256)
    M = np.zeros(len(ue) * 256); np.maximum.at(M, idx, amp); M = M.reshape(len(ue), 256)
    order = np.argsort(POSITION_MM[:256])                       # spatial order of the X channels
    Qs, Ms, chs = Q[:, order], M[:, order], np.arange(256)[order]
    assert np.allclose(np.diff(POSITION_MM[:256][order]), PITCH_MM)
    c = np.argmax(Qs, axis=1)
    rows = np.arange(len(ue))
    ok = (c >= 2) & (c <= 253)
    cc = np.clip(c, 2, 253)
    q0, qm, qp = Qs[rows, cc], Qs[rows, cc - 1], Qs[rows, cc + 1]
    ok &= (q0 > QMIN) & (Ms[rows, cc] < SAT) & (qm > 0) & (qp > 0) & (Qs[rows, cc - 2] > 0) & (Qs[rows, cc + 2] > 0)
    aQ = (qm - qp) / (qm + qp)
    loc = chs[cc] % 64
    out = {'n_events': int(len(ue)), 'n_used': int(ok.sum())}
    cls = {'interior': ok & (loc >= 2) & (loc <= 61), 'local0': ok & (loc == 0), 'local63': ok & (loc == 63),
           'local1': ok & (loc == 1), 'local62': ok & (loc == 62)}
    for nm, m in cls.items():
        if m.sum() >= 10:
            out[nm] = dict(n=int(m.sum()), aQ=boot(aQ[m]), r=boot(((qm + qp) / (2 * q0))[m]),
                           qm=boot((qm / q0)[m]), qp=boot((qp / q0)[m]))
            r = out[nm]
            print(f'{nm:9s} n {r["n"]:6d}  aQ {r["aQ"][0]:+.3f} ± {r["aQ"][1]:.3f}   Q-1/Q0 {r["qm"][0]:.3f} ± {r["qm"][1]:.3f}   '
                  f'Q+1/Q0 {r["qp"][0]:.3f} ± {r["qp"][1]:.3f}', flush=True)
    # by connector / position, interior only
    pos = POSITION_MM[chs[cc]]
    out['by_pos'] = []
    for lo in np.arange(100, 300, 10.0):
        m = cls['interior'] & (pos >= lo) & (pos < lo + 10)
        if m.sum() >= 50:
            out['by_pos'].append(dict(x=float(lo + 5), n=int(m.sum()), aQ=boot(aQ[m])))
    json.dump(out, open(os.path.join(HERE, 'results', 'xasym_boundary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
