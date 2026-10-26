#!/usr/bin/env python3
"""m3match.py -- does the M3 telescope see det3's second particle? (question 1, the reference side)

    ../../.venv/bin/python m3match.py      -> multiparticle/m3match.parquet, m3match.json

For every full (2x2) multi-particle event, the second particle's line (det3 forward fit, both views) is extrapolated to
each of the eight M3 planes (Z 1302 / 1185 above, 144 / 24 below) and compared with every M3 cluster on that plane
(DataReader cluster trees rebuilt from raw for C22, rerun/outliers/m3clus; strip -> mm by m3feat.calib). The cluster
the recipe ray used (nearest the M3 ray) is excluded. A plane 'sees' the second particle when another cluster lies
within MATCH mm of the extrapolation.

Two nulls on the same events and the same clusters:
  mirror  the point on the other side of the M3 ray at the same offset (ray - (second - ray)): clusters that sit near
          the muon but not on the second particle's line (the muon's own delta rays, noise near the hit).
  swap    the second particle's offset from ITS ray, borrowed from another full event: chance level for this
          event's cluster multiplicity.
The extrapolation error grows with the lever arm: ~0.02 in tan (det3 angle resolution of a single track, 1.0-1.3 deg)
times 470-690 mm is 10-15 mm before multiple scattering, hence MATCH = 20 mm.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mplib as M  # noqa: E402
import m3feat as MF  # noqa: E402

MATCH = 20.0


def clusters():
    an_all, ry_all = MF.load_run('det3')
    out = {}
    for an, ry in zip(an_all, ry_all):
        C = MF.calib(an, ry)
        for i, e in enumerate(an['evn']):
            pl = {}
            for p, (v, l) in MF.PLANES.items():
                n = min(int(an['MGv2_NClus'][i][p]), 300)
                pl[(v, l)] = np.array([C[p][0] * an['MGv2_ClusPos'][i, p, k] + C[p][1] for k in range(n)])
            out[int(e)] = pl
    return out


def main():
    E = pd.read_parquet(os.path.join(M.OUT, 'events.parquet')); F = E[E.full].copy(); a = M.align()
    CL = clusters()
    # second particle and M3 ray in the M3 frame
    xo, yo = M.raw_to_m3(F.o_p0x, F.o_p0y, a); txo, tyo = M.tan_raw_to_m3(F.o_tx, F.o_ty, a)
    xr, yr = M.raw_to_m3(F.mesh_x, F.mesh_y, a); txr, tyr = M.tan_raw_to_m3(F.tan_x, F.tan_y, a)
    ids = F.index.to_numpy(); rng = np.random.default_rng(5); perm = rng.permutation(len(ids))
    rows = []
    for k, e in enumerate(ids):
        if e not in CL:
            continue
        r = dict(event_id=int(e))
        for (v, l), pos in CL[e].items():
            z = M.ZM3[l]
            ray = (xr[k] + txr[k] * (z - M.Z0)) if v == 'x' else (yr[k] + tyr[k] * (z - M.Z0))
            sec = (xo[k] + txo[k] * (z - M.Z0)) if v == 'x' else (yo[k] + tyo[k] * (z - M.Z0))
            j = perm[k]
            sw = ray + (((xo[j] + txo[j] * (z - M.Z0)) - (xr[j] + txr[j] * (z - M.Z0))) if v == 'x'
                        else ((yo[j] + tyo[j] * (z - M.Z0)) - (yr[j] + tyr[j] * (z - M.Z0))))
            if len(pos) and np.min(np.abs(pos - ray)) < 3.0:
                pos = np.delete(pos, np.argmin(np.abs(pos - ray)))
            r[f'n_{v}{l}'] = len(pos); r[f'off_{v}{l}'] = sec - ray
            for nm, pt in (('sec', sec), ('mir', 2 * ray - sec), ('swap', sw)):
                r[f'{nm}_{v}{l}'] = float(np.min(np.abs(pos - pt))) if len(pos) else np.inf
        rows.append(r)
    T = pd.DataFrame(rows).set_index('event_id')
    T.to_parquet(os.path.join(M.OUT, 'm3match.parquet'))
    out = dict(match_mm=MATCH, n=int(len(T)), planes={})
    for v in 'xy':
        for l in range(4):
            k = f'{v}{l}'
            out['planes'][k] = dict(z=M.ZM3[l], extra_clusters=float((T[f'n_{k}'] > 0).mean()),
                                    **{nm: float((T[f'{nm}_{k}'] <= MATCH).mean()) for nm in ('sec', 'mir', 'swap')},
                                    off_med=float(np.median(np.abs(T[f'off_{k}']))))
    for grp, ls in (('upper', (0, 1)), ('lower', (2, 3))):
        for nm in ('sec', 'mir', 'swap'):
            # seen on both planes of a doublet in both views (a full M3 point pair)
            ok = np.ones(len(T), bool)
            for v in 'xy':
                for l in ls:
                    ok &= T[f'{nm}_{v}{l}'] <= MATCH
            out[f'{grp}_all4_{nm}'] = float(ok.mean())
            anyv = np.zeros(len(T), bool)
            for v in 'xy':
                for l in ls:
                    anyv |= T[f'{nm}_{v}{l}'] <= MATCH
            out[f'{grp}_any_{nm}'] = float(anyv.mean())
    json.dump(out, open(os.path.join(M.OUT, 'm3match.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
