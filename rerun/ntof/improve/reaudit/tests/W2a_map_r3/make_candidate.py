#!/usr/bin/env python3
"""make_candidate.py -- angle_map_r3_candidate.json = r3_lines.json (lines-only spline: pointing adds nothing on held-out
cosmics, task2) + meta; checks that apply_r3 (pure numpy) reproduces r3lib's sparse evaluation, and how many beam and
cosmic tracks sit where the map extrapolates.  -> angle_map_r3_candidate.json, extrapolation.csv
"""
import json

import numpy as np
import pandas as pd

from r3lib import Spline, HERE, GRID
import apply_r3 as AP
from t6lib import rows_ac, rows_d

T5 = HERE.parent / 'T5_pointing_map'


def flat(m):
    t, s = m['tilt'], m['scale']
    return np.concatenate([np.ravel(t['phi']), np.ravel(s['s_x']), np.ravel(s['s_y'])])


def reclamp(M):
    """clamp every field to its [0.1, 99.9] % range over the clean training lines' positions (both views): the
    spline's values at the extreme corners and outside the lines' support never reach a track."""
    for arm, m in M['maps'].items():
        D = rows_d() if arm == 'D' else rows_ac(arm)
        D = D[D.clean]
        M['maps'][arm]['clamp'] = {'a_x': [-9.0, 9.0], 'a_y': [-9.0, 9.0], 's_x': [-0.99, 9.0], 's_y': [-0.99, 9.0]}
        f = AP.fields(M, arm, D.x.to_numpy(), D.y.to_numpy())
        M['maps'][arm]['clamp'] = {k: [float(np.percentile(v, 0.1)), float(np.percentile(v, 99.9))]
                                   for k, v in zip(('a_x', 'a_y', 's_x', 's_y'), f)}
        M['maps'][arm]['clamp_rule'] = '[0.1, 99.9] percentile of the field over the clean training lines'
    return M


def main():
    for lab in ('lines', 'joint_all', 'joint_g0', 'joint_g1'):
        Mx = reclamp(json.load(open(HERE / f'r3_{lab}.json')))
        json.dump(Mx, open(HERE / f'r3_{lab}.json', 'w'))
    M = json.load(open(HERE / 'r3_lines.json'))
    # D: B-D lines + horizontal pointing (all runs); on held-out B-D lines it ties lines-only, on the independent
    # pointing split (beam.py r3_joint_split) it lowers the detrended cell rms 0.0082 -> 0.0065
    M['maps']['D'] = json.load(open(HERE / 'r3_joint_all.json'))['maps']['D']
    rng = np.random.default_rng(3)
    rows = []
    late = pd.read_parquet(T5 / 'late_tracks.parquet', columns=['arm', 'x_local', 'y_local'])
    for arm, m in M['maps'].items():
        sp = Spline(tilt='pot', scale='sep', ht=m['tilt_basis']['h'], hs=m['scale_basis']['h'])
        sp.beta = flat(m)
        if m['fit'].get('pointing'):
            sp.beta = flat(m)       # nuisances (alpha, gamma, kappa) are not part of the map
        x, y = rng.uniform(-210, 210, 20000), rng.uniform(-210, 210, 20000)
        f1 = sp.fields(np.clip(x, -200, 200), np.clip(y, -200, 200))
        ax, ay, sx, sy = AP.fields(M, arm, x, y)
        c = m['clamp']
        maxdiff = max(float(np.max(np.abs(np.clip(f1[k], *c[k]) - v))) for k, v in
                      (('a_x', ax), ('a_y', ay), ('s_x', sx), ('s_y', sy)))
        sup = np.array(m['support']['mask'], bool)

        def outside(xx, yy):
            ix = np.clip(np.round((np.asarray(xx) - GRID[0]) / 10).astype(int), 0, len(GRID) - 1)
            iy = np.clip(np.round((np.asarray(yy) - GRID[0]) / 10).astype(int), 0, len(GRID) - 1)
            ok = np.isfinite(xx) & np.isfinite(yy)
            return ok & ~sup[ix, iy]
        g = late[late.arm == arm]
        D = rows_d() if arm == 'D' else rows_ac(arm)
        D = D[D.view == 0]
        X, Y = np.meshgrid(GRID, GRID, indexing='ij')
        F = sp.fields(X.ravel(), Y.ravel())
        unc = {k: (float(np.max(np.abs(v[~sup.ravel()] - np.clip(v[~sup.ravel()], *c[k])))) if (~sup).any() else 0.0)
               for k, v in F.items()}
        sx_ = X[sup]; sy_ = Y[sup]
        rows.append(dict(arm=arm, apply_vs_r3lib_maxdiff=maxdiff, support_frac_of_square=float(sup.mean()),
                         support_x=[float(sx_.min()), float(sx_.max())], support_y=[float(sy_.min()), float(sy_.max())],
                         beam_frac_outside_support=float(outside(g.x_local.to_numpy(), g.y_local.to_numpy()).mean()),
                         cosmic_all_lines_frac_outside=float(outside(D.x.to_numpy(), D.y.to_numpy()).mean()),
                         max_clamp_effect=unc, clamp=c))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / 'extrapolation.csv', index=False)
    M['meta'].update(
        variant='candidate r3: A, C lines-only penalised spline (potential tilt + per-view scale); D the same model '
                'fitted to the B-D lines + horizontal capsule pointing cells (nuisances alpha, gamma, kappa per chamber, '
                'not applied: maps[D].fit.pointing)',
        status='PROTOTYPE from test W2a_map_r3 (2026-11-24); not in any product',
        validation='r3_validation.csv; held-out by run parity on cosmic lines, beam pointing (beam_*.csv)',
        pointing='A, C: not used (task 2: no held-out gain on cosmics; cross-check only). D: used (fit.pointing)',
        extrapolation={r['arm']: dict(support_x=r['support_x'], support_y=r['support_y'],
                                      beam_frac_outside_support=r['beam_frac_outside_support']) for r in rows})
    json.dump(M, open(HERE / 'angle_map_r3_candidate.json', 'w'))


if __name__ == '__main__':
    main()
