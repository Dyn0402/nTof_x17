#!/usr/bin/env python3
"""T4 part 3: the T map from beam-off cosmic tracks (a different angle-position correlation, muons, no beam) against
the beam (capsule) T map, 50 mm cells; and the cosmic within-cell angle dependence.  Writes results_cosmic.json and
adds C_<arm>_xy_T50 etc. to tmaps_cosmic.npz."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tmaps as TM
Z = dict(np.load(HERE / 'tmaps.npz'))
C = pd.read_parquet(HERE / 'cache_cosmic.parquet')
out, npz = {}, {}
for arm in 'ACDB':
    d = TM.prep(C, arm)
    r = dict(n=int(len(d)), v_t_finite=float(np.isfinite(d.v_t).mean()) if arm != 'B' else None,
             tan_spread=None)
    if arm != 'B':
        d = d[np.isfinite(d.v_t)]
        r['tanx_q'] = np.nanpercentile(d.tanx, [5, 50, 95]).tolist(); r['tany_q'] = np.nanpercentile(d.tany, [5, 50, 95]).tolist()
    for tag, edges in (('50', TM.E50), ('100', TM.E100)):
        o = TM.build(d, arm, 'xy', edges=edges)
        npz[f'{arm}_xy_T{tag}'] = np.exp(o['lnmap']); npz[f'{arm}_xy_err{tag}'] = o['err']; npz[f'{arm}_xy_n{tag}'] = o['n']
        mb, eb, nb = np.log(Z[f'{arm}_xy_T{tag}']), Z[f'{arm}_xy_err{tag}'], Z[f'{arm}_xy_n{tag}']
        mc, ec, nc = o['lnmap'], o['err'], o['n']
        ok = np.isfinite(mb) & np.isfinite(mc) & (nb >= 100) & (nc >= 100)
        cc = 0.5 * (edges[1:] + edges[:-1])
        inner = (np.abs(cc)[:, None] < 150) & (np.abs(cc)[None, :] < 150)
        res = {}
        for nm, m in (('all', ok), ('interior', ok & inner)):
            if m.sum() < 4:
                continue
            a, b = mb[m] - np.median(mb[m]), mc[m] - np.median(mc[m])
            w = 1 / (eb[m] ** 2 + ec[m] ** 2)
            slope = np.sum(w * a * b) / np.sum(w * a * a)
            res[nm] = dict(n_cells=int(m.sum()), corr=float(np.corrcoef(a, b)[0, 1]),
                           chi2_diff=float(np.sum(w * (a - b) ** 2)), slope_cosmic_on_beam=float(slope),
                           rms_beam_pct=float(100 * a.std()), rms_cosmic_pct=float(100 * b.std()),
                           rms_diff_pct=float(100 * (a - b).std()), med_err_cosmic_pct=float(100 * np.median(ec[m])))
        r[f'cells{tag}'] = res
        if tag == '50':
            r['quad50'] = TM.quad_fit(mc, ec, edges)
    r['prof'] = {k: {str(i): float(v) for i, v in p.items()} for k, p in o['prof'].items() if k != 'table'}
    out[arm] = r
    print(arm, json.dumps({k: v for k, v in r.items() if k != 'prof'}, default=float)[:1800], flush=True)
np.savez(HERE / 'tmaps_cosmic.npz', **npz)
json.dump(out, open(HERE / 'results_cosmic.json', 'w'), indent=1, default=float)
