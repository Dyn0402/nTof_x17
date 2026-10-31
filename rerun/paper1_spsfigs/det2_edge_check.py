#!/usr/bin/env python3
"""det2_edge_check.py <headon.pkl> -- quick look at the det2 edge-T tension of rerun/consistency (2026-11-01):
S4 re-check head-on stack out30 - in30 = -86 +- 33 ns against the edge study's column duration -4 ns.

Both are T(edge) - T(interior), but over different distance distributions of a strongly non-monotonic T(d)
(edge study prof_T: ~584 ns at 0-5 mm, ~915-950 at 5-15 mm, ~800-860 beyond).  This weights the edge study's own
T(d) profile with the S4 sample's distance distribution (min distance of the M3 reference to the active-area box, the
S4 'edge' rule) and reports what the column duration predicts for the S4 split.
"""
import json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
X0, X1, Y0, Y1 = 0.0, 398.6, 19.7, 378.8          # det2 box used by s4r_specs (TRUE_ACTIVE_BY_DET mx17_2)


def main():
    H = pickle.load(open(sys.argv[1], 'rb'))
    E = json.load(open(os.path.join(REPO, 'rerun', 'paper1_figs', 'results', 'edge_study.json')))['det2']
    prof = E['prof_T']
    lo = np.array([p['d_lo'] for p in prof]); hi = np.array([p['d_hi'] for p in prof]); m = np.array([p['m'] for p in prof])
    d, side = [], []
    for e in H.values():
        dd = [e['ref_mesh_x'] - X0, X1 - e['ref_mesh_x'], e['ref_mesh_y'] - Y0, Y1 - e['ref_mesh_y']]
        d.append(min(dd)); side.append('xy'[int(np.argmin(dd)) // 2])
    d = np.array(d); side = np.array(side)

    def pred(sel):
        k = np.clip(np.searchsorted(hi, d[sel], side='right'), 0, len(m) - 1)
        return float(np.mean(m[k]))
    out_ = d < 30; in_ = ~out_
    r = dict(n_out=int(out_.sum()), n_in=int(in_.sum()),
             out_frac_0_5mm=round(float(np.mean(d[out_] < 5)), 3), out_frac_5_15mm=round(float(np.mean((d[out_] >= 5) & (d[out_] < 15))), 3),
             out_frac_nearest_y_side=round(float(np.mean(side[out_] == 'y')), 3),
             colsum_T_out_pred=round(pred(out_), 1), colsum_T_in_pred=round(pred(in_), 1),
             colsum_dT_pred=round(pred(out_) - pred(in_), 1),
             edge_study_dT=round(E['dT_edge'], 1), s4_dT=round(727.63 - 813.54, 1))
    print(json.dumps(r, indent=1))
    json.dump(r, open(os.path.join(HERE, 'results', 'det2_edge_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
