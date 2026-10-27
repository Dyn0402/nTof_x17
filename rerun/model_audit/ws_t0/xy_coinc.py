#!/usr/bin/env python3
"""xy_coinc.py -- width of the x-y time coincidence on the golden rows, per timing variable (F5, workstream D).

The x and y views of one chamber read the SAME drifting electrons, so any time the two fits assign to the column must
agree up to a fixed FEU offset once each view is phase-corrected (t + 10 ns * ftst_view).  Variables:
  t0    the fitted mesh time (free: slides by whole depth bins, the 'tooth')
  t50   t0 + u50, the median arrival of the fitted charge (invariant under the tooth shift)
  t90   t0 + u90
Sample: both views fitted, quality_ok, |tan| < 0.45 both, i.e. the single-track population the gate must accept.
Prints, per chamber: median, robust sigma, and the fraction inside 120 ns and inside 2.5 rsig of the median.
"""
import json, os
import numpy as np
import pandas as pd

M = '/home/dylan/x17/cosmic_bench/model_audit'
HERE = os.path.dirname(os.path.abspath(__file__))


def rsig(a):
    q = np.percentile(a, [16, 84]); return 0.5 * (q[1] - q[0])


def main():
    out = {}
    for det in ['det2', 'det3', 'det4', 'det6', 'det7']:
        d = pd.read_parquet(f'{M}/{det}/final_physmask/events.parquet')
        ok = (d.x_ok.astype(bool) & d.y_ok.astype(bool) & d.x_quality_ok.astype(bool) & d.y_quality_ok.astype(bool)
              & (d.x_tan_theta.abs() < 0.45) & (d.y_tan_theta.abs() < 0.45) & (d.x_ftst >= 0) & (d.y_ftst >= 0))
        s = d[ok]; out[det] = {}
        for var, ux, uy in (('t0', None, None), ('t50', 'x_q_u50', 'y_q_u50'), ('t90', 'x_q_u90', 'y_q_u90')):
            tx = s.x_t0 + 10 * s.x_ftst + (s[ux] if ux else 0)
            ty = s.y_t0 + 10 * s.y_ftst + (s[uy] if uy else 0)
            dt = (tx - ty).values
            med, rs = float(np.median(dt)), float(rsig(dt))
            out[det][var] = dict(n=int(len(dt)), median=med, rsig=rs,
                                 in120=float(np.mean(np.abs(dt - med) <= 120)),
                                 in2p5=float(np.mean(np.abs(dt - med) <= 2.5 * rs)),
                                 tol_2p5=2.5 * rs,
                                 q95=float(np.percentile(np.abs(dt - med), 95)),
                                 q98=float(np.percentile(np.abs(dt - med), 98)))
            o = out[det][var]
            print(f'{det} {var:4s} n={o["n"]:5d} median {med:+6.1f} rsig {rs:5.1f}  inside 120 ns {100 * o["in120"]:5.1f} %  '
                  f'inside 2.5 rsig ({2.5 * rs:5.0f} ns) {100 * o["in2p5"]:5.1f} %  |dt| q95 {o["q95"]:4.0f} q98 {o["q98"]:4.0f}')
    json.dump(out, open(os.path.join(HERE, 'xy_coinc.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
