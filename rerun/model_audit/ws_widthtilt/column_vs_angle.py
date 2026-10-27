#!/usr/bin/env python3
"""column_vs_angle.py — does the fitted charge column get LONGER for steep tracks?  The fit measures the lateral
span L well (it is on the strips); w = L / (duration).  If the column duration the fit assigns grows with |tan|,
w is compressed by exactly that factor.  Per |tan_ref| bin: median q_u50, q_u90, q_uend, and the scale
r = (w − w0)/(v·tan) together with r·u90/u90(shallow) (does the duration growth account for the compression?).
Writes ws_widthtilt/column_vs_angle.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_split import load, ROOT

BINS = [(0.0, 0.04), (0.04, 0.08), (0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]


def main():
    out = {'bins': BINS}
    for det in ['det2', 'det3', 'det4', 'det6', 'det7']:
        b, T = load(det)
        for p in 'xy':
            df = T[p]; a = np.abs(df.t)
            rows = []
            for lo, hi in BINS:
                m = (a >= lo) & (a < hi)
                s = df[m]
                rows.append(dict(n=int(m.sum()), u50=float(s.q_u50.median()), uend=float(s.q_uend.median()),
                                 r=float(np.median(s.r)) if lo >= 0.08 else np.nan, chi2dof=float(s.chi2dof.median()),
                                 q=float(s.q_sum.median())))
            out[f'{det}_{p}'] = dict(rows=rows, T_gap=30000.0 / b['v_drift'])
            print(f'{det} {p} T30={30000/b["v_drift"]:.0f}  u50: ' + ' '.join(f'{r["u50"]:5.0f}' for r in rows)
                  + '  uend: ' + ' '.join(f'{r["uend"]:5.0f}' for r in rows) + '  r: ' + ' '.join(f'{r["r"]:.3f}' for r in rows))
    json.dump(out, open(f'{ROOT}/ws_widthtilt/column_vs_angle.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
