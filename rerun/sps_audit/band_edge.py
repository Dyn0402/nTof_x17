#!/usr/bin/env python3
"""band_edge.py --tel DIR [--out results/band_edge.json]

Does det4 X resolve worse near the edges of its live amplification bands?  The live-band map comes from the data: the
fraction of telescope tracks whose det4 X agrees with the telescope (|r| < 1.5 mm) in 0.5 mm bins of the PREDICTED det4 X
(the telescope y through a linear map fitted on the agreeing tracks).  A bin is live above 50 %.  The X core width is
then compared for tracks whose predicted X has every bin within +-margin live (interior) and the rest.  The Y residual
(the view not being measured) defines the sample, as in analyze_final.py.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_final import load, residuals, gauss_sigma, Z_BACK, Z_DET4      # noqa: E402

SETS = {'run53': (['G53'], None), 'run63 flat700': (['G6302', 'G6303'], ['run63_flat__flat700']),
        'run66 all': (['G66'], None), 'run71 raw700': (['G71'], ['run71_raw__raw700'])}
BIN = 0.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tel', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'band_edge.json'))
    a = ap.parse_args()
    out = {}
    for name, (pre, keys) in SETS.items():
        d, _ = load(a.tel, pre, keys)
        if d is None or not len(d):
            continue
        tel = np.all([(d[f'{p}_n'] == 1).values for p in ('fx', 'fy', 'bx', 'by')], axis=0)
        d = d[tel].reset_index(drop=True)
        ty = (d.fy_p + (d.by_p - d.fy_p) * Z_DET4 / Z_BACK).values
        for e in ('c_pos', 'p0', 'p_mid'):
            ex, ey = d[f'd4_x_{e}'].values, d[f'd4_y_{e}'].values
            ok = np.isfinite(ex) & np.isfinite(ey)
            D = d[ok].reset_index(drop=True); rx, ry, _ = residuals(D, ex[ok], ey[ok])
            tyo = ty[ok]; agree = np.abs(ry) < 1.5
            sl, ic = np.polyfit(tyo[agree], ex[ok][agree], 1); xp = sl * tyo + ic
            edges = np.arange(90.0, 300.0, BIN); k = np.clip(np.searchsorted(edges, xp) - 1, 0, len(edges) - 2)
            nb = len(edges) - 1
            eff = np.array([agree[k == j].mean() if (k == j).sum() >= 10 else np.nan for j in range(nb)])
            live = np.nan_to_num(eff) > 0.5
            fy = np.abs(rx) < 1.5
            rows = {}
            for m in (0, 2, 4, 6):
                inn = np.array([0 <= kk - m and kk + m < nb and live[kk - m:kk + m + 1].all() for kk in k])
                s, er, n = gauss_sigma(ry[fy & inn]); s2, er2, n2 = gauss_sigma(ry[fy & ~inn])
                rows[f'{m * BIN:.1f}'] = dict(interior=[float(s), float(er), int(n)], rest=[float(s2), float(er2), int(n2)])
                print(f'{name:14s} {e:6s} margin {m * BIN:.1f} mm: X core interior {s * 1e3:.0f}+-{er * 1e3:.0f} (n {n}) | rest {s2 * 1e3:.0f} (n {n2})')
            out[f'{name}|{e}'] = dict(live_fraction=float(live.mean()), margins=rows)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
