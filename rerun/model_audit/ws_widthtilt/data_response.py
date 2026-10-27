#!/usr/bin/env python3
"""data_response.py — F4 on the finalize tables: (w - w0)/(v·tan_ref) per SIGNED tan_ref bin, both views, five
chambers; i.e. the angle-scale factor as a function of angle with the bundle's w0 removed (kw not applied).
Writes ws_widthtilt/data_response.json."""
import json, os
import numpy as np
ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
EDGES = [-0.45, -0.28, -0.20, -0.14, -0.08, 0.08, 0.14, 0.20, 0.28, 0.45]


def main():
    out = {'edges': EDGES}
    for det in ['det2', 'det3', 'det4', 'det6', 'det7']:
        b = json.load(open(f'{ROOT}/{det}/calib_bundle/bundle.json'))
        d = np.load(f'{ROOT}/{det}/final_physmask/angles/per_event.npz')
        for p in 'xy':
            tr, w = d[f'{p}_tan_ref'], d[f'{p}_w']
            r = (w * 1e3 - b['w0'][p]) / (b['v_drift'] * tr)
            med, err, n = [], [], []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = (tr >= lo) & (tr < hi) & np.isfinite(r)
                n.append(int(m.sum()))
                if lo == -0.08 or m.sum() <= 5:
                    med.append(np.nan); err.append(np.nan); continue
                x = r[m]; q = np.percentile(x, [16, 84])
                med.append(float(np.median(x)))
                err.append(float(1.2533 * 0.5 * (q[1] - q[0]) / np.sqrt(m.sum())))
            out[f'{det}_{p}'] = dict(scale=med, err=err, n=n, v=b['v_drift'], w0=b['w0'][p], kw=b['kw'][p])
            print(det, p, ' '.join(f'{x:.3f}' for x in med), n)
    os.makedirs(f'{ROOT}/ws_widthtilt', exist_ok=True)
    json.dump(out, open(f'{ROOT}/ws_widthtilt/data_response.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
