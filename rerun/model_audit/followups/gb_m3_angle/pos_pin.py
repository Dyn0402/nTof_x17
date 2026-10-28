#!/usr/bin/env python3
"""pos_pin.py -- GB: the two DUTs' POSITIONS pin the M3 slope, with no angle reconstruction involved.

A reference slope error delta_s (M3 fit slope - the true chord between the two chambers) moves the M3 prediction at
z = 232 and z = 702 by different amounts, so the position residuals r_i = det_i - M3(z_i) obey
    r_6 - r_7 = (702 - 232) mm * delta_s + n_6 - n_7.
s68(r_6 - r_7) / 470 mm is therefore an upper bound on the M3 slope error (it also carries both chambers' position
noise), per event, and its cell medians map any position-dependent M3 angle error.  Compared with the cell medians of
the det6-det7 common ANGLE term (pair_cov.py), this says whether that common term is the reference (it would show here
470 mm x its size) or something both chambers do at the same place (FB's bow).

Input: gate_full/<det>/unified/rays_<det>.parquet (unified reco, per-chamber alignment, waveform p0 positions).

    pos_pin.py   -> results/pos_pin.json
"""
import json
import os

import numpy as np
import pandas as pd

from common import RES, AUDIT, load_dut, deg, s68

L = 702.0 - 232.0


def s68mm(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) < 10)]
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def main():
    g = os.path.join(AUDIT, 'gate_full')
    r6 = pd.read_parquet(f'{g}/det6/unified/rays_det6.parquet').set_index('event_id')
    r7 = pd.read_parquet(f'{g}/det7/unified/rays_det7.parquet').set_index('event_id')
    j = r6.join(r7, lsuffix='6', rsuffix='7', how='inner')
    j = j[j.x_ok6.astype(bool) & j.y_ok6.astype(bool) & j.x_ok7.astype(bool) & j.y_ok7.astype(bool)]
    out = {'n': int(len(j))}
    rng = np.random.default_rng(5)
    for c in ('x', 'y'):
        a = (j[f'det_{c}a6'] - j[f'ref_{c}6']).to_numpy()
        b = (j[f'det_{c}a7'] - j[f'ref_{c}7']).to_numpy()
        d = a - b
        ok = (np.abs(a) < 10) & (np.abs(b) < 10)
        exc = s68mm(d[ok]) ** 2 - s68mm(a[ok]) ** 2 - s68mm(b[ok]) ** 2
        bs = []
        for _ in range(300):
            i = rng.integers(0, ok.sum(), ok.sum())
            aa, bb = a[ok][i], b[ok][i]
            bs.append(s68mm(aa - bb) ** 2 - s68mm(aa) ** 2 - s68mm(bb) ** 2)
        res = dict(s68_r6_mm=s68mm(a[ok]), s68_r7_mm=s68mm(b[ok]), s68_diff_mm=s68mm(d[ok]),
                   bound_slope_deg=float(np.degrees(s68mm(d[ok]) / L)),
                   excess_mm2=float(exc), excess_err_mm2=float(np.std(bs)),
                   excess_ul95_slope_deg=float(np.degrees(np.sqrt(max(exc + 1.645 * np.std(bs), 0)) / L)),
                   cov_r6_r7_mm2=float(np.cov(a[ok], b[ok])[0, 1]))
        # cell medians (40 mm, M3 position at the det6 plane) of delta_s, in degrees
        key = (np.floor(j['ref_x6'] / 40).astype(int) * 1000 + np.floor(j['ref_y6'] / 40).astype(int)).to_numpy()
        ds = np.degrees(d / L)
        gg = pd.DataFrame({'k': key[ok], 'ds': ds[ok]}).groupby('k')
        cm = gg.median()[gg.size() >= 25]
        res['cell_delta_s'] = dict(n_cells=int(len(cm)), std_deg=float(cm['ds'].std()),
                                   max_abs_deg=float(cm['ds'].abs().max()))
        out[c] = res
        print(c, json.dumps(res))
    # the common ANGLE cell map vs the delta_s cell map, same cells, per angle view (view x = M3 Y, view y = -M3 X)
    for v, pc in (('x', 'y'), ('y', 'x')):
        d6, _, _ = load_dut('det6', v)
        d7, _, _ = load_dut('det7', v)
        jj = d6.join(d7, lsuffix='6', rsuffix='7', how='inner').join(j, how='inner')
        t6 = deg(jj['tan_fit6']) - deg(jj['tan_ref6'])
        t7 = deg(jj['tan_fit7']) - deg(jj['tan_ref7'])
        dsv = np.degrees(((jj[f'det_{pc}a6'] - jj[f'ref_{pc}6']) - (jj[f'det_{pc}a7'] - jj[f'ref_{pc}7'])).to_numpy() / L)
        key = (np.floor(jj['ref_x6'] / 40).astype(int) * 1000 + np.floor(jj['ref_y6'] / 40).astype(int)).to_numpy()
        ok = (np.abs(t6) < 5) & (np.abs(t7) < 5) & (np.abs(dsv) < 1.2)
        gg = pd.DataFrame({'k': key[ok], 'c': 0.5 * (t6[ok] + t7[ok]), 'ds': dsv[ok]}).groupby('k')
        cm = gg.median()[gg.size() >= 25]
        out[f'angle_view_{v}'] = dict(n_cells=int(len(cm)), common_cell_std_deg=float(cm['c'].std()),
                                      delta_s_cell_std_deg=float(cm['ds'].std()),
                                      r_cells=float(np.corrcoef(cm['c'], cm['ds'])[0, 1]),
                                      r_events=float(np.corrcoef(0.5 * (t6[ok] + t7[ok]), dsv[ok])[0, 1]))
        print('view', v, out[f'angle_view_{v}'])
    json.dump(out, open(os.path.join(RES, 'pos_pin.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
