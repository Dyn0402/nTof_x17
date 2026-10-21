#!/usr/bin/env python3
"""
channel_maps.py -- every recovery measure for each of the 44 channels, so slow
channels can be traced to an arm, a bar end or a PMT.

    python -m ntof_scint_recovery.channel_maps

Production runs, live channels.  Per channel and class:

``t_base3``   baseline back within +-3 mV          (quick.py, recovery.csv)
``under_mV``  deepest baseline excursion against the pulse direction after 20 us (quick curves)
``t_amp90``   pulse amplitude scale back to >= 90 % of its 1-10 ms value (v2 HA)
``t_shape80`` 80 % of the early template-fit misfit excess gone (v2 HR, isolated pulses)
``scale_mV``  the channel's amplitude scale at 1-10 ms (90 % quantile above the floor)

Writes ``out_dir('channels')/channel_maps.csv``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from ntof_scint_recovery.coinc_analysis import load as load_v2
from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.pulse_analysis import AMP_FLOOR, cond_quantile, t_within
from ntof_scint_recovery.quick import DETS

CLS = ('par', 'ded')


def main(argv=None) -> int:
    rec = {(r['det'], r['ch'], r['cls']): r for r in csv.DictReader(open(out_dir('quick') / 'recovery.csv'))}
    C = np.load(out_dir('quick') / 'curves.npz')
    tcq = C['tc_us']
    hd = out_dir('pulsehist_v3') if any(out_dir('pulsehist_v3').glob('*.npz')) else out_dir('pulsehist_v2')
    H = load_v2(hd)
    te = H['t_edges_us']
    tc = np.sqrt(te[:-1] * te[1:])[:100]
    ae, rfe = H['a_edges'], H['rf_edges']
    rfc = np.sqrt(rfe[1:] * np.maximum(rfe[:-1], rfe[1] / 2))
    pl = (tc > 1e3) & (tc < 1e4)
    chans = [tuple(c) for c in H['channels']]
    rows = []
    for i, (d, ch) in enumerate(chans):
        det = DETS[d]
        fam = det[:3]
        for k, c in enumerate(CLS):
            r = dict(det=det, ch=int(ch), cls=c)
            q = rec.get((det, str(ch), c))
            r['t_base3'] = float(q['t3mV_us']) if q else np.nan
            key = f'{det}{ch}_{c}_shift50'
            if key in C.files:
                y = C[key]
                late = (tcq > 20) & (tcq < 1e4)
                r['under_mV'] = float(np.nanmin(y[late]))
            else:
                r['under_mV'] = np.nan
            ha = H['HA'][k, i]
            g = cond_quantile(ha, ae, AMP_FLOOR[fam], 0.9)[:100]
            gpl = np.nanmedian(g[pl])
            r['scale_mV'] = gpl
            r['t_amp90'] = t_within(tc, g / gpl, 0.9, 1e9)
            hr = H['HR'][k, i]
            cc = np.cumsum(hr, axis=1)
            tot = cc[:, -1]
            idx = np.argmax(cc >= 0.5 * np.maximum(tot, 1)[:, None], axis=1)
            mrf = rfc[idx].astype(float)
            mrf[tot < 30] = np.nan
            mrf = mrf[:100]
            rp = np.nanmedian(mrf[pl])
            early = np.nanmax(mrf[(tc > 0.5) & (tc < 100)]) if np.isfinite(mrf[(tc > 0.5) & (tc < 100)]).any() else np.nan
            r['t_shape80'] = t_within(tc, mrf, 0, rp + 0.2 * (early - rp)) if np.isfinite(early) else np.nan
            rows.append(r)
    o = out_dir('channels')
    with open(o / 'channel_maps.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} channel-class rows -> {o}/channel_maps.csv')
    for r in rows:
        if r['cls'] == 'ded':
            print(f"  {r['det']}{r['ch']}: base {r['t_base3']:8.1f}  under {r['under_mV']:6.1f}  amp90 {r['t_amp90']:8.1f}  "
                  f"shape80 {r['t_shape80']:7.1f} us  scale {r['scale_mV']:5.0f} mV")
    return 0


if __name__ == '__main__':
    sys.exit(main())
