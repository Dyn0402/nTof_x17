#!/usr/bin/env python3
"""onset.py -- FD (10a): where does the coarse-bin excess start, and what length sets it?

For every scan_a run (chamber x bin width x clustering x pitch/footprint variant): the floor of that run (mean s68 at
the angles where one bin spans less than half a strip), the excess sqrt(s68^2 - floor^2) at each angle, and the onset
angle where the excess first exceeds ONSET_DEG (linear interpolation).  At the onset the bin's segment across the strips
is L = v dt tan(theta); it is compared with two candidate scales: the strip pitch, and the effective width of one
electron's footprint on the strips at mid-gap, sigma_eff^2 = sigma0^2 + Dp^2 u_mid + pitch^2/12.  The scale that
makes L_onset the same for every run is the mechanism's.

    onset.py   -> results/onset.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'ws_floor'))
import floor_toy as ft  # noqa: E402

ONSET_DEG = 0.30


def main():
    sc = json.load(open(os.path.join(HERE, 'results', 'scan_a.json')))
    out = {}
    for tag, r in sc.items():
        C = ft.CHAMBERS[r['det']]
        angs = np.array([float(a) for a in r['by_angle']])
        s = np.array([b['s68'] for b in r['by_angle'].values()])
        L = r['v'] * r['dt_bin'] * np.tan(np.radians(angs))
        flo = s[L < 0.5 * r['pitch']].mean() if (L < 0.5 * r['pitch']).sum() >= 2 else s[:3].mean()
        exc = np.sqrt(np.clip(s ** 2 - flo ** 2, 0, None))
        u_mid = 0.5 * C['G'] / r['v']
        sig_eff = float(np.sqrt(r['s0'] ** 2 + C['Dp'] ** 2 * u_mid + r['pitch'] ** 2 / 12))
        on = None
        for i in range(1, len(angs)):
            if exc[i] > ONSET_DEG and np.all(exc[i:min(i + 2, len(exc))] > ONSET_DEG * 0.8):
                f = (ONSET_DEG - exc[i - 1]) / max(exc[i] - exc[i - 1], 1e-6)
                on = float(angs[i - 1] + np.clip(f, 0, 1) * (angs[i] - angs[i - 1]))
                break
        Lon = r['v'] * r['dt_bin'] * np.tan(np.radians(on)) if on is not None else None
        out[tag] = dict(det=r['det'], config=r['config'], dt_bin=r['dt_bin'], pitch=r['pitch'], s0=r['s0'], v=r['v'],
                        floor=float(flo), sigma_eff=sig_eff, onset_deg=on, L_onset=Lon,
                        L_over_pitch=Lon / r['pitch'] if on else None, L_over_sig=Lon / sig_eff if on else None,
                        excess_at_45=float(exc[-1]), angles=angs.tolist(), excess=exc.tolist(),
                        L=L.tolist(), s68=s.tolist())
    for cfg in ('full', 'no_landau'):
        for key in ('L_over_pitch', 'L_over_sig'):
            v = [o[key] for o in out.values() if o['config'] == cfg and o[key] is not None and o['dt_bin'] >= 30]
            if v:
                print(cfg, key, f'n {len(v)} mean {np.mean(v):.2f} spread (rms/mean) {np.std(v) / np.mean(v):.2f}')
    for tag, o in out.items():
        print(f"{tag:32s} floor {o['floor']:.2f} onset {o['onset_deg'] if o['onset_deg'] is None else round(o['onset_deg'], 1)}"
              f" L/pitch {o['L_over_pitch'] and round(o['L_over_pitch'], 2)} L/sig {o['L_over_sig'] and round(o['L_over_sig'], 2)}"
              f" excess45 {o['excess_at_45']:.2f}")
    json.dump(out, open(os.path.join(HERE, 'results', 'onset.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
