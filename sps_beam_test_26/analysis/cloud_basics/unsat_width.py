#!/usr/bin/env python3
"""unsat_width.py -- the X width / drift-velocity fit of x_width.py on the unsaturated events only.

At operating HV 30-85 % of the head-on bench events reach the ADC ceiling on the leading strip.  The
det3 amplification scan at 347 V/cm shows the stack-fitted v flat at ~39 um/ns (GAP=30 frame) up to
485 V and falling once saturation sets in (36.5 at 495 V, 24 at 525 V), so a v measured at operating
gain may be saturation-biased -- and the bench 'wet gas' reading rests on v.  Here every dataset is
refitted with all events and with the events whose X window never exceeds SAT_ADC (per-event selection
on the waveform maximum; total charge does not depend on depth, so it cannot select a drift velocity).

    unsat_width.py [--keys k1,k2] [--sat 3500] [--nboot 20]
        -> results/unsat_width.json
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import x_width as xw

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ['sat_det3__long_run_resist_490V_drift_1000V', 'sat_det3',
        'sat_det3__drift_scan_resist_490V_drift_500V', 'sat_det3__drift_scan_resist_490V_drift_700V',
        'sat_det3__drift_scan_resist_490V_drift_900V', 'sat_det3__drift_scan_resist_490V_drift_1100V',
        'sat_det3__hv_scan_resist_445V_drift_1000V', 'sat_det3__hv_scan_resist_465V_drift_1000V',
        'o22_long_det2', 'g_det4', 'g_det6_long', 'g_det7_long', 'o22_long_det3']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=','.join(KEYS))
    ap.add_argument('--sat', type=float, default=3500.0)
    ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    out_path = f'{HERE}/results/unsat_width.json'
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    with ProcessPoolExecutor(a.workers) as pool:
        for key in a.keys.split(','):
            det = xw.det_of(key)
            h, _ = xw.template(det)
            ev = pickle.load(open(f'{xw.CACHES}/headon_{key}.pkl', 'rb'))
            T0 = xw.GAP / (xw.V_BUNDLE[det] * 1e-3)
            if key.startswith(xw.SCAN):
                T0 = xw.GAP / (xw.V_SCAN[int(xw.drift_hv(key))] * 1e-3)
            E = xw.drift_hv(key) / 2.88
            uns = {k: e for k, e in ev.items() if e['x']['W'].max() < a.sat}
            r = dict(det=det, E_Vcm=E, sat_adc=a.sat,
                     frac_unsat=sum(abs(e['tan_x']) < xw.TAN for e in uns.values())
                     / max(sum(abs(e['tan_x']) < xw.TAN for e in ev.values()), 1))
            for lab, sub in (('all', ev), ('unsat', uns)):
                if sum(abs(e['tan_x']) < xw.TAN for e in sub.values()) < xw.MIN_EV:
                    continue
                m = xw.measure(pool, h, sub, T0, a.nboot)
                m['magboltz'] = xw.magboltz(m['v'], E)
                r[lab] = m
                mb = m['magboltz']
                print(f'{key:46s} {lab:5s} n {m["n"]:4d}  v {m["v"]:5.2f}±{m["v_err"]:.2f}  sig0 {m["sig0"]:.3f}  '
                      f'D_T {m["DT"]:4.0f}±{m["DT_err"]:3.0f}  [MB at that v: {mb["DT_um_rtcm"]:4.0f} @ '
                      f'{mb["water_pct"]:.2f} %, dry v {mb["v_dry"]:.1f} D_T {mb["DT_dry"]:.0f}]', flush=True)
            out[key] = r
            json.dump(out, open(out_path, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
