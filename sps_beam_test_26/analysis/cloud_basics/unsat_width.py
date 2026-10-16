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
    unsat_width.py --recompare     redo only the Magboltz comparison on the stored fits
    --tan T changes the head-on cut (adds '@tan<T>').  A --sat other than 3500 stores the result under '<key>@sat<N>' (cut-stability scan); CB_CENTRE=ref
    (stack centred on the M3 reference strip, rc_diffusion.stack) adds '@ref'.

The fits are in the GAP = 30 frame.  The comparison ('gapcorr') rescales to each chamber's measured
gap (GAP_STUDY, bench_fit.py: det2 30.6, det3 27.9, det7 27.5 mm): v x gap/30, D_T x sqrt(30/gap), and
lists every water branch whose Magboltz v matches (v is not monotonic in water at 347 V/cm).
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
GAP_STUDY = {'det2': 30.6, 'det3': 27.9, 'det7': 27.5}           # bench_fit.py; det4/det6 -> 30


def compare(m, det, E):
    """Magboltz at the fitted v, in the GAP-30 frame and at the chamber's own gap."""
    m['magboltz'] = xw.magboltz(m['v'], E)
    gap = GAP_STUDY.get(det, xw.GAP)
    v, dt, de = m['v'] * gap / xw.GAP, m['DT'] * np.sqrt(xw.GAP / gap), m['DT_err'] * np.sqrt(xw.GAP / gap)
    mb = xw.magboltz(v, E)
    for b in mb['branches']:
        b['excess_quad'] = float(np.sign(dt - b['DT_um_rtcm']) * np.sqrt(abs(dt ** 2 - b['DT_um_rtcm'] ** 2)))
        b['pull'] = float((dt - b['DT_um_rtcm']) / de)
    m['gapcorr'] = dict(gap_mm=gap, v=float(v), v_err=float(m['v_err'] * gap / xw.GAP), DT=float(dt),
                        DT_err=float(de), magboltz=mb)
    return m


def show(key, lab, m):
    g = m['gapcorr']
    br = ' | '.join(f'{b["water_pct"]:.2f} %: MB {b["DT_um_rtcm"]:3.0f}, excess {b["excess_quad"]:+4.0f} ({b["pull"]:+.1f} sd)'
                    for b in g['magboltz']['branches']) or 'no Magboltz solution'
    print(f'{key:46s} {lab:5s} n {m["n"]:4d}  gap {g["gap_mm"]:.1f}: v {g["v"]:5.2f}±{g["v_err"]:.2f}  '
          f'D_T {g["DT"]:4.0f}±{g["DT_err"]:3.0f}  [dry v {g["magboltz"]["v_dry"]:.1f}]  {br}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=','.join(KEYS))
    ap.add_argument('--sat', type=float, default=3500.0)
    ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--recompare', action='store_true')
    ap.add_argument('--tan', type=float, default=None, help='head-on cut |tan_x| (x_width.TAN, default 0.03)')
    a = ap.parse_args()
    if a.tan is not None:
        xw.TAN = a.tan                      # before the pool forks
    out_path = f'{HERE}/results/unsat_width.json'
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    if a.recompare:
        for key, r in out.items():
            for lab in ('all', 'unsat'):
                if lab in r:
                    show(key, lab, compare(r[lab], r['det'], r['E_Vcm']))
        json.dump(out, open(out_path, 'w'), indent=1, default=float)
        return
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
                r[lab] = compare(xw.measure(pool, h, sub, T0, a.nboot), det, E)
                show(key, lab, r[lab])
            tag = (('' if a.sat == 3500.0 else f'@sat{a.sat:.0f}') + ('@ref' if os.environ.get('CB_CENTRE') == 'ref' else '')
                   + ('' if a.tan is None else f'@tan{a.tan:g}'))
            out[key + tag] = r
            json.dump(out, open(out_path, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
