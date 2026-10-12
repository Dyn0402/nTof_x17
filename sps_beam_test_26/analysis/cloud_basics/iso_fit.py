#!/usr/bin/env python3
"""iso_fit.py -- can any bench gas (Ar/iso (100-i)/i + water) give the measured v AND X D_T?  (FINDINGS §29)

Magboltz grid results/iso/magboltz_benchi<i>_w<w>_a0_E<E>_c15.json (make_iso_jobs.py).  Data from
x_width.json (stack T-free fits; v = GAP / T, D_T = sigma_drift / sqrt(GAP)), per dataset at its own
field.  det3's 6-27 scan is rescaled to its 27.9 mm gap (driftscan_fit.py): v x 27.9/30,
D_T x sqrt(30/27.9).  For each chamber (its own composition) chi^2 over the grid, three ways:
v only, D_T only, both.  Errors: bootstrap (+3 % on v for the gap/field scale).

    iso_fit.py -> results/iso_fit.json
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN_GAP = 27.9
SATURATED = 505          # amplification-scan sub-runs above this saturate (x_width.py)


def grid():
    G = {}
    for f in glob.glob(f'{HERE}/results/iso/magboltz_benchi*_a0_E*_c15.json'):
        m = re.search(r'benchi([\dp]+)_w([\dp]+)_a0_E(\d+)', f)
        i, w, E = (float(m.group(k).replace('p', '.')) for k in (1, 2, 3))
        p = json.load(open(f))['points'][0]
        G[(i, w, E)] = (p['v_true_um_ns'], p['DT_um_rtcm'])
    return G


def data():
    xw = json.load(open(f'{HERE}/results/x_width.json'))
    D = {}
    for key, r in xw.items():
        q = r['all']
        if q['sig0'] < 0.05 or q['T'] > 1.45 * 30.0 / (r['v_bundle'] * 1e-3) and not key.startswith('sat_det3__'):
            continue                                       # fit at a bound (g_det6_longer sigma_0, o22 det3 T)
        m = re.search(r'resist_(\d+)V', key)
        if m and int(m.group(1)) > SATURATED:
            continue
        v, ve, dt, de = q['v'], q['v_err'], q['DT'], q['DT_err']
        if key.startswith('sat_det3'):
            v, ve = v * SCAN_GAP / 30, ve * SCAN_GAP / 30
            dt, de = dt * np.sqrt(30 / SCAN_GAP), de * np.sqrt(30 / SCAN_GAP)
        if 'drift_500V' in key:
            continue                                       # T beyond the window: v unconstrained
        D.setdefault(r['det'], []).append(dict(key=key, E=round(r['E_Vcm']), v=v, ve=np.hypot(ve, 0.03 * v), DT=dt, DTe=de))
    return D


def main():
    G = grid()
    isos = sorted({k[0] for k in G}); ws = sorted({k[1] for k in G}); Es = sorted({k[2] for k in G})
    wf = np.linspace(0, 1.0, 41)
    D = data()
    out = {}
    for det, rows in D.items():
        res = {}
        for lab in ('v', 'DT', 'both'):
            best = None
            for i in isos:
                for w in wf:
                    c2 = 0.0
                    for d in rows:
                        E = min(Es, key=lambda e: abs(e - d['E']))
                        vv = np.interp(w, ws, [G[(i, x, E)][0] for x in ws])
                        dd = np.interp(w, ws, [G[(i, x, E)][1] for x in ws])
                        if lab in ('v', 'both'):
                            c2 += ((d['v'] - vv) / d['ve']) ** 2
                        if lab in ('DT', 'both'):
                            c2 += ((d['DT'] - dd) / d['DTe']) ** 2
                    if best is None or c2 < best[0]:
                        best = (c2, i, w)
            res[lab] = dict(chi2=best[0], ndf=len(rows) * (2 if lab == 'both' else 1) - 2, iso=best[1], water=float(best[2]))
        # the D_T each composition predicts at the v-only best fit, per row
        i, w = res['v']['iso'], res['v']['water']
        ws5 = sorted({k[1] for k in G if k[0] == 5.0})
        wf5 = np.linspace(0, max(ws5), 61)
        for d in rows:
            E = min(Es, key=lambda e: abs(e - d['E']))
            d['DT_at_vfit'] = float(np.interp(w, ws, [G[(i, x, E)][1] for x in ws]))
            d['v_at_vfit'] = float(np.interp(w, ws, [G[(i, x, E)][0] for x in ws]))
            # the nominal gas: Ar/iso 95/5 with the water that best gives this row's v (v is not monotonic in
            # water at >~300 V/cm, so take the closest of the fine grid, not an inversion)
            v5 = np.interp(wf5, ws5, [G[(5.0, x, E)][0] for x in ws5])
            w5 = float(wf5[np.argmin(abs(v5 - d['v']))])
            d['water_95'] = w5
            d['DT_at_v95'] = float(np.interp(w5, ws5, [G[(5.0, x, E)][1] for x in ws5]))
        out[det] = dict(fits=res, rows=rows)
        print(det, '  '.join(f'{k}: chi2 {r["chi2"]:.1f}/{r["ndf"]} iso {r["iso"]:g} % water {r["water"]:.2f} %' for k, r in res.items()))
        for d in rows:
            print(f'    {d["key"]:48s} E {d["E"]:3d} v {d["v"]:5.1f}±{d["ve"]:.1f} (v-fit {d["v_at_vfit"]:5.1f})  '
                  f'D_T {d["DT"]:4.0f}±{d["DTe"]:3.0f} (v-fit gas {d["DT_at_vfit"]:4.0f}; 95/5 + {d["water_95"]:.2f} % water {d["DT_at_v95"]:4.0f})')
    json.dump(dict(grid_iso=isos, grid_water=ws, fields=Es, chambers=out), open(f'{HERE}/results/iso_fit.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
