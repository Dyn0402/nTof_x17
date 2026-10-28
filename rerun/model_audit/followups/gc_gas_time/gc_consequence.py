#!/usr/bin/env python3
"""gc_consequence.py -- GC: what the measured gas drift does to a reconstruction with ONE v (one angle scale) per run.

tan_reco = (w - w0) / (kw v_applied) = (v_true / v_applied) tan_true = (1 + eps) tan_true.  Per dataset:
  sigma_eps   event-weighted rms of v(t) about the run mean, block noise removed (the smear the constant v leaves);
  h_eps       largest |v_block / v_run - 1| (the worst bias at a run end);
  eps_group   bench rows that inherit their group's kw/v (registry refit = none): T_calib / T_row - 1.
Angle: d(theta) = atan((1 + eps) tan th) - th ~ eps sin th cos th, evaluated at 30 deg (24.8 deg per unit eps) and at
the 45 deg worst case (28.6 deg per unit eps).  Position at mid-gap (15 mm above the mesh): 15 mm x eps x tan th.  Depth
of a slice: z = v u, so dz = eps z (1 x eps x 30 mm at the cathode).  The mesh position p0 is v-free.
Thresholds: 0.2 deg at 30 deg (eps = 0.8 %) and 1 % tan scale.  Resolution reference: bench s68 1.06-1.76 deg (FD).
FA's ladder-vs-T slopes (angle scale moves 1.45x / 1.68x / 1.09x faster than 1/T on det4 / det6 / det7) give the
upper estimate.  Writes results/consequences.json and results/consequences.md.
"""
import json
import os

import numpy as np

from gc_common import NAME, RES, angle_bias_deg

LADDER = {'mx17_4': 1.45, 'mx17_6': 1.68, 'mx17_7': 1.09}
RESOL = {'mx17_2': 1.76, 'mx17_3': 1.06, 'mx17_4': 1.5, 'mx17_6': 1.5, 'mx17_7': 1.5}
EPS_THR = 0.008


def row(name, kind, sig, h, eps_grp=np.nan, slope=1.0, note=''):
    s = sig * slope; hh = h * slope
    d = dict(dataset=name, kind=kind, sigma_eps_pct=sig * 100, h_eps_pct=h * 100, eps_group_pct=eps_grp * 100,
             ladder_factor=slope,
             dtheta30_smear_deg=float(angle_bias_deg(s, 30.0)), dtheta30_worst_deg=float(angle_bias_deg(hh, 30.0)),
             dtheta45_worst_deg=float(angle_bias_deg(hh, 45.0)),
             dx_mid30_smear_mm=float(15.0 * s * np.tan(np.radians(30))), dz_cathode_worst_mm=float(30.0 * hh),
             crosses=bool(hh > EPS_THR or (np.isfinite(eps_grp) and abs(eps_grp) > EPS_THR)), note=note)
    if np.isfinite(eps_grp):
        d['dtheta30_group_deg'] = float(angle_bias_deg(eps_grp, 30.0))
    return d


def main():
    out = []
    bench = json.load(open(os.path.join(RES, 'bench_rows.json')))
    blocks = json.load(open(os.path.join(RES, 'bench_blocks.json')))
    for r in bench:
        if r.get('missing') or r['dur_h'] < 0.5:
            continue
        b = blocks[r['key']]
        v = np.array(b['v_rel']); n = np.array(b['n'])
        if len(v) < 2:
            continue
        vm = np.average(v, weights=n)
        e = np.array(b['err'])
        # worst block, its own noise taken off in quadrature (2 sigma: a maximum over many blocks)
        h = float(np.max(np.sqrt(np.maximum((v / vm - 1) ** 2 - (2 * e) ** 2, 0.0))))
        sig = r['vt_rms_int_pct'] / 100
        note = f"{r['dur_h']:.1f} h, gain range {r['q_range_pct']:.0f} %"
        out.append(row(f"{NAME[r['det']]} {r['run'].replace('mx17_', '')} {r['subrun']}", 'bench row', sig, h,
                       r.get('eps_group_pct', np.nan) / 100 if r['refit'] == 'none' else np.nan,
                       LADDER.get(r['det'], 1.0), note))
        out[-1]['det'] = r['det']; out[-1]['resolution_deg'] = RESOL[r['det']]; out[-1]['refit'] = r['refit']
    for r in json.load(open(os.path.join(RES, 'ladder_rows.json'))):
        if r['refit'] == 'none' and r['valid'] and np.isfinite(r['eps_lad_pct']):
            out.append(row(f"{r['det']} {r['run'].replace('mx17_', '')} {r['subrun']}", 'ladder', 0.0, 0.0,
                           r['eps_lad_pct'] / 100, note='inherits its group kw/v; offset from the M3 ladder'))
    kb = json.load(open(os.path.join(RES, 'ntof_kband.json')))
    noise = {'A': 0.0082, 'C': 0.0111}
    for r in kb['runs']:
        if r['n_sub'] < 3:
            continue
        v = np.array([x['v_rel'] for x in r['subs']])
        sig = float(np.sqrt(max(v.var() - noise[r['arm']] ** 2, 0.0)))
        # the worst sub-run, noise taken off in quadrature (a single sub-run carries the full estimator noise)
        h = float(np.sqrt(max(np.max(np.abs(v / v.mean() - 1)) ** 2 - (2 * noise[r['arm']]) ** 2, 0.0)))
        out.append(row(f"n_TOF {r['run']} arm {r['arm']}", 'n_TOF run (capsule-band k per sub-run)', sig, h,
                       note=f"{r['n_sub']} sub-runs, {(r['t_end'] - r['t_start']) / 3600:.1f} h"))
    p = os.path.join(RES, 'ntof_colsum_runs.json')
    if os.path.exists(p):
        for r in json.load(open(p)):
            out.append(row(f"n_TOF {r['run']} arm {r['arm']} (column end)", 'n_TOF run (column end)', r['sigma'], r['h'],
                           note=r.get('note', '')))
    p = os.path.join(RES, 'sps_runs.json')
    if os.path.exists(p):
        for r in json.load(open(p)):
            out.append(row(f"SPS {r['label']}", 'SPS det4', r['sigma'], r['h'], note=r.get('note', '')))
    json.dump(out, open(os.path.join(RES, 'consequences.json'), 'w'), indent=1)
    L = ['| dataset | sigma_eps % | worst % | eps vs group % | dtheta(30 deg) smear / worst deg | dx mid-gap mm | '
         'dz cathode mm | > 0.8 % |', '|---|---|---|---|---|---|---|---|']
    for d in out:
        L.append(f"| {d['dataset']} | {d['sigma_eps_pct']:.2f} | {d['h_eps_pct']:.2f} | "
                 f"{d['eps_group_pct']:.1f} | {d['dtheta30_smear_deg']:.2f} / {d['dtheta30_worst_deg']:.2f} | "
                 f"{d['dx_mid30_smear_mm']:.3f} | {d['dz_cathode_worst_mm']:.2f} | {'yes' if d['crosses'] else 'no'} |")
    open(os.path.join(RES, 'consequences.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
