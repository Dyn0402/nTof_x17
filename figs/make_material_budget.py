#!/usr/bin/env python3
"""make_material_budget.py -- Paper I v2: material from the target to the end of the drift gap (table).

    .venv/bin/python papers/paper1_v2/figs/make_material_budget.py [--out papers/paper1_v2/figs/out/material_budget]

Writes <out>.json and <out>.tex (a tabular with \\hline, ready to paste). Layers and thicknesses from
docs/detector_spec/spec.py and the chamber-figure caption of Paper I (section 2); X0 and densities PDG (mcs_common).
The window aluminisation (thickness unknown, ~tens of nm) is ignored. The 9 um cathode copper is "assumed" in the
spec: flagged with a dagger, and the totals are also given without it.

theta0: Highland (PDG), plane-projected rms, with the momentum (not the kinetic energy) of an electron of 4, 8 and
16 MeV kinetic energy. Per-layer values are for that layer alone; cumulative rows apply Highland to the summed
x/X0 (the log term makes theta0 non-additive in quadrature, by a few %). Check against the introduction: 200 mm of
air gives 1.70 deg at 8 MeV and 3.24 deg at 4 MeV (text: 1.7 and 3.2).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import mcs_common as mc  # noqa: E402

TS = (4, 8, 16)


def th(x, T):
    return float(np.degrees(mc.highland(x, T)))


def fmt_t(mm):
    return f'{mm:g}\\,mm' if mm >= 1 else f'{mm * 1000:g}\\,\\textmu m'


def fmt_x0(cm):
    return f'{cm / 100:.0f}\\,m' if cm >= 100 else f'{cm:.2f}\\,cm'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'material_budget'))
    a = ap.parse_args()
    layers = list(mc.LAYERS) + [mc.DRIFT]
    rows = []
    for key, label, m, t, z0, asm in layers:
        x = mc.x_over_X0(m, t)
        rows.append(dict(key=key, layer=label, material=m, thickness_mm=t, X0_cm=round(mc.X0_cm(m), 3),
                         x_over_X0_pct=round(100 * x, 5), assumed=asm,
                         theta0_deg={f'{T}MeV': round(th(x, T), 3) for T in TS}))
    xs = {r['key']: r['x_over_X0_pct'] / 100 for r in rows}
    front = ['air', 'window', 'gap5', 'kapton', 'cu']
    cum = {}
    for name, keys in (('front', front), ('front_noCu', [k for k in front if k != 'cu']),
                       ('total', front + ['drift']), ('total_noCu', [k for k in front if k != 'cu'] + ['drift'])):
        x = sum(xs[k] for k in keys)
        cum[name] = dict(layers=keys, x_over_X0_pct=round(100 * x, 5),
                         theta0_deg={f'{T}MeV': round(th(x, T), 3) for T in TS})
    check = dict(air_200mm={f'{T}MeV': round(th(mc.x_over_X0('air', 200.0), T), 3) for T in (4, 8)},
                 air_204mm={f'{T}MeV': round(th(mc.x_over_X0('air', 204.0), T), 3) for T in (4, 8)},
                 introduction_text={'4MeV': 3.2, '8MeV': 1.7})
    out = dict(rows=rows, cumulative=cum, check_introduction=check,
               momenta_MeV={f'{T}MeV': round(float(mc.kin(T)[0]), 4) for T in TS},
               gas=dict(X0_g_cm2=round(mc.MAT['gas']['X0'], 3), rho_g_cm3=round(mc.MAT['gas']['rho'], 6)),
               notes='Highland per projection with momentum; Mylar aluminisation ignored; Cu 9 um assumed (spec).')
    with open(a.out + '.json', 'w') as f:
        json.dump(out, f, indent=1)

    names = {'air': 'Air (target to window)', 'window': 'Entrance window, Mylar$^\\ast$',
             'gap5': 'Gas gap, window to cathode', 'kapton': 'Drift cathode, Kapton',
             'cu': 'Drift cathode, copper$^\\dagger$', 'drift': 'Drift gas (measured gap)'}
    mats = {'air': 'air', 'mylar': 'Mylar', 'gas': 'Ar/iC$_4$H$_{10}$ 95/5', 'kapton': 'Kapton', 'Cu': 'Cu'}
    L = ['\\begin{tabular}{llrrrrrr}', '\\hline',
         ' & & & & & \\multicolumn{3}{c}{$\\theta_0$ (\\textdegree), $e^\\pm$ of} \\\\',
         'Layer & Material & Thickness & $X_0$ & $x/X_0$ (\\%) & 4\\,MeV & 8\\,MeV & 16\\,MeV \\\\', '\\hline']
    for r in rows:
        if r['key'] == 'drift':
            c = cum['front']
            cn = cum['front_noCu']
            L.append('\\hline')
            L.append('In front of the drift volume & & & & ' + f"{c['x_over_X0_pct']:.3f} & " +
                     ' & '.join(f"{c['theta0_deg'][f'{T}MeV']:.2f}" for T in TS) + ' \\\\')
            L.append('\\quad without the copper$^\\dagger$ & & & & ' + f"{cn['x_over_X0_pct']:.3f} & " +
                     ' & '.join(f"{cn['theta0_deg'][f'{T}MeV']:.2f}" for T in TS) + ' \\\\')
            L.append('\\hline')
        L.append(f"{names[r['key']]} & {mats[r['material']]} & {fmt_t(r['thickness_mm'])} & "
                 f"{fmt_x0(r['X0_cm'])} & {r['x_over_X0_pct']:.3f} & " +
                 ' & '.join(f"{r['theta0_deg'][f'{T}MeV']:.2f}" for T in TS) + ' \\\\')
    c, cn = cum['total'], cum['total_noCu']
    L.append('\\hline')
    L.append('Including the drift gas & & & & ' + f"{c['x_over_X0_pct']:.3f} & " +
             ' & '.join(f"{c['theta0_deg'][f'{T}MeV']:.2f}" for T in TS) + ' \\\\')
    L.append('\\quad without the copper$^\\dagger$ & & & & ' + f"{cn['x_over_X0_pct']:.3f} & " +
             ' & '.join(f"{cn['theta0_deg'][f'{T}MeV']:.2f}" for T in TS) + ' \\\\')
    L.append('\\hline')
    L.append('\\end{tabular}')
    L.append('% $^\\ast$ aluminisation (thickness not known, tens of nm) ignored.  $^\\dagger$ thickness assumed '
             '(not in the CAD).')
    L.append('% theta0: Highland, per projection, momentum of the electron; cumulative rows apply Highland to the '
             'summed x/X0. Gases at 1 atm, 20 C. Generated by make_material_budget.py')
    with open(a.out + '.tex', 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(json.dumps(check, indent=1))


if __name__ == '__main__':
    main()
