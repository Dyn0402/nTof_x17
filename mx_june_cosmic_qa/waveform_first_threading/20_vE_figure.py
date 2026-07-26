#!/usr/bin/env python3
"""v(E) capstone figure: forward-fit drift velocities vs prior estimators and
Magboltz mixture curves; RMS match per mixture."""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
GS = '/home/dylan/PycharmProjects/nTof_x17/garfield_sim/results'
DV = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
      'drift_velocity/mx17_3')
GAP_CM = 3.0

ff = json.load(open(os.path.join(BASE, 'drift_scan_v.json')))
ff_low = json.load(open(os.path.join(BASE, 'drift_scan_v_lowhv.json')))
ff.update(ff_low)              # corrected 300/500 points
hv_ff = sorted(int(k) for k in ff)
v_ff = [ff[str(h)]['v'] for h in hv_ff]
# long-run point from hyper_v2
v1000 = json.load(open(os.path.join(BASE, 'hyper_v2.json')))['v']
hv_all = hv_ff + [1000]
v_all = v_ff + [v1000]
o = np.argsort(hv_all)
hv_all = np.array(hv_all)[o]; v_all = np.array(v_all)[o]

slr = pd.read_csv(os.path.join(DV, 'slope_reference_vdrift_scan.csv')).set_index('drift_hv')
hyb = pd.read_csv(os.path.join(DV, 'hybrid_vdrift_scan.csv'))
hyb['hv'] = hyb['drift_hv']; hyb = hyb.set_index('hv')

def load_mix(fn_key, mix_key):
    for fn in glob.glob(os.path.join(GS, '*.json')):
        if fn_key not in os.path.basename(fn):
            continue
        d = json.load(open(fn))
        pts = d.get('mixtures', {}).get(mix_key)
        if pts:
            E = np.array([q['E_Vcm'] for q in pts])
            V = np.array([q['v_um_per_ns'] for q in pts])
            oo = np.argsort(E)
            return E[oo], V[oo]
    return None

CURVES = [
    ('Ar/iso 95/5 dry (Magboltz)', 'attachment_Ar_iso_H2O.json', 'Ar95_iso5', 'tab:blue'),
    ('Ar/iso 95/5 + 1% H2O', 'drift_velocity_candidates.json', 'Ar94_iso5_H2O1', 'tab:green'),
    ('Ar/iso 95/5 + 0.3% H2O', 'drift_velocity_candidates.json', 'Ar95_iso5_H2O0.3', 'tab:cyan'),
    ('Ar/iso 90/10', 'drift_velocity_candidates2.json', 'Ar90_iso10', 'tab:orange'),
]

fig, ax = plt.subplots(figsize=(9, 6))
E_ff = hv_all / GAP_CM
ax.plot(E_ff, v_all, 'ko-', ms=8, lw=2, label='forward fit (this work)', zorder=5)
E_u, v_u, e_u = [], [], []
for hv in slr.index:
    if np.isfinite(slr.loc[hv, 'v_unshared']):
        E_u.append(hv / GAP_CM); v_u.append(slr.loc[hv, 'v_unshared'])
        e_u.append(slr.loc[hv, 'v_unshared_err'])
ax.errorbar(E_u, v_u, yerr=e_u, fmt='s', color='tab:purple', ms=6,
            label='unshared ladder (46-series)', zorder=4)
E_g = [h / GAP_CM for h in hyb.index if hyb.loc[h, 'v_geom'] > 5]
v_g = [hyb.loc[h, 'v_geom'] for h in hyb.index if hyb.loc[h, 'v_geom'] > 5]
ax.plot(E_g, v_g, 'd', color='tab:brown', ms=6, label='geometry estimator (hybrid)')

print('RMS of forward-fit series vs mixtures (all points / >=700V only):')
for lab, fk, mk, col in CURVES:
    cur = load_mix(fk, mk)
    if cur is None:
        print(' missing', lab)
        continue
    E, V = cur
    ax.plot(E, V, '-', color=col, lw=1.5, alpha=0.8, label=lab)
    vi = np.interp(E_ff, E, V)
    rms_all = np.sqrt(np.mean((vi - v_all) ** 2))
    m = hv_all >= 700
    rms_hi = np.sqrt(np.mean((vi[m] - v_all[m]) ** 2))
    print(f'  {lab:34s}: RMS {rms_all:5.2f} / {rms_hi:5.2f} um/ns')
ax.set_xlabel('drift field [V/cm]  (HV / 3.0 cm)')
ax.set_ylabel('drift velocity [um/ns]')
ax.set_xlim(0, 400); ax.set_ylim(0, 45)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower right')
ax.set_title('det3 drift velocity vs field: forward fit vs priors vs Magboltz')
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'v_vs_E_forward.png'), dpi=110)
print('saved v_vs_E_forward.png')
print('forward series:', dict(zip(hv_all.tolist(), np.round(v_all, 1).tolist())))
