#!/usr/bin/env python3
"""Consolidated drift-window figure: v(E) for clean/contaminated/measured gas,
and the sample budget at 600 V for det A (nTOF run_54)."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

GAR='../garfield_sim/results'
import os
if os.path.exists(f'{GAR}/drift_velocity_Ar_iC4H10_90_10_CERN.json'):
    clean=json.load(open(f'{GAR}/drift_velocity_Ar_iC4H10_90_10_CERN.json'))
    ce=np.array([p['E_Vcm'] for p in clean['points']]); cv=np.array([p['v_um_per_ns'] for p in clean['points']])
    clean_lbl='clean 90/10 Magboltz (CERN P)'
else:  # fallback: candidates2 has Ar90_iso10 at Saclay P (~3% slower, same shape)
    c2=json.load(open(f'{GAR}/drift_velocity_candidates2.json'))
    pts=c2['mixtures']['Ar90_iso10']
    ce=np.array([p['E_Vcm'] for p in pts]); cv=np.array([p['v_um_per_ns'] for p in pts])
    clean_lbl='clean 90/10 Magboltz (Saclay P; CERN ~3% faster)'
c95=json.load(open(f'{GAR}/drift_velocity_Ar_iC4H10_95_5_Saclay.json'))
e95=np.array([p['E_Vcm'] for p in c95['points']]); v95=np.array([p['v_um_per_ns'] for p in c95['points']])
# June Saclay measured (30 mm gap, 95/5 + ~1% H2O + ~1% air)
jE=np.array([167,233,300,333,367]); jv=np.array([12.4,21.6,30.0,33.9,35.1])
# nTOF run_54 measured det A @600V (E=200), v from full-gap span p95..p99 (840..960 ns)
nE=200; nv=30000/np.array([960,900,840]); nv_c=30000/900.

fig,ax=plt.subplots(1,2,figsize=(14,5.2))
a=ax[0]
a.plot(ce,cv,'-',color='C0',lw=2,label=clean_lbl)
m=(e95>120)&(e95<380)
a.plot(e95[m],v95[m],'--',color='C2',lw=1.5,label='clean 95/5 Magboltz (Saclay P)')
a.plot(jE,jv,'s-',color='C3',lw=1.5,ms=7,label='June Saclay MEASURED 95/5 +~1%H2O (30mm)')
a.errorbar([nE],[nv_c],yerr=[[nv_c-nv.min()],[nv.max()-nv_c]],fmt='*',color='k',ms=18,
           capsize=5,label='nTOF run_54 MEASURED det A @600V',zorder=5)
for V,E in [(600,200),(700,233),(800,267)]:
    a.axvline(E,color='gray',ls=':',lw=0.8); a.text(E,4,f'{V}V',rotation=90,va='bottom',fontsize=8,color='gray')
a.set_xlabel('drift field E [V/cm]  (= HV / 3 cm for 30 mm gap)'); a.set_ylabel('drift velocity [µm/ns]')
a.set_xlim(120,380); a.set_ylim(0,50); a.legend(fontsize=8,loc='lower right')
a.set_title('Drift velocity: theory vs measured\nwater halves v at low field; nTOF gas is fairly DRY')

b=ax[1]
# det A @600V occupancy (run_54): prompt sample ~11, deep edge percentiles
smp=np.arange(32)
occ=np.zeros(32)  # schematic occupancy from measured percentiles
occ[9:28]=1
b.bar(smp,[0.15]*32,color='#eee',edgecolor='#ccc',width=1.0)
b.axvspan(0,9,color='C7',alpha=0.25,label='empty pre-prompt baseline (~8 smp)\nreclaim by lowering latency')
b.axvspan(9,11,color='C0',alpha=0.4,label='prompt / near-mesh (sample≈latency−24=11)')
b.axvspan(11,27,color='C2',alpha=0.35,label='drift column 600V (prompt→deep-edge p99≈27)')
b.axvspan(27,32,color='C3',alpha=0.2,label='tail margin (p99..ceiling; <1% truncated)')
b.axvline(31,color='r',ls=':',label='window ceiling (32 smp)')
b.set_xlim(0,32); b.set_ylim(0,0.2); b.set_yticks([])
b.set_xlabel('sample index (× 60 ns)')
b.set_title('det A @600V sample budget (run_54 cosmics)\nfull-gap drift ≈ 16 smp; window 32 → trim via latency')
b.legend(fontsize=7.5,loc='upper right')
# top axis in ns
bt=b.secondary_xaxis('top',functions=(lambda s:s*60,lambda t:t/60)); bt.set_xlabel('time [ns]')
fig.tight_layout(); fig.savefig('drift_window_summary.png',dpi=120)
print('saved drift_window_summary.png')
