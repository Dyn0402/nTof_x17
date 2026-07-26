#!/usr/bin/env python3
"""Fit the forward-fit v(E) against the full Magboltz grid incl. water_grid2.

Outputs: v_vs_E_forward_refined.png (best curves + data), rms_vs_h2o.png
(RMS parabola -> best humidity per gap convention), printed table.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
     'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
GS = '/home/dylan/PycharmProjects/nTof_x17/garfield_sim/results'

ff = json.load(open(f'{B}/drift_scan_v.json'))
ff.update(json.load(open(f'{B}/drift_scan_v_lowhv.json')))
hv = sorted(int(k) for k in ff)
v = [ff[str(h)]['v'] for h in hv]
hv += [1000]; v += [json.load(open(f'{B}/hyper_v2.json'))['v']]
o = np.argsort(hv)
HV = np.array(hv)[o]; VM = np.array(v)[o]

mixes = {}
for f in glob.glob(os.path.join(GS, '*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for k, pts in d.get('mixtures', {}).items():
        if not pts or 'v_um_per_ns' not in pts[0]:
            continue
        E = np.array([q['E_Vcm'] for q in pts])
        V = np.array([q['v_um_per_ns'] for q in pts])
        oo = np.argsort(E)
        key = (k, os.path.basename(f))
        mixes[key] = (E[oo], V[oo])

def rms_vs(curve, gap_cm):
    E = HV / gap_cm
    Ec, Vc = curve
    vi = np.interp(E, Ec, Vc)
    m = HV >= 700
    return (np.sqrt(np.mean((vi - VM) ** 2)),
            np.sqrt(np.mean((vi[m] - VM[m]) ** 2)))

rows = []
for key, cur in mixes.items():
    r30 = rms_vs(cur, 3.0)
    r29 = rms_vs(cur, 2.9)
    rows.append((r30[0], r30[1], r29[0], r29[1], key))
rows.sort()
print(f'{"mixture":26s} {"file":26s}  RMS30_all RMS30_hi  RMS29_all RMS29_hi')
for r in rows[:15]:
    print(f'{r[4][0]:26s} {r[4][1]:26s}  {r[0]:7.2f} {r[1]:7.2f}   {r[2]:7.2f} {r[3]:7.2f}')

# ---- RMS vs water fraction at 95/5 (both gap conventions) ----
h2o_series = {}
for key, cur in mixes.items():
    k = key[0]
    if k.startswith('Ar_iso5_H2O') and 'N2' not in k and 'air' not in k:
        try:
            frac = float(k.replace('Ar_iso5_H2O', ''))
        except ValueError:
            continue
        h2o_series[frac] = cur
    if k == 'Ar95_iso5_H2O0.3':
        h2o_series[0.3] = cur
    if k == 'Ar94_iso5_H2O1':
        h2o_series[1.0] = cur

fig2, ax2 = plt.subplots(figsize=(7, 5))
best = {}
for gap, col in ((3.0, 'k'), (2.9, 'tab:red')):
    fr = sorted(h2o_series)
    rr_all = [rms_vs(h2o_series[f], gap)[0] for f in fr]
    rr_hi = [rms_vs(h2o_series[f], gap)[1] for f in fr]
    ax2.plot(fr, rr_all, 'o-', color=col, label=f'all points, gap {gap*10:.0f} mm')
    ax2.plot(fr, rr_hi, 's--', color=col, alpha=0.6,
             label=f'>=700 V only, gap {gap*10:.0f} mm')
    # parabola around the minimum of the hi series
    j = int(np.argmin(rr_hi))
    if 0 < j < len(fr) - 1:
        cf = np.polyfit(fr[j - 1:j + 2], rr_hi[j - 1:j + 2], 2)
        fbest = -cf[1] / (2 * cf[0])
    else:
        fbest = fr[j]
    best[gap] = fbest
ax2.set_xlabel('H2O fraction [%] (Ar/iso 95/5 base)')
ax2.set_ylabel('RMS(model - forward fit) [um/ns]')
ax2.set_title(f'best H2O: {best[3.0]:.2f}% (30mm) / {best[2.9]:.2f}% (29mm)')
ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(os.path.join(B, 'rms_vs_h2o.png'), dpi=110)
print('best H2O fraction: gap30 %.2f%%  gap29 %.2f%%' % (best[3.0], best[2.9]))

# ---- refined v(E) figure ----
fig, ax = plt.subplots(figsize=(9, 6))
E30 = HV / 3.0
ax.plot(E30, VM, 'ko-', ms=8, lw=2, label='forward fit (E = HV/3.0cm)', zorder=5)
shown = 0
for r in rows[:4]:
    key = r[4]
    E, V = mixes[key]
    ax.plot(E, V, '-', lw=1.5, alpha=0.85,
            label=f'{key[0]} (RMS {r[0]:.2f}/{r[1]:.2f})')
    shown += 1
for k, lab, ls in ((('Ar94_iso5_H2O1', 'attachment_Ar_iso_H2O.json'),
                    'Ar/iso 95/5 + 1% H2O', ':'),):
    if k in mixes:
        E, V = mixes[k]
        ax.plot(E, V, ls, lw=1.2, color='gray', label=lab)
ax.set_xlabel('drift field [V/cm]'); ax.set_ylabel('v [um/ns]')
ax.set_xlim(0, 420); ax.set_ylim(0, 45)
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
ax.set_title('det3 forward-fit v(E) vs refined Magboltz grid')
fig.tight_layout(); fig.savefig(os.path.join(B, 'v_vs_E_forward_refined.png'), dpi=110)
print('saved figures')
