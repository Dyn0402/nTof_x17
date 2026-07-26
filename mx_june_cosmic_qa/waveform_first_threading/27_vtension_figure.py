#!/usr/bin/env python3
"""Four-panel v-tension resolution figure."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
     'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
prof = json.load(open(os.path.join(B, 'profile_v.json')))
gap = json.load(open(os.path.join(B, 'gap_vs_hv.json')))

fig, axs = plt.subplots(2, 2, figsize=(13, 9))

# (a) profile likelihood valley
v = np.array(sorted(float(k) for k in prof))
c = np.array([prof[str(k) if str(k) in prof else repr(k)]['chi2']
              for k in v])
ax = axs[0, 0]
ax.plot(v, (c - c.min()) / 1e5, 'ko-')
ax.axvline(36.7, color='C0', ls='-', lw=1, label='forward fit 36.7')
ax.axvline(34.3, color='tab:orange', ls='--', lw=1, label='geometry est. 34.3')
ax.set_xlabel('v [um/ns]')
ax.set_ylabel(r'$\Delta\chi^2$ / $10^5$  (sharing re-fit at each v)')
ax.set_title('(a) profile likelihood: sharp minimum at 36.7')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (b) implied gap vs HV
ax = axs[0, 1]
hvv = sorted(int(k) for k in gap)
g = [gap[str(h)]['gap_mm'] for h in hvv]
ge = [gap[str(h)]['v'] * gap[str(h)]['dU'] * 1e-3 for h in hvv]
ax.errorbar(hvv, g, yerr=ge, fmt='ko-', capsize=4)
ax.axhline(29, color='r', ls='--', label='assumed working gap 29 mm')
ax.axhline(30, color='gray', ls=':', label='mechanical 30 mm')
ax.axhspan(24.2, 25.2, color='C0', alpha=0.15, label='measured 24.7 mm')
ax.set_xlabel('drift HV [V]'); ax.set_ylabel('v_fit x U50 [mm]')
ax.set_title('(b) charge-visible column: 24.7 mm, constant vs HV')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(15, 33)

# (c) toy calibration-bias test
ax = axs[1, 0]
ax.bar(['truth\n(v=34, alt kernel)', 'calibrated\non toys', 'calibrated\non data'],
       [34.0, 33.11, 36.60], color=['gray', 'C2', 'C0'])
ax.axhline(34, color='gray', ls='--', lw=1)
for i, (val, lab) in enumerate([(34.0, '34.0'), (33.11, '33.1'), (36.60, '36.6')]):
    ax.text(i, val + 0.3, lab, ha='center')
ax.set_ylim(30, 38.5)
ax.set_ylabel('v [um/ns]')
ax.set_title('(c) toy bias test: mismatch deflates v (-0.9), cannot inflate')

# (d) estimator family
ax = axs[1, 1]
names = ['geometry\n(21/23)', 'unshared\nladder', 'extent 24.5mm\n/ U50 674ns',
         'forward fit\n(this work)']
vals = [34.3, 34.7, 36.2, 36.7]
errs = [0.6, 0.7, 1.5, 0.95]
cols = ['tab:orange', 'tab:purple', 'tab:green', 'C0']
ax.bar(names, vals, yerr=errs, color=cols, capsize=4)
ax.axhspan(36.7 - 0.95, 36.7 + 0.95, color='C0', alpha=0.12)
ax.set_ylim(31, 39)
ax.set_ylabel('v(1000 V) [um/ns]')
ax.set_title('(d) estimator family: tension = extent/floor systematics')
fig.suptitle('det3 drift-velocity tension resolution (2026-07-26)', y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(B, 'vtension_resolution.png'), dpi=110)
print('saved vtension_resolution.png')
