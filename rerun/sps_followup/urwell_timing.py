#!/usr/bin/env python3
"""urwell_timing.py FEU1.root [N]: where in the RAW window the uRWELL pulses sit, per plane-view, both polarities:
sample-of-maximum histogram of strips above 100 ADC, and the mean waveform of the strongest strip of events whose
strongest strip exceeds 150 ADC.  Diagnostic for urwell_scan.py's in-time window."""
import sys
import numpy as np
import uproot
from urwell_run71 import urwell_map, grid_of, baseline, RAW_PRE

f = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
view, pos, det = urwell_map()
arr = uproot.open(f)['nt'].arrays(['eventId', 'sample', 'channel', 'amplitude'], entry_stop=N, library='np')
G = np.array([baseline(grid_of(arr['channel'][j].astype(int), arr['sample'][j].astype(int), arr['amplitude'][j].astype(np.float32)))
              for j in range(len(arr['eventId']))])
G = np.nan_to_num(G)
osc = (G[:, :, RAW_PRE:].max(2) > 60).mean(0) > 0.3
print('oscillating strips:', np.where(osc)[0].tolist())
for sgn, name in ((1, 'positive'), (-1, 'negative')):
    H = sgn * G
    H[:, osc, :] = 0
    for d in 'fb':
        for v in 'xy':
            m = (det == d) & (view == v)
            W = H[:, m, :]
            amax = W[:, :, RAW_PRE:].max(2); tm = W[:, :, RAW_PRE:].argmax(2) + RAW_PRE
            print(name, d, v, 'tmax of strips >100 ADC:', np.bincount(tm[amax > 100], minlength=64).tolist())
            i = amax.argmax(1); top = amax.max(1) > 150
            if top.sum():
                mw = W[np.arange(len(W))[top], i[top], :].mean(0)
                print('    n=%d mean lead waveform:' % top.sum(), np.round(mw).astype(int).tolist())
