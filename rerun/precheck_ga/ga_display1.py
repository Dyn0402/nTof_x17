#!/usr/bin/env python3
"""ga_display1.py <posmaps.json> <bench_root> <out_dir> -- F7 displays for the det4 GA channels the two-sided flank
selection could not show (94 400 435 439): events where a live +-1 neighbour is the cluster peak (> 300 ADC)."""
import glob, os, sys, json
import numpy as np
from ga_check import RUN_DIR, BIG, MAX_FILES, wave_peaks, neighbours, display

pm = json.load(open(sys.argv[1]))['g_det4']; root, out = sys.argv[2], sys.argv[3]
pos = np.array([np.nan if p is None else p for p in pm['pos']['6']])
L, R, o = neighbours(pos)
df = sorted(glob.glob(os.path.join(root, RUN_DIR['g_det4'], 'decoded_root_fa80c06', '*_datrun_*_06.root')))[:MAX_FILES]
P, W = wave_peaks(df, keep_wf=6000)
P = P[:len(W)]
for c in (94, 400, 435, 439):
    nb = [x for x in (L[c], R[c]) if x >= 0]
    sel = [i for i in range(len(W)) if any(P[i, x] > BIG and P[i, x] == P[i, [y for y in (L[x], x, R[x]) if y >= 0]].max() for x in nb)]
    L2 = [L[L[c]] if L[c] >= 0 else -1, L[c], c, R[c], R[R[c]] if R[c] >= 0 else -1]
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    oo = sorted([x for x in L2 if x >= 0], key=lambda x: pos[x])
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for k, i in enumerate(sel[:6]):
        a = ax.flat[k]
        for x in oo:
            a.plot(W[i][x], lw=2.2 if x == c else 1.0, color='k' if x == c else None, label=f'ch {x} ({pos[x] - pos[c]:+.2f} mm)')
        a.set_title(f'event {i}', fontsize=8)
        if k == 0: a.legend(fontsize=6)
    fig.suptitle(f'g_det4 x ch {c}: events with a live neighbour as cluster peak ({len(sel)} of {len(W)})', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'figures', f'ga_g_det4_x_{c}_onesided.png'), dpi=90); plt.close(fig)
    print(c, len(sel), flush=True)
