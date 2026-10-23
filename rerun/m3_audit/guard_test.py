#!/usr/bin/env python3
"""guard_test.py <decoded.root> -- how often the masked-CM coherent-event guard falls back, and in which events."""
import os, sys
import numpy as np, uproot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import wft.io as wio

t = uproot.open(sys.argv[1])['nt']
A = t.arrays(['amplitude'], entry_stop=4000, library='np')['amplitude']
X = np.stack([a.reshape(-1, 512) for a in A if len(a) == 32 * 512]).astype(np.float32)
ped = np.median(X[:300], axis=(0, 1)); X -= ped
cm = np.median(X.reshape(len(X), 32, 8, 64), axis=3)
U = X - np.repeat(cm, 64, axis=2)
noise = 1.4826 * np.median(np.abs(U[:300]), axis=(0, 1))
nfb = nmask = 0; fb_cm = []; ok_cm = []; diffs_ok = []
for w, c in zip(X, cm):
    blk = w.reshape(32, 8, 64)
    plain = w - np.repeat(np.median(blk, axis=2), 64, axis=1)
    g = wio.CM_GUARD
    wio.CM_GUARD = 1e9; nog = wio.masked_block_median(w.copy(), noise, 5)
    wio.CM_GUARD = g; withg = wio.masked_block_median(w.copy(), noise, 5)
    for b in range(8):
        sl = slice(b * 64, b * 64 + 64)
        changed = np.any(np.abs(nog[:, sl] - plain[:, sl]) > 1e-3)
        if not changed:
            continue
        nmask += 1
        fell = np.allclose(withg[:, sl], plain[:, sl])
        peak = np.abs(c[:, b]).max()
        if fell:
            nfb += 1; fb_cm.append(peak)
        else:
            ok_cm.append(peak); diffs_ok.append(np.abs(withg[:, sl] - plain[:, sl]).max())
print(f'{os.path.basename(sys.argv[1])}: chips with a masked correction {nmask}, guard fell back {nfb} ({100*nfb/max(nmask,1):.1f} %)')
if fb_cm: print('  fallback chips: |plain chip median| max per event pctl 10/50/90:', np.percentile(fb_cm, [10, 50, 90]).round(0))
if ok_cm: print('  kept chips:     |plain chip median| pctl 10/50/90/99:', np.percentile(ok_cm, [10, 50, 90, 99]).round(0),
                '  |masked - plain| max pctl 50/90/99:', np.percentile(diffs_ok, [50, 90, 99]).round(0))
