#!/usr/bin/env python3
"""inspect_lost.py -- for events the masked-CM hits lose, show what sits at the masked-CM cluster in the waveforms."""
import glob, json, os, sys
import numpy as np, uproot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge as J
sys.path.insert(0, J.REPO)
from wft.io import masked_block_median

sub, tag, feu = sys.argv[1], sys.argv[2], int(sys.argv[3])
hits = {g: uproot.open(glob.glob(os.path.join(sub, f'combined_hits_root_rr1_{g}', f'*_{tag}_feu-combined_hits.root'))[0])['hits']
        .arrays(['eventId', 'feu', 'channel', 'amplitude'], library='np') for g in ('cm0', 'cm5')}
dec = glob.glob(os.path.join(sub, 'decoded_root_fa80c06', f'*_datrun_{tag}_{feu:02d}.root'))[0]
t = uproot.open(dec)['nt']
A = t.arrays(['eventId', 'amplitude'], entry_stop=3000, library='np')
X = np.stack([a.reshape(-1, 512) for a in A['amplitude'] if len(a) == 32 * 512]).astype(np.float32)
eids = np.array([e for e, a in zip(A['eventId'], A['amplitude']) if len(a) == 32 * 512])
ped = np.median(X[:300], axis=(0, 1))
X -= ped
cm = np.median(X.reshape(len(X), 32, 8, 64), axis=3)
U = X - np.repeat(cm, 64, axis=2)
noise = 1.4826 * np.median(np.abs(U[:300]), axis=(0, 1))
# events with many more hits on this FEU under cm5
n0 = {}; n5 = {}
for g, d in (('cm0', n0), ('cm5', n5)):
    h = hits[g]; m = h['feu'] == feu
    e, c = np.unique(h['eventId'][m], return_counts=True)
    d.update(zip(e.tolist(), c.tolist()))
cand = sorted(((n5.get(e, 0) - n0.get(e, 0), e) for e in eids.tolist()), reverse=True)[:5]
for dn, e in cand:
    i = int(np.where(eids == e)[0][0])
    w = X[i]
    plain = U[i]
    mk = masked_block_median(w.copy(), noise, 5)
    h5 = hits['cm5']; m5 = (h5['eventId'] == e) & (h5['feu'] == feu)
    h0 = hits['cm0']; m0 = (h0['eventId'] == e) & (h0['feu'] == feu)
    new = np.setdiff1d(h5['channel'][m5], h0['channel'][m0])
    print(f'event {e}: hits cm0 {m0.sum()} cm5 {m5.sum()} (+{dn}); new-hit channels {new[:40]}')
    blk = sorted(set((new // 64).tolist()))
    for b in blk[:3]:
        sl = slice(b * 64, b * 64 + 64)
        sigch = np.flatnonzero((plain[:, sl] > 5 * noise[sl]).sum(0) >= 2) + b * 64
        print(f'   block {b}: plain-CM signal channels {sigch[:30]} ({len(sigch)})')
        print(f'      block CM (plain) per sample: {np.round(np.median(w[:, sl], axis=1)[::3]).astype(int)}')
        print(f'      masked - plain baseline per sample: {np.round((plain - mk)[::3, b * 64]).astype(int)}')
        c = new[(new // 64) == b][:3]
        for ch in c:
            print(f'      ch {ch}: plain {np.round(plain[::3, ch]).astype(int)}')
            print(f'      ch {ch}: masked {np.round(mk[::3, ch]).astype(int)}  noise {noise[ch]:.1f}')
