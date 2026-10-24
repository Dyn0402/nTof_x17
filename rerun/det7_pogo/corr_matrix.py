"""Amplitude correlation matrix of det7 X (FEU 6) over events, same definition as
rerun/seedfit/channel_census.py (cosine of amplitude products, events with 2-40 hits,
significance > 5, unsaturated). Hits are used for QA only (which channels move together),
no geometry is reconstructed from them. Writes corr_feu6.npz."""
import glob, os, numpy as np, uproot
RUN = '/home/dylan/x17/cosmic_bench/det6_det7/mx17_det6_det7_overnight_6-26-26'
SUBS = ['long_run', 'longer_run']
S = np.zeros((512, 512)); N = np.zeros(512); nev = 0
for s in SUBS:
    for f in sorted(glob.glob(f'{RUN}/{s}/hits_root/*_06_hits.root')):
        a = uproot.open(f)['hits'].arrays(['eventId', 'channel', 'amplitude', 'significance', 'saturated'], library='np')
        o = np.argsort(a['eventId'], kind='stable'); a = {k: v[o] for k, v in a.items()}
        e, c, q = a['eventId'], a['channel'].astype(int), a['amplitude'].astype(float)
        good = (a['significance'] > 5) & ~a['saturated']
        b = np.flatnonzero(np.diff(e)) + 1
        for s0, s1 in zip(np.r_[0, b], np.r_[b, len(e)]):
            g = good[s0:s1]; cc, qq = c[s0:s1][g], q[s0:s1][g]
            if 2 <= len(cc) <= 40:
                S[np.ix_(cc, cc)] += np.outer(qq, qq); N[cc] += qq ** 2; nev += 1
C = S / np.sqrt(np.maximum(np.outer(N, N), 1e-9)); np.fill_diagonal(C, 0)
np.savez_compressed('corr_feu6.npz', C=C.astype(np.float32), nev=nev)
print('events', nev)
