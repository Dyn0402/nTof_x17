#!/usr/bin/env python3
"""urwell_scan.py --out O.npz FEU1.root [FEU1.root ...]

Telescope-first pass over run_71 FEU 1 files (both uRWELL planes): for EVERY event, per plane-view (front/back x/y)
the strongest 3-strip sum of the in-time strip maxima and the charge-weighted centroid of the 5 strips around it.
Baseline as in urwell_run71.py (per-channel pre-sample median, 64-channel block common mode per sample, pre median
again).  In-time = the maximum falls in samples T_LO..T_HI (the uRWELL pulse peaks at sample 12 of the RAW
window).  Oscillating strips (> 30 % of events above 5 sigma) are dropped.  Writes eventId, file group and the four
(position, 3-strip sum, number of separate 3-strip peaks above PEAK_THR) per event; the track selection is offline.
"""
import argparse, os, re
import numpy as np
from urwell_run71 import urwell_map, grid_of, baseline, RAW_PRE

T_LO, T_HI = 11, 14
PEAK_THR = 100.0


def main():
    import uproot
    ap = argparse.ArgumentParser(); ap.add_argument('files', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--max-events', type=int, default=0)
    a = ap.parse_args()
    view, pos, det = urwell_map()
    planes = []
    for d in 'fb':
        for v in 'xy':
            m = np.where((det == d) & (view == v))[0]
            planes.append((d + v, m[np.argsort(pos[m])]))
    out = {k: [] for k in ['eid', 'group']}
    for p, _ in planes:
        for s in ('_p', '_s3', '_npk'):
            out[p + s] = []
    for f in a.files:
        grp = re.search(r'_(\d{3})_01\.root$', f).group(1)
        tr = uproot.open(f)['nt']
        noise, bad = None, None
        nmax = tr.num_entries if not a.max_events else min(a.max_events, tr.num_entries)
        for lo in range(0, nmax, 2000):
            arr = tr.arrays(['eventId', 'sample', 'channel', 'amplitude'], entry_start=lo, entry_stop=min(lo + 2000, nmax), library='np')
            G = [baseline(grid_of(arr['channel'][j].astype(int), arr['sample'][j].astype(int), arr['amplitude'][j].astype(np.float32)))
                 for j in range(len(arr['eventId']))]
            if noise is None:            # strip noise + oscillating strips from this file's first block
                P = np.concatenate([g[:, :RAW_PRE] for g in G[:400]], axis=1)
                noise = 1.4826 * np.nanmedian(np.abs(P - np.nanmedian(P, axis=1, keepdims=True)), axis=1)
                with np.errstate(all='ignore'):
                    amx = np.array([np.nanmax(g[:, RAW_PRE:], axis=1) for g in G[:400]])
                bad = ~np.isfinite(noise) | (np.nanmean(amx > 5 * noise, axis=0) > 0.3)
            for j, g in enumerate(G):
                with np.errstate(all='ignore'):
                    w = np.nan_to_num(g[:, RAW_PRE:], nan=-1e9)
                    amp = w.max(1); t = w.argmax(1) + RAW_PRE
                a_in = np.where((t >= T_LO) & (t <= T_HI) & ~bad, np.clip(amp, 0, None), 0.0)
                out['eid'].append(int(arr['eventId'][j])); out['group'].append(int(grp))
                for p, idx in planes:
                    aa = a_in[idx]; pp = pos[idx]
                    s3 = aa[:-2] + aa[1:-1] + aa[2:]
                    i = int(np.argmax(s3))
                    k = np.clip(np.arange(i - 1, i + 4), 0, len(pp) - 1)
                    out[p + '_p'].append(float((aa[k] * pp[k]).sum() / max(aa[k].sum(), 1e-9)))
                    out[p + '_s3'].append(float(s3[i]))
                    # separate peaks: local maxima of s3 above threshold, at least 4 strips apart
                    pk = np.where((s3 > PEAK_THR) & (s3 >= np.r_[0, s3[:-1]]) & (s3 >= np.r_[s3[1:], 0]))[0]
                    out[p + '_npk'].append(int(len(pk) - (np.diff(pk) < 4).sum()) if len(pk) else 0)
        print(f'{os.path.basename(f)}: {nmax} events, {int(bad.sum())} oscillating strips', flush=True)
    np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()})
    print('wrote', a.out)


if __name__ == '__main__':
    main()
