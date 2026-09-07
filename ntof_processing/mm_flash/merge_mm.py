#!/usr/bin/env python3
"""Merge the per-file MM extractions into one npz per run, joined to the
n_TOF index tree (PulseIntensity, PSpulse, wall-clock timestamp)."""
import glob
import os
import re

import numpy as np

SRC = '/tmp/mmx'
OUT = '/tmp/mmx_merged'
CHANNELS = ('MMA', 'MMB', 'PKUP', 'WALL')
SCALARS = ('base', 'peak', 'peak_t', 'integral', 'n_pos_rail', 'n_neg_rail',
           'recov', 'nsamp')


def seq(p):
    m = re.search(r'_(\d+)_s1\.npz$', p)
    return int(m.group(1)) if m else -1


def stamps(run):
    a = np.load(f'/tmp/index_{run}.npz')
    d = a['Date'].astype(np.int64)
    t = a['Time'].astype(np.int64)
    yy = 2000 + (d // 10000) % 100
    mm = (d // 100) % 100
    dd = d % 100
    hh, mi, ss = t // 10000, (t // 100) % 100, t % 100
    # seconds since 2026-07-01 00:00:00, good enough as a monotone clock
    doy = (mm - 7) * 31 + (dd - 1)
    secs = doy * 86400 + hh * 3600 + mi * 60 + ss
    return a['BunchNumber'].astype(np.int64), a['PulseIntensity'], a['PSpulse'], secs


def main():
    os.makedirs(OUT, exist_ok=True)
    for run in (224302, 224325, 224327):
        files = sorted(glob.glob(f'{SRC}/run{run}_*_s1.npz'), key=seq)
        bunches, per = [], {f'{c}_{k}': [] for c in CHANNELS for k in SCALARS}
        traces = {c: [] for c in CHANNELS}
        tracebunch = {c: [] for c in CHANNELS}
        zs = {c: [] for c in CHANNELS}
        for p in files:
            d = np.load(p)
            b = d['bunch']
            bunches.append(b)
            for c in CHANNELS:
                if f'{c}_ev' not in d.files:
                    continue
                ev = d[f'{c}_ev'].astype(int)
                cb = b[ev]
                for k in SCALARS:
                    per[f'{c}_{k}'].append(np.stack([cb, d[f'{c}_{k}']]))
                traces[c].append(d[f'{c}_trace'])
                tracebunch[c].append(cb)
                if f'{c}_zs' in d.files:
                    z = d[f'{c}_zs']
                    if len(z):
                        z = z.copy()
                        z[:, 0] = b[z[:, 0].astype(int)]      # ev index -> bunch
                        zs[c].append(z)
        out = {'bunch': np.concatenate(bunches)}
        for key, chunks in per.items():
            if chunks:
                m = np.concatenate(chunks, axis=1)
                out[key + '_bunch'] = m[0].astype(np.int64)
                out[key] = m[1]
        for c in CHANNELS:
            if traces[c]:
                out[f'{c}_trace'] = np.concatenate(traces[c])
                out[f'{c}_trace_bunch'] = np.concatenate(tracebunch[c]).astype(np.int64)
            if zs[c]:
                out[f'{c}_zs'] = np.concatenate(zs[c])
        ib, pi, ps, secs = stamps(run)
        out['idx_bunch'] = ib
        out['idx_intensity'] = pi
        out['idx_pspulse'] = ps
        out['idx_secs'] = secs
        np.savez_compressed(f'{OUT}/mm_{run}.npz', **out)
        nb = len(out['bunch'])
        matched = np.isin(out['bunch'], ib).sum()
        print(f'{run}: {len(files)} files, {nb} raw events, index has {len(ib)}, '
              f'raw bunches found in index: {matched} ({100*matched/nb:.1f} %)')
        for c in CHANNELS:
            k = f'{c}_zs'
            print(f'   {c}: traces {out.get(f"{c}_trace", np.empty(0)).shape} '
                  f'zs blocks {0 if k not in out else len(out[k])}')


if __name__ == '__main__':
    main()
