#!/usr/bin/env python3
"""Merge the per-file 224709 extractions into one npz (run on lxplus)."""
import glob
import re

import numpy as np

SRC = '/tmp/mm709'
OUT = '/tmp/mm_224709.npz'


def seq(p):
    m = re.search(r'_(\d+)_s1\.npz$', p)
    return int(m.group(1)) if m else -1


def main():
    files = sorted(glob.glob(f'{SRC}/run224709_*_s1.npz'), key=seq)
    bunch, wall, flash, stats, zs, pkup = [], [], [], [], [], []
    base_ev = 0
    for p in files:
        d = np.load(p)
        n = len(d['bunch'])
        if n == 0:
            continue
        bunch.append(d['bunch'])
        wall.append(d['wall'])
        flash.append(d['flash'])
        s = d['stats']
        if len(s):
            s = s.copy()
            s[:, 0] += base_ev
            stats.append(s)
        z = d['zs']
        if len(z):
            z = z.copy()
            z[:, 0] += base_ev
            zs.append(z)
        k = d['pkup']
        if len(k):
            k = k.copy()
            k[:, 0] += base_ev
            pkup.append(k)
        base_ev += n
    out = dict(bunch=np.concatenate(bunch), wall=np.concatenate(wall),
               flash=np.concatenate(flash), stats=np.concatenate(stats),
               zs=np.concatenate(zs), pkup=np.concatenate(pkup))
    np.savez(OUT, **out)          # uncompressed: the traces dominate and are noisy
    print(f'{len(files)} files -> {OUT}')
    for k, v in out.items():
        print(f'   {k}: {v.shape}')


if __name__ == '__main__':
    main()
