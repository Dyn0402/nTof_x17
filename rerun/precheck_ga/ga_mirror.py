#!/usr/bin/env python3
"""ga_mirror.py <posmaps.json> <bench_root> <out.json> -- F7 one-sided test for GA channels the flank test cannot judge.

For GA channel c and each live position neighbour n (+-1 pitch), on events where n is the cluster peak (A_n > 300 and
A_n >= both its neighbours): mirror ratio = sum A_c / sum A_m, m = the strip on n's other side.  Tracks are symmetric
about n on average, so a healthy c reads ~1.  Reference: the same ratio for every channel of the view, local median
(+-32 strips).  Hits (rr1_cm0) and waveform peaks (fa80c06).  q = ratio / local median: weak if q < 0.6 in both.
"""
import json
import sys

import numpy as np

from ga_check import GA, RUN_DIR, BIG, MAX_FILES, PITCH, hits_matrix, wave_peaks, local_median
import glob
import os


def mirror(A, pos):
    """per channel and side (-1, +1): mirror ratio and the number of events."""
    ok = np.flatnonzero(np.isfinite(pos)); o = ok[np.argsort(pos[ok])]; ps = pos[o]
    idx = {int(c): i for i, c in enumerate(o)}
    r = {s: np.full(512, np.nan) for s in (-1, 1)}; n = {s: np.zeros(512, int) for s in (-1, 1)}
    for c in o:
        i = idx[int(c)]
        for s in (-1, 1):
            j, k = i + s, i + 2 * s
            if not (0 <= k < len(o)) or abs(abs(ps[k] - ps[i]) - 2 * PITCH) > 0.3 * PITCH:
                continue
            nb, m = o[j], o[k]
            sel = (A[:, nb] > BIG) & (A[:, nb] >= A[:, m]) & (A[:, nb] >= A[:, c])
            n[s][c] = sel.sum()
            if n[s][c] >= 20 and A[sel, m].sum() > 0:
                r[s][c] = A[sel, c].sum() / A[sel, m].sum()
    return r, n, o


def main():
    pm = json.load(open(sys.argv[1])); root = sys.argv[2]
    res = {}
    for key, views in GA.items():
        info = pm[key]; sd = os.path.join(root, RUN_DIR[key])
        for view, lists in views.items():
            feu = info['feu_x'] if view == 'x' else info['feu_y']
            pos = np.array([np.nan if p is None else p for p in info['pos'][str(feu)]])
            hf = sorted(glob.glob(os.path.join(sd, 'hits_root_rr1_cm0', f'*_datrun_*_{feu:02d}_hits.root')))[:MAX_FILES]
            df = sorted(glob.glob(os.path.join(sd, 'decoded_root_fa80c06', f'*_datrun_*_{feu:02d}.root')))[:MAX_FILES]
            mats = {'hits': hits_matrix(hf), 'wf': wave_peaks(df, keep_wf=0)[0]}
            for src, A in mats.items():
                r, n, o = mirror(A, pos)
                ref = {s: local_median(r[s], o) for s in r}
                for kind in ('dead', 'hot'):
                    for c in lists[kind]:
                        e = res.setdefault(f'{key}:{view}:{c}', dict(key=key, view=view, ch=c, ga=kind))
                        for s in (-1, 1):
                            e[f'{src}_q{s:+d}'] = float(r[s][c] / ref[s][c]); e[f'{src}_n{s:+d}'] = int(n[s][c])
    for e in res.values():
        q = {src: np.nanmean([e[f'{src}_q-1'], e[f'{src}_q+1']]) for src in ('hits', 'wf')}
        e['q_hits'], e['q_wf'] = float(q['hits']), float(q['wf'])
        e['verdict'] = ('not judged' if not np.isfinite(q['wf']) else
                        'weak' if (q['wf'] < 0.6 and q['hits'] < 0.6) else 'hot' if q['wf'] > 2 else 'healthy')
        print(json.dumps(e), flush=True)
    json.dump(res, open(sys.argv[3], 'w'), indent=1)


if __name__ == '__main__':
    main()
