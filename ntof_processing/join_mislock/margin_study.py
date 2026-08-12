#!/usr/bin/env python3
"""pulse_match count-margin study, fleet-wide.

For every sub-run in margin_targets.txt, replicate pulse_match's offset scan
(clusters -> beam pulses, count within TOL_S) but keep the WHOLE lock
structure: every local maximum's offset, match count, and intensity
correlation. The margin (best count minus second-best lock's count) is the
per-sub-run confidence that the silent tie-break bug erased.

Prediction on record (2026-08-12): FITTED sub-runs at margin >= +2-3,
FAILED whole-hours at 0 or +-1, error always toward the more negative lock.

Writes margin_results.csv: run, subrun, kind, n_clusters, best_off, best_n,
second_off, second_n, margin, r_best, r_second, n_locks.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ntof_july_analysis import pulse_match as pm            # noqa: E402

TOL_S = 0.05


def locks_for(run, subrun):
    eid, t_rel, anchor = pm._event_times(run, subrun)
    if eid is None or anchor is None:
        return None
    starts = np.concatenate([[0], np.where(np.diff(t_rel) > pm.GAP_S)[0] + 1])
    c_t = t_rel[starts]
    span = float(t_rel.max()) if len(t_rel) else 600.0
    pt, pe = pm._load_pulses([anchor + s for s in
                              np.arange(0, span + 600 + 43200, 43200)])
    if pt.size == 0:
        return None

    offs = np.arange(-120.0, 120.0, 0.02)
    counts = np.zeros(len(offs), int)
    for i, off in enumerate(offs):
        cand = anchor + c_t + off
        j = np.searchsorted(pt, cand)
        j0 = np.clip(j - 1, 0, len(pt) - 1)
        j1 = np.clip(j, 0, len(pt) - 1)
        d = np.minimum(np.abs(pt[j0] - cand), np.abs(pt[j1] - cand))
        counts[i] = int((d < TOL_S).sum())

    # cluster contiguous high-count offsets into locks
    thr = max(3, int(0.5 * counts.max()))
    hot = counts >= thr
    locks = []
    i = 0
    while i < len(offs):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < len(offs) and (hot[j] or (j - i) < 5):
            j += 1
        seg = slice(i, j)
        k = i + int(counts[seg].argmax())
        # intensity correlation at this lock
        off = offs[k]
        cand = anchor + c_t + off
        jj = np.searchsorted(pt, cand)
        j0 = np.clip(jj - 1, 0, len(pt) - 1)
        j1 = np.clip(jj, 0, len(pt) - 1)
        pick = np.where(np.abs(pt[j0] - cand) <= np.abs(pt[j1] - cand), j0, j1)
        d = np.abs(pt[pick] - cand)
        m = d < TOL_S
        # cluster size = events per cluster as intensity proxy
        sizes = np.diff(np.r_[starts, len(t_rel)])
        r = (float(np.corrcoef(sizes[m], pe[pick[m]])[0, 1])
             if m.sum() > 10 else float('nan'))
        locks.append((float(off), int(counts[k]), r))
        i = j
    locks.sort(key=lambda x: -x[1])
    return dict(n_clusters=len(c_t), locks=locks)


def main():
    targets = [l.split() for l in
               Path('margin_targets.txt').read_text().splitlines() if l]
    with open('margin_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'subrun', 'kind', 'n_clusters', 'best_off',
                    'best_n', 'second_off', 'second_n', 'margin',
                    'r_best', 'r_second', 'n_locks'])
        for run, sub, kind in targets:
            try:
                res = locks_for(run, sub)
            except Exception as e:
                print(f'{run}/{sub}: ERROR {e}', flush=True)
                continue
            if res is None or not res['locks']:
                print(f'{run}/{sub}: no locks', flush=True)
                continue
            L = res['locks']
            b = L[0]
            s = L[1] if len(L) > 1 else (float('nan'), 0, float('nan'))
            w.writerow([run, sub, kind, res['n_clusters'],
                        f'{b[0]:.2f}', b[1], f'{s[0]:.2f}', s[1],
                        b[1] - s[1], f'{b[2]:.3f}', f'{s[2]:.3f}', len(L)])
            f.flush()
            print(f'{run}/{sub} {kind}: best {b[1]}@{b[0]:+.1f}s '
                  f'second {s[1]}@{s[0]:+.1f}s margin {b[1]-s[1]} '
                  f'r {b[2]:.3f}/{s[2]:.3f} locks {len(L)}', flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
