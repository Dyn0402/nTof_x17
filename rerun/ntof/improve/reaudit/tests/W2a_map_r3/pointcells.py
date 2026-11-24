#!/usr/bin/env python3
"""pointcells.py -- horizontal pointing cells (T5 hmap.run_arm, 25 mm, unmapped slopes, late >= 25 ms tracks) for all
runs and split into two run groups g0/g1 balanced in track count (run_groups.json; run parity is 80/20), for the
joint fit and the independent beam check.  -> pc_<arm>_<all|g0|g1>.npz
Cell validity: T5's own flags (reach, near-normal, dead bands, core, n >= 1000) evaluated on each split; the joint fit
uses cells valid in the split AND in T5's all-run map.
"""
import sys
from pathlib import Path

import numpy as np

T5 = Path('/media/ucla/work/claude-jobs/eb8200b3/T5_pointing_map')
sys.path.insert(0, str(T5))
import pm  # noqa: E402
import hmap  # noqa: E402

OUT = Path(__file__).resolve().parent


def main():
    d = pm.load()
    # two run groups balanced in track count (greedy by size): run parity is 80/20, too unbalanced
    cnt = d.groupby('rn').size().sort_values(ascending=False)
    grp, load = {}, [0, 0]
    for r, n in cnt.items():
        k = int(load[1] < load[0]); grp[r] = k; load[k] += n
    d['grp'] = d.rn.map(grp)
    import json; json.dump({int(k): int(v) for k, v in grp.items()}, open(OUT / 'run_groups.json', 'w'))
    print('group loads', load, flush=True)
    for arm in 'ACD':
        nom = dict(np.load(T5 / f'hmap_{arm}_25.npz'))
        for lab, sel in (('all', None), ('g0', lambda g: g.grp == 0), ('g1', lambda g: g.grp == 1)):
            o = hmap.run_arm(d, arm, pm.E25, '25', sel=sel, boot=False)
            v = o['valid'] & nom['valid']
            np.savez(OUT / f'pc_{arm}_{lab}.npz', edges=pm.E25, c=o['c'], err=o['err'], e=o['e'], n=o['n'],
                     sig=o['sig'], valid=v, valid_own=o['valid'], e_normal=o['e_normal'])
            print(arm, lab, int(v.sum()), float(np.nanmedian(o['err'][v])), flush=True)


if __name__ == '__main__':
    main()
