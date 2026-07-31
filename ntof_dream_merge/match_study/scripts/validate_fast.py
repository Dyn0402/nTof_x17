#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prove fast_singles reproduces dream_trigger exactly, on a small bunch set.

The rewrite exists only for speed, so the acceptance test is equality, not
agreement: same offsets, same candidate (bunch, time) lists, both legs, all four
arms. Anything less and the full-statistics numbers are not comparable with the
07-29 ones.
"""
import sys
import time

import numpy as np

from study_common import use_variant, dream_events, NTOF_RUN, DREAM_RUN

use_variant()

from ntof_dream_merge import dream_trigger as dt          # noqa: E402
from ntof_dream_merge import fast_singles as fs           # noqa: E402

# dream_trigger reads through ntof_io's default, i.e. with the laptop tflash
# repair ON. Match it here -- this test is about the rewrite, not the time base.
fs.REPAIR_TFLASH = True


def main() -> int:
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    _, bunches = dream_events(sub, nb=nb)
    thr, adc = dt.load_thresholds(DREAM_RUN, sub), dt.load_adc_mv()
    print(f'{len(bunches)} bunches ({bunches.min()}-{bunches.max()})\n')

    bad = 0
    for arm in dt.ARMS:
        t0 = time.time()
        off_old = dt.measure_tb_offsets(NTOF_RUN, bunches, arm)
        t_old_off = time.time() - t0
        t0 = time.time()
        off_new = fs.measure_tb_offsets(NTOF_RUN, bunches, arm)
        t_new_off = time.time() - t0
        same_off = all(abs(off_old[g] - off_new[g]) < 1e-9 for g in range(4))
        bad += not same_off

        for req in (True, False):
            t0 = time.time()
            cb, ct = dt.singles_candidates(NTOF_RUN, bunches, arm, thr, adc,
                                           tb_off=off_old, require_plastic=req)
            t_old = time.time() - t0
            o = np.lexsort((ct, cb))
            cb, ct = cb[o], ct[o]
            t0 = time.time()
            d = fs.singles_candidates(NTOF_RUN, bunches, arm, thr, adc,
                                      tb_off=off_old, require_plastic=req)
            t_new = time.time() - t0
            same = (cb.size == d['bunch'].size
                    and np.array_equal(cb, d['bunch'])
                    and np.allclose(ct, d['t'], rtol=0, atol=1e-9))
            bad += not same
            leg = 'wall AND plastic' if req else 'wall only       '
            print(f'  {arm} {leg}: {cb.size:7,} vs {d["bunch"].size:7,} '
                  f'-> {"IDENTICAL" if same else "*** DIFFERS ***"}   '
                  f'({t_old:6.1f} s -> {t_new:5.1f} s, {t_old/max(t_new,1e-6):5.1f}x)')
        print(f'  {arm} tb offsets   : {[round(off_new[g], 1) for g in range(4)]} '
              f'-> {"IDENTICAL" if same_off else "*** DIFFERS ***"}   '
              f'({t_old_off:6.1f} s -> {t_new_off:5.1f} s)\n')

    print('VERDICT:', 'all identical' if bad == 0 else f'{bad} MISMATCHES')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
