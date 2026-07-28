#!/usr/bin/env python3
"""Efficiency + purity of the thresholded SINGLES matcher (wall AND plastic)
with the repaired time base, per time-since-flash bin, using a +100 us
shifted control for the accidental rate."""
import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from ntof_dream_merge.bunch_join import dream_event_to_bunch
from ntof_dream_merge.dream_trigger import (load_thresholds, load_adc_mv,
                                            measure_tb_offsets,
                                            singles_candidates, ARMS)

K, T0 = 1.089e-4, -197.5
BANDS = ((-150.0, 150.0), (250.0, 450.0))
SHIFT = 100_000.0

def main():
    run, sub, nt, nb = 'run_79', 'stat090_0000', 224572, 100
    if len(sys.argv) > 1:
        nb = int(sys.argv[1])

    ev_all = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev_all.loc[ev_all['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]
    ev = ev_all[(ev_all['BunchNumber'].isin(bunches)) & (~ev_all['is_flash'])].reset_index(drop=True)
    thr, adc = load_thresholds(run, sub), load_adc_mv()

    CB, CT = [], []
    for arm in ARMS:
        off = measure_tb_offsets(nt, bunches, arm)
        for rp in (True,):
            cb, ct = singles_candidates(nt, bunches, arm, thr, adc, tb_off=off,
                                        require_plastic=True)
        CB.append(cb); CT.append(ct)
    cb = np.concatenate(CB); ct = np.concatenate(CT)
    o = np.lexsort((ct, cb)); cb, ct = cb[o], ct[o]
    print(f'{len(ev):,} events, {nb} bunches, SINGLES candidates: '
          f'{ct.size:,} ({ct.size/nb:.0f}/bunch)')


    def match(shift):
        hit = np.zeros(len(ev), bool)
        for b, g in ev.groupby('BunchNumber'):
            s, e = np.searchsorted(cb, [b, b + 1])
            tt = ct[s:e]
            if tt.size == 0:
                continue
            et = g['t_since_flash_ns'].to_numpy().astype(float)
            pred = et + K * et + T0 + shift
            lo = np.searchsorted(tt, pred - 1000)
            hi = np.searchsorted(tt, pred + 1000)
            for j, i in enumerate(g.index):
                if hi[j] <= lo[j]:
                    continue
                r = tt[lo[j]:hi[j]] - pred[j]
                for blo, bhi in BANDS:
                    if ((r >= blo) & (r <= bhi)).any():
                        hit[i] = True
                        break
        return hit


    hit = match(0.0)
    ctl = match(SHIFT)
    ets = ev['t_since_flash_ns'].to_numpy().astype(float)
    print('\n  t bin (ms)      n    efficiency   control(false)')
    for lo, hi in ((1, 3), (3, 10), (10, 20), (20, 40), (40, 80)):
        m = (ets >= lo * 1e6) & (ets < hi * 1e6)
        if m.sum():
            print(f'  {lo:4d}-{hi:<4d} {m.sum():7d}   {hit[m].mean():9.1%}   {ctl[m].mean():9.1%}')
    print(f'\n  overall: eff {hit.mean():.1%}   control {ctl.mean():.1%}')


if __name__ == '__main__':
    main()
