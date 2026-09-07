#!/usr/bin/env python3
"""Write a real slim file for a block of bunches and measure what it costs.

Two layouts are written so the size question is answered with bytes on disk
rather than an estimate:

  full   every hit branch that carries information, tof kept as float64
  lean   dt to the DREAM prediction as int16 (tof is quantised to 1 ns), the
         per-bunch-constant branches dropped, amplitudes as float32
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import uproot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ntof_dream_merge' / 'match_study' / 'scripts'))

import study_common as sc                                  # noqa: E402
from ntof_dream_merge.calibration import load as load_cal  # noqa: E402
import ntof_dream_merge.ntof_io as ntof_io                 # noqa: E402

OUT = Path(__file__).resolve().parent / 'figures'
TREES = ('WALA', 'WALB', 'WALC', 'WALD',
         'PSSA', 'PSSB', 'PSSC', 'PSSD',
         'LIQA', 'LIQB', 'LIQC', 'LIQD')
KEY = 1e9
W = 250.0            # slim half-window, ns -- see resid_envelope.py
NBUNCH = 200         # how many bunches of the reference pair to prototype on

BR = ('BunchNumber', 'detn', 'tof', 'amp', 'area_0', 'amp_0', 'fwhm',
      'risetime', 'chi2', 'satuflag', 'pileup1', 'pulseshape')


def main():
    sc.use_variant()
    cal = load_cal()

    ev = np.load(sc.DATA / 'events_stat090_0000.npz')
    bsel = np.unique(ev['bunch'])[:NBUNCH]
    m = np.isin(ev['bunch'], bsel)
    et, eb, eid = ev['t'][m].astype(np.float64), ev['bunch'][m], ev['eventId'][m]
    tp = cal.predict(et, arm=None)
    order = np.argsort(eb.astype(np.float64) * KEY + tp)
    pk = (eb.astype(np.float64) * KEY + tp)[order]
    pid = eid[order]
    print(f'{et.size:,} triggers over {bsel.size} bunches, +-{W:g} ns')

    cols = {k: [] for k in ('eventId', 'det', 'detn', 'dt_ns', 'tof',
                            'amp', 'area_0', 'amp_0', 'fwhm', 'risetime',
                            'chi2', 'satuflag', 'pileup1', 'pulseshape')}
    n_src = 0
    t0 = time.time()
    for ti, tree in enumerate(TREES):
        a = ntof_io.read_bunches(sc.NTOF_RUN, tree, bsel, branches=BR,
                                 repair_tflash=False)
        if a['tof'].size == 0:
            continue
        n_src += a['tof'].size
        hk = a['BunchNumber'].astype(np.float64) * KEY + a['t_since_flash_ns']
        j = np.searchsorted(pk, hk)
        j0, j1 = np.clip(j - 1, 0, pk.size - 1), np.clip(j, 0, pk.size - 1)
        d0, d1 = pk[j0] - hk, pk[j1] - hk
        take = np.abs(d0) <= np.abs(d1)
        pick = np.where(take, j0, j1)
        dt = np.where(take, -d0, -d1)          # hit - prediction
        keep = np.abs(dt) <= W
        if not keep.any():
            continue
        k = np.nonzero(keep)[0]
        cols['eventId'].append(pid[pick[k]])
        cols['det'].append(np.full(k.size, ti, np.uint8))
        cols['dt_ns'].append(dt[k])
        cols['tof'].append(a['tof'][k])
        for b in ('detn', 'amp', 'area_0', 'amp_0', 'fwhm', 'risetime',
                  'chi2', 'satuflag', 'pileup1', 'pulseshape'):
            cols[b].append(a[b][k])
    print(f'read {n_src:,} source hits in {time.time()-t0:.0f}s')

    c = {k: np.concatenate(v) for k, v in cols.items() if v}
    n = c['eventId'].size
    print(f'kept {n:,} hits  ({n/n_src:.4%} of the source)')

    o = np.argsort(c['eventId'].astype(np.int64) * 100 + c['det'])
    c = {k: v[o] for k, v in c.items()}

    full = dict(eventId=c['eventId'].astype(np.uint64),
                det=c['det'], detn=c['detn'].astype(np.int32),
                tof=c['tof'].astype(np.float64),
                dt_ns=c['dt_ns'].astype(np.float32),
                amp=c['amp'].astype(np.float32),
                area_0=c['area_0'].astype(np.float32),
                amp_0=c['amp_0'].astype(np.float32),
                fwhm=c['fwhm'].astype(np.float32),
                risetime=c['risetime'].astype(np.float32),
                chi2=c['chi2'].astype(np.float32),
                satuflag=c['satuflag'].astype(np.int32),
                pileup1=c['pileup1'].astype(np.int32),
                pulseshape=c['pulseshape'].astype(np.int32))

    q = np.rint(c['dt_ns']).astype(np.int16)
    print(f'dt quantisation residual: max {np.abs(c["dt_ns"] - q).max():.3f} ns')
    lean = dict(eventId=c['eventId'].astype(np.uint64),
                det=c['det'], detn=c['detn'].astype(np.uint8),
                dt_ns=q,
                amp=c['amp'].astype(np.float32),
                amp_0=c['amp_0'].astype(np.float32),
                area_0=c['area_0'].astype(np.float32),
                fwhm=c['fwhm'].astype(np.float32),
                chi2=c['chi2'].astype(np.float32),
                flags=(c['satuflag'].astype(np.uint8)
                       | (c['pileup1'].astype(np.uint8) << 1)
                       | (c['pulseshape'].astype(np.uint8) << 2)))

    for name, d in (('full', full), ('lean', lean)):
        p = OUT / f'slim_{name}.root'
        with uproot.recreate(p, compression=uproot.ZLIB(4)) as f:
            f['ntof_hits'] = d
        print(f'{name:5s} {p.stat().st_size/1e6:8.2f} MB   '
              f'{p.stat().st_size/n:6.2f} B/hit   '
              f'{p.stat().st_size/et.size:8.1f} B/DREAM event')

    # what the same hits cost inside the source file, for the ratio
    print(f'\nsource: {n_src:,} hits in these {bsel.size} bunches')


if __name__ == '__main__':
    main()
