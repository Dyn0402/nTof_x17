#!/usr/bin/env python3
"""
bump.py -- the ~1 µs bump on the plastic and liquid traces: fast neutrons, not a reflection of the flash.

    python -m ntof_scint_recovery.bump <a local flash_<run>_<idx>.root>

Per plastic and liquid channel, the mean trace over the dedicated and the parasitic bunches of one file
(0-4 µs after the flash), and from it the bump's peak time (50 ns running mean) and height.  Three checks:

* the peak time is the same in every channel; a reflection would sit at a delay set by each channel's cabling,
  and the flash arrival already differs by ~25 ns between plastic and liquid cables;
* on single dedicated traces the bump is made of separate fast pulses (count of peaks > 30 mV above a 200 ns
  running median in 0.8-2 µs, and their width), not a smooth copy of the flash;
* its height scales with the beam intensity like a particle count (liquids: dedicated / parasitic ~2), and on the
  plastics by less, as the early gain loss (gain_ref.py) predicts.

1.0 µs after the flash is ~1.8 MeV neutrons at 19.5 m.  Writes out_dir('early')/bump.csv and bump_traces.npz.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import uproot
from scipy.ndimage import median_filter

from ntof_scint_recovery.common import DEDICATED_PPP, MV_PER_ADC, out_dir
from ntof_scint_recovery.flash_summary import C_FALLBACK, DET_NAMES

N_NS = 3900


def main(path: str) -> int:
    f = uproot.open(path)
    T = f['blocks']
    meta = f['bunches'].arrays(['bunch', 'PulseIntensity'], library='np')
    bb = T['bunch'].array(library='np')
    starts = np.r_[0, np.nonzero(np.diff(bb))[0] + 1]
    stops = np.r_[starts[1:], len(bb)]
    traces, single = {}, {}
    for bi, (e0, e1) in enumerate(zip(starts, stops)):
        cls = 'ded' if meta['PulseIntensity'][bi] > DEDICATED_PPP else 'par'
        a = T.arrays(['det', 'detn', 'tof0', 'samples'], entry_start=e0, entry_stop=e1)
        det, detn, tof0 = (np.asarray(a[k]).astype(int) for k in ('det', 'detn', 'tof0'))
        k = np.nonzero((det == 12) & (tof0 == 0))[0]
        if not len(k):
            continue
        x = np.asarray(a['samples'][k[0]]).astype(float)
        t_pk = int(np.argmax(x - np.median(x[:5000])))
        for s in np.nonzero((det >= 4) & (det < 12) & (tof0 == 0))[0]:
            xh = np.asarray(a['samples'][s]).astype(float)
            th = (np.arange(len(xh)) - t_pk - C_FALLBACK[DET_NAMES[det[s]]]) / 1e3
            pre = (th > -9) & (th < -1)
            m = (th >= 0) & (th < 4)
            if pre.sum() < 100 or m.sum() < N_NS:
                continue
            y = -(xh - np.median(xh[pre])) * MV_PER_ADC
            key = (DET_NAMES[det[s]], int(detn[s]))
            traces.setdefault(key + (cls,), []).append(y[m][:N_NS])
            if cls == 'ded':
                single.setdefault(key, y[m][:N_NS])
    tt = np.arange(N_NS) / 1e3
    rows = []
    for (name, ch, cls), v in sorted(traces.items()):
        sm = np.convolve(np.mean(v, axis=0), np.ones(50) / 50, 'same')
        w = (tt > 0.6) & (tt < 3)
        i = int(np.argmax(np.where(w, sm, -1e9)))
        r = dict(det=name, ch=ch, cls=cls, n_bunch=len(v), peak_us=tt[i], peak_mV=sm[i])
        if cls == 'ded':
            y = single[(name, ch)]
            res = y - median_filter(y, 201)
            seg = (tt > 0.8) & (tt < 2.0)
            pk = np.nonzero(seg[1:-1] & (res[1:-1] > 30) & (res[1:-1] >= res[:-2]) & (res[1:-1] > res[2:]))[0] + 1
            pk = pk[np.r_[True, np.diff(pk) > 10]] if len(pk) else pk
            fw = []
            for q in pk:
                lo, hi = q, q
                while lo > 0 and res[lo] > res[q] / 2:
                    lo -= 1
                while hi < N_NS - 1 and res[hi] > res[q] / 2:
                    hi += 1
                fw.append(hi - lo)
            r.update(n_pulses_one_trace=len(pk), pulse_fwhm_ns=float(np.median(fw)) if fw else np.nan)
        rows.append(r)
    B = pd.DataFrame(rows)
    piv = B.pivot_table(index=['det', 'ch'], columns='cls', values='peak_mV')
    B = B.merge((piv['ded'] / piv['par']).rename('ded_over_par').reset_index(), on=['det', 'ch'])
    o = out_dir('early')
    B.to_csv(o / 'bump.csv', index=False)
    np.savez_compressed(o / 'bump_traces.npz', t_us=tt,
                        **{f'{n}{c}_{k}': np.mean(v, axis=0) for (n, c, k), v in traces.items()})
    pd.set_option('display.width', 200)
    print(B.round(2).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
