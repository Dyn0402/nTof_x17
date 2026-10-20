#!/usr/bin/env python3
"""
psa_tail.py -- every PSA hit of run 224572 (v12 reprocessing, 3018 bunches)
binned in time since the flash: amplitude and FWHM, per channel, split into
parasitic and dedicated bunches.

The PSA finds pulses over the whole 80 ms, flash tail included, so this is
the hits-side view of the recovery: when pulses start being found again, when
their shape (FWHM) returns to its late-time plateau, and how busy each channel
is.  It cannot tell gain from population -- the hits at 10 us are not the same
particles as the hits at 20 ms -- see the report's limitations.

    python -m ntof_scint_recovery.psa_tail --jobs 4
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import uproot

from ntof_scint_recovery.common import C_NS, DEDICATED_PPP, DETS, PSA_DIR, out_dir

TB_NS = np.concatenate([[-2e4, -1e3], np.logspace(1, np.log10(8e7), 140)])
AB = np.logspace(1, 5, 81)          # |amp|, ADC
WB = np.logspace(0, 3.5, 71)        # fwhm, ns
CLASSES = ('par', 'ded')


def one_segment(path) -> tuple:
    f = uproot.open(path)
    pk = f['PKUP'].arrays(['BunchNumber', 'tof', 'amp', 'PulseIntensity'], library='np')
    o = np.argsort(-pk['amp'])                       # the flash: largest PKUP hit
    b, i = np.unique(pk['BunchNumber'][o], return_index=True)
    t_pk = dict(zip(b, pk['tof'][o][i]))
    ppp = dict(zip(b, pk['PulseIntensity'][o][i]))
    nb = {c: sum((v > DEDICATED_PPP) == (c == 'ded') for v in ppp.values()) for c in CLASSES}
    H = {}
    for det in DETS:
        a = f[det].arrays(['BunchNumber', 'detn', 'tof', 'amp', 'fwhm'], library='np')
        t = a['tof'] - (np.array([t_pk.get(x, np.nan) for x in a['BunchNumber']]) + C_NS[det])
        hi = np.array([ppp.get(x, 0) for x in a['BunchNumber']]) > DEDICATED_PPP
        for ch in np.unique(a['detn']):
            for c, sel in zip(CLASSES, (~hi, hi)):
                m = (a['detn'] == ch) & np.isfinite(t) & sel
                hA, *_ = np.histogram2d(t[m], np.abs(a['amp'][m]), [TB_NS, AB])
                hW, *_ = np.histogram2d(t[m], a['fwhm'][m], [TB_NS, WB])
                H[(det, int(ch), c)] = (hA, hW)
    return nb, H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    files = sorted(PSA_DIR.glob('run*_*.root'))
    nb = {c: 0 for c in CLASSES}
    H: dict = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for n, h in ex.map(one_segment, files):
            for c in CLASSES:
                nb[c] += n[c]
            for k, (hA, hW) in h.items():
                if k in H:
                    H[k][0] += hA
                    H[k][1] += hW
                else:
                    H[k] = [hA, hW]
    keys = sorted(H)
    np.savez_compressed(
        out_dir() / 'psa_tail.npz', tb_ns=TB_NS, ab=AB, wb=WB,
        keys=np.array([f'{d}|{ch}|{c}' for d, ch, c in keys]),   # det|detn|class
        hA=np.array([H[k][0] for k in keys], dtype=np.float32),
        hW=np.array([H[k][1] for k in keys], dtype=np.float32),
        n_par=nb['par'], n_ded=nb['ded'])
    print(f'{len(files)} segments, bunches {nb} -> {out_dir()}/psa_tail.npz')
    return 0


if __name__ == '__main__':
    sys.exit(main())
