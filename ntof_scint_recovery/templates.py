#!/usr/bin/env python3
"""
templates.py -- per-family pulse templates from clean late pulses, for the
template fit in pulses.py.

    python -m ntof_scint_recovery.templates <flash_*.root> [...]

Clean = in a zero-suppressed block 1-10 ms after the flash (baseline long flat),
30-300 mV, the only pulse found in its block.  Each is cut at [-PRE, +POST] ns
around its peak sample, baseline-subtracted (block pre-samples), divided by its
peak height; the template is the median over pulses, per family and per channel.

Writes ``ntof_scint_recovery/templates.npz`` (small, versioned with the code):
``<FAM>`` the family median, ``<FAM>_<det>_<detn>`` per channel, ``t_ns`` per family.
"""
from __future__ import annotations

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from ntof_scint_recovery.pulses import FAMILY, NPRE, find

WIN = {'WAL': (150, 350), 'PSS': (40, 120), 'LIQ': (30, 100)}
OUT = Path(__file__).with_name('templates.npz')
MV = 0.0306


def collect(path: str, acc: dict):
    f = uproot.open(path)
    T = f['blocks']
    bb = T['bunch'].array(library='np')
    starts = np.r_[0, np.nonzero(np.diff(bb))[0] + 1]
    stops = np.r_[starts[1:], len(bb)]
    for e0, e1 in zip(starts, stops):
        a = T.arrays(['det', 'detn', 'tof0', 'n', 'samples'], entry_start=e0, entry_stop=e1)
        det, detn = np.asarray(a['det']).astype(int), np.asarray(a['detn']).astype(int)
        tof0, n = np.asarray(a['tof0']), np.asarray(a['n'])
        flat = np.asarray(ak.flatten(a['samples'])).astype(np.float32)
        offs = np.r_[0, np.cumsum(n)[:-1]]
        sel = np.nonzero((det < 12) & (tof0 > 1.012e6) & (tof0 < 1.0e7) & (n > NPRE + 400))[0]
        for s in sel:
            d = det[s]
            fam = FAMILY[d]
            pol = 1 if d < 4 else -1
            y = pol * (flat[offs[s]:offs[s] + n[s]] - np.median(flat[offs[s]:offs[s] + NPRE])) * MV
            pk = find(y, fam)
            if len(pk) != 1:
                continue
            i = pk[0]
            amp = y[i]
            pre, post = WIN[fam]
            if not (30 < amp < 300) or i - pre < 0 or i + post >= len(y):
                continue
            w = y[i - pre:i + post + 1] / amp
            acc.setdefault(fam, []).append(w)
            acc.setdefault(f'{fam}_{d}_{detn[s]}', []).append(w)


def main(argv=None) -> int:
    acc: dict = {}
    for p in (argv or sys.argv[1:]):
        collect(p, acc)
    out = {}
    for k, v in acc.items():
        v = np.array(v)
        out[k] = np.median(v, axis=0).astype(np.float32)
        out[k + '_n'] = len(v)
    for fam, (pre, post) in WIN.items():
        out[f't_ns_{fam}'] = np.arange(-pre, post + 1)
    np.savez_compressed(OUT, **out)
    for fam in WIN:
        t = out[f't_ns_{fam}']
        tp = out[fam]
        hm = t[tp > 0.5]
        print(f'{fam}: {out[fam + "_n"]} pulses, FWHM {hm[-1] - hm[0] + 1} ns, '
              f'value at +{t[-1]} ns: {tp[-1]:.3f}, min {tp.min():.3f}')
    print(f'-> {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
