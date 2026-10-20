#!/usr/bin/env python3
"""
examples.py -- single-bunch example waveforms from one flash_pull product, as
min/max envelopes that an SVG slide can draw: one parasitic and one dedicated
bunch, one channel per detector.

    python -m ntof_scint_recovery.examples <flash_<run>_<idx>.root>

Writes ``out_dir('quick')/examples.npz``.  Two grids per trace: linear 0.1 us
steps over -3..45 us (the flash), and 1200 log columns over 0.3 us..10 ms (the
recovery).  Envelope = min and max of pol*(x - pre-flash baseline) in mV over the
samples in a column; columns with no kept sample (a ZS gap) are 0 -- the signal
was inside the zero-suppression band there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from ntof_scint_recovery.common import C_NS, MV_PER_ADC, out_dir

DET_NAMES = ('WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD',
             'LIQA', 'LIQB', 'LIQC', 'LIQD', 'PKUP')
#: the example channel of each detector: wall end 1 (bar 1 top), plastic bar 1, the liquid
EX_CH = {d: 1 for d in range(12)}
LIN = np.arange(-3, 45.0001, 0.1)
LOG = np.logspace(np.log10(0.3), 4, 1201)


def envelope(t_us, y, edges):
    ib = np.digitize(t_us, edges) - 1
    ok = (ib >= 0) & (ib < len(edges) - 1)
    lo = np.zeros(len(edges) - 1)
    hi = np.zeros(len(edges) - 1)
    np.minimum.at(lo, ib[ok], y[ok])
    np.maximum.at(hi, ib[ok], y[ok])
    return lo, hi


def col_median(t_us, y, edges):
    """Per-column median of the samples, gaps (no kept sample) at 0."""
    ib = np.digitize(t_us, edges) - 1
    ok = (ib >= 0) & (ib < len(edges) - 1)
    ib, y = ib[ok], y[ok]
    o = np.argsort(ib, kind='stable')
    ib, y = ib[o], y[o]
    med = np.zeros(len(edges) - 1)
    cut = np.r_[0, np.nonzero(np.diff(ib))[0] + 1, len(ib)]
    for a, b in zip(cut[:-1], cut[1:]):
        if b > a:
            med[ib[a]] = np.median(y[a:b])
    return med


def main(argv=None) -> int:
    path = Path((argv or sys.argv[1:])[0])
    run = json.loads(path.with_name(path.stem + '_provenance.json').read_text())['run']
    f = uproot.open(path)
    meta = f['bunches'].arrays(['bunch', 'PulseIntensity'], library='np')
    want = {}
    for b, p in zip(meta['bunch'], meta['PulseIntensity']):
        c = 'ded' if p >= 6.5e12 else 'par' if p >= 1e11 else None
        if c and c not in want:
            want[c] = int(b)
    bb = f['blocks']['bunch'].array(library='np')
    out = {'lin_edges_us': LIN, 'log_edges_us': LOG, 'run': run}
    for c, bunch in want.items():
        idx = np.nonzero(bb == bunch)[0]
        a = f['blocks'].arrays(['det', 'detn', 'tof0', 'n', 'samples'], entry_start=idx[0], entry_stop=idx[-1] + 1)
        det, detn = np.asarray(a['det']), np.asarray(a['detn'])
        tof0, n = np.asarray(a['tof0']), np.asarray(a['n'])
        flat = np.asarray(ak.flatten(a['samples'])).astype(np.float32)
        offs = np.r_[0, np.cumsum(n)[:-1]]
        k = np.nonzero((det == 12) & (tof0 == 0))[0][0]
        x = flat[offs[k]:offs[k] + n[k]]
        t_pk = int(np.argmax(x - np.median(x[:5000])))
        out[f'{c}_bunch'] = bunch
        out[f'{c}_ppp'] = float(meta['PulseIntensity'][meta['bunch'] == bunch][0])
        for d in range(12):
            sel = np.nonzero((det == d) & (detn == EX_CH[d]))[0]
            if not len(sel):
                continue
            pol = 1 if d < 4 else -1
            t_fl = t_pk + C_NS[DET_NAMES[d]]
            ts = np.concatenate([np.arange(tof0[s], tof0[s] + n[s]) for s in sel]) - t_fl
            xs = np.concatenate([flat[offs[s]:offs[s] + n[s]] for s in sel])
            head = sel[tof0[sel] == 0][0]
            th = (np.arange(n[head]) - t_fl) / 1e3
            base = np.median(flat[offs[head]:offs[head] + n[head]][(th > -9) & (th < -1)])
            y = pol * (xs - base) * MV_PER_ADC
            for g, e in (('lin', LIN), ('log', LOG)):
                lo, hi = envelope(ts / 1e3, y, e)
                out[f'{c}_{DET_NAMES[d]}_{g}_lo'] = lo.astype(np.float32)
                out[f'{c}_{DET_NAMES[d]}_{g}_hi'] = hi.astype(np.float32)
                out[f'{c}_{DET_NAMES[d]}_{g}_med'] = col_median(ts / 1e3, y, e).astype(np.float32)
    np.savez_compressed(out_dir('quick') / 'examples.npz', **out)
    print(f'run {run}: bunches {want} -> {out_dir("quick")}/examples.npz')
    return 0


if __name__ == '__main__':
    sys.exit(main())
