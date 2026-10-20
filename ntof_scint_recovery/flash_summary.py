#!/usr/bin/env python3
"""
flash_summary.py -- reduce flash_pull products to per-bunch, per-channel traces
of time since the flash, small enough to analyse on the desktop.

    python -m ntof_scint_recovery.flash_summary <flash_*.root> [...] --out summary_<run>.npz

Per bunch and per scintillator channel, on the log grid ``T_EDGES_US`` (0.05 us
to 10 ms after the flash) plus the 30-32 ms reference window:

``shift``  baseline shift from the pre-flash level, mV, + = pulse direction.
           Inside the flash head block: the per-bin median of the samples (on the
           plastics this includes pile-up, i.e. the baseline a discriminator sees).
           After it: the median, over ZS blocks starting in the bin, of each
           block's first 200 samples (259 pre-samples precede the trigger).
``mean``   mean signal over the bin, mV, + = pulse direction, the ZS gaps counted
           as zero.  This is the average waveform: flash tail + pile-up current.
``nblk``   ZS blocks starting in the bin (a hit-rate proxy that needs no PSA).

plus ``lin``: the mean signal on a linear 100 ns grid over -5..60 us, for the
average flash waveform.  Bunch metadata: run, raw file, bunch, PulseIntensity
(official; NaN if the run was never processed), the PKUP flash amplitude and
time from the product itself.

Time since the flash for channel c of detector d: ``tof - (t_PKUP + C[d])``,
``t_PKUP`` the PKUP head-block maximum, C from ``common.C_NS`` (per detector, good
to ~10 ns, far below any bin here).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import uproot

try:
    from ntof_scint_recovery.common import C_NS, MV_PER_ADC
except Exception:                     # lxplus sandbox: no sept26 paths package
    C_NS = None
    MV_PER_ADC = 0.0306

DET_NAMES = ('WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD',
             'LIQA', 'LIQB', 'LIQC', 'LIQD', 'PKUP')
#: fallback constants, identical to common.C_NS (walls from flash_timing json)
C_FALLBACK = {'WALA': -1717.48, 'WALB': -1719.28, 'WALC': -1715.53, 'WALD': -1714.12,
              'PSSA': -1679.49, 'PSSB': -1690.64, 'PSSC': -1682.14, 'PSSD': -1682.68,
              'LIQA': -1707.91, 'LIQB': -1710.19, 'LIQC': -1695.53, 'LIQD': -1701.43}
CHANNELS = [(d, c) for d, n in zip(range(12), [8] * 4 + [2] * 4 + [1] * 4) for c in range(1, n + 1)]
CH_INDEX = {k: i for i, k in enumerate(CHANNELS)}

T_EDGES_US = np.concatenate([np.logspace(np.log10(0.05), 4, 121), [3e4, 3.05e4, 3.1e4, 3.15e4, 3.2e4]])
REF_BINS = slice(121, 125)                        # the 30-32 ms reference bins
LIN_EDGES_US = np.arange(-5, 60.0001, 0.1)
PRE_US = (-9.0, -1.0)                             # pre-flash baseline window
NPRE = 200


def polarity(det: int) -> int:
    return 1 if det < 4 else -1                   # walls positive-going, PSS/LIQ negative


def summarise_file(path: str, C: dict) -> dict:
    f = uproot.open(path)
    T = f['blocks']
    meta = f['bunches'].arrays(['bunch', 'PulseIntensity', 'has_official'], library='np')
    bb = T['bunch'].array(library='np')
    # blocks are written bunch by bunch: find each bunch's entry range
    starts = np.r_[0, np.nonzero(np.diff(bb))[0] + 1]
    stops = np.r_[starts[1:], len(bb)]
    nT, nL, nC = len(T_EDGES_US) - 1, len(LIN_EDGES_US) - 1, len(CHANNELS)
    out = {k: [] for k in ('bunch', 'ppp', 'has_official', 'pk_amp', 'pk_t', 'shift', 'mean',
                           'nblk', 'lin', 'base_mV', 'noise_mV')}
    ppp = dict(zip(meta['bunch'].tolist(), meta['PulseIntensity'].tolist()))
    has = dict(zip(meta['bunch'].tolist(), meta['has_official'].tolist()))
    for e0, e1 in zip(starts, stops):
        a = T.arrays(['bunch', 'det', 'detn', 'tof0', 'n', 'samples'], entry_start=e0, entry_stop=e1)
        bunch = int(a['bunch'][0])
        det = np.asarray(a['det']).astype(int)
        detn = np.asarray(a['detn']).astype(int)
        tof0 = np.asarray(a['tof0']).astype(np.int64)
        n = np.asarray(a['n']).astype(np.int64)
        import awkward as ak
        flat = np.asarray(ak.flatten(a['samples'])).astype(np.float32)
        offs = np.r_[0, np.cumsum(n)[:-1]]
        pk = np.nonzero((det == 12) & (tof0 == 0))[0]
        if not len(pk):
            continue
        x = flat[offs[pk[0]]:offs[pk[0]] + n[pk[0]]]
        b0 = np.median(x[:5000])
        t_pk = int(np.argmax(x - b0))
        pk_amp = float((x[t_pk] - b0) * MV_PER_ADC)
        shift = np.full((nC, nT), np.nan, np.float32)
        mean = np.zeros((nC, nT), np.float32)
        nblk = np.zeros((nC, nT), np.uint16)
        lin = np.zeros((nC, nL), np.float32)
        base_mV = np.full(nC, np.nan, np.float32)
        noise_mV = np.full(nC, np.nan, np.float32)
        for (d, c), ci in CH_INDEX.items():
            sel = np.nonzero((det == d) & (detn == c))[0]
            if not len(sel):
                continue
            pol = polarity(d)
            t_fl = t_pk + C[DET_NAMES[d]]
            head = sel[tof0[sel] == 0]
            if not len(head):
                continue
            h = head[0]
            xh = flat[offs[h]:offs[h] + n[h]]
            th_us = (np.arange(n[h]) - t_fl) / 1e3
            pre = xh[(th_us > PRE_US[0]) & (th_us < PRE_US[1])]
            if len(pre) < 1000:
                continue
            base = float(np.median(pre))
            base_mV[ci], noise_mV[ci] = base * MV_PER_ADC, float(np.std(pre)) * MV_PER_ADC
            # --- head block: per-bin median and sums
            ib = np.digitize(th_us, T_EDGES_US) - 1
            okb = (ib >= 0) & (ib < nT)
            y = pol * (xh - base) * MV_PER_ADC
            o = np.argsort(ib[okb], kind='stable')
            ibs, ys = ib[okb][o], y[okb][o]
            cut = np.r_[0, np.nonzero(np.diff(ibs))[0] + 1, len(ibs)]
            for k0, k1 in zip(cut[:-1], cut[1:]):
                if k1 > k0:
                    shift[ci, ibs[k0]] = np.median(ys[k0:k1])
            # --- every block: mean signal and block counts
            nn = n[sel]
            ts = np.repeat(tof0[sel] - t_fl, nn) + (np.arange(nn.sum()) - np.repeat(np.r_[0, np.cumsum(nn)[:-1]], nn))
            xs = np.concatenate([flat[offs[s]:offs[s] + n[s]] for s in sel])
            ys = pol * (xs - base) * MV_PER_ADC
            tus = ts / 1e3
            ib = np.digitize(tus, T_EDGES_US) - 1
            okb = (ib >= 0) & (ib < nT)
            w = np.diff(T_EDGES_US) * 1e3                       # bin width, ns = samples
            mean[ci] = np.bincount(ib[okb], ys[okb], minlength=nT)[:nT] / w
            il = np.digitize(tus, LIN_EDGES_US) - 1
            okl = (il >= 0) & (il < nL)
            lin[ci] = np.bincount(il[okl], ys[okl], minlength=nL)[:nL] / 100.0
            # --- ZS blocks after the head: pre-sample baseline
            zs = sel[(tof0[sel] > 0) & (n[sel] >= NPRE)]
            if len(zs):
                idx = offs[zs][:, None] + np.arange(NPRE)[None, :]
                pm = pol * (np.median(flat[idx], axis=1) - base) * MV_PER_ADC
                tb = (tof0[zs] - t_fl) / 1e3
                ibz = np.digitize(tb, T_EDGES_US) - 1
                okz = (ibz >= 0) & (ibz < nT)
                nblk[ci] = np.minimum(np.bincount(ibz[okz], minlength=nT)[:nT], 65535)
                o = np.argsort(ibz[okz], kind='stable')
                ibs, ps = ibz[okz][o], pm[okz][o]
                cut = np.r_[0, np.nonzero(np.diff(ibs))[0] + 1, len(ibs)]
                for k0, k1 in zip(cut[:-1], cut[1:]):
                    if k1 > k0 and np.isnan(shift[ci, ibs[k0]]):
                        shift[ci, ibs[k0]] = np.median(ps[k0:k1])
        out['bunch'].append(bunch)
        out['ppp'].append(ppp.get(bunch, np.nan))
        out['has_official'].append(has.get(bunch, 0))
        out['pk_amp'].append(pk_amp)
        out['pk_t'].append(t_pk)
        for k, v in (('shift', shift), ('mean', mean), ('nblk', nblk), ('lin', lin),
                     ('base_mV', base_mV), ('noise_mV', noise_mV)):
            out[k].append(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('products', nargs='+')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)
    C = C_NS if C_NS else C_FALLBACK
    allv: dict = {}
    files = []
    for p in a.products:
        prov = json.loads(Path(p[:-5] + '_provenance.json').read_text()) if Path(p[:-5] + '_provenance.json').exists() else {}
        r = summarise_file(p, C)
        nb = len(r['bunch'])
        r['run'] = [prov.get('run', -1)] * nb
        r['raw_index'] = [prov.get('raw_index', -1)] * nb
        for k, v in r.items():
            allv.setdefault(k, []).extend(v)
        files.append(Path(p).name)
        print(f'{Path(p).name}: {nb} bunches', flush=True)
    if not allv.get('bunch'):
        print('no bunches with a PKUP flash block')
        return 1
    np.savez_compressed(
        a.out, t_edges_us=T_EDGES_US, lin_edges_us=LIN_EDGES_US,
        channels=np.array(CHANNELS), files=np.array(files),
        **{k: (np.array(v, np.float16) if k in ('shift', 'mean', 'lin') else np.array(v))
           for k, v in allv.items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
