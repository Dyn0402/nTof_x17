#!/usr/bin/env python3
"""Per-bunch MMA products from a large n_TOF raw run, at full sample resolution.

Written for run 224709 (1.5 TB, 52 channels, ~8 bunches per 4.9 GB file), where
the July decimated-trace approach is unnecessary: MMA alone is ~175 MB for the
whole run, so the flash block is kept sample for sample.

Same stream1 semantics as extract_mm.py: signed int16, ZS fill -32768,
259 pre-samples, 1 sample = 1 ns.
"""
import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh, ZS_FILL_CODE, PRE_SAMPLES  # noqa: E402

POS_RAIL = 32767
TRACE_NS = 30_000
WANT = ('MMA', 'PKUP')


def clean(det):
    return str(det).strip(' \t\r\n\x00')


def process(path):
    bunch, flash, stats, zs = [], [], [], []
    wall = []                       # (HHMMSS, 1YYMMDD) from the event header
    pk_peak, pk_t, pk_ev = [], [], []
    cur = -1
    for _off, tag, _ver, payload in iter_banks(path):
        if tag == 'EVEH':
            w = parse_eveh(payload)['words']
            # w[1] event counter, w[3] wall clock HHMMSS, w[4] date as 1YYMMDD
            bunch.append(w[1])
            wall.append((w[3], w[4]))
            cur = len(bunch) - 1
        elif tag == 'ACQC' and cur >= 0:
            det = clean(payload[0:4].decode('ascii', 'replace'))
            if det not in WANT:
                continue
            _d, _c, blks = parse_acqc(payload, with_samples=True)
            for start, s in blks:
                if det == 'PKUP':
                    if start == 0:
                        a = s.astype(np.int32)
                        base = float(np.median(a[:2000]))
                        dev = a - base            # PKUP is positive-going
                        pk_peak.append(float(dev.max()))
                        pk_t.append(int(np.argmax(dev)))
                        pk_ev.append(cur)
                    continue
                if start == 0:
                    a = s.astype(np.int32)
                    base = float(np.median(a[:2000]))
                    dev = base - a                # MMA is negative-going
                    keep = np.full(TRACE_NS, ZS_FILL_CODE, dtype=np.int16)
                    n = min(len(s), TRACE_NS)
                    keep[:n] = s[:n]
                    flash.append(keep)
                    pi = int(np.argmax(dev))
                    stats.append((cur, base, float(dev[pi]), pi, float(dev.sum()),
                                  int((a >= POS_RAIL).sum()), int((a <= ZS_FILL_CODE).sum()),
                                  len(s)))
                else:
                    a = s.astype(np.int32)
                    fill = a <= ZS_FILL_CODE
                    good = a[~fill]
                    if good.size < 8:
                        continue
                    b = float(np.median(good))
                    d = b - good
                    zs.append((cur, int(start) - PRE_SAMPLES, len(s), int(fill.sum()),
                               float(d.max()), float(d.sum()),
                               int((good >= POS_RAIL).sum())))
    return dict(bunch=np.asarray(bunch, dtype=np.int64),
                wall=np.asarray(wall, dtype=np.int64) if wall else np.zeros((0, 2), np.int64),
                flash=np.asarray(flash, dtype=np.int16) if flash else np.zeros((0, TRACE_NS), np.int16),
                stats=np.asarray(stats, dtype=np.float64) if stats else np.zeros((0, 8)),
                zs=np.asarray(zs, dtype=np.float64) if zs else np.zeros((0, 7)),
                pkup=np.asarray(list(zip(pk_ev, pk_peak, pk_t)), dtype=np.float64)
                if pk_ev else np.zeros((0, 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('outdir')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tag = os.path.basename(a.raw).replace('.raw.finished', '')
    dst = os.path.join(a.outdir, tag + '.npz')
    if os.path.exists(dst):
        print('skip', tag, flush=True)
        return
    out = process(a.raw)
    np.savez_compressed(dst + '.tmp.npz', **out)
    os.replace(dst + '.tmp.npz', dst)
    print('ok', tag, len(out['bunch']), 'events', len(out['zs']), 'zs', flush=True)


if __name__ == '__main__':
    main()
