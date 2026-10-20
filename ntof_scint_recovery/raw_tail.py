#!/usr/bin/env python3
"""
raw_tail.py -- the baseline of every scintillator channel after the gamma
flash, from the raw stream1 samples.

The DAQ keeps an unsuppressed head (30 us; 50 us for PKUP) from t = 0, and
extends it while the signal stays outside the zero-suppression band, so the
flash and its tail are complete.  After it, every ZS block starts
PRE_SAMPLES = 259 samples before its threshold crossing, so the median of a
block's first 200 samples is the local baseline.  Together they give the
baseline of each channel from the flash out to the end of the 80 ms window.

Per (bunch, channel) this writes the baseline shift from the pre-flash level,
in log bins of time since the flash, signed so that + is the pulse direction
(walls positive-going, plastics and liquids negative-going).  Inside the head
block the per-bin MEDIAN of the samples is used; at the pulse rates of the
plastics in the first ~100 us that median is pulled in the pulse direction by
pile-up, so it is the effective baseline a discriminator sees, not the
amplifier's.

    python -m ntof_scint_recovery.raw_tail --jobs 7
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import uproot

from ntof_scint_recovery.common import (C_NS, DEDICATED_PPP, MV_PER_ADC, PSA_DIR,
                                        RAW_FILES, out_dir)

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

TB_US = np.logspace(-1, np.log10(7e4), 60)
#: a bunch whose last event in the chunk is truncated has fewer channels
MIN_CHANNELS = 45


def scan(path: Path) -> list:
    """[(bunch, det, ch, head_end_us, noise_mV, shift_mV[len(TB_US)-1])]."""
    blocks: dict = {}
    ev = None
    for _o, tag, _v, p in iter_banks(str(path)):
        if tag == 'EVEH':
            ev = parse_eveh(p)['words'][1]
        elif tag == 'ACQC' and ev is not None:
            det, ch, blks = parse_acqc(p, with_samples=True)
            if det in C_NS or det == 'PKUP':
                blocks.setdefault((ev, det, ch), []).extend(
                    (int(s), np.asarray(x, dtype=np.int16)) for s, x in blks)
    out = []
    for ev in sorted({k[0] for k in blocks}):
        keys = [k for k in blocks if k[0] == ev]
        if len(keys) < MIN_CHANNELS or (ev, 'PKUP', 1) not in blocks:
            continue
        pk = blocks[(ev, 'PKUP', 1)][0][1].astype(float)
        t_pk = np.argmax(pk - np.median(pk[:5000]))
        for k in keys:
            det = k[1]
            if det not in C_NS:
                continue
            pol = 1 if det.startswith('WAL') else -1
            t0 = t_pk + C_NS[det]
            bl = sorted(blocks[k], key=lambda b: b[0])
            s0, x0 = bl[0][0], bl[0][1].astype(float)
            pre = x0[:int(t0) - 500]
            base, noise = np.median(pre), np.std(pre)
            vals = np.full(len(TB_US) - 1, np.nan)
            tt = (np.arange(len(x0)) + s0 - t0) / 1e3
            idx = np.digitize(tt, TB_US) - 1
            for b in np.unique(idx[(idx >= 0) & (idx < len(vals))]):
                vals[b] = np.median(x0[idx == b])
            head_end = s0 + len(x0)
            later = [(s, np.median(x[:200].astype(float))) for s, x in bl[1:] if s >= head_end]
            if later:
                ts = (np.array([s for s, _ in later]) - t0) / 1e3
                bs = np.array([b for _, b in later])
                ib = np.digitize(ts, TB_US) - 1
                for b in np.unique(ib[(ib >= 0) & (ib < len(vals))]):
                    if np.isnan(vals[b]):
                        vals[b] = np.median(bs[ib == b])
            out.append((ev, det, k[2], (head_end - t0) / 1e3, noise * MV_PER_ADC,
                        pol * (vals - base) * MV_PER_ADC))
    return out


def bunch_intensity() -> dict:
    d = {}
    for f in sorted(PSA_DIR.glob('run*_*.root')):
        a = uproot.open(f)['PKUP'].arrays(['BunchNumber', 'PulseIntensity'], library='np')
        d.update(zip(a['BunchNumber'].tolist(), a['PulseIntensity'].tolist()))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--jobs', type=int, default=7)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for part in ex.map(scan, RAW_FILES) for r in part]
    inten = bunch_intensity()
    np.savez_compressed(
        out_dir() / 'raw_tail.npz', tb_us=TB_US,
        bunch=np.array([r[0] for r in rows]),
        det=np.array([r[1] for r in rows]), ch=np.array([r[2] for r in rows]),
        head_end_us=np.array([r[3] for r in rows]),
        noise_mV=np.array([r[4] for r in rows]),
        shift_mV=np.array([r[5] for r in rows]),
        ppp=np.array([inten.get(r[0], np.nan) for r in rows]))
    nb = len({r[0] for r in rows})
    nd = sum(inten.get(b, 0) > DEDICATED_PPP for b in {r[0] for r in rows})
    print(f'{len(rows)} channel-bunches, {nb} bunches ({nd} dedicated) -> {out_dir()}/raw_tail.npz')
    return 0


if __name__ == '__main__':
    sys.exit(main())
