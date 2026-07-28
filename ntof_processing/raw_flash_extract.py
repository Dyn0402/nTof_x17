#!/usr/bin/env python3
"""Extract the gamma-flash region of every detector channel from a raw stream1 chunk.

The DAQ always keeps an initial un-suppressed block (30 us for most detectors,
50 us for PKUP) covering the flash, so a single raw head chunk gives the flash
waveform of every configured channel for a handful of bunches.  That is all we
need to choose the PSA G-FLASH parameters and to see what the SiPM walls
actually record while their signal is diverted.

Usage:
    python raw_flash_extract.py <raw_head.bin> <out.npz> [--tmax 60000]

Output npz keys: '<DET>_<chan>_ev<event>' -> int32 samples of the flash block,
plus '<key>__start' scalars (first sample index) and a 'meta' json string.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path.home()
                       / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh, parse_modh  # noqa: E402


def extract(path, tmax=60000, max_events=None):
    """Return (blocks, modh, events).

    blocks: dict key -> (start_sample, samples int32)
    Only blocks starting before `tmax` ns are kept (the flash region).
    """
    blocks, modh, events = {}, None, []
    cur = None
    for _off, tag, _ver, payload in iter_banks(path):
        if tag == 'MODH' and modh is None:
            modh = parse_modh(payload)
        elif tag == 'EVEH':
            hdr = parse_eveh(payload)
            cur = hdr['words'][1]          # BunchNumber [verified]
            events.append({'bunch': cur, 'event': hdr['event']})
            if max_events is not None and len(events) > max_events:
                break
        elif tag == 'ACQC' and cur is not None:
            det, chan, blks = parse_acqc(payload, with_samples=True)
            for start, samples in blks:
                if start < tmax:
                    key = f'{det}_{chan}_b{cur}'
                    blocks.setdefault(key, []).append(
                        (int(start), np.asarray(samples, dtype=np.int32)))
    # keep the earliest block per key (the mandatory flash block)
    out = {k: sorted(v)[0] for k, v in blocks.items()}
    return out, modh, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('out')
    ap.add_argument('--tmax', type=int, default=60000)
    ap.add_argument('--max-events', type=int, default=None)
    a = ap.parse_args()

    blocks, modh, events = extract(a.raw, a.tmax, a.max_events)
    payload = {}
    for k, (start, samples) in blocks.items():
        payload[k] = samples
        payload[k + '__start'] = np.array([start], dtype=np.int64)
    payload['meta'] = np.array([json.dumps({'modh': modh, 'events': events,
                                            'raw': a.raw, 'tmax': a.tmax})])
    np.savez_compressed(a.out, **payload)
    print(f'{len(blocks)} flash blocks from {len(events)} events -> {a.out}')
    for k in sorted(blocks)[:200]:
        s, arr = blocks[k]
        print(f'  {k:18s} start={s:7d} n={len(arr):7d} '
              f'min={arr.min():6d} max={arr.max():6d}')


if __name__ == '__main__':
    main()
