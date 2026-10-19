#!/usr/bin/env python3
"""List the detector channels present in a raw stream1 chunk.

Use this for runs that have no official processed file (so `DAQsettings` is not
available) -- the ACQC banks carry the detector name and channel directly.

    head -c 120000000 <run>/stream1/run<run>_0_s1.raw.finished > head.bin
    python3 detectors_from_raw.py head.bin

Prints `nev=<events seen> DET,DET,...`.  With `--channels` it prints DET/chan
pairs, and with `--blocks DET` it dumps the block starts and sample ranges for
that detector, which is how MMA/MMB were shown to carry real waveforms.

Note on interpretation: the DAQ writes one mandatory un-suppressed block per
configured channel per event (30 us for most detectors, 50 us for PKUP), so a
channel absent from a *complete* event is not configured.  A 120 MB head chunk
may only hold one partial event on the big July/August configurations -- read
more of the file before concluding a channel is absent.

Requires `ntof_raw.py` from nTof_x17_DAQ/stream1_monitor on sys.path.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.home()
                       / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', help='raw stream1 chunk (may be a truncated head)')
    ap.add_argument('--channels', action='store_true',
                    help='report DET/chan instead of DET')
    ap.add_argument('--blocks', metavar='DET', default=None,
                    help='dump block start/length/min/max for this detector')
    ap.add_argument('--max-events', type=int, default=None)
    a = ap.parse_args()

    names, nev = set(), 0
    for _off, tag, _ver, payload in iter_banks(a.raw):
        if tag == 'EVEH':
            nev += 1
            if a.max_events is not None and nev > a.max_events:
                break
        elif tag == 'ACQC':
            want_samples = a.blocks is not None
            det, chan, blks = parse_acqc(payload, with_samples=want_samples)
            # raw names are padded to 4 chars with spaces *or* NULs -- MMA/MMB
            # come back as 'MMA\x00', which a bare .strip() leaves alone
            det = str(det).strip(' \t\r\n\x00')
            names.add(f'{det}/{chan}' if a.channels else det)
            if a.blocks and det == a.blocks:
                for start, samples in blks:
                    lo, hi = min(samples), max(samples)
                    print(f'  {det}/{chan} start={start:9d} n={len(samples):6d} '
                          f'min={lo:7d} max={hi:7d}')
    print(f'nev={nev} ' + ','.join(sorted(names)))


if __name__ == '__main__':
    main()
