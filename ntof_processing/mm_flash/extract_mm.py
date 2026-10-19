#!/usr/bin/env python3
"""Per-bunch micromegas products from the secured n_TOF raw.

Run on lxplus over /eos/experiment/ntof/data/x17/mm_raw_2026-07.

Sample semantics (all from ntof_raw.py, do not re-derive):
  * samples are SIGNED int16
  * ZS fill code is -32768, bit-identical to the negative rail
  * a block's payload starts PRE_SAMPLES=259 before its `start`; the always-kept
    flash block has start == 0 and no pre-samples
  * 1 GS/s -> 1 sample = 1 ns

Outputs one npz per (run, file) into --outdir; merge afterwards.
"""
import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh, ZS_FILL_CODE, PRE_SAMPLES  # noqa: E402

POS_RAIL = 32767
DEC = 10                      # decimation factor for the stored flash trace
TRACE_NS = 30_000             # uniform stored capture; PKUP keeps 50 us, MM 30 us
CHANNELS = ('MMA', 'MMB', 'PKUP', 'WALL')


def clean(det):
    return str(det).strip(' \t\r\n\x00')


def flash_stats(s):
    """Stats for the always-kept un-suppressed block (start == 0)."""
    s = s.astype(np.int32)
    base = float(np.median(s[:2000]))          # 2 us of pre-flash baseline
    dev = base - s                             # positive = negative-going pulse
    n_pos_rail = int((s >= POS_RAIL).sum())
    n_neg_rail = int((s <= ZS_FILL_CODE).sum())
    peak_i = int(np.argmax(dev))
    # recovery: last sample deviating by more than 1 % of the peak
    thr = 0.01 * dev[peak_i]
    over = np.flatnonzero(dev > thr)
    recov = int(over[-1]) if over.size else -1
    fixed = np.full(TRACE_NS, np.nan, dtype=np.float64)
    take = min(len(dev), TRACE_NS)
    fixed[:take] = dev[:take]
    trace = fixed.reshape(-1, DEC).mean(axis=1).astype(np.float32)
    return dict(base=base, peak=float(dev[peak_i]), peak_t=peak_i,
                integral=float(dev.sum()), n_pos_rail=n_pos_rail,
                n_neg_rail=n_neg_rail, recov=recov, nsamp=len(s), trace=trace)


def zs_stats(start, s):
    """Stats for a zero-suppressed block; -32768 here may be fill, so mask it."""
    s = s.astype(np.int32)
    fill = s <= ZS_FILL_CODE
    good = s[~fill]
    if good.size < 8:
        return None
    base = float(np.median(good))
    dev = base - good
    return dict(t=int(start) - PRE_SAMPLES, n=int(len(s)), nfill=int(fill.sum()),
                peak=float(dev.max()), integral=float(dev.sum()),
                n_pos_rail=int((good >= POS_RAIL).sum()))


def process(path):
    ev_bunch = []
    rows = {c: [] for c in CHANNELS}          # per-event flash stats
    traces = {c: [] for c in CHANNELS}
    zs = {c: [] for c in CHANNELS}            # (event_index, t, n, nfill, peak, integral, npos)
    cur = -1
    for _off, tag, _ver, payload in iter_banks(path):
        if tag == 'EVEH':
            hdr = parse_eveh(payload)
            ev_bunch.append(hdr['words'][1])
            cur = len(ev_bunch) - 1
        elif tag == 'ACQC' and cur >= 0:
            det, _chan, blks = parse_acqc(payload, with_samples=True)
            det = clean(det)
            if det not in CHANNELS:
                continue
            for start, s in blks:
                if start == 0:
                    st = flash_stats(s)
                    traces[det].append(st.pop('trace'))
                    st['ev'] = cur
                    rows[det].append(st)
                else:
                    z = zs_stats(start, s)
                    if z is not None:
                        zs[det].append((cur, z['t'], z['n'], z['nfill'],
                                        z['peak'], z['integral'], z['n_pos_rail']))
    out = {'bunch': np.asarray(ev_bunch, dtype=np.int64)}
    for c in CHANNELS:
        if rows[c]:
            keys = [k for k in rows[c][0] if k != 'ev']
            out[f'{c}_ev'] = np.asarray([r['ev'] for r in rows[c]], dtype=np.int32)
            for k in keys:
                out[f'{c}_{k}'] = np.asarray([r[k] for r in rows[c]], dtype=np.float64)
            out[f'{c}_trace'] = np.asarray(traces[c], dtype=np.float32)
        if zs[c]:
            out[f'{c}_zs'] = np.asarray(zs[c], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('outdir')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tag = os.path.basename(a.raw).replace('.raw.finished', '')
    dst = os.path.join(a.outdir, tag + '.npz')
    if os.path.exists(dst):
        print('skip', tag)
        return
    out = process(a.raw)
    np.savez_compressed(dst + '.tmp.npz', **out)
    os.replace(dst + '.tmp.npz', dst)
    print('ok', tag, len(out['bunch']), 'events')


if __name__ == '__main__':
    main()
