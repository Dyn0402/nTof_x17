#!/usr/bin/env python3
"""Per-bunch lag scan for a segment that fails the global coarse search.

Hypothesis: in the failed sub-runs the burst's tagged flash reference is not
the gamma flash but an arbitrary trigger, so every bunch carries its own
~ms-scale time offset. Globally the coincidence smears into nothing, but
WITHIN one bunch every event shares the same offset, so a per-bunch
cross-correlation must show a sharp peak at that bunch's own lag.

For each bunch: FFT cross-correlation at 1 us bins over +-80 ms, record the
tallest lag and its robust z. For every bunch with z >= 8, refine: histogram
candidate-minus-prediction residuals near that lag at 100 ns and then 20 ns
bins, and record whether it SHARPENS (a real per-bunch coincidence) or not.

Usage: perbunch_lag.py <dream_run> <dream_subrun> <ntof_run> --ntof-source DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ntof_processing.slim_pipeline import clockfit as cf        # noqa: E402
from ntof_processing.slim_pipeline.slim import (                # noqa: E402
    Segment, _bind_ntof, join_events, bunch_table, pass1_candidates)

BIN_NS = 1000.0
BURST_MS = 80.0


def perbunch_lag(te, tc, bin_ns=BIN_NS, burst_ms=BURST_MS):
    nb = int(burst_ms * 1e6 / bin_ns)
    a = np.bincount(np.clip((te / bin_ns).astype(int), 0, nb - 1),
                    minlength=nb).astype(float)
    c = np.bincount(np.clip((tc / bin_ns).astype(int), 0, nb - 1),
                    minlength=nb).astype(float)
    acc = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * nb))
                       * np.fft.rfft(c, 2 * nb), 2 * nb)
    lags = np.arange(2 * nb) * bin_ns
    lags[nb:] -= 2 * nb * bin_ns
    o = np.argsort(lags)
    lags, acc = lags[o], acc[o]
    med = float(np.median(acc))
    mad = float(np.median(np.abs(acc - med))) * 1.4826
    i = int(np.argmax(acc))
    return float(lags[i]), float((acc[i] - med) / max(mad, 1e-9))


def refine(te, tc, lag, k=cf.K_SEED):
    """Peak height in 100 ns and 20 ns bins around the lag, +-20 us."""
    pred = te * (1.0 + k)
    out = {}
    for bw in (100.0, 20.0):
        d = []
        for t in pred:
            lo = np.searchsorted(tc, t + lag - 20_000)
            hi = np.searchsorted(tc, t + lag + 20_000)
            d.append(tc[lo:hi] - t - lag)
        d = np.concatenate(d) if d else np.zeros(0)
        if d.size == 0:
            out[int(bw)] = dict(peak=0, floor=0.0, sigma=0.0, at=0.0)
            continue
        edges = np.arange(-20_000, 20_000 + bw, bw)
        h, _ = np.histogram(d, bins=edges)
        i = int(h.argmax())
        centres = 0.5 * (edges[:-1] + edges[1:])
        far = np.abs(centres - centres[i]) > 2000
        floor = float(np.median(h[far]))
        sig = (h[i] - floor) / np.sqrt(max(floor, 1.0))
        out[int(bw)] = dict(peak=int(h[i]), floor=floor, sigma=float(sig),
                            at=float(centres[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dream_run')
    ap.add_argument('dream_subrun')
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('--ntof-source', default=None)
    args = ap.parse_args()

    seg = Segment(args.dream_run, args.dream_subrun, args.ntof_run,
                  ntof_source=Path(args.ntof_source) if args.ntof_source
                  else None)
    _bind_ntof(seg)
    ev = join_events(seg)
    btbl, keep = bunch_table(ev)
    if not keep.all():
        ev = ev[keep].reset_index(drop=True)
    phys = ~ev['is_flash'].to_numpy()
    ev_b = ev['BunchNumber'].to_numpy().astype(np.int64)[phys]
    ev_t = ev['t_since_flash_ns'].to_numpy().astype(np.float64)[phys]
    cd, _, _ = pass1_candidates(seg, np.unique(ev_b))
    cb, ct = cd['bunch'], cd['t']
    order = np.lexsort((ct, cb))
    cb, ct = cb[order], ct[order]
    print(f'{ev_t.size:,} events, {ct.size:,} candidates')

    rows = []
    for b in np.unique(ev_b):
        te = ev_t[ev_b == b]
        lo, hi = np.searchsorted(cb, [b, b + 1])
        tc = ct[lo:hi]
        if te.size < 10 or tc.size < 50:
            continue
        lag, z = perbunch_lag(te, tc)
        row = dict(bunch=int(b), n_ev=int(te.size), n_cd=int(tc.size),
                   lag_ns=lag, z=z)
        if z >= 8.0:
            row['refine'] = refine(te, np.sort(tc), lag)
        rows.append(row)

    z = np.array([r['z'] for r in rows])
    lag = np.array([r['lag_ns'] for r in rows])
    sig = z >= 8.0
    print(f'\n{len(rows)} bunches scanned; {sig.sum()} with per-bunch z >= 8')
    if sig.any():
        l = lag[sig]
        print(f'lags of significant bunches: median {np.median(l)/1e6:+.4f} '
              f'ms, spread p10 {np.percentile(l,10)/1e6:+.4f} / '
              f'p90 {np.percentile(l,90)/1e6:+.4f} ms')
        sharp = [r for r in rows if r.get('refine')
                 and r['refine'][20]['sigma'] >= 5.0]
        print(f'{len(sharp)} of {int(sig.sum())} sharpen to 20 ns bins '
              f'(sigma >= 5)')
        for r in sharp[:10]:
            f = r['refine'][20]
            print(f'  bunch {r["bunch"]}: lag {r["lag_ns"]/1e6:+.4f} ms, '
                  f'20ns peak {f["peak"]} over floor {f["floor"]:.1f} '
                  f'({f["sigma"]:.0f} sigma) at {f["at"]:+.0f} ns')
    with open(f'perbunch_{args.dream_run}_{args.dream_subrun}_'
              f'{args.ntof_run}.json', 'w') as fh:
        json.dump(rows, fh, indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
