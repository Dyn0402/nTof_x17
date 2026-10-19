#!/usr/bin/env python3
"""Where did the mystery-class DREAM events' true partners go?

For a mystery-class sliver (join proven correct, no coincidence at any
per-bunch lag), cross-correlate EVERY DREAM bunch against EVERY n_TOF
bunch of the run. If the n_TOF payloads are shifted against their bunch
headers by K bunches (constant or drifting -- a drifting K evades the
flat +-200 constant-shift scan), the matrix shows a sharp off-diagonal
ridge at c = b + K(b). If no n_TOF bunch matches any DREAM bunch, the
events have no recorded counterpart at all.

Method: 1 us-bin histograms over +-80 ms, FFT cross-correlation of every
pair (precompute each side's FFT once), robust-z of the tallest lag.
For each DREAM bunch report the best-matching n_TOF bunch and whether the
match sharpens to 20 ns bins (the signature of a REAL coincidence; the
envelope artifact never sharpens).

Usage: cross_bunch_matrix.py <dream_run> <dream_subrun> <ntof_run>
           --ntof-source DIR --delta-hint S [--ntof-bunches N]

Two-host split: extraction needs the repo + root files; the FFT matrix
needs only numpy. `--dump X.npz` extracts the four arrays (plus the
clock-skew seed) and exits; `--load X.npz` skips extraction and imports
nothing from the repo, so the heavy stage can run on any box with numpy.
refine() is inlined (duplicated from perbunch_lag.py) for the same reason.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BIN_NS = 1000.0
BURST_MS = 80.0


C_BLOCK = 128     # candidate bunches per correlation block (memory bound)


def hist(t, nb):
    return np.bincount(np.clip((t / BIN_NS).astype(int), 0, nb - 1),
                       minlength=nb).astype(np.float32)


def refine(te, tc, lag, k):
    """Peak height in 100 ns and 20 ns bins around the lag, +-20 us.

    Inlined from perbunch_lag.refine so --load mode has no repo imports.
    """
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


def extract(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ntof_processing.slim_pipeline.slim import (
        Segment, _bind_ntof, join_events, bunch_table, pass1_candidates)
    from ntof_processing.slim_pipeline import clockfit as cf

    seg = Segment(args.dream_run, args.dream_subrun, args.ntof_run,
                  ntof_source=Path(args.ntof_source) if args.ntof_source
                  else None, delta_hint_s=args.delta_hint)
    _bind_ntof(seg)
    ev = join_events(seg)
    btbl, keep = bunch_table(ev)
    if not keep.all():
        ev = ev[keep].reset_index(drop=True)
    phys = ~ev['is_flash'].to_numpy()
    ev_b = ev['BunchNumber'].to_numpy().astype(np.int64)[phys]
    ev_t = ev['t_since_flash_ns'].to_numpy().astype(np.float64)[phys]

    want = np.arange(1, args.ntof_bunches + 1)
    cd, _, _ = pass1_candidates(seg, want)
    cb, ct = cd['bunch'], cd['t']
    o = np.lexsort((ct, cb))
    return ev_b, ev_t, cb[o].astype(np.int64), ct[o].astype(np.float64), \
        float(cf.K_SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dream_run')
    ap.add_argument('dream_subrun')
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('--ntof-source', default=None)
    ap.add_argument('--delta-hint', type=float, default=None)
    ap.add_argument('--ntof-bunches', type=int, default=1000,
                    help='scan n_TOF bunches 1..N')
    ap.add_argument('--dump', default=None,
                    help='extract arrays, save to this .npz, and exit')
    ap.add_argument('--load', default=None,
                    help='load arrays from this .npz instead of extracting')
    args = ap.parse_args()

    if args.load:
        z = np.load(args.load)
        ev_b, ev_t, cb, ct = z['ev_b'], z['ev_t'], z['cb'], z['ct']
        k_seed = float(z['k_seed'])
    else:
        ev_b, ev_t, cb, ct, k_seed = extract(args)
        if args.dump:
            np.savez_compressed(args.dump, ev_b=ev_b, ev_t=ev_t,
                                cb=cb, ct=ct, k_seed=k_seed)
            print(f'DUMPED -> {args.dump}')
            return
    print(f'{ev_t.size:,} DREAM events in {np.unique(ev_b).size} bunches; '
          f'{ct.size:,} candidates in {np.unique(cb).size} n_TOF bunches')

    nb = int(BURST_MS * 1e6 / BIN_NS)
    n2 = 2 * nb
    lags = np.arange(n2) * BIN_NS
    lags[nb:] -= n2 * BIN_NS

    d_bunches = np.unique(ev_b)
    d_fft, d_te = {}, {}
    for b in d_bunches:
        te = ev_t[ev_b == b]
        if te.size < 10:
            continue
        d_te[b] = te
        d_fft[b] = np.conj(np.fft.rfft(hist(te, nb), n2))
    c_fft, c_tc = {}, {}
    for c in np.unique(cb):
        lo, hi = np.searchsorted(cb, [c, c + 1])
        tc = ct[lo:hi]
        if tc.size < 50:
            continue
        c_tc[c] = np.sort(tc)
        c_fft[c] = np.fft.rfft(hist(tc, nb), n2)
    print(f'{len(d_fft)} DREAM x {len(c_fft)} n_TOF bunches to correlate')

    c_keys = np.array(sorted(c_fft))
    # block the candidate axis: the full matrix at complex128 OOMs the box
    c_blocks = [(c_keys[lo:lo + C_BLOCK],
                 np.stack([c_fft[c] for c in c_keys[lo:lo + C_BLOCK]]))
                for lo in range(0, len(c_keys), C_BLOCK)]
    rows = []
    for i, b in enumerate(sorted(d_fft)):
        z = np.empty(len(c_keys))
        pk_lag = np.empty(len(c_keys))
        for blo, (bkeys, bmat) in zip(
                range(0, len(c_keys), C_BLOCK), c_blocks):
            acc = np.fft.irfft(d_fft[b][None, :] * bmat, n2, axis=1)
            med = np.median(acc, axis=1, keepdims=True)
            mad = np.median(np.abs(acc - med), axis=1) * 1.4826
            pk = acc.max(axis=1)
            z[blo:blo + len(bkeys)] = \
                (pk - med[:, 0]) / np.maximum(mad, 1e-9)
            pk_lag[blo:blo + len(bkeys)] = \
                lags[np.argmax(acc, axis=1)]
        j = int(np.argmax(z))
        best_c = int(c_keys[j])
        best_lag = float(pk_lag[j])
        row = dict(dream_bunch=int(b), best_ntof_bunch=best_c,
                   shift=int(best_c - b), z=float(z[j]), lag_ns=best_lag,
                   z_self=float(z[np.searchsorted(c_keys, b)]
                                if b in c_fft else np.nan))
        r = refine(d_te[b], c_tc[best_c], best_lag, k_seed)
        row['sharp20_sigma'] = r[20]['sigma']
        row['sharp20_peak'] = r[20]['peak']
        rows.append(row)
        if i % 40 == 0:
            print(f'  {i}/{len(d_fft)}: bunch {b} -> best {best_c} '
                  f'(shift {best_c - b:+d}, z {z[j]:.0f}, '
                  f'sharp20 {r[20]["sigma"]:.0f} sigma)')

    sh = np.array([r['shift'] for r in rows])
    sharp = np.array([r['sharp20_sigma'] for r in rows])
    real = sharp >= 8.0
    print(f'\n{len(rows)} DREAM bunches; {real.sum()} with a SHARP '
          f'(>=8 sigma at 20 ns) best match')
    if real.any():
        print('shift distribution of sharp matches: '
              f'median {np.median(sh[real]):+.0f}, '
              f'p10 {np.percentile(sh[real], 10):+.0f} / '
              f'p90 {np.percentile(sh[real], 90):+.0f}')
    print(f'shift distribution of ALL best matches: '
          f'zero-shift fraction {np.mean(sh == 0):.2f}, '
          f'median {np.median(sh):+.0f}')
    out = (f'crossbunch_{args.dream_run}_{args.dream_subrun}_'
           f'{args.ntof_run}.json')
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print(f'DONE -> {out}')


if __name__ == '__main__':
    main()
