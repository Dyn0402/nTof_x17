#!/usr/bin/env python3
"""What does a plastic hit with `amp` ABOVE the ADC ceiling actually look like?

The census finds 211-6 417 hits per PSS channel with `amp` > 63 800 -- above
everything the digitiser can represent -- reaching 3.9e8 on PSSC. Nothing in the
raw data can exceed the rail, so the phrase "runs past the ADC rail" can only
describe the FIT, not the waveform. This pulls up the raw traces behind those
hits so the failure mode is visible rather than inferred.

Each panel shows one over-ceiling hit: the stored samples (signed int16), the
rails, the pre-pulse baseline, and every PSA hit the same block contains, with
the offending one marked. The annotation gives the fitted `amp` next to `amp_0`
(the PSA's own pre-fit maximum) and `area_0` (pre-fit integration), which is the
honest comparison -- see the PSA guide, "Finding the amplitude and area".

Time base: absolute sample time is `start + j - 259` for a zero-suppressed block
and `start + j` for the flash block, which carries no pre-samples
(FINDINGS_2026-07-29_signed_decoding.md §4).

    python pss_over_ceiling_waveforms.py <reproc_dir> <raw_head.bin> [det] [n] [-o out.png]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

CEILING = 63_800.0
RAIL_POS, RAIL_NEG = 32_767, -32_768
PRESAMPLES = 259
BR = ['amp', 'amp_0', 'area', 'area_0', 'tof', 'peak_tof', 'satuflag', 'chi2',
      'fwhm', 'fwtm', 'pulseshape', 'segment', 'BunchNumber', 'detn',
      'pileup1', 'pileup2']


def segment_of(path):
    digits = ''.join(c for c in Path(path).stem if c.isdigit() or c == '_')
    tail = digits.rsplit('_', 1)[-1]
    return int(tail) if tail else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reproc')
    ap.add_argument('raw')
    ap.add_argument('det', nargs='?', default='PSSC')
    ap.add_argument('n', nargs='?', type=int, default=6)
    ap.add_argument('-o', '--out', default='pss_over_ceiling.png')
    ap.add_argument('--no-spread', dest='spread', action='store_false',
                    help='show the highest-amp hits instead of a spread')
    ap.add_argument('--ceiling', type=float, default=CEILING,
                    help='amp threshold to select on; use 34600 for WAL*')
    ap.add_argument('--before', type=float, default=200.0,
                    help='ns of trace to show before the hit')
    ap.add_argument('--after', type=float, default=600.0,
                    help='ns of trace to show after the hit')
    args = ap.parse_args()

    seg = segment_of(args.raw)
    part = seg // 10 + 1
    with uproot.open(Path(args.reproc) / f'run224572_{part:04d}.root') as fh:
        a = fh[args.det].arrays(BR, library='np')
    in_seg = a['segment'] == seg
    ceiling = args.ceiling
    over = in_seg & (a['amp'] > ceiling)
    print(f'{args.det} segment {seg}: {int(in_seg.sum()):,} hits, '
          f'{int(over.sum())} with amp > {ceiling:,.0f} '
          f'({int((over & (a["tof"] > 1e6)).sum())} at physics time)')
    if not over.any():
        print('nothing to plot in this segment')
        return 0

    # blocks of this detector in this chunk, with their absolute time span
    blocks, bunch = [], -1
    for _o, tag, _v, pay in iter_banks(args.raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            continue
        if tag != 'ACQC':
            continue
        d, _c, blks = parse_acqc(pay, with_samples=True)
        if d != args.det:
            continue
        for start, s in blks:
            off = -PRESAMPLES if start > 0 else 0
            v = s.view('<i2').astype(np.int64)
            blocks.append(dict(bunch=bunch, chan=_c, t0=start + off, v=v,
                               t1=start + off + len(v), flash=start == 0))
    have = sorted({b['bunch'] for b in blocks})
    print(f'{len(blocks)} raw blocks for {args.det} in {Path(args.raw).name}, '
          f'bunches {have[0] if have else "-"}-{have[-1] if have else "-"} '
          f'({len(have)} bunches), channels {sorted({b["chan"] for b in blocks})}')

    # A head_*.bin holds only the first bunches of its segment, so restrict the
    # candidates to bunches actually present before ranking -- otherwise the
    # highest-amp hits are all in bunches the chunk does not contain.
    over = over & np.isin(a['BunchNumber'], have)
    print(f'{int(over.sum())} of those hits are in bunches the chunk contains')
    if not over.any():
        print('nothing to plot from this chunk')
        return 0

    # Spread the panels over the whole over-ceiling population rather than
    # showing only the most extreme: the tail (amp ~ 1e8, and amp_0 impossible
    # too) and the bulk (amp_0 sane, amp 100x it) are different failures.
    ranked = np.flatnonzero(over)[np.argsort(-a['amp'][over])]
    if args.spread and ranked.size > args.n:
        take = np.unique(np.linspace(0, ranked.size - 1, args.n * 3).astype(int))
        idx = ranked[take]
    else:
        idx = ranked
    picked = []
    for i in idx:
        t, b, dn = float(a['tof'][i]), int(a['BunchNumber'][i]), int(a['detn'][i])
        for blk in blocks:
            # each plastic is two channels; `detn` says which one the hit is on
            if (blk['bunch'] == b and blk['chan'] == dn
                    and blk['t0'] <= t < blk['t1']):
                picked.append((i, blk))
                break
        if len(picked) >= args.n:
            break
    print(f'located {len(picked)} of {min(args.n, idx.size)} in the raw stream')
    if not picked:
        print('none of the over-ceiling hits fall in a block of this chunk')
        return 0

    ncol = 2
    nrow = int(np.ceil(len(picked) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax, (i, blk) in zip(axes.ravel(), picked):
        vfull, t0 = blk['v'], blk['t0']
        base = float(np.median(vfull[:40]))
        # The first block runs 120-220 us and holds thousands of hits, so plot a
        # window around the hit rather than the block.
        tof_i = float(a['tof'][i])
        lo = max(0, int(tof_i - t0 - args.before))
        hi = min(vfull.size, int(tof_i - t0 + args.after))
        v = vfull[lo:hi]
        x = np.arange(lo, hi) + t0
        ax.plot(x, v, lw=0.9, color='tab:blue', marker='.', ms=2)
        n_rail = int((v == RAIL_NEG).sum())
        if n_rail:
            m = v == RAIL_NEG
            ax.plot(x[m], v[m], '.', ms=5, color='tab:red',
                    label=f'at negative rail ({n_rail} samples in view, '
                          f'{int((vfull == RAIL_NEG).sum())} in block)')
        ax.axhline(RAIL_NEG, color='r', lw=0.8, ls='--')
        ax.axhline(RAIL_POS, color='purple', lw=0.8, ls='--')
        ax.axhline(base, color='k', lw=0.6, ls=':')
        # every PSA hit inside this block, the offending one in red
        same = ((a['BunchNumber'] == blk['bunch']) & in_seg
                & (a['detn'] == blk['chan'])
                & (a['tof'] >= x[0]) & (a['tof'] <= x[-1]))
        for k in np.flatnonzero(same):
            ax.axvline(a['tof'][k], color='0.75', lw=0.6)
        ax.axvline(a['tof'][i], color='tab:red', lw=1.3)
        span = max(RAIL_POS - v.min(), 2000)
        ax.set_ylim(min(v.min(), RAIL_NEG) - 0.05 * span, RAIL_POS + 0.05 * span)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel('absolute sample time in the bunch [ns]')
        ax.set_ylabel('sample [ADC, signed]')
        when = ('FLASH block' if blk['flash']
                else 't = %.2f ms' % (a['tof'][i] / 1e6))
        ax.set_title(f'{args.det} bunch {blk["bunch"]} {when}  |  '
                     f'{int(same.sum())} PSA hits in view '
                     f'({int(((a["BunchNumber"] == blk["bunch"]) & in_seg & (a["detn"] == blk["chan"])).sum()):,} in the block)',
                     fontsize=9)
        txt = (f'fitted amp = {a["amp"][i]:,.0f}'
               f'  ({a["amp"][i] / ceiling:,.1f}x the threshold)\n'
               f'amp_0 (pre-fit max) = {a["amp_0"][i]:,.0f}\n'
               f'area_0 = {a["area_0"][i]:,.0f}   area = {a["area"][i]:,.0f}\n'
               f'satuflag = {int(a["satuflag"][i])}   chi2 = {a["chi2"][i]:,.0f}\n'
               f'fwhm = {a["fwhm"][i]:.1f} ns   shape = {a["pulseshape"][i]}   '
               f'pileup = {a["pileup1"][i]}/{a["pileup2"][i]}\n'
               f'baseline = {base:,.0f}   deepest sample = {v.min():,}')
        ax.text(0.015, 0.03, txt, transform=ax.transAxes, fontsize=7.5,
                va='bottom', ha='left',
                bbox=dict(fc='white', ec='0.7', alpha=0.85, boxstyle='round'))
        if n_rail:
            ax.legend(fontsize=7, loc='upper right')
    for ax in axes.ravel()[len(picked):]:
        ax.axis('off')
    fig.suptitle(f'{args.det}: raw traces behind hits with amp > {ceiling:,.0f} '
                 f'(run 224572 segment {seg}, samples decoded signed)', y=0.997)
    fig.tight_layout()
    fig.savefig(args.out, dpi=115)
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
