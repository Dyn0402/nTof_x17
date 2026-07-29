#!/usr/bin/env python3
"""Show what an ADC under-range WRAP looks like, on real late-time liquid pulses.

`adc_range_census.py` counts them; this one draws them, because the failure mode
is not obvious from a number. The liquids are NEGATIVE-going on a baseline near
ADC 31 100, and the samples are unsigned 16-bit, so a pulse taller than the
baseline needs a sample below zero and instead reappears near 65 535. In
pulse-height coordinates (baseline - sample) the recorded trace therefore does
NOT flat-top: it spikes to about -34 000 for one or two samples at the very peak
and then continues normally.

This is the liquid (and plastic) direction. The walls and PKUP are
POSITIVE-going on a baseline near 34 000 and wrap the other way, off the TOP of
the range and back to near 0 -- see the polarity column of the census. This
script only handles the liquids, so it tests for a sample above 60 000; that
test would be wrong on a wall, where such a sample is an ordinary large pulse.

Only late-time (post-flash) blocks are used, so these are ordinary physics pulses
-- not the gamma flash, where wall/plastic saturation is expected and understood.

    python adc_wrap_examples.py <outdir> <raw_head.bin> [...] [--as-recorded]

--as-recorded draws only what is in the file: raw ADC samples, no baseline
subtraction and no wrap undone. That is the view to check the story against,
since the corrected trace is our interpretation and the raw one is not.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc  # noqa: E402

TREES = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
PAD = 32768              # zero-suppression fill value, not a measurement
HIGH = 60000             # a negative-going detector never legitimately goes here
T_LATE = 1_000_000       # ns; well past the flash and its recovery
WRAP = 65536
PRE, POST = 60, 260      # ns drawn around the peak
N_SHOW = 6


def collect(paths, want=N_SHOW * 6):
    """Late-time liquid blocks containing at least one wrapped sample."""
    found, n_late = [], 0
    for path in paths:
        for _o, tag, _v, pay in iter_banks(path):
            if tag != 'ACQC':
                continue
            det, _chan, blks = parse_acqc(pay, with_samples=True)
            if det not in TREES:
                continue
            for start, s in blks:
                if start < T_LATE or len(s) < 80:
                    continue
                n_late += 1
                real = s != PAD
                hi = (s > HIGH) & real
                if hi.any():
                    found.append((det, start, s.astype(np.int64), hi))
            if len(found) >= want:
                return found, n_late
    return found, n_late


def prepare(s, hi):
    """Pulse-height view: recorded, and with the wrap undone."""
    real = s != PAD
    base = float(np.median(s[real][:40])) if real.any() else float(np.median(s))
    rec = base - s.astype(float)                 # recorded, pulses up
    true = base - np.where(hi, s - WRAP, s).astype(float)
    i = int(np.argmax(hi))                       # the wrapped sample IS the peak
    return base, rec, true, i


def main():
    argv = [a for a in sys.argv if a != '--as-recorded']
    as_rec = len(argv) != len(sys.argv)
    if len(argv) < 3:
        print(__doc__)
        return 1
    outdir = Path(argv[1])
    outdir.mkdir(parents=True, exist_ok=True)

    found, n_late = collect(argv[2:])
    if not found:
        print('no late-time wrapped liquid blocks in these chunks')
        return 1
    print(f'{len(found)} late-time wrapped liquid blocks '
          f'(of {n_late} late-time liquid blocks scanned)')

    # one example per detector first, then the rest, so the figure is not all LIQA
    order, seen = [], set()
    for k, (det, *_ ) in enumerate(found):
        if det not in seen:
            seen.add(det)
            order.append(k)
    order += [k for k in range(len(found)) if k not in order]
    show = order[:N_SHOW]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, k in zip(axes.ravel(), show):
        det, start, s, hi = found[k]
        base, rec, true, i = prepare(s, hi)
        lo, up = max(0, i - PRE), min(len(s), i + POST)
        t = np.arange(lo, up)                    # ns within the block
        if as_rec:
            # literally the stored samples: unsigned 16-bit, pulses point DOWN
            ax.axhline(0, color='0.6', lw=1.0)
            ax.axhline(WRAP - 1, color='0.6', lw=1.0)
            ax.axhline(base, color='crimson', lw=1.0, ls=':',
                       label=f'baseline ({base:.0f} ADC)')
            ax.plot(t, s[lo:up], color='tab:blue', lw=1.2, label='stored samples')
            w = np.flatnonzero(hi[lo:up]) + lo
            ax.plot(w, s[w], 'x', color='tab:red', ms=9, mew=2,
                    label='sample above 60 000')
            ax.set_ylim(-2500, WRAP + 2500)
            ax.set_yticks([0, 16384, 32768, 49152, 65535])
            ax.set_title(f'{det}  t = {(start + i) / 1e6:.2f} ms   '
                         f'{int(hi.sum())} sample(s) above 60 000', fontsize=9)
            ax.set_xlabel('sample within block [ns]')
            ax.set_ylabel('stored sample value [ADC]')
            ax.legend(fontsize=7, loc='center right')
            continue
        ax.axhline(0, color='0.8', lw=0.8)
        ax.axhline(base, color='crimson', lw=1.0, ls=':',
                   label=f'wrap ceiling = baseline ({base:.0f} ADC)')
        ax.plot(t, true[lo:up], color='tab:orange', lw=2.2, alpha=0.75,
                label='true pulse (wrap undone)')
        ax.plot(t, rec[lo:up], color='tab:blue', lw=1.2,
                label='as recorded')
        w = np.flatnonzero(hi[lo:up]) + lo
        ax.plot(w, rec[w], 'x', color='tab:blue', ms=8, mew=2)
        ax.set_title(f'{det}  t = {(start + i) / 1e6:.2f} ms   '
                     f'{int(hi.sum())} wrapped sample(s), '
                     f'true peak {true[i]:.0f} ADC', fontsize=9)
        ax.set_xlabel('sample within block [ns]')
        ax.set_ylabel('pulse height  (baseline - sample) [ADC]')
        ax.legend(fontsize=7, loc='lower right')
        # the full-scale spike squashes the pulse itself; zoom on the peak
        zl, zu = max(0, i - 15), min(len(s), i + 60)
        tz = np.arange(zl, zu)
        ins = ax.inset_axes([0.55, 0.52, 0.43, 0.44])
        ins.axhline(base, color='crimson', lw=0.9, ls=':')
        ins.plot(tz, true[zl:zu], color='tab:orange', lw=2.0, alpha=0.8)
        ins.plot(tz, rec[zl:zu], color='tab:blue', lw=1.0)
        ins.set_ylim(-0.05 * true[i], 1.12 * true[i])
        ins.tick_params(labelsize=6)
        ins.set_title('zoom on the peak', fontsize=6)
    fig.suptitle(('Late-time liquid blocks with a sample above 60 000, exactly '
                  'as stored in stream1 (run 224572)' if as_rec else
                  'ADC under-range wrap on late-time (physics) liquid pulses, '
                  'run 224572 stream1'), fontsize=12)
    fig.tight_layout()
    p = outdir / ('adc_wrap_as_recorded.png' if as_rec else 'adc_wrap_examples.png')
    fig.savefig(p, dpi=130)
    print('wrote', p)
    if as_rec:
        return 0

    # what the wrap costs: how far over the ceiling, and how many samples
    over, nwrap = [], []
    for det, start, s, hi in found:
        _b, _r, true, i = prepare(s, hi)
        over.append(true[i])
        nwrap.append(int(hi.sum()))
    over, nwrap = np.array(over), np.array(nwrap)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(over, bins=30, color='tab:orange')
    a1.set_xlabel('true peak height [ADC]')
    a1.set_ylabel('blocks')
    a1.set_title(f'how far past the ceiling ({len(found)} blocks)\n'
                 f'median {np.median(over):.0f} ADC, max {over.max():.0f}',
                 fontsize=9)
    a2.hist(nwrap, bins=np.arange(0.5, nwrap.max() + 1.5), color='tab:blue')
    a2.set_xlabel('wrapped samples per pulse')
    a2.set_ylabel('blocks')
    a2.set_title('a wrap is 1-2 samples wide -- there is no flat top\n'
                 'to trigger a clipping test', fontsize=9)
    fig.tight_layout()
    p = outdir / 'adc_wrap_summary.png'
    fig.savefig(p, dpi=130)
    print('wrote', p)

    print(f'\ntrue peak height: median {np.median(over):.0f} ADC, '
          f'max {over.max():.0f} (ceiling is the baseline, ~31 000)')
    print(f'wrapped samples per pulse: {np.bincount(nwrap)[1:].tolist()} '
          f'(1, 2, 3, ... samples)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
