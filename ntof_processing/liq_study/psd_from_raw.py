#!/usr/bin/env python3
"""Do the liquid scintillators show two pulse classes in the RAW waveforms?

A liquid scintillator is a pulse-shape-discriminating detector: recoil protons
(from neutrons) produce a larger slow component than electrons (from gammas),
so the tail-to-total ratio separates them. If both classes are present, a
single averaged template cannot fit both -- which would explain why every
averaged template we built made the liquid fits WORSE (chi2 more than doubled)
while the same method clearly helped the walls.

This measures, on clean isolated late-time pulses straight from stream1:
  * the tail/total ratio distribution, and whether it is bimodal
  * the median normalised pulse of each class, if two are found
  * how the shipped templates (FWHM 1 ns and 7 ns) sit relative to them

    python psd_from_raw.py <outdir> <raw_head.bin> [...]
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
T_LO, T_HI = 1_000_000, 18_000_000
PRE, POST = 20, 200          # ns kept around the 50 % crossing
AMP_MIN = 150.0
TAIL_START = 12              # ns after the peak: start of the slow component


def collect(paths):
    out = {}
    for path in paths:
        for _o, tag, _v, pay in iter_banks(path):
            if tag != 'ACQC':
                continue
            det, chan, blks = parse_acqc(pay, with_samples=True)
            if det not in TREES:
                continue
            for start, s in blks:
                if T_LO <= start < T_HI and len(s) > 300:
                    out.setdefault(det, []).append(s.astype(np.float64))
    return out


def pulses(blocks):
    """Clean isolated negative pulses, aligned, normalised; plus their PSD ratio."""
    rows, amps, psd = [], [], []
    for s in blocks:
        base = np.median(np.concatenate([s[:50], s[-50:]]))
        d = -(s - base)                       # liquids are negative-going
        i = int(np.argmax(d))
        a = d[i]
        if a < AMP_MIN or i < PRE + 5 or i + POST + 5 > len(d):
            continue
        rms = np.std(s[:50] - np.median(s[:50]))
        # isolation: nothing before, and monotone decay after (no second pulse)
        before = d[:max(0, i - 8)]
        if before.size and before.max() > max(0.10 * a, 5 * rms):
            continue
        after = d[i + 8:]
        if after.size and (after - np.minimum.accumulate(after)).max() > max(0.06 * a, 5 * rms):
            continue
        seg = d[i - PRE:i + POST]
        tot = seg.sum()
        if tot <= 0:
            continue
        tail = d[i + TAIL_START:i + POST].sum()
        rows.append(seg / a)
        amps.append(a)
        psd.append(tail / tot)
    return (np.array(rows), np.array(amps), np.array(psd))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    blocks = collect(sys.argv[2:])

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    summary = {}
    for k, tree in enumerate(TREES):
        if tree not in blocks:
            continue
        rows, amps, psd = pulses(blocks[tree])
        summary[tree] = (rows, amps, psd)
        ax = axes[0][k]
        ax.hist2d(np.log10(amps), psd, bins=(70, 70),
                  range=[[np.log10(AMP_MIN), 4.3], [0.0, 0.8]],
                  cmap='viridis', cmin=1)
        ax.set_xlabel('log10 amplitude [ADC]')
        ax.set_ylabel('tail / total')
        ax.set_title(f'{tree}: PSD vs amplitude  (n={len(psd):,})')

        ax = axes[1][k]
        big = amps > 500
        ax.hist(psd[big], bins=90, range=(0, 0.8), histtype='step', color='C0',
                label=f'amp>500 (n={big.sum():,})')
        ax.hist(psd[~big], bins=90, range=(0, 0.8), histtype='step', color='C1',
                label=f'amp<500 (n={(~big).sum():,})')
        ax.set_xlabel('tail / total')
        ax.legend(fontsize=7)
        m = psd[big]
        if m.size > 100:
            print(f'{tree}: n={len(psd):6d}  tail/total p16={np.percentile(m,16):.3f} '
                  f'p50={np.percentile(m,50):.3f} p84={np.percentile(m,84):.3f} '
                  f'p99={np.percentile(m,99):.3f}')
    plt.tight_layout()
    plt.savefig(out / 'liq_psd.png', dpi=110)
    print(f'wrote {out}/liq_psd.png')

    # median pulse in PSD slices -- do the shapes actually differ?
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for k, tree in enumerate(TREES):
        if tree not in summary:
            continue
        rows, amps, psd = summary[tree]
        sel = amps > 500
        if sel.sum() < 200:
            continue
        r, p = rows[sel], psd[sel]
        lo, hi = np.percentile(p, [15, 85])
        t = np.arange(-PRE, POST)
        ax = axes[k]
        for m, lab, col in ((p < lo, f'low tail (<{lo:.2f})', 'C0'),
                            (p > hi, f'high tail (>{hi:.2f})', 'C3')):
            if m.sum() > 50:
                ax.plot(t, np.median(r[m], axis=0), lw=1.1, color=col,
                        label=f'{lab}  n={m.sum()}')
        ax.set_yscale('log'); ax.set_ylim(1e-3, 1.4); ax.set_xlim(-20, 200)
        ax.set_xlabel('ns from peak'); ax.set_title(tree)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axes[0].set_ylabel('normalised amplitude')
    plt.tight_layout()
    plt.savefig(out / 'liq_shapes_by_psd.png', dpi=110)
    print(f'wrote {out}/liq_shapes_by_psd.png')

    np.savez_compressed(out / 'liq_pulses.npz',
                        **{f'{t}_rows': summary[t][0] for t in summary},
                        **{f'{t}_amps': summary[t][1] for t in summary},
                        **{f'{t}_psd': summary[t][2] for t in summary})
    print(f'wrote {out}/liq_pulses.npz')
    return 0


if __name__ == '__main__':
    sys.exit(main())
