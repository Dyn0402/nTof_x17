#!/usr/bin/env python3
"""What ADC saturation actually looks like on the liquids, decoded as SIGNED int16.

This supersedes `adc_range_census.py` / `adc_wrap_examples.py`, which read the
stream1 samples as UNSIGNED 16-bit and therefore reported an "under-range wrap"
that does not exist. The samples are `int16_t` -- that is how ntoflib reads them
(`ReaderStructACQC.h:41`) and it is what the DAQ settings say:

  * LIQ/PSS/SILI carry `baselineOffsetmV = +950` of a ~2004 mV full scale, so
    their baseline sits at about +31 000 of +-32 768 and their (negative-going)
    pulses swing down through zero toward -32 768;
  * WAL/PKUP carry -950 mV, baseline near -31 400, and swing up.

Decoded signed, every trace is continuous: what looked like a wrap at 65 535 is
just a pulse that crossed zero. The usable amplitude is therefore ~63 800 ADC,
about twice what the unsigned reading suggested, and a pulse is saturated only
when it reaches the rail at -32 768 (liquids) or +32 767 (walls).

CAUTION, and the reason this script does not simply test `== -32768`: the
zero-suppression FILL value is 0x8000, which is the same code as the negative
rail. A fill sample is told apart from a clipped one by its neighbours -- a clip
is approached, a fill is not.

    python saturation_examples.py <outdir> <raw_head.bin> [...]

Writes sat_examples_liq.png (individual pulses), sat_population_liq.png (run
lengths, times, and flat-top width vs depth) and prints a per-detector census
with the provenance of every clipped liquid block.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

LIQ = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
RAIL_LO, RAIL_HI = -32768, 32767
FILL = -32768            # 0x8000 zero-suppression fill == the negative rail
APPROACH = 5000          # a genuine clip has a neighbour this close to the rail
T_LATE = 1_000_000       # ns; past the flash and its recovery
MV_PER_ADC = 2004.0 / 65536
PRE, POST = 80, 200      # ns drawn around the clip
N_SHOW = 6
DEEP = 20000             # "deep" pulse: at least this far from baseline


def segment(path):
    digits = ''.join(c for c in Path(path).stem if c.isdigit() or c == '_')
    tail = digits.rsplit('_', 1)[-1]
    return int(tail) if tail else -1


def rail_runs(v, rail):
    """Contiguous runs of samples at `rail`, split into (genuine, fill).

    Genuine means the sample just outside the run is already within APPROACH of
    the rail, i.e. the trace walked into it. A zero-suppression fill sits next to
    baseline-level samples and fails that test.
    """
    at = np.flatnonzero(v == rail)
    if at.size == 0:
        return [], []
    genuine, fill = [], []
    for grp in np.split(at, np.flatnonzero(np.diff(at) != 1) + 1):
        i0, i1 = int(grp[0]), int(grp[-1])
        near = False
        if i0 > 0:
            near |= abs(int(v[i0 - 1]) - rail) < APPROACH
        if i1 + 1 < len(v):
            near |= abs(int(v[i1 + 1]) - rail) < APPROACH
        (genuine if near else fill).append((i0, i1))
    return genuine, fill


def flat_top(v, base):
    """Width of the plateau at the extreme of the deepest pulse in the block.

    Counts samples contiguous with the extremum that are within 2 % of the peak
    depth of it. A front-end (not ADC) saturation shows up here: a flat top that
    never reaches the rail.
    """
    j = int(np.argmin(v)) if base > 0 else int(np.argmax(v))
    depth = abs(float(v[j]) - base)
    if depth < 1:
        return 0, j, depth
    tol = 0.02 * depth
    lo = j
    while lo > 0 and abs(float(v[lo - 1]) - float(v[j])) <= tol:
        lo -= 1
    hi = j
    while hi + 1 < len(v) and abs(float(v[hi + 1]) - float(v[j])) <= tol:
        hi += 1
    return hi - lo + 1, j, depth


def scan(paths):
    """One pass: per-detector census, liquid clip examples, deep-pulse shapes."""
    census = defaultdict(lambda: dict(blocks=0, clip=0, clip_late=0, fill=0,
                                      deepest=0.0, base=[]))
    examples, deep = [], defaultdict(list)
    for path in paths:
        seg, bunch, event = segment(path), -1, -1
        for _o, tag, _v, pay in iter_banks(path):
            if tag == 'EVEH':
                h = parse_eveh(pay)
                bunch, event = int(h['words'][1]), int(h['event'])
                continue
            if tag != 'ACQC':
                continue
            det, _chan, blks = parse_acqc(pay, with_samples=True)
            for start, s in blks:
                if len(s) < 40:
                    continue
                v = s.view('<i2').astype(np.int64)     # SIGNED, the whole point
                c = census[det]
                c['blocks'] += 1
                base = float(np.median(v[:40]))
                if len(c['base']) < 500:
                    c['base'].append(base)
                rail = RAIL_LO if base > 0 else RAIL_HI
                gen, fil = rail_runs(v, rail)
                if base < 0:                            # walls also clip DOWNWARD
                    gen2, _ = rail_runs(v, RAIL_LO)     # into the fill code, so
                    gen += gen2                         # the same caution applies
                c['fill'] += len(fil)
                width, j, depth = flat_top(v, base)
                c['deepest'] = max(c['deepest'], depth)
                if depth > DEEP and len(deep[det]) < 4000:
                    deep[det].append((depth, width, bool(gen), int(start + j)))
                if not gen:
                    continue
                c['clip'] += 1
                late = (start + gen[0][0]) > T_LATE
                c['clip_late'] += int(late)
                if det in LIQ:
                    examples.append(dict(det=det, start=start, v=v, base=base,
                                         runs=gen, seg=seg, bunch=bunch,
                                         event=event, late=late))
    return census, examples, deep


def draw_examples(examples, outdir):
    """Prefer late-time (physics) clips, and spread over detectors."""
    ex = sorted(examples, key=lambda b: (not b['late'], b['det']))
    order, seen = [], set()
    for k, b in enumerate(ex):
        if (b['det'], b['late']) not in seen:
            seen.add((b['det'], b['late']))
            order.append(k)
    order += [k for k in range(len(ex)) if k not in order]
    show = order[:N_SHOW]
    if not show:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))
    for ax, k in zip(axes.ravel(), show):
        b = ex[k]
        v, base, (i0, i1) = b['v'], b['base'], b['runs'][0]
        lo, up = max(0, i0 - PRE), min(len(v), i1 + POST)
        t = np.arange(lo, up)
        ax.axhline(RAIL_LO, color='crimson', lw=1.2,
                   label=f'ADC rail ({RAIL_LO})')
        ax.axhline(base, color='0.4', lw=1.0, ls=':',
                   label=f'baseline ({base:.0f})')
        ax.axhline(0, color='0.85', lw=0.8)
        ax.plot(t, v[lo:up], color='tab:blue', lw=1.3, label='samples (int16)')
        at = np.arange(i0, i1 + 1)
        ax.plot(at, v[at], color='crimson', lw=2.4,
                label=f'at rail ({i1 - i0 + 1} samples)')
        ax.set_ylim(RAIL_LO - 2500, max(base + 2500, 2000))
        ax.set_xlabel('sample within block [ns]')
        ax.set_ylabel('sample value [ADC, signed]')
        sec = ax.secondary_yaxis(
            'right', functions=(lambda a: (a - base) * MV_PER_ADC,
                                lambda m: m / MV_PER_ADC + base))
        sec.set_ylabel('pulse height [mV]', fontsize=8)
        sec.tick_params(labelsize=7)
        where = (f'seg {b["seg"]}  bunch {b["bunch"]}  trig {b["event"]}'
                 if b['bunch'] >= 0 else f'seg {b["seg"]}')
        ax.set_title(f'{b["det"]}   {where}\n'
                     f't = {(b["start"] + i0) / 1e6:.3f} ms '
                     f'({"physics" if b["late"] else "flash region"}), '
                     f'{len(b["runs"])} clipped run(s)', fontsize=8)
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Liquid pulses that reach the ADC rail, run 224572 stream1 '
                 '(samples decoded as signed int16)', fontsize=12)
    fig.tight_layout()
    p = outdir / 'sat_examples_liq.png'
    fig.savefig(p, dpi=130)
    print('wrote', p)


def draw_population(examples, deep, outdir):
    runs = [i1 - i0 + 1 for b in examples for i0, i1 in b['runs']]
    times = [(b['start'] + b['runs'][0][0]) / 1e6 for b in examples]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))

    if runs:
        a1.hist(runs, bins=np.arange(0.5, max(runs) + 1.5), color='crimson')
        a1.set_xlabel('samples at the rail, per clipped run [ns]')
        a1.set_ylabel('runs')
        a1.set_title(f'clipped pulses sit at the rail\n'
                     f'median {np.median(runs):.0f} ns, max {max(runs)} ns',
                     fontsize=9)
    if times:
        a2.hist(times, bins=40, color='tab:blue')
        a2.axvline(T_LATE / 1e6, color='crimson', ls='--', lw=1,
                   label='flash region ends')
        a2.set_xlabel('time in the 20 ms window [ms]')
        a2.set_ylabel('clipped liquid blocks')
        a2.set_title('when the liquids clip', fontsize=9)
        a2.legend(fontsize=7)

    for det, col in zip(LIQ, ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')):
        d = np.array([(x[0], x[1], x[2]) for x in deep.get(det, [])])
        if not len(d):
            continue
        a3.scatter(d[:, 0], d[:, 1], s=8, alpha=0.45, color=col, label=det)
    a3.set_xlabel('pulse depth from baseline [ADC]')
    a3.set_ylabel('flat-top width at the peak [ns]')
    a3.set_yscale('symlog', linthresh=10)
    a3.axvline(abs(RAIL_LO) + 31000, color='crimson', ls='--', lw=1,
               label='rail (from a ~31 000 baseline)')
    a3.set_title('a front-end saturation would show as a wide flat top\n'
                 'well before the rail', fontsize=9)
    a3.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / 'sat_population_liq.png'
    fig.savefig(p, dpi=130)
    print('wrote', p)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    census, examples, deep = scan(sys.argv[2:])

    print(f'\n{"det":6s} {"blocks":>8s} {"baseline":>9s} {"deepest":>9s} '
          f'{"clipped":>8s} {"of which physics":>17s} {"fill runs":>10s}')
    for det in sorted(census):
        c = census[det]
        base = np.median(c['base']) if c['base'] else float('nan')
        print(f'{det:6s} {c["blocks"]:8d} {base:9.0f} {c["deepest"]:9.0f} '
              f'{c["clip"]:8d} {c["clip_late"]:17d} {c["fill"]:10d}')

    if examples:
        print(f'\nclipped liquid blocks ({len(examples)}), newest scan order:')
        print(f'{"det":5s} {"seg":>4s} {"bunch":>7s} {"trig":>9s} {"t [ms]":>9s} '
              f'{"runs":>5s} {"widest":>7s} {"region":>9s}')
        for b in sorted(examples, key=lambda r: (r['seg'], r['bunch'], r['start'])):
            widest = max(i1 - i0 + 1 for i0, i1 in b['runs'])
            print(f'{b["det"]:5s} {b["seg"]:4d} {b["bunch"]:7d} {b["event"]:9d} '
                  f'{(b["start"] + b["runs"][0][0]) / 1e6:9.3f} '
                  f'{len(b["runs"]):5d} {widest:7d} '
                  f'{"physics" if b["late"] else "flash":>9s}')

    draw_examples(examples, outdir)
    draw_population(examples, deep, outdir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
