#!/usr/bin/env python3
"""Generate UserInput variants from a base, by explicit column edits.

Hand-editing these files is how you end up with a variant that is not what its
name says. Every variant below is a base plus a named list of (detector-family,
column, new value) edits, so the diff is auditable and the header comment is
generated from the same spec.

    python make_variants.py            # write all variants
    python make_variants.py v6_lowthr  # just one

Column names follow the UserInput header.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UI = HERE / 'userinputs'

COLS = ['NAME', 'NUMBER', 'CLASS', 'STEP SIZE', 'TIMING FILTER',
        'MIXED POLARITY', 'EXPAND PULSES', 'SMOOTHING FILTER', 'TIME LIMIT',
        'G-FLASH OPTION', 'G-FLASH THRESHOLD', 'G-FLASH MIN_WIDTH',
        'G-FLASH WINDOW', 'BASELINE OPTION', 'BASELINE FILTER',
        'AMPLITUDE OPTION', 'AMPLITUDE THRESHOLD', 'AREA/AMP LOW',
        'AREA/AMP HIGH', 'SIGNAL WIDTH LOW', 'SIGNAL WIDTH HIGH',
        'NUMBER OF PULSE SHAPES']
IDX = {c: i for i, c in enumerate(COLS)}

VARIANTS = {
    'v6_lowthr': dict(
        base='v4_walshapes',
        why="""The amplitude threshold is BINDING on the plastics and the liquids
and NOT on the walls -- measured on the v4 output with
ntof_processing/threshold_headroom.py:

    tree   amp p1   <2x cut   <3x cut     (cut = 50)
    WAL     68-82    1.7-3.6%  4.9-8.7%   spectrum dies BEFORE the cut
    PSS     52-53   11.2-22.9% 28.1-42.7% spectrum piles UP against the cut
    LIQ     53-54    8.4-28.2% 15.5-55.3% same

So there is signal below 50 channels on PSS/LIQ and none on WAL (the DAQ
zero-suppression is the wall's floor, not the PSA). Halve the plastic and
liquid thresholds and open the AREA/AMP low edge, which also sits right at the
PSS p1 (1.30-1.55 against a cut of 1.0). Walls deliberately untouched.

Noise is the thing to watch: the PSS baseline RMS is ~20 channels, so 25 is
~1.2 sigma and the width/area conditions are doing the rejecting. If the
singles-matcher EFFICIENCY rises this was signal; if only the false rate rises
it was noise.""",
        edits=[('PSS', 'AMPLITUDE THRESHOLD', '25'),
               ('PSS', 'AREA/AMP LOW', '0.2'),
               ('LIQ', 'AMPLITUDE THRESHOLD', '25')]),

    'v7_step': dict(
        base='v4_walshapes',
        why="""Pileup resolution, which is the other lever on how much signal is
recovered -- and at early times it is the dominant one. The DREAM regression
measures an n_TOF rate of 13.76 hits/us in the 1-3 ms bin, i.e. a mean spacing
of 73 ns, against a wall pulse of 74 ns FWHM. The walls are therefore
self-piled-up exactly where the matcher is weakest.

The PSA guide's first practical advice: "Reducing the STEP SIZE -- even at the
price of worsening the signal-to-noise ratio in the derivative -- can often
help in resolving pileups."

WAL 8/7 -> 5/5, PSS 3/4 -> 2/3, LIQ 2/4 -> 2/3.""",
        edits=[('WAL', 'STEP SIZE', '5/5'),
               ('PSS', 'STEP SIZE', '2/3'),
               ('LIQ', 'STEP SIZE', '2/3')]),

    'v8_pssfit': dict(
        base='v4_walshapes',
        why="""Turn on pulse-shape fitting for the plastics (AMPLITUDE OPTION
1 -> 2) with the measured 101 ns averaged templates, one per amplitude regime.

The plastics currently use the parabolic-top option, i.e. no deconvolution at
all. They are the leg the wall AND plastic trigger is limited by, they are the
highest-rate tree in the file, and their pulse is 13 ns FWHM -- so pileup
resolution should be exactly where their remaining inefficiency lives. This is
the change the earlier rounds deferred as "riskier"; it gets its own variant so
a regression can be attributed.""",
        edits=[('PSS', 'AMPLITUDE OPTION', '2'),
               ('PSS', 'NUMBER OF PULSE SHAPES', '3')],
        shapes={'PSS': ['X17_{tree}_Signal_avg0.txt', 'X17_{tree}_Signal_avg1.txt',
                        'X17_{tree}_Signal_avg2.txt']}),

    'v9_liqaug': dict(
        base='v4_walshapes',
        why="""The liquid retry, done the other way round. Replacing the shipped
liquid templates lost twice (551 ns in v3_shapes, 81 ns in v5_liqshort), and
the length hypothesis is dead. The measured difference that survives is basis
diversity: the shipped pair is a normal pulse (LIQA_Signal_7, FWHM 7 ns) AND a
near-delta spike (LIQB_Signal_0, FWHM 1 ns), while every set I built spanned
only 5-7 ns.

So AUGMENT instead of replace: keep both shipped shapes and add one measured
per-detector average as a third. If this wins, diversity was the story; if it
is neutral, the liquids are limited by something other than the template and we
stop spending variants on it.""",
        edits=[('LIQ', 'NUMBER OF PULSE SHAPES', '3')],
        shapes={'LIQ': ['X17_LIQA_Signal_7.txt', 'X17_LIQB_Signal_0.txt',
                        'X17_{tree}_Signal_avg2.txt']}),
}


def edit_line(line, edits, shapes):
    toks = [(m.group(0), m.start(), m.end()) for m in re.finditer(r'\S+', line)]
    fam = toks[0][0][:3]
    out = line
    # apply column edits right-to-left so earlier spans stay valid
    for col, val in sorted(edits.get(fam, {}).items(),
                           key=lambda kv: IDX[kv[0]], reverse=True):
        i = IDX[col]
        if i >= len(toks):
            continue
        _, s, e = toks[i]
        out = out[:s] + val + out[e:]
    if fam in shapes:
        tree = toks[0][0]
        files = [f.format(tree=tree) for f in shapes[fam]]
        # replace everything from the first template address to end of line
        m = re.search(r'X17_\S+\.txt(\s+X17_\S+\.txt)*\s*$', out)
        if m:
            out = out[:m.start()] + ' '.join(files)
        else:
            out = out.rstrip() + '\t\t' + ' '.join(files)
    return out


def build(name, spec):
    base = (UI / spec['base'] / 'UserInput.h').read_text()
    edits, shapes = {}, spec.get('shapes', {})
    for fam, col, val in spec.get('edits', []):
        edits.setdefault(fam, {})[col] = val

    hdr = [f'# X17 EAR2 2026 -- variant {name}  =  {spec["base"]}  +']
    for fam in sorted(set(list(edits) + list(shapes))):
        for col, val in edits.get(fam, {}).items():
            hdr.append(f'#   {fam}*  {col} -> {val}')
        if fam in shapes:
            hdr.append(f'#   {fam}*  pulse shapes -> {len(shapes[fam])} '
                       f'({", ".join(shapes[fam])})')
    hdr.append('#')
    hdr += ['# ' + l if l else '#' for l in spec['why'].strip().splitlines()]
    hdr.append('#')

    lines, done_hdr = [], False
    for line in base.splitlines():
        if line.startswith('#'):
            continue                       # drop the base variant's rationale
        if not done_hdr and re.match(r'^(PKUP|SILI|WAL|PSS|LIQ)', line):
            lines += hdr
            done_hdr = True
        if re.match(r'^(WAL|PSS|LIQ)[ABCD]', line):
            line = edit_line(line, edits, shapes)
        lines.append(line)

    out = UI / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'UserInput.h').write_text('\n'.join(lines) + '\n')
    return out


def main():
    want = sys.argv[1:] or list(VARIANTS)
    for name in want:
        d = build(name, VARIANTS[name])
        print(f'wrote {d}/UserInput.h')
    return 0


if __name__ == '__main__':
    sys.exit(main())
