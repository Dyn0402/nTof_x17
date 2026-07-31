#!/usr/bin/env python3
"""Would an AMPLITUDE-binned liquid template basis buy anything?

`is_it_photon_statistics.py` shows two things at once: `resid/sqrt(A)` is flat
at low amplitude (photon statistics, irreducible) but rises above ~6000 ADC,
and the residual becomes increasingly COHERENT there -- the same shape in every
pulse, which is by definition something a template could absorb. LIQB is the
worst case (resid/sqrt(A) 0.62 -> 1.59, coherence 7x random).

Coherence says a shape is *present*. It does not say a basis can *capture* it,
because a coherent residual can also come from a common contamination (residual
pileup, baseline droop) that no legitimate template should learn. This scores
the thing itself: fit with one template, then with one template per amplitude
octile, always on a HELD-OUT half so a richer basis cannot win by memorising.

Decision this feeds (../archive/PRE_SHIP_TESTS.md T6): a large held-out gain would reopen
the liquid template question for the affected detectors. A small one confirms
that the floor is real and the shipped templates stay.

    python amp_binned_basis.py <liq_pulses.npz> [TREE ...]
"""
import sys

import numpy as np

PRE = 20
TAIL = slice(PRE + 12, PRE + 180)


def fit_resid(pulses, amps, template):
    """Residual in ADC units for an amplitude-free fit at the best integer shift."""
    out = []
    for p, a in zip(pulses, amps):
        best, br = np.inf, None
        for t0 in (-1, 0, 1):
            mm = np.roll(template, t0)
            sc = (p * mm).sum() / (mm * mm).sum()
            r = p - sc * mm
            v = float((r * r).sum())
            if v < best:
                best, br = v, r
        out.append(br[TAIL] * a)
    return np.array(out)


def main():
    z = np.load(sys.argv[1])
    trees = sys.argv[2:] or sorted(k[:-5] for k in z.files if k.endswith('_rows'))
    rng = np.random.default_rng(20260729)
    print(f'{"tree":6s} {"n(held-out)":>12s} {"1 template":>11s} '
          f'{"8 by amplitude":>15s} {"gain":>7s}')
    print('-' * 56)
    for tree in trees:
        rows, amps = z[f'{tree}_rows'], z[f'{tree}_amps']
        keep = (amps > 800) & (rows.min(axis=1) > -0.5)   # drop under-range wraps
        rows, amps = rows[keep], amps[keep]
        if len(rows) < 200:
            print(f'{tree:6s} {len(rows):12d}   too few')
            continue
        half = rng.random(len(rows)) < 0.5
        tr_r, tr_a = rows[half], amps[half]
        te_r, te_a = rows[~half], amps[~half]

        med = np.median(tr_r, axis=0)
        one = med / med.max()
        r1 = fit_resid(te_r, te_a, one)

        # one template per amplitude octile of the TRAINING half; each test pulse
        # is fitted with the template of the octile its own amplitude falls in
        edges = np.percentile(tr_a, np.linspace(0, 100, 9))
        edges[0], edges[-1] = -np.inf, np.inf
        r8 = np.empty_like(r1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            s_tr = (tr_a >= lo) & (tr_a < hi)
            s_te = (te_a >= lo) & (te_a < hi)
            if s_tr.sum() < 20 or not s_te.any():
                if s_te.any():
                    r8[s_te] = r1[s_te]
                continue
            m = np.median(tr_r[s_tr], axis=0)
            r8[s_te] = fit_resid(te_r[s_te], te_a[s_te], m / m.max())

        a1 = float(np.sqrt(np.mean(r1 ** 2)))
        a8 = float(np.sqrt(np.mean(r8 ** 2)))
        print(f'{tree:6s} {(~half).sum():12d} {a1:11.1f} {a8:15.1f} '
              f'{a8 / a1 - 1:+6.1%}')

        # where the gain (if any) lives: the top amplitude octile is where the
        # coherent term was, so quote it separately from the bulk
        hi_cut = np.percentile(te_a, 75)
        for lab, m in (('  bulk (amp<p75)', te_a < hi_cut),
                       ('  top   (amp>p75)', te_a >= hi_cut)):
            b1 = float(np.sqrt(np.mean(r1[m] ** 2)))
            b8 = float(np.sqrt(np.mean(r8[m] ** 2)))
            print(f'{lab:6s} {m.sum():10d} {b1:11.1f} {b8:15.1f} '
                  f'{b8 / b1 - 1:+6.1%}')
    print('\nA gain of a few per cent is the photon-statistics floor asserting '
          'itself:\nthere is no shape left to learn. A large gain concentrated '
          'in the top octile\nwould mean the high-amplitude pulses are '
          'genuinely a different shape.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
