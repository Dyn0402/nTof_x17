#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for pulse_match.select_lock — the 2026-08-12 fix.

The failure being guarded against (ntof_processing/join_mislock/): the
count-only offset scan is degenerate under the supercycle, tied between the
true lock and a shifted one, and silently kept the most negative — 25.7 % of
the July campaign beam. These cases assert the three behaviours that replace
it: count wins when it can, intensity fluctuations arbitrate near-ties, and
anything the instruments cannot separate RAISES.

Synthetic beam: pulses on a supercycle-periodic schedule (so shifted locks
tie on count by construction), with per-pulse intensity fluctuations that
only the true alignment can reproduce in the cluster sizes.

Run: ../.venv/bin/python test_pulse_match_locks.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ntof_july_analysis import pulse_match as pm             # noqa: E402

rng = np.random.default_rng(20260812)

SC_S = 39.6                  # supercycle
PATTERN_S = [0.0, 4.8, 9.6, 18.0, 26.4, 31.2]   # pulses per cycle
N_CYCLES = 80                # ~53 min of beam
ANCHOR = 1_785_000_000.0
TRUE_OFF = +37.3             # truth, off the minute anchor as in real life


def make_beam(fluct=0.08):
    """(pt, pe): supercycle-periodic pulse train with intensity fluctuations.
    The SCHEDULE intensities repeat per cycle (dedicated/parasitic pattern);
    the fluctuations are per-pulse and unique."""
    base = np.array([850.0, 420.0, 850.0, 420.0, 850.0, 420.0])
    pt, pe = [], []
    for c in range(N_CYCLES):
        for k, s in enumerate(PATTERN_S):
            pt.append(c * SC_S + s)
            pe.append(base[k] * (1.0 + fluct * rng.standard_normal()))
    pt = np.array(pt) + ANCHOR + TRUE_OFF
    return pt, np.array(pe)


def make_clusters(pt, pe, coupling=1.0):
    """DREAM clusters at the true pulse times, sizes tracking intensity."""
    c_t = pt - ANCHOR - TRUE_OFF + rng.normal(0, 0.005, len(pt))
    # cluster size = detector response ~ intensity (+ Poisson-ish noise)
    sizes = np.maximum(1, (0.11 * (coupling * pe
                                   + (1 - coupling) * pe.mean())
                           + rng.normal(0, 2.0, len(pe)))).astype(float)
    return c_t, sizes


def expect(name, cond):
    print(f'  {"PASS" if cond else "FAIL"}  {name}')
    if not cond:
        sys.exit(1)


def main():
    print('1. clean beam, unique fluctuations -> count picks truth outright')
    pt, pe = make_beam(fluct=0.30)
    # break the count tie: drop two pulses from one cycle so the true lock
    # matches every cluster and shifted locks miss the orphans
    keep = np.ones(len(pt), bool)
    keep[[13, 14]] = False
    c_t, sizes = make_clusters(pt, pe)
    off, locks, diag = pm.select_lock(c_t, sizes, ANCHOR, pt[keep], pe[keep])
    expect('offset within 0.1 s of truth', abs(off - TRUE_OFF) < 0.1)
    expect('multiple supercycle locks seen', len(locks) >= 3)

    print('2. perfect periodic tie, strong fluctuations -> intensity '
          'arbitration picks truth')
    pt, pe = make_beam(fluct=0.10)
    c_t, sizes = make_clusters(pt, pe)
    off, locks, diag = pm.select_lock(c_t, sizes, ANCHOR, pt, pe)
    expect('offset within 0.1 s of truth', abs(off - TRUE_OFF) < 0.1)
    expect("chosen_by == 'intensity'", diag['chosen_by'] == 'intensity')
    expect('r_sig recorded and >= threshold',
           diag['r_sig'] is not None and diag['r_sig'] >= pm.R_SIG)

    print('3. tie with NO usable fluctuations -> AmbiguousLock, never a '
          'silent pick (the margin-0-but-correct case must not be decided '
          'by scan order)')
    pt, pe = make_beam(fluct=0.0)          # schedule only, repeats exactly
    c_t, sizes = make_clusters(pt, pe, coupling=0.0)   # sizes uncorrelated
    try:
        off, locks, diag = pm.select_lock(c_t, sizes, ANCHOR, pt, pe)
        expect('raised AmbiguousLock', False)
    except pm.AmbiguousLock:
        expect('raised AmbiguousLock', True)

    print('4. beam record does not cover the hour -> NoLock, never the '
          'scan edge (the old code returned offset = -120.000 with 0 '
          'matches)')
    pt, pe = make_beam()
    c_t, sizes = make_clusters(pt, pe)
    far = pt + 9_000.0                      # pulses exist, hours away
    try:
        pm.select_lock(c_t, sizes, ANCHOR, far, pe)
        expect('raised NoLock', False)
    except pm.NoLock:
        expect('raised NoLock', True)

    print('5. schedule irregularity -> count wins without touching r')
    # a genuine count margin needs APERIODICITY, not missing pulses:
    # removing pulses starves every periodic lock equally (that symmetry is
    # the degeneracy). Append an irregular 25-pulse stretch that only the
    # true alignment can match.
    pt, pe = make_beam(fluct=0.30)
    t_irr = pt[-1] + np.cumsum(rng.uniform(2.0, 7.5, 25))
    pt = np.r_[pt, t_irr]
    pe = np.r_[pe, 600.0 * (1 + 0.3 * rng.standard_normal(25))]
    c_t, sizes = make_clusters(pt, pe)
    off, locks, diag = pm.select_lock(c_t, sizes, ANCHOR, pt, pe)
    expect("chosen_by == 'count'", diag['chosen_by'] == 'count')
    expect('margin at least MARGIN_CLEAR', diag['margin'] >= pm.MARGIN_CLEAR)
    expect('offset within 0.1 s of truth', abs(off - TRUE_OFF) < 0.1)

    print('all cases pass')


if __name__ == '__main__':
    main()
