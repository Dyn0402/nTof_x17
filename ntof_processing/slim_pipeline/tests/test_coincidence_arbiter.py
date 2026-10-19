#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the arbiter pick the right lock, refuse when it should, and stay cheap?

The measurement it arbitrates on is injected, so these cases are exact: a
"right" lock returns the coincidence fraction the campaign measures (96 %), a
"wrong" one returns the accidental rate (~0 %). What is under test is the
decision logic and the cost, not the physics.

    python3 test_coincidence_arbiter.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ntof_processing.slim_pipeline.coincidence_arbiter import (   # noqa: E402
    ACCEPT_FRAC, arbitrate, rank_candidates)


def locks(*spec):
    """spec: (off_s, n, r) triples, in the order pulse_match found them."""
    return [dict(off_s=o, n=n, r=r) for o, n, r in spec]


def measurer(truth, calls, ambiguous=()):
    """0.96 at the true lock, ~0 elsewhere; `ambiguous` locks return 0.5."""
    def m(off, n):
        calls.append((off, n))
        if off in ambiguous:
            return (0.50, n, n * 90)
        return (0.962, n, n * 90) if off == truth else (0.0005, n, n * 90)
    return m


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case('picks the true lock when it is ranked first')
def _():
    calls = []
    v = arbitrate(locks((50.72, 788, 0.931), (-69.32, 788, 0.902)),
                  measurer(50.72, calls), log=lambda *a: None)
    assert v.ok and v.offset_s == 50.72, v
    assert v.tested == 1, f'should stop at the first pass, tested {v.tested}'
    return f'accepted {v.offset_s:+.2f}s after {v.tested} test'


@case('finds the true lock when the ranking MISLEADS')
def _():
    # Below ~200 clusters the intensity correlation is noise, so it can rank a
    # wrong lock above the right one. That is the case the coincidence has to
    # rescue, and the one where every cheap screen has already failed.
    calls = []
    v = arbitrate(locks((-69.32, 788, 0.931), (50.72, 788, 0.902)),
                  measurer(50.72, calls), log=lambda *a: None)
    assert v.ok and v.offset_s == 50.72, v
    assert len(v.rejected) == 1, v.rejected
    assert v.rejected[0][0] == -69.32
    return f'rejected {v.rejected[0][0]:+.2f}s, accepted {v.offset_s:+.2f}s'


@case('refuses when NO lock shows the coincidence')
def _():
    v = arbitrate(locks((10.0, 500, 0.5), (20.0, 495, 0.5)),
                  measurer(999.0, []), log=lambda *a: None)
    assert not v.ok, v
    assert 'no candidate lock reached' in v.reason
    return v.reason


@case('drops hopeless locks without testing them')
def _():
    calls = []
    v = arbitrate(locks((1.0, 800, 0.9), (2.0, 100, 0.9), (3.0, 50, 0.9)),
                  measurer(1.0, calls), log=lambda *a: None)
    assert v.ok
    assert all(off == 1.0 for off, _ in calls), calls
    return f'{len(calls)} measurement(s) for 3 candidates'


@case('an ambiguous result pays for a bigger sample, once')
def _():
    calls = []
    v = arbitrate(locks((7.0, 300, 0.6), (8.0, 300, 0.6)),
                  measurer(8.0, calls, ambiguous=(7.0,)),
                  log=lambda *a: None)
    assert v.ok and v.offset_s == 8.0
    sizes = [n for off, n in calls if off == 7.0]
    assert len(sizes) == 2 and sizes[1] > sizes[0], sizes
    return f'resampled the ambiguous lock at {sizes[1]} pulses, then moved on'


@case('empty candidate list is refused, not crashed')
def _():
    v = arbitrate([], measurer(1.0, []), log=lambda *a: None)
    assert not v.ok and 'no candidate' in v.reason
    return v.reason


@case('ranking prefers count, then the intensity correlation')
def _():
    r = rank_candidates(locks((1.0, 500, 0.10), (2.0, 500, 0.90),
                              (3.0, 600, 0.05)))
    assert [x['off_s'] for x in r] == [3.0, 2.0, 1.0], r
    return 'count first, r as tie-break'


@case('an unmeasurable candidate is not a rejection')
def _():
    """A candidate the join refused is NOT evidence against that lock.

    Measured 2026-08-13: 16 of the first 20 candidates on the unmatched
    campaign never reached the measurement (pulse_match's confident-selection
    guard, and bunch_join's delta-scan ambiguity). Reporting those as
    "coincidence 0 %" blames the physics for an upstream blockage.
    """
    calls = []

    def m(off, n):
        calls.append((off, n))
        return float('nan'), 0, 0        # the join refused every candidate

    v = arbitrate(locks((+50.72, 40, 0.9), (-69.32, 40, 0.9)), m, log=lambda *a: None)
    assert not v.ok, 'an unmeasurable candidate must not be accepted'
    assert v.rejected == [], f'unmeasurable landed in rejected: {v.rejected}'
    assert len(v.unmeasured) == 2, f'unmeasured = {v.unmeasured}'
    assert 'never got to decide' in v.reason, f'reason blames physics: {v.reason}'
    return f'2 unmeasurable, 0 rejected, reason names the blockage'


@case('an unmeasurable candidate does not hide a good lock')
def _():
    """One candidate unmeasurable, the next one right -> still accepted."""
    def m(off, n):
        if abs(off + 69.32) < 0.01:
            return float('nan'), 0, 0
        return (0.95, 8, 700) if abs(off - 50.72) < 0.01 else (0.001, 8, 700)

    v = arbitrate(locks((-69.32, 41, 0.91), (+50.72, 40, 0.90)), m,
                  log=lambda *a: None)
    assert v.ok and abs(v.offset_s - 50.72) < 0.01, f'picked {v.offset_s}'
    assert v.unmeasured == [-69.32], f'unmeasured = {v.unmeasured}'
    return f'skipped the unmeasurable, accepted {v.offset_s:+.2f}'


def main() -> int:
    fails = []
    for name, fn in CASES:
        try:
            detail = fn()
        except AssertionError as e:
            fails.append(f'{name}: {e}')
            print(f'  FAIL  {name}\n        {e}')
        else:
            print(f'  ok    {name:52s} {detail}')
    print('-' * 78)
    if fails:
        print(f'{len(fails)} FAILURE(S)')
        return 1
    print(f'all {len(CASES)} cases passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
