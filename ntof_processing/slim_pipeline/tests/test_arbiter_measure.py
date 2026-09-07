#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the measurer feed the PRODUCTION clock fit correctly?

`arbiter_measure.make_measurer` no longer models any physics of its own. It
joins at a candidate offset, samples bunches, and hands them to
`pass1_candidates` -> `fit_global` -> `fit_perbunch` -> `efficiency`, then reads
the per-pulse coincidence fraction off the matched mask. Everything the earlier
parallel model got wrong -- the rate walk, per-bunch da/dk, per-arm offsets, the
search centre -- is a term `clockfit` already carries, and re-deriving them cost
six corrections in two days before that model was deleted on 2026-08-13.

So what is left to test here is the PLUMBING, which is where the remaining
failure modes live:

  * the top/bottom calibration must not be measured on the scoring sample
    (it needs ~1e5 late-hit pairs; on a handful of bunches it lands on noise);
  * it must be measured ONCE per n_TOF run, not once per candidate lock;
  * it must be drawn from BEAM bunches;
  * a candidate the join refuses must report NOT MEASURED, never a measured
    zero -- the two are different answers and only one is evidence;
  * candidate recovery from a cached refusal must not silently drop locks;
  * the arbiter must never hand the fit a sample too small to bootstrap.

Everything below `measure` is stubbed -- no EOS, no n_TOF files, no clockfit --
because that physics is production code with its own tests, and duplicating it
here would recreate the very parallel-model problem the deletion removed.

    python3 test_arbiter_measure.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ntof_dream_merge import dream_trigger as dt          # noqa: E402
from ntof_dream_merge import fast_singles as fs           # noqa: E402
import ntof_dream_merge.ntof_io as io                     # noqa: E402

from ntof_processing.slim_pipeline import arbiter_measure as AM  # noqa: E402
from ntof_processing.slim_pipeline import clockfit as cf         # noqa: E402
from ntof_processing.slim_pipeline import config as C            # noqa: E402
from ntof_processing.slim_pipeline import coincidence_arbiter as CA  # noqa: E402

N_BUNCH = 400            # synthetic segment size
N_TRIG = 90              # triggers per pulse, as the real segments carry
TRUE_OFFSET_S = +27.9175  # run_79/stat090_0000's known-correct lock


class Spy:
    """Records what the stubs were asked for."""

    def __init__(self):
        self.tb_calls = []       # bunch-sample size of each tb measurement
        self.tb_bunches = []     # the bunch numbers it was handed
        self.cand_calls = []     # (n_bunches, offsets-or-None) per pass-1 call
        self.fit_calls = 0       # how often the production fit was invoked


def install(spy, monkey, join_raises=None, matched_frac=0.96):
    """Stub everything below `measure`: EOS, the n_TOF reader, and clockfit."""

    def put(obj, name, value):
        monkey.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    # --- n_TOF side -------------------------------------------------------
    def pkup_bunches(run):
        n = 3000
        inten = np.full(n, 60.0)
        inten[::7] = 0.0                       # empty pulses, must be skipped
        return dict(BunchNumber=np.arange(1, n + 1, dtype=np.int64),
                    intensity_e10=inten)

    def measure_tb_offsets(run, bunches, arm, **kw):
        spy.tb_calls.append(len(bunches))
        spy.tb_bunches.append(np.asarray(bunches))
        return {g: 34.0 for g in range(4)}

    put(io, 'pkup_bunches', pkup_bunches)
    put(fs, 'measure_tb_offsets', measure_tb_offsets)

    # --- slim side --------------------------------------------------------
    put(AM, '_bind_ntof', lambda seg: (io, []))

    def join_events(seg, log=None, events=None):
        if join_raises is not None:
            raise join_raises
        b = np.repeat(np.arange(1, N_BUNCH + 1, dtype=np.int64), N_TRIG)
        t = np.tile(np.arange(N_TRIG, dtype=float) * 1e5, N_BUNCH)
        ev = pd.DataFrame(dict(BunchNumber=b, is_flash=np.zeros(b.size, int),
                               t_since_flash_ns=t,
                               bunch_intensity_e10=np.full(b.size, 60.0)))
        if seg.bunches is not None:
            ev = ev[ev['BunchNumber'].isin(seg.bunches)].reset_index(drop=True)
        return ev

    def bunch_table(ev, log=None):
        return {}, np.ones(len(ev), bool)

    def pass1_candidates(seg, bunches, log=None, offsets=None):
        spy.cand_calls.append((len(bunches), offsets))
        n = len(bunches) * N_TRIG
        return (dict(bunch=np.repeat(np.asarray(bunches), N_TRIG),
                     t=np.arange(n, dtype=float) * 1e3,
                     arm=(np.arange(n) % 4).astype(np.int8)), None, None)

    put(AM, 'join_events', join_events)
    put(AM, 'bunch_table', bunch_table)
    put(AM, 'pass1_candidates', pass1_candidates)

    # --- the production clock fit -----------------------------------------
    def fit_global(ev_b, ev_t, cb, ct, ca, log=None):
        spy.fit_calls += 1
        return 1.1e-4, -250.0, np.zeros(4), {}

    def fit_perbunch(ev_b, ev_t, cb, ct, ca, K, T0, off, log=None):
        return None, None, {}

    def efficiency(ev_b, ev_t, cb, ct, ca, K, T0, off, corr, win):
        m = np.zeros(ev_b.size, bool)
        keep = int(round(matched_frac * N_TRIG))
        for u in np.unique(ev_b):
            idx = np.flatnonzero(ev_b == u)
            m[idx[:keep]] = True
        return dict(matched=m, efficiency=float(m.mean()), accidental=0.0005)

    put(cf, 'fit_global', fit_global)
    put(cf, 'fit_perbunch', fit_perbunch)
    put(cf, 'efficiency', efficiency)


def run_case(fn, **kw):
    spy, monkey = Spy(), []
    install(spy, monkey, **kw)
    try:
        return fn(spy)
    finally:
        for obj, name, orig in reversed(monkey):
            setattr(obj, name, orig)


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

def case_production_fit_is_what_scores(spy):
    """The score must come from clockfit's matched mask, not a local model."""
    m = AM.make_measurer('run_79', 'stat090_0000', 224572, log=AM.QUIET)
    frac, npulse, ntrig = m(TRUE_OFFSET_S, AM.PROD_SAMPLE)
    assert spy.fit_calls == 1, (
        f'fit_global called {spy.fit_calls} times; the measurer must run the '
        f'PRODUCTION fit exactly once per candidate')
    assert abs(frac - 0.96) < 0.02, (
        f'scored {frac:.2f}; should be the matched fraction clockfit reported')
    assert npulse > 0 and ntrig > 0, f'{npulse} pulses, {ntrig} triggers'
    return f'{frac:.0%} of {npulse} pulses, straight off the matched mask'


def case_tb_sample_is_not_the_scoring_sample(spy):
    """Scoring 32 pulses must not calibrate the wall on 32 bunches.

    The top/bottom offsets are ~0 or +-32..40 ns instrumental constants needing
    ~1e5 late-hit pairs; on the scoring sample they land on noise, and
    `dream_trigger` documents that a wrong pairing window keeps only 27.6 % of
    genuine pairs. That gutted the wall leg and scored known-correct locks at
    0 % (2026-08-13).
    """
    m = AM.make_measurer('run_79', 'stat090_0000', 224572, log=AM.QUIET)
    m(TRUE_OFFSET_S, AM.PROD_SAMPLE)
    assert spy.tb_calls, 'the offsets were never measured at all'
    n_tb = spy.tb_calls[0]
    assert n_tb == AM.OFFSET_BUNCHES, (
        f'tb offsets measured on {n_tb} bunches, not OFFSET_BUNCHES='
        f'{AM.OFFSET_BUNCHES}')
    n_score, offs = spy.cand_calls[0]
    assert n_score <= AM.PROD_SAMPLE, f'scored on {n_score} bunches'
    assert offs is not None, 'pass1_candidates was left to measure them itself'
    return f'calibrated on {n_tb}, scored on {n_score}'


def case_calibration_is_once_per_segment(spy):
    """Four candidate locks must not pay four calibrations."""
    m = AM.make_measurer('run_96', 'stat090_0001', 224597, log=AM.QUIET)
    for off in (+50.72, -69.32, +12.0, -3.5):
        m(off, AM.PROD_SAMPLE)
    n_cal = len(spy.tb_calls) / len(dt.ARMS)     # one measurement per arm
    assert n_cal == 1, (
        f'{n_cal:g} calibrations for 4 candidates; the offsets are read from '
        f'n_TOF alone and do not depend on the lock')
    assert len(spy.cand_calls) == 4, f'{len(spy.cand_calls)} pass-1 calls'
    assert len({id(o) for _, o in spy.cand_calls}) == 1, \
        'each candidate got a different offset table'
    return f'1 calibration, {len(spy.cand_calls)} scored candidates'


def case_calibration_sample_is_beam_bunches(spy):
    """An empty pulse has no flash for a late hit to be late of."""
    m = AM.make_measurer('run_79', 'stat090_0000', 224572, log=AM.QUIET)
    m(TRUE_OFFSET_S, AM.PROD_SAMPLE)
    empty = np.arange(1, 3001, dtype=np.int64)[::7]   # the stub's empties
    bad = np.intersect1d(spy.tb_bunches[0], empty)
    assert bad.size == 0, f'{bad.size} empty pulses in the calibration sample'
    assert spy.tb_calls[0] == AM.OFFSET_BUNCHES, 'short calibration sample'
    return f'{spy.tb_calls[0]} beam bunches, 0 empty'


def case_join_refusal_is_not_a_measured_zero(spy):
    """NOT MEASURED must be NaN, never 0.0.

    Returning 0.0 when the join refused made an un-joinable candidate
    indistinguishable from one whose pulses genuinely show no coincidence. On
    the 2026-08-13 campaign 16 of the first 20 candidates never reached the
    measurement at all, and every one was reported as a rejection on the
    physics.
    """
    m = AM.make_measurer('run_79', 'stat090_0000', 224572, log=AM.QUIET)
    frac, npulse, ntrig = m(TRUE_OFFSET_S, AM.PROD_SAMPLE)
    assert frac != frac, f'join refusal scored {frac!r}, expected NaN'
    assert (npulse, ntrig) == (0, 0), f'{npulse} pulses, {ntrig} triggers'
    v = CA.arbitrate([dict(off_s=TRUE_OFFSET_S, n=40, r=0.9)],
                     lambda o, n: (float('nan'), 0, 0), log=lambda *a: None)
    assert not v.ok and v.rejected == [] and len(v.unmeasured) == 1, \
        f'arbiter mis-filed it: rejected={v.rejected} unmeasured={v.unmeasured}'
    return 'NaN, and the arbiter files it as unmeasured'


def case_nan_correlation_locks_survive_text_recovery(spy):
    """A candidate with an undefined or negative r must not vanish.

    `select_lock` formats r with `:.3f`, so an undefined correlation prints
    `r=nan` and an anti-correlated lock prints `r=-0.204`; a numeric-only
    pattern matched neither and skipped those candidates. Short segments are
    exactly the ambiguous ones AND the ones where r is undefined, so the locks
    most needing arbitration were the ones dropped. Only pre-2026-08-13 cached
    refusals take this path -- AmbiguousLock now carries `.locks`.
    """
    msg = ('count margin 2 (< 10) and intensity correlation cannot separate '
           'the top locks (r_sig None, need 3): '
           '+50.72s n=41 r=nan; -69.32s n=41 r=-0.204; +12.10s n=33 r=0.512. '
           'This sub-run needs a bunch-shift scan.')
    got = AM._locks_from_text(msg)
    assert len(got) == 3, f'recovered {len(got)} of 3 candidates'
    assert sorted(round(g['off_s'], 2) for g in got) == [-69.32, 12.10, 50.72]
    assert [g['r'] for g in got if abs(g['off_s'] - 12.10) < 0.01] == [0.512], \
        'trailing punctuation broke r'
    assert [g['r'] for g in got if abs(g['off_s'] + 69.32) < 0.01] == [-0.204], \
        'a negative correlation was dropped'
    assert [g['r'] for g in got if abs(g['off_s'] - 50.72) < 0.01] == [None], \
        'an undefined r did not come back as None'
    ranked = CA.rank_candidates(got)
    assert any(abs(l['off_s'] - 50.72) < 0.01 for l in ranked), \
        'the r=nan lock was dropped from the ranking'
    return '3 candidates recovered, all 3 ranked'


def case_sample_floor_protects_the_bootstrap(spy):
    """The arbiter must not hand the production fit a sample it cannot boot.

    `clockfit.bootstrap` needs BOOT_MIN_PEAK counts in its tallest 20 ns bin and
    histograms at K_SEED, so the un-fitted walk smears the peak before it is
    counted. Under-sampling makes a CORRECT lock raise "no peak", which is
    indistinguishable from a wrong one -- the worst available failure direction.
    """
    assert CA.SAMPLE_SMALL >= AM.PROD_SAMPLE, (
        f'arbiter would call the production measurer with '
        f'{CA.SAMPLE_SMALL} bunches, below its {AM.PROD_SAMPLE} floor')
    assert CA.SAMPLE_LARGE >= CA.SAMPLE_SMALL, 'resample is smaller than first'
    n_trig = CA.SAMPLE_SMALL * N_TRIG
    assert n_trig >= C.MIN_EVENTS, (
        f'{n_trig} triggers is below production MIN_EVENTS={C.MIN_EVENTS}')
    return (f'{CA.SAMPLE_SMALL} bunches ~ {n_trig:,} triggers, over '
            f'MIN_EVENTS={C.MIN_EVENTS} and the peak floor')


CASES = [
    ('the production fit is what scores', case_production_fit_is_what_scores, {}),
    ('tb sample is not the scoring sample',
     case_tb_sample_is_not_the_scoring_sample, {}),
    ('calibration is once per segment, not per lock',
     case_calibration_is_once_per_segment, {}),
    ('calibration sample is beam bunches',
     case_calibration_sample_is_beam_bunches, {}),
    ('a join refusal is NOT a measured zero',
     case_join_refusal_is_not_a_measured_zero,
     dict(join_raises=RuntimeError('delta ambiguous'))),
    ('nan/negative-r locks survive text recovery',
     case_nan_correlation_locks_survive_text_recovery, {}),
    ('the sample floor protects the bootstrap',
     case_sample_floor_protects_the_bootstrap, {}),
]


def main() -> int:
    fails = []
    for name, fn, kw in CASES:
        try:
            detail = run_case(fn, **kw)
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
