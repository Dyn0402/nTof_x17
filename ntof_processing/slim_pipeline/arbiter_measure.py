#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arbiter_measure.py -- the real-data measurement `coincidence_arbiter` needs.

The arbiter decides between candidate burst-to-pulse locks by asking, for each,
what fraction of a pulse's DREAM triggers actually have an n_TOF partner. That
question needs data, so it lives here rather than in the arbiter, which stays
IO-free and unit-testable.

WHY THIS NEEDS NO PRIOR KNOWLEDGE OF THE CLOCK. A correct lock puts every
trigger's partner at the SAME dt (T0, drifting slowly with K); a wrong lock
scatters them uniformly over the 80 ms burst. So the test is "is there a peak",
not "is there a peak at the expected place" -- and the peak's position is an
output, not an input. That is what lets the coincidence choose the lock instead
of merely confirming one after the clock has already been fitted around it.

COST. Only `n_sample` bunches are read, not the segment. At ~90 triggers per
pulse, eight pulses is ~700 chances to see a 96 % effect against a 0.05 %
background, which is why the arbiter's screens can usually stop after one call.

    python3 arbiter_measure.py <dream_run> <dream_subrun> <ntof_run> [--source D]
        -- resolve one segment and print the winning lock
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ntof_processing.slim_pipeline import clockfit as cf             # noqa: E402
from ntof_processing.slim_pipeline import coincidence_arbiter as CA  # noqa: E402
from ntof_processing.slim_pipeline import config as C                # noqa: E402
from ntof_processing.slim_pipeline.slim import (                     # noqa: E402
    OFFSET_BUNCHES, Segment, _bind_ntof, bunch_table, join_events,
    pass1_candidates)

QUIET = lambda *a, **k: None            # noqa: E731
# clock_qa accepts a fitted K anywhere in 0.9e-4..1.3e-4 against K_SEED = 1.1e-4,
# so the un-fitted walk reaches 80e6 ns * 2e-5 = 1600 ns. Everything inside this
# band is a candidate partner; the line fit below then removes the walk.


def _per_pulse(ev_b, hit):
    """Per-bunch fraction of triggers flagged, in bunch order."""
    order = np.argsort(ev_b, kind='stable')
    b_, h_ = ev_b[order], hit[order]
    starts = np.r_[0, np.flatnonzero(np.diff(b_)) + 1]
    tot = np.diff(np.r_[starts, b_.size])
    got = np.add.reduceat(h_.astype(np.int64), starts)
    return got / np.maximum(tot, 1)


# Bunches for the production measurer. NOT 8. `clockfit.bootstrap` needs
# BOOT_MIN_PEAK = 150 counts in its tallest 20 ns bin, and it histograms at
# K_SEED, so the un-fitted rate walk SMEARS the peak across bins before it is
# counted -- measured walks run to -168 ns/burst, which at 20 ns bins spreads
# ~650 matched pairs over ~8 bins and leaves ~80 counts, UNDER the floor. The
# failure would be "no peak", i.e. indistinguishable from a wrong lock: a
# correct lock refused because the sample was too small to bootstrap. 32
# bunches is 4x the counts (~2,880 triggers, comfortably over MIN_EVENTS = 500)
# and clears the floor even at the worst observed walk.
PROD_SAMPLE = 32


def make_measurer(dream_run, dream_subrun, ntof_run,
                             ntof_source=None, log=QUIET,
                             offset_bunches=OFFSET_BUNCHES):
    """`measure` built from the PRODUCTION chain, not a parallel model.

    Same signature and contract as `make_measurer`, so `coincidence_arbiter`
    cannot tell them apart -- but every physical term comes from the code that
    writes the products: `pass1_candidates` for the candidates, `fit_global`
    for K / T0 / per-arm offsets, `fit_perbunch` for the per-bunch clock, and
    `efficiency` for the matched mask. The per-pulse coincidence fraction is
    then read off that mask.

    WHY THIS EXISTS. `make_measurer` re-derives those same terms approximately,
    and every one of them cost a defect: the rate walk, the per-bunch scatter,
    the search centre, the top/bottom sample. Each was a term `clockfit`
    already carried correctly. Re-deriving physics that exists is how you get
    six corrections in two days.

    A WRONG LOCK FAILS HERE HONESTLY. `bootstrap` raises when there is no peak
    above its floor, which is exactly the state a wrong lock produces, so the
    rejection needs no threshold of its own -- and it is an INDEPENDENT check,
    not the same statistic re-read.
    """
    cal = {}

    def _tb(seg):
        if 'offs' in cal:
            return cal['offs']
        import time
        from ntof_dream_merge import dream_trigger as dt
        from ntof_dream_merge import fast_singles as fs
        import ntof_dream_merge.ntof_io as io
        fs.REPAIR_TFLASH = False
        pk = io.pkup_bunches(ntof_run)
        beam = ~(pk['intensity_e10'] < C.EMPTY_PULSE_E10)
        bn = pk['BunchNumber'][beam] if beam.any() else pk['BunchNumber']
        take = np.asarray(bn[:offset_bunches], np.int64)
        t0 = time.time()
        cal['offs'] = {a: fs.measure_tb_offsets(ntof_run, take, a)
                       for a in dt.ARMS}
        log(f'    tb offsets on {take.size} beam bunches '
            f'[{time.time()-t0:.0f} s]')
        return cal['offs']

    def measure(offset_s, n_sample):
        import time as _time
        _t0 = _time.time()
        seg = Segment(dream_run, dream_subrun, ntof_run,
                      ntof_source=ntof_source, accept_offset_s=offset_s)
        _bind_ntof(seg)
        try:
            ev = join_events(seg, log=QUIET)
        except (RuntimeError, ValueError, KeyError) as e:
            log(f'      join failed at {offset_s:+.2f}s after '
                f'{_time.time()-_t0:.0f} s: {type(e).__name__}: {e}')
            return float('nan'), 0, 0
        # empty pulses out, exactly as run_segment does -- their triggers are
        # background against a background trigger and can only dilute the score
        _tbl, keep = bunch_table(ev, log=QUIET)
        if len(ev) and not keep.all():
            ev = ev[keep].reset_index(drop=True)
        ev = ev[ev['is_flash'] == 0]
        if not len(ev):
            return float('nan'), 0, 0
        b_all = ev['BunchNumber'].to_numpy().astype(np.int64)
        ub = np.unique(b_all)
        if ub.size == 0:
            return float('nan'), 0, 0
        n = max(int(n_sample), 1)
        take = np.unique(ub[np.linspace(0, ub.size - 1,
                                        min(n, ub.size)).astype(int)])
        m = np.isin(b_all, take)
        ev_b = b_all[m]
        ev_t = ev['t_since_flash_ns'].to_numpy()[m]
        seg.bunches = take
        try:
            cd, _o, _thr = pass1_candidates(seg, take, log=QUIET,
                                            offsets=_tb(seg))
            K, T0, arm_off, _gi = cf.fit_global(ev_b, ev_t, cd['bunch'],
                                                cd['t'], cd['arm'], log=QUIET)
            corr_in, _cv, _pb = cf.fit_perbunch(ev_b, ev_t, cd['bunch'],
                                                cd['t'], cd['arm'], K, T0,
                                                arm_off, log=QUIET)
            qa = cf.efficiency(ev_b, ev_t, cd['bunch'], cd['t'], cd['arm'],
                               K, T0, arm_off, corr_in, C.ACCEPT_NS)
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            # `bootstrap` raising "no peak" IS the wrong-lock verdict, so this
            # is a measured 0, not an un-measurement. Narrow on purpose: a
            # NameError here must crash, not read as a rejection.
            log(f'      no clock at {offset_s:+.2f}s after '
                f'{_time.time()-_t0:.0f} s: {type(e).__name__}: {e}')
            return 0.0, int(take.size), int(ev_t.size)
        per = _per_pulse(ev_b, qa['matched'])
        frac = float(np.median(per))
        log(f'      {offset_s:+.2f}s: K {K:.6e} T0 {T0:+.1f} ns, '
            f'efficiency {qa["efficiency"]:.1%}, accidental '
            f'{qa["accidental"]:.3%}, median pulse {frac:.0%} over {per.size} '
            f'pulses; {_time.time()-_t0:.0f} s')
        return frac, int(per.size), int(ev_t.size)

    return measure


def resolve(dream_run, dream_subrun, ntof_run, ntof_source=None, log=print):
    """Candidate locks from pulse_match, then let the coincidence choose."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                           'ntof_july_analysis'))
    import pulse_match as pm

    try:
        d = pm.match_subrun(dream_run, dream_subrun)
        locks = d.get('locks') or [dict(off_s=d['offset_s'],
                                        n=d.get('n_matched', 0), r=None)]
        log(f'  pulse_match resolved it itself at {d["offset_s"]:+.3f}s '
            f'({d.get("lock_chosen_by")})')
    except pm.AmbiguousLock as e:
        locks = getattr(e, 'locks', None) or _locks_from_text(str(e))
        log(f'  pulse_match refused; {len(locks)} candidate lock(s) to test')
    except Exception as e:
        log(f'  pulse_match failed: {type(e).__name__}: {e}')
        return CA.Verdict(reason=f'{type(e).__name__}')
    measure = make_measurer(dream_run, dream_subrun, ntof_run, ntof_source,
                            log=log)
    return CA.arbitrate(locks, measure, log=log)


def _locks_from_text(msg):
    """Recover the candidate list from the AmbiguousLock message. FALLBACK ONLY.

    `AmbiguousLock` now carries `.locks`, and so does its cache entry, so this
    runs only against refusals cached before 2026-08-13.

    `r` MUST accept `nan` AND a minus sign. `select_lock` formats it with
    `:.3f`, so an undefined correlation prints `r=nan` and an anti-correlated
    lock prints `r=-0.204`; a bare `[\\d.]+` matches neither and skips those
    candidates entirely. Short segments are both the ambiguous ones and the
    ones where the correlation is undefined (below ~200 clusters it has no
    power in principle), so the strict pattern dropped exactly the candidates
    that most needed arbitrating -- silently, leaving the arbiter to report
    "no candidate locks" for a segment that had several.

    The numeric branch is anchored rather than greedy: the lock table is
    followed by `. This sub-run needs...`, and `[\\d.]+` swallows that full
    stop and then fails to parse.
    """
    import re
    out = []
    for m in re.finditer(
            r'([+-]?\d+\.?\d*)s n=(\d+) r=(nan|-?\d*\.?\d+)', msg):
        r = m.group(3)
        out.append(dict(off_s=float(m.group(1)), n=int(m.group(2)),
                        r=None if r == 'nan' else float(r)))
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('dream_run')
    ap.add_argument('dream_subrun')
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('--source', default=None)
    a = ap.parse_args()
    v = resolve(a.dream_run, a.dream_subrun, a.ntof_run, a.source)
    print(f'\nVERDICT: {"accept " + format(v.offset_s, "+.3f") + " s" if v.ok else "REFUSE"}'
          f'  ({v.reason}; {v.tested} measurement(s))')
    return 0 if v.ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
