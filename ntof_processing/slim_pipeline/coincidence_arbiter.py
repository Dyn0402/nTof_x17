#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coincidence_arbiter.py -- decide a burst-to-pulse lock by the COINCIDENCE, not
by cluster counts.

The campaign's matching failures all have the same shape: `pulse_match` picks
the lock by counting DREAM clusters against beam pulses at ~50 ms tolerance,
several candidate locks tie on that count, and the tie is broken by something
that carries no information (scan order, or an intensity correlation that is
flat below ~200 clusters). The count is a weak instrument used to SEARCH; the
25 ns wall-plastic coincidence is a decisive instrument used only to CONFIRM,
downstream, after a lock has already been committed to.

That ordering is a historical artifact, not a constraint. This module inverts
it. Measured over 209,316 beam pulses of the 2026-08-12 campaign:

    right lock -> 96.2 % of a pulse's triggers have a wall AND plastic hit
                  on the same arm within the accept window (5th pct 89 %)
    wrong lock -> the accidental rate, ~0.05 %

Three orders of magnitude. Nothing else in this pipeline separates that well,
so it should be what chooses the lock.

COST. The coincidence test needs n_TOF hits, so it is not free -- but it does
not need the whole segment. Reading a SAMPLE of bunches is enough: at ~90
triggers per pulse, eight pulses already give ~700 chances to see a 96 %
effect against a 0.05 % background. The screens below exist so the sample is
read as few times as possible:

  1. candidates come from `pulse_match`'s own lock enumeration -- already
     computed, no extra IO;
  2. locks whose cluster count is far below the best cannot be right, and are
     dropped without ever being tested;
  3. survivors are tried in rank order (count, then intensity correlation),
     and the search STOPS at the first lock that passes -- for a healthy
     segment that is the first test;
  4. the first pass uses a small sample; only a result in the ambiguous band
     between the accept and reject thresholds pays for a larger one.

A lock that passes is returned with the evidence that passed it. If every
candidate is rejected the caller gets None -- which is a real answer ("none of
these locks is right"), not a silent guess, and is what the whole exercise is
about.

THREE OUTCOMES, NOT TWO. A candidate can also be UNMEASURABLE: the join refuses
it before any coincidence can be computed, so nothing was learned about it
either way. That is not evidence against the lock and is kept in
`Verdict.unmeasured`, apart from `rejected`. It matters more than it sounds:
on the 2026-08-13 unmatched campaign, 16 of the first 20 candidates never
reached the measurement at all -- 10 blocked by pulse_match's own
confident-selection guard, 6 by bunch_join's delta-scan ambiguity, which is a
SECOND count-based decision downstream of the one this module replaces. Folding
those into "coincidence 0 %" would have reported an architectural blockage as a
verdict from the physics.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ntof_processing.slim_pipeline import config as C          # noqa: E402

ARMS = ('A', 'B', 'C', 'D')

# The bar the MEDIAN PULSE must clear: the fraction of a typical pulse's
# physics triggers with a wall AND plastic hit on the same arm inside the
# accept window. 0.80 sits in the empty gap between the two populations
# (96.2 % median vs ~0.05 %), so its exact value cannot matter.
#
# It is deliberately applied to ONE statistic. Scoring instead "what fraction
# of pulses individually clear 0.80" applies the cut twice, and compounding two
# cuts near the measurement's ceiling refuses good locks: at the known-correct
# lock of run_102/stat090_0003 the median pulse was 95 % while only 75 % of
# pulses cleared the inner bar (2026-08-13). Measured across six correct locks
# the median pulse ran 87-95 %, against ~0 % at a wrong one.
#
# Defined in config so clock_qa, the dashboard and the pulse ledger read the
# same number; this name is kept as the local alias.
ACCEPT_FRAC = C.PULSE_MIN_FRAC
REJECT_FRAC = 0.20
# Sample sizes, small then large. The large one is only paid when a result
# lands between REJECT_FRAC and ACCEPT_FRAC -- which over ~190 candidate
# evaluations has never once happened, so treat it as a safety net rather than
# a working path.
#
# 32 IS A FLOOR, NOT A PREFERENCE. The measurement runs the production clock
# fit, and `clockfit.bootstrap` needs BOOT_MIN_PEAK = 150 counts in its tallest
# 20 ns bin. It histograms at K_SEED, so the un-fitted rate walk (measured to
# -168 ns/burst) smears the peak across bins before it is counted: at 8 bunches
# that leaves ~80 counts, under the floor, and the fit raises "no peak" -- which
# is INDISTINGUISHABLE from a wrong lock. A correct lock would be refused for
# being under-sampled. 32 bunches is ~2,880 triggers, over MIN_EVENTS = 500 and
# clear of the peak floor even at the worst observed walk.
SAMPLE_SMALL = 32
SAMPLE_LARGE = 64
# A lock matching far fewer clusters than the best cannot be the right one; it
# is not worth an IO round trip. Generous on purpose -- the whole failure mode
# is near-ties, and this only drops the obviously-worse tail.
COUNT_FLOOR_FRAC = 0.70


@dataclass
class Verdict:
    offset_s: float | None = None
    frac: float = 0.0
    n_pulses: int = 0
    n_triggers: int = 0
    tested: int = 0
    rejected: list = field(default_factory=list)
    # candidates whose coincidence could NOT be measured -- the join refused,
    # the candidates could not be read. Kept apart from `rejected` because
    # "we looked and there is no coincidence" and "we never got to look" are
    # different answers, and only the first is evidence about the lock.
    unmeasured: list = field(default_factory=list)
    reason: str = ''

    @property
    def ok(self) -> bool:
        return self.offset_s is not None


def rank_candidates(locks, count_floor_frac: float = COUNT_FLOOR_FRAC):
    """Screen 1+2: drop hopeless locks, order the rest best-first.

    `locks` is pulse_match's own list of dicts with off_s / n / r. Both keys
    are already computed by the count scan, so this costs nothing.
    """
    if not locks:
        return []
    best_n = max(l['n'] for l in locks)
    keep = [l for l in locks if l['n'] >= count_floor_frac * best_n]
    # count first, then the intensity correlation as a tie-break: r is weak
    # (it is flat below ~200 clusters) but it is free and it is better than
    # scan order, which is what produced the campaign's mislocks.
    return sorted(keep, key=lambda l: (-l['n'],
                                       -(l.get('r') if l.get('r') is not None
                                         else -1.0)))


def pulse_coincidence(ev_bunch, ev_id, hits_det, hits_eid, hits_dt,
                      accept_ns: float = C.ACCEPT_NS):
    """Fraction of triggers, per pulse, with a same-arm wall+plastic hit.

    This is the physical question -- is the coincidence present -- and NOT
    whether the offline N1081B emulation rebuilds it. Measured on
    run_79/stat090_0000, 99.5 % of the triggers the emulator calls unmatched do
    have both legs inside +-25 ns, so requiring the emulation here would import
    a several-percent inefficiency into a decision that does not need it.
    """
    det = {t: i for i, t in enumerate(C.SCINT_TREES)}
    near = np.abs(hits_dt) <= accept_ns
    eid, d = hits_eid[near], hits_det[near]
    coinc = set()
    for a in ARMS:
        w = set(np.unique(eid[d == det[f'WAL{a}']]).tolist())
        p = set(np.unique(eid[d == det[f'PSS{a}']]).tolist())
        coinc |= (w & p)
    if not len(ev_id):
        return 0.0, 0, 0
    hit = np.fromiter((int(e) in coinc for e in ev_id), bool, len(ev_id))
    order = np.argsort(ev_bunch, kind='stable')
    b, h = ev_bunch[order], hit[order]
    starts = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
    tot = np.diff(np.r_[starts, len(b)])
    got = np.add.reduceat(h.astype(np.int64), starts)
    per = got / np.maximum(tot, 1)
    return float(np.mean(per >= ACCEPT_FRAC)), int(len(per)), int(len(b))


def arbitrate(locks, measure, sample_small: int = SAMPLE_SMALL,
              sample_large: int = SAMPLE_LARGE, log=print) -> Verdict:
    """Pick the lock whose pulses actually show the coincidence.

    `measure(offset_s, n_sample) -> (median_pulse_coincidence, n_pulses,
    n_triggers)` is injected so this module stays free of IO and can be tested
    on synthetic input. The first element is the MEDIAN pulse's coincidence
    fraction -- see ACCEPT_FRAC for why it is a median and not a pass rate.
    It is the only expensive call, and screens 3+4 above exist to keep the
    number of invocations near one.
    """
    v = Verdict()
    ranked = rank_candidates(locks)
    if not ranked:
        v.reason = 'no candidate locks'
        return v
    for lock in ranked:
        off = lock['off_s']
        frac, npulse, ntrig = measure(off, sample_small)
        v.tested += 1
        if frac != frac:                      # NaN -- no measurement was made
            v.unmeasured.append(off)
            log(f'    lock {off:+.2f}s NOT MEASURED (the join refused it); '
                f'this is not evidence against the lock')
            continue
        if frac >= ACCEPT_FRAC:
            v.offset_s, v.frac, v.n_pulses, v.n_triggers = off, frac, npulse, ntrig
            v.reason = f'coincidence {frac:.0%} of {npulse} sampled pulses'
            log(f'    lock {off:+.2f}s ACCEPTED: {v.reason}')
            return v
        if frac > REJECT_FRAC:
            # ambiguous band -- the only case that pays for a bigger sample
            frac2, npulse2, ntrig2 = measure(off, sample_large)
            v.tested += 1
            log(f'    lock {off:+.2f}s ambiguous at {frac:.0%}, '
                f'resampled -> {frac2:.0%}')
            if frac2 >= ACCEPT_FRAC:
                v.offset_s, v.frac = off, frac2
                v.n_pulses, v.n_triggers = npulse2, ntrig2
                v.reason = (f'coincidence {frac2:.0%} of {npulse2} pulses '
                            f'(resampled)')
                return v
            frac = frac2
        v.rejected.append((off, frac))
        log(f'    lock {off:+.2f}s rejected: coincidence {frac:.0%}')
    n_un = len(v.unmeasured)
    if v.rejected:
        v.reason = (f'no candidate lock reached {ACCEPT_FRAC:.0%}; '
                    f'best {max(f for _, f in v.rejected):.0%} of '
                    f'{len(v.rejected)} measured')
        if n_un:
            v.reason += f', {n_un} could not be measured'
    elif n_un:
        # THE COINCIDENCE NEVER GOT A VOTE. Saying "no lock reached 80 %" here
        # would blame the physics for what is an upstream refusal.
        v.reason = (f'none of the {n_un} candidate lock(s) could be measured '
                    f'-- the coincidence never got to decide this segment')
    else:
        v.reason = 'no candidate tested'
    return v
