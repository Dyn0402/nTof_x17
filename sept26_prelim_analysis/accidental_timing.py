#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
accidental_timing.py -- are the two-track pairs real coincidences, or two
particles that happened to share a DREAM window?  The scintillator answer.

Follows HANDOFF_ACCIDENTAL_TIMING.md (2026-09-08), which established the
question is answerable with what the slim already carries -- `dt_ns` at the
full +-1000 ns range, and the unused `is_control` accidental control -- and ran
a first pass that came out on the accidental side but was known-biased. This
module redoes it per the handoff's item (a): unbiased hit choice, the full
range, and a two-component fit for f, the true-coincidence fraction.

THE THREE THINGS BUILT HERE.

  1. `single_arm_hit_classes` -- per scintillator hit, dt_ns tagged by whether
     its own family's partner (wall<->plastic, same arm, same event) ALSO
     fired in-window, restricted to events active in exactly one arm.  This is
     the picture behind the accept-window choice: does requiring the
     coincidence buy a cleaner sample than either element alone?

  2. `arm_tag_time` -- one unbiased time per (event, arm) with a wall+plastic
     coincidence: the mean of a RANDOMLY chosen hit from each family, not the
     one nearest dt_ns = 0.  HANDOFF sec 3.1: "closest to zero" pulls |dt| down
     whenever an arm has more than one candidate hit, which biased the first
     pass toward finding coincidence.

  3. `two_arm_delta_t` + `fit_prompt_fraction` -- for the real (not mixed)
     inter-chamber MM pairs, the arm1-arm2 scintillator time difference, and
     an unbinned two-component MLE for f: PROMPT template = bootstrap
     difference of two independent draws from the single-arm reference (the
     trigger's own resolution function); ACCIDENTAL template = one draw from
     the reference, one from `is_control` (dead flat, HANDOFF sec 2.2 -- the
     missing S4 background normalisation, unused until now).

DOES NOT use event mixing (HANDOFF sec 3.2: dt_ns is relative to each event's
OWN trigger, so a singly-tagged event's hit is at zero by construction and
mixing inverts the answer here). DOES read `key1`/`key2` from
`source_imaging.vertices`, not `key` (HANDOFF sec 3.3 -- the latent bug that
silently mis-attributed a mixed pair's second track, fixed 2026-09-08).

    python -m sept26_prelim_analysis.accidental_timing --run run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.scintillators import DT_WINDOW, read_slim  # noqa: E402

SCHEMA = 'sept26_prelim/accidental_timing/1'
#: The peak's own core, HANDOFF sec 2.1 -- a candidate re-optimised window,
#: reported alongside the production one but NOT installed anywhere: changing
#: `candidate_filter.py`'s window touches the stage-1/stage-2 production chain,
#: which is on Dylan's hold.
CORE_WINDOW = (-30.0, 30.0)


# --------------------------------------------------------------------------- #
# 1. single-arm events: does the coincidence requirement buy anything?
# --------------------------------------------------------------------------- #
def _fam_flags(slim: pd.DataFrame, window: tuple) -> pd.DataFrame:
    """Per (event, arm): does WAL / PSS have >=1 in-window hit."""
    s = slim[slim.family.isin(('WAL', 'PSS'))].copy()
    s['in_win'] = ((s.is_control == 0) & (s.dt_ns >= window[0])
                   & (s.dt_ns <= window[1]))
    g = (s[s.in_win].groupby(['subrun', 'eventId', 'arm', 'family'])
         .size().unstack('family', fill_value=0))
    for f in ('WAL', 'PSS'):
        if f not in g.columns:
            g[f] = 0
    g = g.reset_index()
    g['wal_in'] = g['WAL'] > 0
    g['pss_in'] = g['PSS'] > 0
    return g[['subrun', 'eventId', 'arm', 'wal_in', 'pss_in']]


def single_arm_hit_classes(slim: pd.DataFrame, window: tuple = DT_WINDOW,
                           ) -> pd.DataFrame:
    """Every WAL/PSS hit in a single-active-arm event, classed by its partner.

    "single-active-arm" means exactly one of the four arms has any in-window
    wall or plastic hit at all -- the reference population, uncontaminated by
    a second particle elsewhere.  Within that arm, ``cls`` is 'both' if the
    partner family also fired in-window, else 'wall_only' / 'plastic_only'.
    The returned `dt_ns` is the hit's own, over the FULL +-1000 ns range, so
    a caller can see the peak-over-pedestal shape for each class rather than
    just a window-truncated count.
    """
    flags = _fam_flags(slim, window)
    flags['active'] = flags.wal_in | flags.pss_in
    act = flags[flags.active].copy()
    n_arms = act.groupby(['subrun', 'eventId'])['arm'].transform('nunique')
    single = act[n_arms == 1].copy()
    single['cls'] = np.select(
        [single.wal_in & single.pss_in, single.wal_in, single.pss_in],
        ['both', 'wall_only', 'plastic_only'], default='none')
    # `is_control` is a SEPARATE random-sampling stream (HANDOFF sec 2.2), not
    # a candidate "this element fired" hit -- excluded here too, or a handful
    # would leak into the 'wall_only'/'plastic_only' buckets despite playing
    # no part in classifying them, which is what produced negative "purity"
    # entries on first pass.
    real_hits = slim[slim.family.isin(('WAL', 'PSS')) & (slim.is_control == 0)]
    return real_hits.merge(single[['subrun', 'eventId', 'arm', 'cls']],
                           on=['subrun', 'eventId', 'arm'], how='inner')


def single_arm_summary(hits: pd.DataFrame, core: tuple = CORE_WINDOW,
                       ) -> pd.DataFrame:
    """Per (family, class): how many hits sit in the peak core vs the flat
    pedestal -- the same peak/pedestal metric as HANDOFF sec 2.1, now split by
    whether the coincidence partner fired."""
    ped_lo, ped_hi = -200.0, -100.0
    #: The only (family, class) pairs that mean anything -- a family's own
    #: hits in its own "solo" class, plus "both". The complementary pairs
    #: (e.g. a WAL hit inside a 'plastic_only' event) are real rows in `hits`
    #: but describe the OTHER element's absence, not this one's timing, so
    #: they are dropped here rather than left to confuse the table.
    relevant = {('WAL', 'both'), ('WAL', 'wall_only'),
               ('PSS', 'both'), ('PSS', 'plastic_only')}
    rows = []
    for (fam, cls), g in hits.groupby(['family', 'cls']):
        if (fam, cls) not in relevant:
            continue
        n_ped = int(((g.dt_ns >= ped_lo) & (g.dt_ns < ped_hi)).sum())
        n_core = int(((g.dt_ns >= core[0]) & (g.dt_ns < core[1])).sum())
        ped_rate = n_ped / (ped_hi - ped_lo)
        exp_bg = ped_rate * (core[1] - core[0])
        rows.append(dict(family=fam, cls=cls, n=int(len(g)),
                         n_core=n_core, n_pedestal_per_10ns=10 * ped_rate,
                         purity_core=1 - exp_bg / n_core if n_core else np.nan))
    return pd.DataFrame(rows).sort_values(['family', 'cls'], ignore_index=True)


# --------------------------------------------------------------------------- #
# item (c): where should the accept window actually sit
# --------------------------------------------------------------------------- #
def window_scan(slim: pd.DataFrame,
                centers=tuple(range(-60, 41, 5)),
                half_widths=(10, 15, 20, 25, 30, 40, 50, 60, 80, 100),
                ) -> pd.DataFrame:
    """Peak/pedestal purity for a grid of (center, half-width) windows.

    The pedestal rate is measured OUTSIDE any candidate window, on a fixed
    -200..-100 ns reference slice that HANDOFF sec 2.1 shows is flat, so it is
    not circular with the window being scanned.
    """
    s = slim[slim.family.isin(('WAL', 'PSS')) & (slim.is_control == 0)]
    dt = s.dt_ns.to_numpy()
    ped_rate = np.sum((dt >= -200) & (dt < -100)) / 100.0
    rows = []
    for c in centers:
        for hw in half_widths:
            lo, hi = c - hw, c + hw
            n = int(np.sum((dt >= lo) & (dt < hi)))
            exp_bg = ped_rate * (hi - lo)
            rows.append(dict(center=c, half_width=hw, lo=lo, hi=hi, n=n,
                             width_ns=hi - lo, exp_bg=exp_bg,
                             purity=1 - exp_bg / n if n else np.nan,
                             n_signal=n - exp_bg))
    return pd.DataFrame(rows)


def recommend_window(scan: pd.DataFrame, min_purity: float = 0.99) -> dict:
    """The narrowest-signal-cost window that clears ``min_purity``.

    Among windows at or above the purity floor, the one with the MOST signal
    is what the accept window should be -- purity alone would just pick the
    narrowest window on the grid.
    """
    ok = scan[scan.purity >= min_purity]
    if ok.empty:
        best = scan.loc[scan.purity.idxmax()]
    else:
        best = ok.loc[ok.n_signal.idxmax()]
    return dict(center=float(best.center), half_width=float(best.half_width),
               lo=float(best.lo), hi=float(best.hi),
               purity=float(best.purity), n_signal=float(best.n_signal),
               min_purity=min_purity)


# --------------------------------------------------------------------------- #
# 2. one unbiased time per tagged (event, arm)
# --------------------------------------------------------------------------- #
def arm_tag_time(slim: pd.DataFrame, window: tuple = DT_WINDOW,
                 require_both: bool = True, seed: int = 7) -> pd.DataFrame:
    """One (event, arm) row per scintillator tag, unbiased.

    `require_both=True` is the production tag (wall AND plastic, both
    in-window) used everywhere else in this analysis (`efficiency.py`,
    `scintillators.py`) -- `t_tag` averages a RANDOMLY chosen in-window hit
    from each family, not the one nearest dt_ns = 0 (HANDOFF sec 3.1: nearest
    pulls |dt| down whenever an arm has more than one candidate hit, and that
    is exactly what biased the first pass of this test).

    `require_both=False` tags on EITHER element alone -- a real, weaker
    per-arm definition, needed because the strict tag leaves only 8 of
    run_145's 464 real inter-chamber pairs with both arms tagged (see
    :func:`main`), too few to fit. `t_tag` is then one randomly chosen
    in-window hit from either family. Both counts are reported so the choice
    is visible rather than silently made.

    Implemented as a full shuffle followed by `drop_duplicates(keep='first')`,
    which gives every in-window hit an equal chance of being "first".
    """
    s = slim[slim.family.isin(('WAL', 'PSS')) & (slim.is_control == 0)
             & (slim.dt_ns >= window[0]) & (slim.dt_ns <= window[1])]
    s = s.sample(frac=1.0, random_state=seed)
    if require_both:
        picked = s.drop_duplicates(['subrun', 'eventId', 'arm', 'family'],
                                   keep='first')
        piv = picked.pivot_table(index=['subrun', 'eventId', 'arm'],
                                 columns='family', values='dt_ns',
                                 aggfunc='first')
        for f in ('WAL', 'PSS'):
            if f not in piv.columns:
                piv[f] = np.nan
        piv = piv.dropna(subset=['WAL', 'PSS'])
        piv['t_tag'] = 0.5 * (piv['WAL'] + piv['PSS'])
        return piv.reset_index().rename(columns={'WAL': 'wal_dt',
                                                  'PSS': 'pss_dt'})
    picked = s.drop_duplicates(['subrun', 'eventId', 'arm'], keep='first')
    nfam = s.groupby(['subrun', 'eventId', 'arm'])['family'].nunique()
    out = picked[['subrun', 'eventId', 'arm', 'dt_ns', 'family']].rename(
        columns={'dt_ns': 't_tag', 'family': 'picked_family'})
    return out.merge(nfam.rename('n_families'),
                     on=['subrun', 'eventId', 'arm'])


def single_arm_reference(tag: pd.DataFrame) -> pd.DataFrame:
    """`t_tag` for events tagged in EXACTLY one arm -- the trigger's own
    resolution function, unbiased."""
    n = tag.groupby(['subrun', 'eventId'])['arm'].transform('nunique')
    return tag[n == 1].copy()


# --------------------------------------------------------------------------- #
# 3. the two-arm test
# --------------------------------------------------------------------------- #
def _inter_pairs(run: str, subruns, dca_max: float,
                 src=None) -> pd.DataFrame:
    """The real (not mixed) inter-chamber MM pairs, keyed for a slim merge.

    Pairs come from `source_imaging.vertices`, which is what feeds the
    published S4 spectrum, so this test uses exactly the sample whose null it
    is checking. Reads `key1`/`key2` (HANDOFF sec 3.3): for the real sample
    they are equal by construction, asserted here so a future change upstream
    cannot silently break it again.
    """
    from sept26_prelim_analysis import source_imaging as SI
    real, _ = SI.vertices(run, subruns, dca_max, src=src)
    inter = real[real.topology == 'inter'].copy()
    if inter.empty:
        return inter
    assert (inter.key1 == inter.key2).all(), \
        'real pairs must share one trigger -- key1/key2 mismatch'
    ek = inter.key1.str.split(':', n=1, expand=True)
    inter['subrun'] = ek[0]
    inter['eventId'] = ek[1].astype(np.int64)
    return inter


def two_arm_delta_t(run: str, subruns, tag: pd.DataFrame,
                    dca_max: float = 30.0, src=None) -> pd.DataFrame:
    """arm1 - arm2 scintillator tag time, for the REAL inter-chamber MM pairs."""
    from sept26_prelim_analysis import acceptance as AC
    inter = _inter_pairs(run, subruns, dca_max, src=src)
    if inter.empty:
        return inter

    t1 = tag.rename(columns={'arm': 'arm1', 't_tag': 't1'})
    t2 = tag.rename(columns={'arm': 'arm2', 't_tag': 't2'})
    m = inter.merge(t1[['subrun', 'eventId', 'arm1', 't1']],
                    on=['subrun', 'eventId', 'arm1'], how='inner')
    m = m.merge(t2[['subrun', 'eventId', 'arm2', 't2']],
               on=['subrun', 'eventId', 'arm2'], how='inner')
    m['delta_t'] = m.t1 - m.t2
    m['topo'] = [AC.topology(a, b) for a, b in zip(m.arm1, m.arm2)]
    return m


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
def bootstrap_templates(single_ref: pd.DataFrame, control_dt_in_window: np.ndarray,
                        n: int = 200_000, seed: int = 11) -> tuple:
    """PROMPT: difference of two independent draws from the single-arm
    reference -- both sides born together, so both look like the trigger.
    ACCIDENTAL: one draw from the reference, one from `is_control` (HANDOFF
    sec 2.2's dead-flat control) RESTRICTED to the same accept window used to
    build `t_tag` -- an accidental hit only ever enters `t_tag` if it happens
    to land in that window, so the null must be truncated the same way or it
    is too wide by construction. Both signed orders are pooled, since a real
    pair's "arm1"/"arm2" is alphabetical, not "trigger"/"other".
    """
    rng = np.random.default_rng(seed)
    ref = single_ref.t_tag.to_numpy()
    prompt = rng.choice(ref, n) - rng.choice(ref, n)
    c, d = rng.choice(ref, n), rng.choice(control_dt_in_window, n)
    acc = np.concatenate([c - d, d - c])
    return prompt, acc


def fit_prompt_fraction(delta_obs: np.ndarray, prompt_samp: np.ndarray,
                        acc_samp: np.ndarray, f_grid: np.ndarray = None,
                        ) -> dict:
    """Unbinned two-component MLE for f, the true-coincidence fraction.

    p(delta) and a(delta) are Gaussian-KDE densities fit to the bootstrap
    templates, and f maximises sum(log(f p + (1-f) a)) over the observed
    pairs. The 1-sigma interval is where -2 log L rises by 1 from the maximum
    (Wilks, asymptotic -- stated as approximate given N of order 10-100).
    """
    from scipy.stats import gaussian_kde
    if len(delta_obs) == 0:
        return dict(f_hat=np.nan, f_lo=np.nan, f_hi=np.nan, n=0)
    kde_p = gaussian_kde(prompt_samp)
    kde_a = gaussian_kde(acc_samp)
    p = kde_p(delta_obs)
    a = kde_a(delta_obs)
    if f_grid is None:
        f_grid = np.linspace(0.0, 1.0, 2001)
    ll = np.array([np.sum(np.log(np.clip(f * p + (1 - f) * a, 1e-300, None)))
                   for f in f_grid])
    i = int(np.argmax(ll))
    within = f_grid[ll >= ll[i] - 0.5]
    return dict(f_hat=float(f_grid[i]), f_lo=float(within.min()),
               f_hi=float(within.max()), n=int(len(delta_obs)),
               f_grid=f_grid, ll=ll, kde_p=kde_p, kde_a=kde_a)


def fit_by_topology(m: pd.DataFrame, single_ref: pd.DataFrame,
                    control_dt: np.ndarray) -> pd.DataFrame:
    """The fit repeated per topology, where statistics allow (>=15 pairs)."""
    prompt, acc = bootstrap_templates(single_ref, control_dt)
    rows = []
    for topo, g in [('all', m)] + list(m.groupby('topo')):
        if len(g) < 15:
            rows.append(dict(topo=topo, n=int(len(g)), f_hat=np.nan,
                             f_lo=np.nan, f_hi=np.nan,
                             note='too few pairs (<15)'))
            continue
        fit = fit_prompt_fraction(g.delta_t.to_numpy(), prompt, acc)
        rows.append(dict(topo=topo, n=fit['n'], f_hat=fit['f_hat'],
                         f_lo=fit['f_lo'], f_hi=fit['f_hi'], note=''))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# split by whether stage 1 would have kept the event at all
# --------------------------------------------------------------------------- #
#: The stage-1 classes `allowlist.py` reconstructs in full.  Everything else
#: reaches stage 2 only through the prescaled control (SINGLE 5 %, NONE 1 %),
#: so on run_145 -- the one run with a blind full pass -- the complement is the
#: population the campaign pass essentially never saw.
SELECTED_CLASSES = ('INTER', 'INTRA', 'IMPLIED')


def stage1_class(run: str, subruns) -> pd.DataFrame:
    """``(subrun, eventId) -> cls``, straight from the stage-1 candidate tables.

    One row per trigger.  The tables carry a `tag` column and a trigger can
    appear once per tag, so duplicates are dropped on the identity that
    actually matters here -- `(subrun, eventId)` -- rather than trusted to be
    unique (DATASET.md: `event_id` is unique within a sub-run, not across one).
    """
    out = []
    for sub in subruns:
        p = paths.require(
            paths.out('stage1') / f'candidates_{run}_{sub}.parquet',
            f'stage-1 candidates for {sub}')
        out.append(pd.read_parquet(p, columns=['subrun', 'eventId', 'cls']))
    return pd.concat(out, ignore_index=True).drop_duplicates(
        ['subrun', 'eventId'])


def fit_by_stage1_class(m: pd.DataFrame, single_ref: pd.DataFrame,
                        control_dt: np.ndarray, run: str, subruns,
                        min_n: int = 15) -> tuple:
    """Is the prompt component carried by the events stage 1 KEPT, or the ones
    it threw away?

    THE QUESTION THIS ANSWERS.  Stage 1 selects 12.8 % of the triggers that the
    run_145 full pass turns into a two-track event; the rest sit in `NONE` and
    `SINGLE` and reach stage 2 only at 1 % / 5 %.  Scaling the campaign to a
    blind full pass therefore rests on whether those missed events are real
    pairs or fits to noise -- and the scintillators answer it without the
    Micromegas, because a pair born at the capsule lights both arms promptly
    and two unrelated tracks do not.

    ``f`` is the same true-coincidence fraction :func:`fit_prompt_fraction`
    measures, so the split numbers are directly comparable to the published
    pooled one.  ``z_vs_zero`` is how far the fit sits from *pure* accidental,
    sqrt(2 dlogL) at f = 0 -- the number that decides the question, since the
    interesting null is "these are not coincidences at all", not "f = 29 %".

    Returns ``(table, m)`` with the class joined onto the pairs.
    """
    cls = stage1_class(run, subruns)
    m = m.merge(cls, on=['subrun', 'eventId'], how='left')
    m['stage1'] = np.where(m.cls.isin(SELECTED_CLASSES), 'selected', 'missed')
    prompt, acc = bootstrap_templates(single_ref, control_dt)

    groups = [('all', m),
              ('stage-1 selected', m[m.stage1 == 'selected']),
              ('stage-1 missed', m[m.stage1 == 'missed'])]
    groups += [(f'missed / {t}', g) for t, g
               in m[m.stage1 == 'missed'].groupby('topo')]
    groups += [(f'class {c}', g) for c, g in m.groupby('cls')]

    rows = []
    for name, g in groups:
        if len(g) < min_n:
            rows.append(dict(group=name, n=int(len(g)), f_hat=np.nan,
                             f_lo=np.nan, f_hi=np.nan, z_vs_zero=np.nan,
                             note=f'too few pairs (<{min_n})'))
            continue
        fit = fit_prompt_fraction(g.delta_t.to_numpy(), prompt, acc)
        ll, fg = fit['ll'], fit['f_grid']
        d = float(ll.max() - ll[int(np.argmin(np.abs(fg)))])
        rows.append(dict(group=name, n=fit['n'], f_hat=fit['f_hat'],
                         f_lo=fit['f_lo'], f_hi=fit['f_hi'],
                         z_vs_zero=float(np.sqrt(2 * max(d, 0.0))), note=''))
    return pd.DataFrame(rows), m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--dca', type=float, default=30.0)
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    slim = read_slim(a.run, subs)
    control_dt = slim.loc[slim.family.isin(('WAL', 'PSS')) & (slim.is_control == 1),
                          'dt_ns'].to_numpy()
    control_dt_win = control_dt[(control_dt >= DT_WINDOW[0])
                                & (control_dt <= DT_WINDOW[1])]

    # (1) single-arm events: does requiring the coincidence buy purity, over
    # the peak's own core -- the PRODUCTION window is wide enough that almost
    # every single-active-arm event already reads as "both", which is itself
    # the headline finding for item (c) below.
    hits_core = single_arm_hit_classes(slim, window=CORE_WINDOW)
    single_sum = single_arm_summary(hits_core, core=CORE_WINDOW)
    hits_prod = single_arm_hit_classes(slim, window=DT_WINDOW)
    n_solo_prod = int((hits_prod.drop_duplicates(
        ['subrun', 'eventId', 'arm']).cls != 'both').sum())
    n_single_prod = int(hits_prod.drop_duplicates(
        ['subrun', 'eventId', 'arm']).shape[0])

    # (c) the accept window itself
    scan = window_scan(slim)
    rec = recommend_window(scan, min_purity=0.95)

    # (2) the trigger's own resolution function -- the STRICT (wall AND
    # plastic) tag, same definition as efficiency.py, used only as the prompt
    # template's source.
    tag_strict = arm_tag_time(slim, require_both=True)
    ref = single_arm_reference(tag_strict)

    # (3) the two-arm test.  The strict tag leaves too few pairs to fit (see
    # printed count below), so the LOOSE (either element, HANDOFF sec 2.3's
    # own per-arm criterion loosened for statistics) tag is what is fit;
    # both counts are reported so the choice is not silent.
    tag_loose = arm_tag_time(slim, require_both=False)
    m_strict = two_arm_delta_t(a.run, subs, tag_strict, a.dca)
    m = two_arm_delta_t(a.run, subs, tag_loose, a.dca)
    fits = fit_by_topology(m, ref, control_dt_win)
    by_cls, m = fit_by_stage1_class(m, ref, control_dt_win, a.run, subs)
    prompt_t, acc_t = bootstrap_templates(ref, control_dt_win)

    od = paths.out('accidental_timing')
    single_sum.to_csv(od / f'single_arm_summary_{a.run}.csv', index=False)
    scan.to_csv(od / f'window_scan_{a.run}.csv', index=False)
    fits.to_csv(od / f'fit_by_topology_{a.run}.csv', index=False)
    by_cls.to_csv(od / f'fit_by_stage1_class_{a.run}.csv', index=False)
    hits_core[['subrun', 'eventId', 'arm', 'family', 'dt_ns', 'cls']].to_parquet(
        od / f'single_arm_hits_{a.run}.parquet', index=False)
    ref.to_parquet(od / f'single_arm_reference_{a.run}.parquet', index=False)
    m.to_parquet(od / f'two_arm_pairs_{a.run}.parquet', index=False)
    np.savez_compressed(od / f'templates_{a.run}.npz',
                        prompt=prompt_t[:50_000], acc=acc_t[:50_000],
                        delta_obs=m.delta_t.to_numpy())

    ref_iw = ref.t_tag.to_numpy()
    med_ref = float(np.median(np.abs(ref_iw)))
    frac20_ref = float((np.abs(ref_iw) < 20).mean())

    json.dump(dict(
        schema=SCHEMA, run=a.run, subruns=subs, dca_max=a.dca,
        production_window=list(DT_WINDOW), core_window=list(CORE_WINDOW),
        recommended_window=rec,
        n_slim_hits=int(len(slim)), n_control_hits=int(len(control_dt)),
        n_single_active_arm_events_prod_window=n_single_prod,
        n_single_active_arm_solo_prod_window=n_solo_prod,
        n_single_arm_reference=int(len(ref)),
        single_ref_median_abs_dt=med_ref, single_ref_frac_within_20ns=frac20_ref,
        n_inter_pairs_total=int(len(_inter_pairs(a.run, subs, a.dca))),
        n_two_arm_tagged_strict=int(len(m_strict)),
        n_two_arm_tagged_loose=int(len(m)),
        overall_fit=fits[fits.topo == 'all'].iloc[0].to_dict()
        if 'all' in set(fits.topo) else {},
        n_two_arm_tagged_by_stage1=m.stage1.value_counts().to_dict(),
        n_two_arm_tagged_by_class=m.cls.value_counts().to_dict(),
        fit_stage1_missed=by_cls[by_cls.group == 'stage-1 missed'].iloc[0].to_dict()
        if 'stage-1 missed' in set(by_cls.group) else {},
    ), open(od / f'accidental_timing_{a.run}.meta.json', 'w'), indent=1,
       default=str)

    print('SINGLE-ARM HIT CLASSES, peak core '
         f'({CORE_WINDOW[0]:.0f}, {CORE_WINDOW[1]:.0f}) ns '
         '-- coincidence vs one element only')
    print(single_sum.to_string(index=False))
    print(f'\nUnder the PRODUCTION window {DT_WINDOW}: {n_solo_prod} of '
          f'{n_single_prod} single-active-arm events ({100 * n_solo_prod / max(n_single_prod, 1):.2f}%) '
          f'are anything OTHER than a full wall+plastic "both" -- the window is '
          f'wide enough that a coincidence is nearly automatic.')
    print(f'\nsingle-arm reference (strict tag): n={len(ref)}, '
          f'median|dt|={med_ref:.1f} ns, {100 * frac20_ref:.1f}% within 20 ns')
    print(f'\nrecommended window (>= {rec["min_purity"]:.0%} purity, max signal): '
          f'({rec["lo"]:.0f}, {rec["hi"]:.0f}) ns  purity={rec["purity"]:.3f}  '
          f'vs production {DT_WINDOW}')
    print(f'\nTWO-ARM PAIRS -- strict tag: {len(m_strict)}, loose tag (fit '
          f'uses this): {len(m)} of the real inter-chamber sample tagged in '
          'both arms')
    print('\nFIT: prompt fraction f, per topology (loose tag, unbinned MLE)')
    print(fits.to_string(index=False))
    print('\nFIT: the same f, split by whether STAGE 1 would have kept the '
          'event -- the campaign pass reconstructs `selected` in full and\n'
          '     reaches `missed` only through the 1 % / 5 % control prescale')
    print(by_cls.to_string(index=False))
    print('\ntwo-arm-tagged pairs by stage-1 class')
    print(m.cls.value_counts().to_string())
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
