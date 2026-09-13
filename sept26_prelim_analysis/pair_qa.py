#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pair_qa.py -- the campaign pair table, with the quality of both legs attached.

`campaign_angle.py` answers *what is the opening angle*.  This answers the
question that has to come first: **what kind of object is each pair**, so the
question "is there filtering still to do?" can be looked at rather than
guessed.  One row per pair, the same pairs `campaign_angle` makes -- same
`_track_table` selection, same `_pairs_real`, same event-mixed null -- with
every per-leg quality number carried through instead of thrown away.

WHY THIS IS A SEPARATE TABLE.  `campaign_angle` keeps five columns per pair and
drops the track rows behind them, so nothing downstream can ask why a pair is
in the sample.  Rebuilding here rather than widening that output keeps the
published angle products byte-identical; the two are checked against each other
by `--verify`, which is a real comparison and not a comment.

THE LEG ORDER IS THE ARM ORDER.  `_vertex_frame` sorts each pair so that
``arm1 <= arm2`` alphabetically, and the ``_1`` / ``_2`` suffixes here follow
that sort, not the order the tracks happened to be stored in.  Otherwise an
A-D pair would put A's chi2 in ``chi2_1`` only half the time and every
per-chamber QA plot would silently be a mixture.

WHAT IS AND IS NOT A COINCIDENCE.  Three timing columns are written and only
one of them is a coincidence measurement.  Measured on the campaign, not
assumed:

  ``delta_t``       **the coincidence.**  The two arms' SCINTILLATOR times,
                    joined from `tight_coincidence.py`'s tagged sample.  It
                    exists only for the pairs in which both arms carry a
                    scintillator hit -- a few thousand of sixty-odd thousand --
                    and it is the only number here that compares two
                    independent clocks.
  ``dt_track_ns``   the difference of the two legs' fitted track times
                    (``x_t0``/``y_t0``).  **Not a coincidence.**  Both legs of
                    a real pair share one trigger and one sampling window, and
                    a leg's ``t0`` moves with the depth at which it crossed the
                    gap -- up to the full ~1.2 us drift time.  So this is
                    dominated by the drift-depth difference and is a QA
                    variable about the reconstruction, not about simultaneity.
  ``dt_flash_ns``   **identically zero for every real pair**, because
                    ``t_since_flash_ns`` is a property of the TRIGGER and both
                    legs share it.  Kept anyway, and only as the check that the
                    event mixing did what it claims: it must be 0 on the real
                    sample and wide on the mixed one.  It is not plotted as a
                    quality metric and must not be read as one.

``t_flash_ns`` -- the pair's own neutron arrival time, and ``e_neutron_keV``
with it -- is per trigger and therefore well defined for a real pair.  Whether
the six arm pairs are drawn from the same neutron-time population is a real
question and this is what answers it.

The event-mixed null is built and kept throughout, and it is the thing to
compare against: for the mixed sample the two legs come from DIFFERENT
triggers, so whatever a cut does to the mixed distribution is what that cut
does to pure accidentals.

    python -m sept26_prelim_analysis.pair_qa --jobs 8
    python -m sept26_prelim_analysis.pair_qa --verify     # against campaign_angle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import acceptance as AC  # noqa: E402
from sept26_prelim_analysis.tight_coincidence import (  # noqa: E402
    BACK_TO_BACK_DEG, PRE_ACCESS_RUNS)

SCHEMA = 'sept26_prelim/pair_qa/1'

#: Per-track stage-3 columns to carry onto both legs.  Every one of these is a
#: quantity a cut could plausibly be placed on; nothing here is decoration.
LEG_COLS = [
    'chi2dof_x', 'chi2dof_y',        # fit quality, per view
    'x_n_strips', 'y_n_strips',      # how much of the track was actually seen
    'q_total', 'q_per_len',          # charge, and charge per unit path
    'drift_len_mm', 'drift_railed',  # where in the gap, and whether it railed
    'dca_axis_mm',                   # how well the leg points at the beam axis
    'angle_to_beam_deg',
    'n_cand_x', 'n_cand_y',          # how ambiguous the X<->Y pairing was
    't_since_flash_ns',              # the n_TOF time base
    'x_t0', 'y_t0',                  # the fitted track times, per view
    'e_neutron_keV',
]

#: Derived per-PAIR quantities, and the reduction that makes each.  A cut acts
#: on the WORST leg, not on an average, so these are the shapes to look at:
#: ``chi2dof_worst`` is what a chi2 cut would actually remove.
#: ``dt_flash_ns`` is deliberately absent -- it is the mixing check, not a
#: metric; see the module docstring.
PAIR_DERIVED = (
    'chi2dof_worst', 'n_strips_min', 'q_total_min', 'q_total_max',
    'dca_worst', 'sep_mm', 'v_r', 'dt_track_ns', 't_flash_ns',
    'e_neutron_keV', 'drift_railed_any',
)


def _leg_frame(t: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """The QA columns of one leg, positionally indexed, index reset."""
    return t.loc[idx, LEG_COLS].reset_index(drop=True)


def _qa_frame(t: pd.DataFrame, pr: pd.DataFrame, V: pd.DataFrame
              ) -> pd.DataFrame:
    """Attach both legs' QA to the vertex frame ``V``, in ``arm1``/``arm2`` order.

    ``V`` comes from ``source_imaging._vertex_frame`` and is already sorted so
    that ``arm1 <= arm2``; the legs are re-ordered here by the SAME comparison
    so the suffixes and the arm columns cannot disagree.
    """
    a = _leg_frame(t, pr.i.to_numpy())
    b = _leg_frame(t, pr.j.to_numpy())
    swap = (t.loc[pr.i.to_numpy(), 'arm'].to_numpy()
            > t.loc[pr.j.to_numpy(), 'arm'].to_numpy())
    out = V.reset_index(drop=True).copy()
    for c in LEG_COLS:
        av, bv = a[c].to_numpy(), b[c].to_numpy()
        out[f'{c}_1'] = np.where(swap, bv, av)
        out[f'{c}_2'] = np.where(swap, av, bv)

    def four(stem_x: str, stem_y: str, how):
        return how(np.vstack([out[f'{stem_x}_1'], out[f'{stem_y}_1'],
                              out[f'{stem_x}_2'], out[f'{stem_y}_2']]), axis=0)

    out['chi2dof_worst'] = four('chi2dof_x', 'chi2dof_y', np.nanmax)
    out['n_strips_min'] = four('x_n_strips', 'y_n_strips', np.nanmin)
    out['q_total_min'] = np.nanmin(
        np.vstack([out.q_total_1, out.q_total_2]), axis=0)
    out['q_total_max'] = np.nanmax(
        np.vstack([out.q_total_1, out.q_total_2]), axis=0)
    out['dca_worst'] = np.nanmax(
        np.vstack([out.dca_axis_mm_1, out.dca_axis_mm_2]), axis=0)
    out['drift_railed_any'] = out.drift_railed_1 | out.drift_railed_2
    # Signed, not absolute: a systematic offset between two arms and a
    # symmetric spread about zero are different diagnoses, and |dt| hides
    # which one you have.
    out['dt_track_ns'] = (0.5 * (out.x_t0_1 + out.y_t0_1)
                          - 0.5 * (out.x_t0_2 + out.y_t0_2))
    # The mixing check, not a metric -- 0 on every real pair by construction.
    out['dt_flash_ns'] = out.t_since_flash_ns_1 - out.t_since_flash_ns_2
    # Per TRIGGER, so a real pair has exactly one of each and leg 1 carries it.
    out['t_flash_ns'] = out.t_since_flash_ns_1
    out['e_neutron_keV'] = out.e_neutron_keV_1
    return out


def one_run(run: str, subruns, src: str, dca_max: float) -> tuple:
    """Every pair of one run with both legs' QA, plus the mixed null.  Worker."""
    from sept26_prelim_analysis import source_imaging as SI
    try:
        t = SI._track_table(run, subruns, dca_max, src=Path(src),
                            extra_cols=LEG_COLS)
        real = SI._pairs_real(t)
        if real.empty:
            return run, None, 'no pairs -- check angle_calibrated for this run'
        mix = SI._pairs_mixed(t, real, seed=5)
        out = []
        for pr, is_mixed in ((real, False), (mix, True)):
            if pr.empty:
                continue
            V = SI._vertex_frame(t, pr, is_mixed)
            out.append(_qa_frame(t, pr, V))
        d = pd.concat(out, ignore_index=True)
        d['run'] = run
        d['topo'] = [AC.topology(a, b) for a, b in zip(d.arm1, d.arm2)]
        d['pair'] = d.arm1 + '–' + d.arm2
        d['back_to_back'] = ((d.topo == 'opposing')
                             & (d.open_deg > BACK_TO_BACK_DEG))
        return run, d, ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def campaign(src: Path, dca_max: float, jobs: int,
             include_pre_access: bool) -> tuple:
    """Every run, pooled.  Same discovery and same exclusions as campaign_angle."""
    import re
    rs = {}
    for p in sorted(src.glob('tracks_run_*_stat090_*.parquet')):
        m = re.match(r'tracks_(run_\d+)_(stat090_\d+)\.parquet$', p.name)
        if m:
            rs.setdefault(m.group(1), []).append(m.group(2))
    if not include_pre_access:
        for r in PRE_ACCESS_RUNS:
            rs.pop(r, None)
    print(f'{len(rs)} run(s) from {src}'
          f'{"" if include_pre_access else " (pre-access excluded)"}\n')
    out, bad = [], {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one_run, r, subs, str(src), dca_max): r
                for r, subs in rs.items()}
        for f in as_completed(futs):
            run, d, err = f.result()
            if err:
                bad[run] = err
                print(f'  {run:<10} --      {err}', flush=True)
                continue
            out.append(d)
            print(f'  {run:<10} ok      {int((~d.mixed).sum()):>7,} real pairs',
                  flush=True)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), bad


#: The join key for the scintillator times.  THE ARMS ARE PART OF IT AND MUST
#: BE.  ``key1``/``key2`` are TRIGGER keys (``subrun:event_id``), not track
#: keys, so both legs of a real pair carry the same one -- joining on those
#: alone matches every pair in a trigger to every tagged pair in that trigger,
#: which silently invented a scintillator ``delta_t`` for intra pairs, a
#: quantity that does not exist (one arm, one time, no difference).
SCINT_KEY = ['run', 'key1', 'key2', 'arm1', 'arm2', 'mixed']


def attach_scint(d: pd.DataFrame, tight: Path) -> pd.DataFrame:
    """Join the scintillator times on, where the pair is tagged in both arms.

    ``t1``/``t2`` are per (trigger, ARM) and not per track, so two A-C pairs in
    one trigger legitimately share them: the join is many-to-one by
    construction and the duplicate drop below is removing repeats of the same
    (trigger, arm pair), not choosing between candidates.

    Intra pairs come out NaN, and that is correct -- `tight_coincidence.py`
    writes no intra rows at all, because a pair with both legs in one chamber
    has one arm and therefore no ``t1 - t2``.
    """
    for c in ('t1', 't2', 'delta_t', 'tight_pair'):
        d[c] = np.nan if c != 'tight_pair' else False
    if not tight.exists():
        print(f'  scintillator times NOT joined: {tight} is absent')
        return d
    m = pd.read_parquet(tight)[SCINT_KEY + ['t1', 't2', 'delta_t',
                                            'tight_pair']]
    m = m.drop_duplicates(subset=SCINT_KEY)
    j = d.drop(columns=['t1', 't2', 'delta_t', 'tight_pair']).merge(
        m, on=SCINT_KEY, how='left')
    j['tight_pair'] = j.tight_pair.fillna(False).astype(bool)
    n = int(j.delta_t.notna().sum())
    print(f'  scintillator times joined onto {n:,} of {len(j):,} pairs '
          f'({100 * n / max(len(j), 1):.2f} %)')
    bad = int(j.loc[j.topo == 'intra', 'delta_t'].notna().sum())
    print(f'  intra pairs given a delta_t: {bad} (must be 0 -- an intra pair '
          f'has one arm and so no arm-to-arm difference)')
    return j


def summary(d: pd.DataFrame, metrics=PAIR_DERIVED) -> pd.DataFrame:
    """Per arm pair, per metric: the percentiles a cut would be chosen from."""
    rows = []
    real = d[~d.mixed]
    for (topo, pair), g in real.groupby(['topo', 'pair']):
        for m in metrics:
            v = pd.to_numeric(g[m], errors='coerce').to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                rows.append(dict(topology=topo, pair=pair, metric=m, n=0))
                continue
            p = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
            rows.append(dict(topology=topo, pair=pair, metric=m, n=len(v),
                             p01=p[0], p05=p[1], p25=p[2], median=p[3],
                             p75=p[4], p95=p[5], p99=p[6],
                             mean=float(v.mean()), max=float(v.max())))
    return pd.DataFrame(rows)


def verify(d: pd.DataFrame, angle_pairs: Path) -> pd.DataFrame:
    """Check this table is the SAME sample campaign_angle published.

    Compares the per (arm1, arm2, mixed) pair counts.  A table that is only
    nearly the same sample is worse than no table, because every QA
    distribution drawn from it would be about a population nobody else has.
    """
    if not angle_pairs.exists():
        print(f'  cannot verify: {angle_pairs} is absent')
        return pd.DataFrame()
    a = pd.read_parquet(angle_pairs).groupby(
        ['arm1', 'arm2', 'mixed']).size().rename('n_angle')
    b = d.groupby(['arm1', 'arm2', 'mixed']).size().rename('n_qa')
    c = pd.concat([a, b], axis=1).fillna(0).astype(int).reset_index()
    c['delta'] = c.n_qa - c.n_angle
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'stage3_fullpass')),
                    help='stage-3 track tables (default: the full pass)')
    ap.add_argument('--dca', type=float, default=30.0)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--include-pre-access', action='store_true',
                    help='keep run_79/81, which are the other side of the '
                         '27 July access')
    ap.add_argument('--tight',
                    default=str(paths.spell('out', 'tight_coincidence',
                                            'pairs_tight_campaign.parquet')),
                    help='tagged pairs, for the scintillator times')
    ap.add_argument('--verify', action='store_true',
                    help='also compare the sample against campaign_angle')
    a = ap.parse_args()

    src = paths.require(Path(a.src), 'the stage-3 track tables')
    od = paths.out('pair_qa')
    d, bad = campaign(src, a.dca, a.jobs, a.include_pre_access)
    if d.empty:
        print('no pairs at all -- nothing written')
        return 1
    d = attach_scint(d, Path(a.tight))

    S = summary(d)
    d.to_parquet(od / 'pairs_qa_campaign.parquet', index=False)
    S.to_csv(od / 'pair_qa_summary.csv', index=False)
    json.dump(dict(schema=SCHEMA, src=str(src), tight=a.tight,
                   dca_max=a.dca, include_pre_access=a.include_pre_access,
                   back_to_back_deg=BACK_TO_BACK_DEG,
                   leg_cols=LEG_COLS, pair_derived=list(PAIR_DERIVED),
                   n_pairs_real=int((~d.mixed).sum()),
                   n_pairs_mixed=int(d.mixed.sum()),
                   n_with_scint=int(d.delta_t.notna().sum()),
                   runs_failed=bad),
              open(od / 'pair_qa.meta.json', 'w'), indent=1)

    # The mixing check.  Both legs of a real pair share a trigger and so share
    # `t_since_flash_ns`; both legs of a mixed pair must not.  Printed rather
    # than asserted: a sub-run whose trigger time never backfilled has NaN
    # here, which is a staging gap and not a broken null.
    rf = d.loc[~d.mixed, 'dt_flash_ns'].dropna()
    mf = d.loc[d.mixed, 'dt_flash_ns'].dropna()
    print(f'\nmixing check  real dt_flash != 0: {int((rf != 0).sum())} of '
          f'{len(rf):,} (must be 0)   mixed dt_flash != 0: '
          f'{int((mf != 0).sum())} of {len(mf):,} (must be all)')

    print(f'\nREAL pairs by arm pair')
    print(d[~d.mixed].groupby(['topo', 'pair']).size().to_string())
    if a.verify:
        V = verify(d, paths.spell('out', 'angle_campaign', 'pairs.parquet'))
        if len(V):
            print('\nSAME SAMPLE AS campaign_angle?')
            print(V.to_string(index=False))
            print('  all match' if (V.delta == 0).all() else
                  '  *** MISMATCH -- do not use these QA plots to describe '
                  'the published angle sample ***')
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
