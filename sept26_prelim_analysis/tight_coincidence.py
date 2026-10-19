#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tight_coincidence.py -- require a pair's two arms to be PROMPT, and prompt
with EACH OTHER, before calling it a coincidence; then look at the opening
angle of what survives.

WHY. `HANDOFF_ACCIDENTAL_TIMING.md` measured that in a two-arm event one arm
is the trigger (median |dt| 7 ns against a 5.2 ns single-arm reference) and
**the other fires at essentially a random time** -- median 171 ns, 47 % beyond
200 ns.  The production accept window is (-100, +60) ns, wide enough that the
plastic family's own accidental rate lands something in it almost every time:
only 1 of 147 216 single-active-arm events fails to show a "coincidence".  So
the sample feeding the S4 opening-angle spectrum is dominated by pairs whose
second leg is uncorrelated, which is exactly what the S4 null found.

This module applies the cut that the null implies, and asks the one question
the published page could not: **does the opening-angle spectrum of genuinely
coincident pairs look different from the event-mixed shape the full sample
follows?**

THE CUT (Dylan, 2026-09-08 -- see OVERNIGHT_2026-09-08.md sec 1):

    per arm    |t_tag| <= 30 ns     the measured peak core, HANDOFF sec 2.1
    mutual     |t1 - t2| <= 20 ns   the two arms prompt with EACH OTHER

Both halves are needed and neither is redundant.  The per-arm cut alone is
nearly free for the trigger arm (it IS the trigger, at zero by construction)
and does the real work on the second arm.  The mutual cut is what a genuine
pair must satisfy and an accidental need not, and it is the half that does not
care which arm was the trigger.

DOWNSTREAM ONLY, BY DECISION.  `DT_WINDOW` in `candidate_filter.py`,
`efficiency.py` and `scintillators.py` is UNCHANGED at the production
(-100, +60): stage 1 keeps every hit and the full +-1000 ns `dt_ns`, so this
window is re-tunable offline without another campaign pass.  Nothing here
writes into the production chain.

WHAT IS AND IS NOT A LEGITIMATE NULL HERE.  Event mixing is the WRONG null for
the *timing* variable (HANDOFF sec 3.2: `dt_ns` is measured relative to each
event's own trigger, so two hits drawn from two singly-triggered events are
both at zero by construction and mixing inverts the answer).  It remains the
RIGHT null for the *opening angle*, which is what S4 uses it for and what this
module compares against.  So: no mixed Delta-t anywhere below; mixed open_deg
throughout.

    python -m sept26_prelim_analysis.tight_coincidence --run run_145
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
from sept26_prelim_analysis import accidental_timing as AT  # noqa: E402
from sept26_prelim_analysis.scintillators import DT_WINDOW, read_slim  # noqa: E402
from sept26_prelim_analysis.slim_export import read_export  # noqa: E402

SCHEMA = 'sept26_prelim/tight_coincidence/1'

#: Per-arm prompt window -- the scintillator peak's own core (HANDOFF sec 2.1:
#: peak core is about +-15 ns on a flat pedestal; +-30 keeps the tail without
#: admitting the ~100 ns of flat background the production window carries).
ARM_CORE_NS = 30.0
#: The two arms must be prompt with each other. Looser than the per-arm cut on
#: purpose: t1 - t2 is a DIFFERENCE of two measurements each with the trigger's
#: ~5 ns resolution, so its own width is sqrt(2) larger, and 20 ns is ~3 sigma
#: of that -- tight against a 171 ns accidental median, loose against the
#: resolution.
MUTUAL_NS = 20.0

#: A single charged particle that crosses the target and punches through BOTH
#: opposite chambers registers as a "pair" at open_deg ~ 180, and it is
#: perfectly time-coincident because it is ONE PARTICLE. So the tight cut does
#: not just remove accidentals -- it ENRICHES this background, which is
#: `HANDOFF_ACCIDENTAL_TIMING.md` sec 5's D12 caveat arriving in the data.
#: Measured on run_145: 8 of the 11 tight opposing pairs sit above 150 deg and
#: four above 175 deg, against 25 % above 150 in the loose-tagged sample.
#: Flagged, never silently dropped -- it is also the cleanest available
#: calibration line for back-to-back geometry.
BACK_TO_BACK_DEG = 170.0

#: The scan grid for the systematic. The published number is (30, 20); these
#: bracket it by a factor ~3 either way so the reader can see whether the
#: answer is a cliff or a plateau.
SCAN_ARM = (10.0, 20.0, 30.0, 50.0, 100.0)
SCAN_MUTUAL = (5.0, 10.0, 20.0, 40.0, 100.0)

#: Runs before the 27 July access. Their per-trigger occupancy is a different
#: detector condition (arm A ~10 clean strips per trigger against 0 after,
#: C ~2 against 0) and they carry about TWICE the INTER and INTRA fraction of
#: every other run -- see OVERNIGHT_2026-09-08.md. Excluded from the pooled
#: campaign spectrum by default, and reported separately, because INTER is the
#: signal topology and pooling would put a detector artefact into it.
PRE_ACCESS_RUNS = ('run_79', 'run_81')


def campaign_pairs(run_subruns: dict, dca_max: float = 30.0,
                   window: tuple = DT_WINDOW, require_both: bool = False,
                   seed: int = 7, src=None,
                   include_pre_access: bool = False) -> pd.DataFrame:
    """`tagged_pairs` pooled over many runs, with the condition stamped.

    `run_subruns` maps run -> list of sub-runs. Runs are processed
    independently (the slim, the pairs and the tag are all per run) and
    concatenated; a run that has no stage-3 tracks yet, or whose `k_arm` never
    certified, contributes nothing and is reported rather than skipped
    silently -- `source_imaging._track_table` filters on `angle_calibrated`,
    so an uncalibrated run looks exactly like an empty one otherwise.
    """
    out, skipped = [], {}
    for run in sorted(run_subruns):
        if run in PRE_ACCESS_RUNS and not include_pre_access:
            skipped[run] = 'pre-27-Jul-access condition (excluded by default)'
            continue
        try:
            m = tagged_pairs(run, run_subruns[run], dca_max=dca_max,
                             window=window, require_both=require_both,
                             seed=seed, src=src)
        except FileNotFoundError as exc:
            skipped[run] = f'missing input: {str(exc).splitlines()[0]}'
            continue
        if m.empty:
            skipped[run] = ('no two-arm-tagged inter-chamber pairs -- check '
                            'angle_calibrated for this run')
            continue
        m['run'] = run
        out.append(m)
    if skipped:
        print('\nruns contributing nothing, and why:')
        for r, why in sorted(skipped.items()):
            print(f'  {r:<10} {why}')
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d['condition'] = np.where(d.run.isin(PRE_ACCESS_RUNS),
                              'pre_access_27jul', 'post_access_27jul')
    return d


# --------------------------------------------------------------------------- #
def _slim(run: str, subruns, prefer_export: bool = True) -> pd.DataFrame:
    """The slim, from the exported parquet if it is there, else the ROOT.

    Campaign-wide only the parquet comes home (`slim_export.py`); on run_145
    the ROOT is still local. Preferring the export means the campaign and the
    single-run case run the same code, and the ROOT path stays as the fallback
    that made run_145 work before the export existed.
    """
    if prefer_export:
        try:
            return read_export(run, subruns)
        except FileNotFoundError:
            pass
    return read_slim(run, subruns)


def tagged_pairs(run: str, subruns, dca_max: float = 30.0,
                 window: tuple = DT_WINDOW, require_both: bool = False,
                 seed: int = 7, src=None) -> pd.DataFrame:
    """Real inter-chamber pairs with a scintillator time on BOTH arms.

    `require_both=False` is the loose wall-OR-plastic per-arm tag, and is the
    default for the same reason `accidental_timing` made it one: the strict
    wall-AND-plastic tag leaves only 8 of run_145's 464 real inter-chamber
    pairs with both arms tagged, which is too few to say anything. The loose
    tag reaches ~16 % of them.

    The returned frame is `two_arm_delta_t`'s -- so it carries `open_deg`,
    `arm1`, `arm2`, `topo`, `t1`, `t2`, `delta_t` -- plus the cut columns.
    """
    slim = _slim(run, subruns)
    tag = AT.arm_tag_time(slim, window=window, require_both=require_both,
                          seed=seed)
    m = AT.two_arm_delta_t(run, subruns, tag, dca_max=dca_max, src=src)
    if m.empty:
        return m
    m['arm_ok'] = (m.t1.abs() <= ARM_CORE_NS) & (m.t2.abs() <= ARM_CORE_NS)
    m['mutual_ok'] = m.delta_t.abs() <= MUTUAL_NS
    m['tight'] = m.arm_ok & m.mutual_ok
    # Opposing-chamber and nearly collinear: one particle through both, not a
    # pair. Only 'opposing' can produce it -- perpendicular chambers cannot be
    # joined by a straight line through the target.
    m['back_to_back'] = (m.topo == 'opposing') & (m.open_deg > BACK_TO_BACK_DEG)
    m['tight_pair'] = m.tight & ~m.back_to_back
    return m


def cut_census(m: pd.DataFrame) -> pd.DataFrame:
    """How many pairs each half of the cut keeps, per topology.

    Reported per topology because the accidental fraction is topology-ordered
    (HANDOFF sec 0: opposing 42 %, perpendicular 16 %), so a cut that is really
    removing accidentals should bite HARDER on perpendicular than on opposing.
    That asymmetry is a falsifiable prediction of this cut, not a decoration.
    """
    rows = []
    for topo, g in list(m.groupby('topo')) + [('(all)', m)]:
        n = len(g)
        rows.append(dict(
            topo=topo, n_tagged=n,
            n_arm_ok=int(g.arm_ok.sum()),
            n_mutual_ok=int(g.mutual_ok.sum()),
            n_tight=int(g.tight.sum()),
            n_b2b_tagged=int(g.back_to_back.sum()),
            n_b2b_in_tight=int((g.tight & g.back_to_back).sum()),
            n_tight_pair=int(g.tight_pair.sum()),
            frac_tight=round(float(g.tight.mean()), 4) if n else np.nan,
            median_abs_dt=round(float(g.delta_t.abs().median()), 1) if n else np.nan))
    return pd.DataFrame(rows)


def window_scan(m: pd.DataFrame) -> pd.DataFrame:
    """Surviving pairs and their mean opening angle over the (arm, mutual) grid.

    This is the systematic the cut needs: a single (30, 20) number is only
    meaningful if the answer does not swing wildly just outside it.
    """
    rows = []
    for a in SCAN_ARM:
        for mu in SCAN_MUTUAL:
            sel = m[(m.t1.abs() <= a) & (m.t2.abs() <= a)
                    & (m.delta_t.abs() <= mu)]
            rows.append(dict(arm_ns=a, mutual_ns=mu, n=len(sel),
                             frac=round(len(sel) / len(m), 4) if len(m) else np.nan,
                             mean_open=round(float(sel.open_deg.mean()), 2)
                             if len(sel) else np.nan,
                             median_open=round(float(sel.open_deg.median()), 2)
                             if len(sel) else np.nan))
    return pd.DataFrame(rows)


def angle_compare(m: pd.DataFrame, run: str, subruns,
                  dca_max: float = 30.0, bins: int = 12,
                  src=None) -> pd.DataFrame:
    """Opening-angle histogram: tight pairs, loose-tagged pairs, event-mixed.

    Mixed is the S4 null and is legitimate for the ANGLE (HANDOFF sec 3.2 rules
    it out for the timing only). It is normalised to the tight count so the
    comparison is of SHAPE -- the tight sample is small by construction and a
    raw overlay would say nothing.
    """
    from sept26_prelim_analysis import source_imaging as SI
    _, mixed = SI.vertices(run, subruns, dca_max, src=src)
    mixed = mixed[mixed.topology == 'inter']

    edges = np.linspace(0.0, 180.0, bins + 1)
    tight = m[m.tight]
    out = pd.DataFrame({'lo': edges[:-1], 'hi': edges[1:]})
    for name, s in (('tight', tight.open_deg),
                    ('tight_pair', m[m.tight_pair].open_deg),
                    ('tagged', m.open_deg),
                    ('mixed', mixed.open_deg)):
        h, _ = np.histogram(s.dropna(), bins=edges)
        out[f'n_{name}'] = h
    # Shape comparison: mixed and tagged scaled to the tight total.
    for name in ('tagged', 'mixed'):
        tot = out[f'n_{name}'].sum()
        out[f'exp_{name}'] = (out[f'n_{name}'] / tot * out.n_tight.sum()
                              if tot else np.nan)
    return out


def chi2_vs(out: pd.DataFrame, name: str) -> dict:
    """Poisson chi2/dof of the tight histogram against a scaled comparison.

    Bins with an expectation below 1 are dropped rather than kept with a
    hand-waved continuity correction -- with run_145's statistics most of the
    range is empty and keeping them would manufacture agreement.
    """
    o, e = out.n_tight.to_numpy(float), out[f'exp_{name}'].to_numpy(float)
    keep = np.isfinite(e) & (e >= 1.0)
    if keep.sum() < 2:
        return dict(comparison=name, chi2=np.nan, dof=int(keep.sum()),
                    chi2_dof=np.nan, note='too few populated bins to test')
    chi2 = float(np.sum((o[keep] - e[keep]) ** 2 / e[keep]))
    dof = int(keep.sum() - 1)          # -1 for the normalisation
    return dict(comparison=name, chi2=round(chi2, 2), dof=dof,
                chi2_dof=round(chi2 / dof, 2) if dof > 0 else np.nan, note='')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--dca-max', type=float, default=30.0)
    ap.add_argument('--strict-tag', action='store_true',
                    help='wall AND plastic per arm (default: wall OR plastic '
                         '-- strict leaves ~8 pairs on run_145)')
    ap.add_argument('--bins', type=int, default=12)
    ap.add_argument('--all-runs', action='store_true',
                    help='pool every run with stage-3 tracks, instead of one '
                         '--run. Excludes the pre-27-Jul-access runs unless '
                         '--include-pre-access is given.')
    ap.add_argument('--include-pre-access', action='store_true',
                    help='include run_79/run_81 -- a DIFFERENT detector '
                         'condition with ~2x the INTER fraction')
    ap.add_argument('--campaign', action='store_true',
                    help='read tracks from <out>/stage3_campaign instead of '
                         '<out>/stage3_fullpass (the run_145 pass)')
    a = ap.parse_args()
    subruns = a.subruns.split(',')

    src = paths.out('stage3_campaign') if a.campaign else None
    if a.all_runs:
        import re as _re
        base = src or paths.out('stage3_fullpass')
        rs = {}
        for p in sorted(base.glob('tracks_run_*_stat090_*.parquet')):
            mm = _re.match(r'tracks_(run_\d+)_(stat090_\d+)\.parquet$', p.name)
            if mm:
                rs.setdefault(mm.group(1), []).append(mm.group(2))
        print(f'pooling {len(rs)} run(s) from {base}')
        m = campaign_pairs(rs, dca_max=a.dca_max, require_both=a.strict_tag,
                           src=src, include_pre_access=a.include_pre_access)
        stem_runs = 'campaign'
    else:
        m = tagged_pairs(a.run, subruns, dca_max=a.dca_max,
                         require_both=a.strict_tag, src=src)
        stem_runs = a.run
    out = paths.out('tight_coincidence')
    if m.empty:
        print('no two-arm-tagged inter-chamber pairs -- nothing to do')
        return 1

    census = cut_census(m)
    scan = window_scan(m)
    hist = angle_compare(m, a.run, subruns, dca_max=a.dca_max,
                         bins=a.bins, src=src)
    tests = [chi2_vs(hist, 'tagged'), chi2_vs(hist, 'mixed')]

    print(f'\ntight coincidence: |t_arm| <= {ARM_CORE_NS:.0f} ns, '
          f'|t1-t2| <= {MUTUAL_NS:.0f} ns\n')
    print('cut census\n')
    print(census.to_string(index=False))
    print('\nopening angle (counts)\n')
    print(hist.round(2).to_string(index=False))
    print('\nshape tests -- tight against each comparison, normalised\n')
    print(pd.DataFrame(tests).to_string(index=False))
    print('\nwindow scan\n')
    print(scan.to_string(index=False))

    stem = stem_runs
    m.to_parquet(out / f'pairs_tight_{stem}.parquet', index=False)
    census.to_csv(out / f'cut_census_{stem}.csv', index=False)
    scan.to_csv(out / f'window_scan_{stem}.csv', index=False)
    hist.to_csv(out / f'angle_hist_{stem}.csv', index=False)
    meta = dict(schema=SCHEMA, run=a.run, subruns=subruns,
                arm_core_ns=ARM_CORE_NS, mutual_ns=MUTUAL_NS,
                dt_window=list(DT_WINDOW), dca_max=a.dca_max,
                strict_tag=bool(a.strict_tag),
                back_to_back_deg=BACK_TO_BACK_DEG,
                n_tagged=int(len(m)), n_tight=int(m.tight.sum()),
                n_b2b_tagged=int(m.back_to_back.sum()),
                n_b2b_in_tight=int((m.tight & m.back_to_back).sum()),
                n_tight_pair=int(m.tight_pair.sum()),
                tests=tests)
    (out / f'tight_coincidence_{stem}.meta.json').write_text(
        json.dumps(meta, indent=1))
    print(f'\n  -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
