#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
campaign_imaging.py -- image the He-3 capsule ONCE PER RUN, as calibration QA.

WHY THIS IS THE RIGHT QA, AND WHY IT IS THE ONE TO RUN FIRST.  The pointing
crossing `source_imaging.py` measures is **scale-free**: multiplying every angle
by ``k`` scales the band's slope and its intercept together and leaves
``-intercept/slope`` untouched.  So the crossing is the one geometric
observable this analysis has that the angle-scale problem CANNOT touch, and
running it per run separates two questions that have been travelling together:

    does the reconstruction still point at the same place run to run?   <- here
    is the angle scale the same run to run?                             <- k_arm

`k_arm` says no to the second (8-29 % run to run, one contiguous 48-hour
block).  If the crossing says yes to the first, the fault is in the angle
scale alone and the geometry, the alignment and the track finding are sound --
which is what makes an opening-angle spectrum worth building at all.

THE THREE NUMBERS PER RUN, in the order they should be read:

  1. **the crossing per chamber** -- A and C both measure global X, B and D
     both measure global Z, so X is measured twice from opposite sides.
  2. **the mean of A and C** -- the capsule's X.  This is the physics number.
  3. **HALF their difference** -- the relative in-plane alignment of the two
     chambers.  A single chamber cannot produce it and it is the reason the
     opposing pair is worth having.

Z gets only D in practice: B has no drift field, its band is fitted on a few
hundred tracks, and its run-to-run scatter is several millimetres.  B is
reported and never averaged into a verdict.

WHAT IS NOT SCALE-FREE, AND SO IS THE SECOND HALF OF THIS PAGE.  ``target_y_mm``
is built from the calibrated direction, so the y offset against the polycone
forward model DOES move with ``k``.  Reading the two together is the point: a
run whose crossing is normal and whose y offset is not has an angle-scale
fault and not a geometry fault.

    python -m sept26_prelim_analysis.campaign_imaging --jobs 8
    python -m sept26_prelim_analysis.campaign_imaging --runs run_116,run_145
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

SCHEMA = 'sept26_prelim/campaign_imaging/1'
ARMS = ('A', 'B', 'C', 'D')
#: The two chambers that measure global X, from opposite sides.  Their mean is
#: the source and half their difference is the alignment.
X_PAIR = ('A', 'C')
#: Runs before the 27 July access -- a different detector condition (arm A's
#: dead x-view connector).  Reported apart, never pooled into a verdict.
PRE_ACCESS_RUNS = ('run_79', 'run_81')
#: The contiguous 48-hour block, 3-5 Aug, in which every arm's `k` rises
#: together (STATUS.md, 2026-09-10).  The whole reason this module exists is to
#: ask whether a SCALE-FREE observable moves inside it too.
K_BLOCK = (128, 147)


def run_number(run: str) -> int:
    return int(run.split('_')[1])


def condition(run: str) -> str:
    return ('pre_access_27jul' if run in PRE_ACCESS_RUNS
            else 'post_access_27jul')


def in_block(run: str) -> bool:
    return K_BLOCK[0] <= run_number(run) <= K_BLOCK[1]


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def subruns_of(reco: Path, run: str) -> tuple:
    """(usable sub-runs, dropped -> why).

    A sub-run is usable only if ALL FOUR arms have a merged table.  Dropping
    the whole sub-run rather than the missing arm keeps the four crossings on
    the same beam, which is what makes the A-C difference an alignment number
    instead of a comparison of two different samples.  One sub-run in the
    campaign fails this -- ``run_104/stat090_0016``, whose FEUs 02, 03 and 07
    came off the DAQ empty (HANDOFF_FULLPASS sec 4) -- and it costs that run
    one sub-run of twenty.
    """
    d = reco / run
    keep, drop = [], {}
    for p in sorted(x for x in d.iterdir()
                    if x.is_dir() and x.name.startswith('stat090_')):
        miss = [a for a in ARMS
                if not (p / f'mx17_{a}' / 'events_prelim.parquet').exists()]
        if miss:
            drop[p.name] = f'no merged table for arm(s) {"".join(miss)}'
        else:
            keep.append(p.name)
    return keep, drop


def one_run(run: str, reco: str, with_y: bool, n_model: int) -> tuple:
    """(crossings, y_rows, error).  Runs in a worker process."""
    from sept26_prelim_analysis import source_imaging as SI
    try:
        reco = Path(reco)
        subs, dropped = subruns_of(reco, run)
        if not subs:
            return run, None, None, 'no usable sub-runs under the reco tree'
        for s, why in dropped.items():
            print(f'  {run}/{s}: dropped -- {why}', flush=True)
        T = SI.transverse(run, subs, str(reco / run))
        T.insert(0, 'run', run)
        Y = None
        if with_y:
            Y = _y_rows(SI, run, subs, n_model)
            if Y is not None:
                Y.insert(0, 'run', run)
        return run, T, Y, ''
    except Exception:
        return run, None, None, traceback.format_exc(limit=3)


def _y_rows(SI, run: str, subruns, n_model: int):
    """y offset against the polycone model, per arm -- the k-DEPENDENT half.

    `source_imaging.y_compare` is reused rather than reimplemented, except for
    the model size, which is dropped from 400 k to `n_model` because this runs
    36 times and the model's own sampling error at 100 k is already well below
    the run-to-run spread it is being used to measure.
    """
    obs = SI.y_measured(run, subruns, 30.0)
    rows = []
    for arm in ARMS:
        g = obs[obs.arm == arm]
        if len(g) < 200:
            rows.append(dict(arm=arm, n=int(len(g))))
            continue
        pred = SI.y_forward_model(run, arm, n=n_model)
        o = g.target_y_mm.to_numpy()
        o_iqr = float(np.subtract(*np.percentile(o, [75, 25])))
        p_iqr = float(np.subtract(*np.percentile(pred, [75, 25])))
        rows.append(dict(
            arm=arm, n=int(len(g)),
            obs_median=float(np.median(o)), pred_median=float(np.median(pred)),
            offset_mm=float(np.median(o) - np.median(pred)),
            obs_iqr=o_iqr, pred_iqr=p_iqr,
            width_ratio=o_iqr / p_iqr if p_iqr else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# reductions
# --------------------------------------------------------------------------- #
def per_run(T: pd.DataFrame, variant: str = 'baseline') -> pd.DataFrame:
    """Per (run, arm): the crossing averaged over sub-runs, with two errors.

    ``err_stat`` is the bootstrap propagated over sub-runs; ``err_repro`` is the
    spread BETWEEN sub-runs and is the honest one -- it carries everything that
    changes within a run and nothing a bootstrap can see.  A run with one
    sub-run has no ``err_repro`` and says so rather than reporting zero.
    """
    g = T[(T.variant == variant) & T.mm.notna()]
    rows = []
    for (run, arm), h in g.groupby(['run', 'arm']):
        rows.append(dict(
            run=run, num=run_number(run), arm=arm,
            axis=h.axis.iloc[0], n_subruns=int(len(h)), n=int(h.n.sum()),
            mm=float(h.mm.mean()),
            err_stat=float(np.sqrt((h.err ** 2).sum()) / len(h)),
            err_repro=float(h.mm.std(ddof=1)) if len(h) > 1 else np.nan,
            condition=condition(run), k_block=in_block(run)))
    return pd.DataFrame(rows).sort_values(['num', 'arm'], ignore_index=True)


def per_arm(R: pd.DataFrame) -> pd.DataFrame:
    """Campaign stability per arm -- is per-run imaging precise enough to use?

    The number that answers it is ``std_mm`` against ``median_err_repro``: if
    the run-to-run scatter is no larger than what one run's own sub-runs
    already show, there is no run-to-run effect to see and the crossing is
    simply stable.
    """
    rows = []
    for arm, g in R[R.condition == 'post_access_27jul'].groupby('arm'):
        v = g.mm
        rows.append(dict(
            arm=arm, axis=g.axis.iloc[0], n_runs=int(len(g)),
            median_mm=float(v.median()), std_mm=float(v.std(ddof=1)),
            p10_mm=float(v.quantile(0.10)), p90_mm=float(v.quantile(0.90)),
            min_mm=float(v.min()), max_mm=float(v.max()),
            median_err_stat=float(g.err_stat.median()),
            median_err_repro=float(g.err_repro.median()),
            median_n=float(g.n.median())))
    return pd.DataFrame(rows).sort_values('arm', ignore_index=True)


def axis_per_run(R: pd.DataFrame) -> pd.DataFrame:
    """Per run: the source in X, and the A-C alignment that comes with it.

    Only the X pair produces both.  Z is carried as D's single number, with B
    alongside where it exists, and the module refuses to average them: B's
    run-to-run scatter is several millimetres and averaging would import it
    into a number D measures ten times better.
    """
    rows = []
    for run, g in R.groupby('run'):
        v = {r.arm: (r.mm, r.err_stat) for r in g.itertuples()}
        a, c = v.get('A'), v.get('C')
        d, b = v.get('D'), v.get('B')
        rows.append(dict(
            run=run, num=run_number(run), condition=condition(run),
            k_block=in_block(run),
            x_source_mm=(0.5 * (a[0] + c[0]) if a and c else np.nan),
            x_align_half_diff_mm=(0.5 * (a[0] - c[0]) if a and c else np.nan),
            x_err_mm=(0.5 * np.hypot(a[1], c[1]) if a and c else np.nan),
            x_A_mm=a[0] if a else np.nan, x_C_mm=c[0] if c else np.nan,
            z_D_mm=d[0] if d else np.nan, z_B_mm=b[0] if b else np.nan,
            n_arms=int(len(g))))
    return pd.DataFrame(rows).sort_values('num', ignore_index=True)


def block_test(R: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """Does a SCALE-FREE observable move inside the 128-147 `k` excursion?

    The falsifier this module was built for.  If the excursion were a real
    drift-velocity change, ``k`` would move and the crossing would not, because
    the crossing divides the velocity out.  Anything the crossing DOES do
    inside the block is therefore not a velocity effect, and points at the
    illumination or the geometry instead.
    """
    post = R[R.condition == 'post_access_27jul']
    rows = []
    for arm, g in post.groupby('arm'):
        i, o = g[g.k_block].mm, g[~g.k_block].mm
        if len(i) < 2 or len(o) < 2:
            continue
        from scipy import stats
        rows.append(dict(
            quantity=f'crossing {arm}', axis=g.axis.iloc[0],
            n_out=int(len(o)), n_in=int(len(i)),
            median_out=float(o.median()), median_in=float(i.median()),
            shift_mm=float(i.median() - o.median()),
            shift_over_out_sigma=float((i.median() - o.median())
                                       / o.std(ddof=1)) if o.std(ddof=1) else np.nan,
            p_mannwhitney=float(stats.mannwhitneyu(i, o).pvalue)))
    ap = A[A.condition == 'post_access_27jul']
    for col, name in (('x_source_mm', 'X source (A,C mean)'),
                      ('x_align_half_diff_mm', 'A-C alignment half-difference')):
        i, o = ap[ap.k_block][col].dropna(), ap[~ap.k_block][col].dropna()
        if len(i) < 2 or len(o) < 2:
            continue
        from scipy import stats
        rows.append(dict(
            quantity=name, axis='X', n_out=int(len(o)), n_in=int(len(i)),
            median_out=float(o.median()), median_in=float(i.median()),
            shift_mm=float(i.median() - o.median()),
            shift_over_out_sigma=float((i.median() - o.median())
                                       / o.std(ddof=1)) if o.std(ddof=1) else np.nan,
            p_mannwhitney=float(stats.mannwhitneyu(i, o).pvalue)))
    return pd.DataFrame(rows)


def read_k(kcal: Path = None) -> pd.DataFrame:
    """The per-run angle scale, as `k_arm` certified it -- one row per (run, arm).

    Only ``apply`` is read.  A raw fit value that never certified is NOT a
    measurement (STATUS.md, 2026-09-10) and must not appear beside one that is.
    """
    kcal = kcal or paths.out('kcal')
    rows = []
    for p in sorted(kcal.glob('k_arm_run_*.json')):
        stem = p.name[len('k_arm_'):-len('.json')]
        if '.' in stem:          # k_arm_run_86.fullpass.json and friends
            continue
        ap = json.loads(p.read_text()).get('apply', {})
        for arm, k in ap.items():
            rows.append(dict(run=stem, arm=arm, k=float(k)))
    return pd.DataFrame(rows)


def versus_k(R: pd.DataFrame, K: pd.DataFrame) -> pd.DataFrame:
    """Crossing against `k`, per arm.  A scale-free quantity should not care.

    A non-zero correlation does not mean the crossing is scale-dependent -- the
    algebra rules that out -- it means something moved that changed BOTH, which
    is a lead about the excursion and not a fault in this estimator.
    """
    from scipy import stats
    m = R[R.condition == 'post_access_27jul'].merge(K, on=['run', 'arm'])
    rows = []
    for arm, g in m.groupby('arm'):
        if len(g) < 6:
            continue
        rho, p = stats.spearmanr(g.mm, g.k)
        rows.append(dict(arm=arm, n_runs=int(len(g)), rho=float(rho),
                         p=float(p),
                         k_median=float(g.k.median()),
                         mm_median=float(g.mm.median())))
    return pd.DataFrame(rows)


def verdict(PA: pd.DataFrame, A: pd.DataFrame, B: pd.DataFrame) -> dict:
    """One paragraph's worth of machine-readable answer."""
    post = A[A.condition == 'post_access_27jul']
    x = post.x_source_mm.dropna()
    al = post.x_align_half_diff_mm.dropna()
    def arm(a, col):
        r = PA[PA.arm == a]
        return float(r[col].iloc[0]) if len(r) else float('nan')
    worst = B.loc[B.shift_over_out_sigma.abs().idxmax()] if len(B) else None
    return dict(
        n_runs=int(len(post)),
        x_source_mm=float(x.median()), x_source_std_mm=float(x.std(ddof=1)),
        x_align_half_diff_mm=float(al.median()),
        x_align_std_mm=float(al.std(ddof=1)),
        z_D_mm=float(post.z_D_mm.dropna().median()),
        z_D_std_mm=float(post.z_D_mm.dropna().std(ddof=1)),
        std_A_mm=arm('A', 'std_mm'), std_C_mm=arm('C', 'std_mm'),
        std_D_mm=arm('D', 'std_mm'), std_B_mm=arm('B', 'std_mm'),
        block_largest=(None if worst is None else dict(
            quantity=str(worst.quantity), shift_mm=float(worst.shift_mm),
            in_sigma=float(worst.shift_over_out_sigma),
            p=float(worst.p_mannwhitney))),
        stable=bool(x.std(ddof=1) < 1.0 and al.std(ddof=1) < 1.5))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--reco', default=None,
                    help='merged reco tree; default <out>/reco_fullpass '
                         '(the condor FULL pass -- NOT <out>/fullpass, which '
                         'is the allowlist pass)')
    ap.add_argument('--runs', default=None,
                    help='comma-separated; default every run under --reco')
    ap.add_argument('--out', default=None, help='default <out>/imaging_campaign')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--n-model', type=int, default=100_000,
                    help='y forward-model throws per (run, arm)')
    ap.add_argument('--no-y', action='store_true',
                    help='skip the k-DEPENDENT y comparison (the scale-free '
                         'crossing alone)')
    a = ap.parse_args()

    reco = Path(a.reco) if a.reco else paths.out('reco_fullpass')
    paths.require(reco, 'the merged reco tree')
    runs = ([r for r in a.runs.split(',') if r] if a.runs else
            sorted((p.name for p in reco.iterdir()
                    if p.is_dir() and p.name.startswith('run_')),
                   key=run_number))
    od = Path(a.out) if a.out else paths.out('imaging_campaign')
    od.mkdir(parents=True, exist_ok=True)

    print(f'{len(runs)} run(s) from {reco}\n')
    Ts, Ys, bad = [], [], {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(one_run, r, str(reco), not a.no_y, a.n_model): r
                for r in runs}
        for f in as_completed(futs):
            run, T, Y, err = f.result()
            if err:
                bad[run] = err.strip().splitlines()[-1]
                print(f'  {run:<10} FAILED  {bad[run]}', flush=True)
                continue
            Ts.append(T)
            if Y is not None and len(Y):
                Ys.append(Y)
            n = int(T[(T.variant == "baseline") & T.mm.notna()].n.sum())
            print(f'  {run:<10} ok      {n:>7,} coincident tracks', flush=True)

    if not Ts:
        print('\nnothing imaged -- every run failed')
        return 1
    T = pd.concat(Ts, ignore_index=True)
    R = per_run(T)
    PA = per_arm(R)
    A = axis_per_run(R)
    B = block_test(R, A)
    K = read_k()
    V = versus_k(R, K)
    Y = pd.concat(Ys, ignore_index=True) if Ys else pd.DataFrame()

    T.to_csv(od / 'crossings.csv', index=False)
    R.to_csv(od / 'per_run.csv', index=False)
    PA.to_csv(od / 'per_arm.csv', index=False)
    A.to_csv(od / 'axis_per_run.csv', index=False)
    B.to_csv(od / 'block_test.csv', index=False)
    V.to_csv(od / 'versus_k.csv', index=False)
    if len(Y):
        Y.to_csv(od / 'y_per_run.csv', index=False)
    vd = verdict(PA, A, B)
    json.dump(dict(schema=SCHEMA, reco=str(reco), runs=runs,
                   n_runs_ok=len(Ts), failed=bad,
                   k_block=list(K_BLOCK), pre_access=list(PRE_ACCESS_RUNS),
                   n_model=a.n_model, with_y=not a.no_y, verdict=vd),
              open(od / 'campaign_imaging.meta.json', 'w'), indent=1,
              default=float)

    print('\nPER ARM, post-access runs -- is the crossing stable?')
    print(PA.round(3).to_string(index=False))
    print('\nPER RUN -- the source in X, and the A-C alignment')
    print(A.round(2).to_string(index=False))
    print('\nTHE 128-147 BLOCK, on a SCALE-FREE observable')
    print(B.round(3).to_string(index=False) if len(B) else '  (not enough runs)')
    print('\nCROSSING vs k -- should be flat if nothing else moved')
    print(V.round(3).to_string(index=False) if len(V) else '  (no k)')
    if len(Y):
        print('\nY OFFSET against the polycone model -- the k-DEPENDENT half')
        print(Y.groupby('arm')[['offset_mm', 'width_ratio']]
              .describe().round(2).to_string())
    print('\nVERDICT')
    print(json.dumps(vd, indent=1, default=float))
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
