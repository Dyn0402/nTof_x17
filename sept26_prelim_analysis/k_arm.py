#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_arm.py -- the in-situ angle scale, per chamber, and whether it is trustworthy.

WHAT k IS.  The waveform fit measures a transverse SPEED, ``w`` [mm/ns]: how
fast the track's position slides across the strips as charge drifts in.  Turning
that into an angle needs the drift velocity, and only there:

    tan(theta) = w / v

``w`` is measured.  ``v`` is a Magboltz prior -- 42.6 um/ns for Ar/iso 90/10 --
transferred from the bench, never measured in these chambers with this gas.  So
every angle carries an unknown multiplicative error, and

    k = v_assumed / v_true,     tan_true = k * tan_reco,     v_true = 42.6 / k

is the one number that fixes it.  k is not a fudge factor: it is a per-chamber
drift-velocity measurement expressed in the units the reconstruction uses.

HOW IT IS MEASURED.  The tracks come from a point-ish source -- the He-3 capsule
on the beam axis, 234.6 mm from each strip plane -- so position and angle are not
independent.  A track crossing the plane at in-plane offset ``u`` must have come
in at tan = (u - foot_x)/d_perp.  Three estimators fall out, and they fail in
different ways, which is exactly why all three are computed:

  band     the slope of median(tan) vs u.  Uses the gradient only, so a constant
           angle offset cannot fool it -- but background tracks that are NOT
           from the target flatten the band and inflate k without bound.
  track    the median of the per-track ratio tan_expected / tan_reco.  Robust to
           a few outliers, biased by any population that is genuinely not
           target-pointing.
  focus    scan k, back-project every track, and take the k that puts the MOST
           tracks within a FIXED radius of the beam axis.  The most physical,
           and the only one that uses both planes -- but it couples x and y.

           The fixed radius is the point.  ``run145_target_imaging``'s own scan
           minimises ``r_core``, the median of the sub-30 mm population -- a
           median conditioned on a cut that k itself moves, so it can be
           "improved" by shrinking the core rather than focusing it.  That is
           why the imaging source says its scan "rails", and why this module
           re-derives the focus estimator instead of reading ``k_opt``.  A
           count inside a fixed radius has no such freedom: the selection does
           not move with the parameter.

AGREEMENT BETWEEN THE THREE IS THE MEASUREMENT.  A single estimator quoting a
number proves nothing; three independent failure modes landing on the same value
is a calibration.  So this module quotes the spread, not a point, and refuses to
certify a chamber whose estimators disagree.

The sample is the pointing-coincident one -- the track extrapolates to the wall
segment AND the plastic bar that actually fired -- because on the full sample the
estimators diverge by an order of magnitude, which is the background talking.

WHAT IS NOT MEASURED HERE.  The SOURCE POSITION (the zero crossing of the band,
-intercept/slope) is scale-free: multiply every angle by k and both the intercept
and the slope scale with it.  It is reported alongside as an independent check on
the geometry, and it is emphatically NOT how k is obtained -- a good target image
in x is not evidence that v is right.

    python -m sept26_prelim_analysis.k_arm --run run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

SCHEMA = 'sept26_prelim/k_arm/1'
ARMS = ('A', 'B', 'C', 'D')

#: The three estimators, and where each lives in ``imaging_summary.json``.
#: All read the POINTING-COINCIDENT sample -- see the module docstring.
ESTIMATORS = ('band', 'track', 'focus')

#: A chamber is certified only if the three estimators agree this well
#: (max/min - 1) AND the value reproduces this well between sub-runs.
SPREAD_MAX = 0.25
REPRO_MAX = 0.10

#: The prior every bundle carries.  Not measured in these chambers.
V_BUNDLE_PRIOR = 42.6


#: The focus scan.  Radii are fixed in mm so the objective's selection cannot
#: move with k; 30 mm is the imaging's own core radius and 10 mm is the He-3
#: capsule's outer radius (``geometry.HE3_R_MAX``).
K_GRID = np.round(np.arange(0.60, 2.61, 0.05), 2)
FOCUS_RADII_MM = (30.0, 10.0)


def coincident_tracks(run: str, sub: str, arm: str, merged_dir: str):
    """(x_local, y_local, tan_x, tan_y) for the pointing-coincident tracks.

    The clean sample: both planes fit, a gated 3D track, and the track
    extrapolates onto the wall segment AND the plastic bar that actually fired.
    Everything here delegates to run145_target_imaging so the frame, the
    in-plane sign and the coincidence geometry are the ones already measured.
    """
    import pandas as pd
    from ntof_tracking import run145_target_imaging as TI

    p = paths.require(os.path.join(merged_dir, sub, f'mx17_{arm}',
                                   'events_prelim.parquet'),
                      f'merged full-pass table for {arm}/{sub}')
    df = pd.read_parquet(p)
    d = os.path.join(str(paths.root('runs')), run, sub, 'ntof_hits')
    slim = sorted(f for f in os.listdir(d) if f.endswith('.root'))
    if not slim:
        raise FileNotFoundError(f'no slim n_TOF file under {d}')
    sel = (df['x_ok'].to_numpy() & df['y_ok'].to_numpy()
           & (df['n_tracks'].to_numpy() > 0))
    coin, _ = TI.pointing_coincidence(os.path.join(d, slim[0]), arm, df, sel,
                                      foot_x=TI.PINWHEEL[arm])
    m = coin & sel
    g = df[m]
    return (TI.local_x(g['x_p0'].to_numpy()),
            g['y_p0'].to_numpy() - TI.STRIP_MAP_HALF,
            g['x_tan_theta'].to_numpy(), g['y_tan_theta'].to_numpy())


def focus_scan(xl, yl, tx, ty, tr, gap_mm: float = 30.0) -> dict:
    """k that maximises the count of tracks pointing within each fixed radius.

    Returns the per-radius optima and the grid, so a flat scan is visible as a
    flat scan rather than collapsing to whichever bin won by one track.
    """
    from ntof_tracking import run145_target_imaging as TI

    counts = {r: [] for r in FOCUS_RADII_MM}
    med = []
    for k in K_GRID:
        P0 = tr.local_to_global(xl, yl, np.zeros_like(xl))
        P1 = tr.local_to_global(xl - tx * k * gap_mm, yl - ty * k * gap_mm,
                                np.full_like(xl, gap_mm))
        D = P1 - P0
        D = D / np.linalg.norm(D, axis=-1, keepdims=True)
        r, _y, _xz = TI.axis_approach(P0, D)
        for rad in FOCUS_RADII_MM:
            counts[rad].append(int((r < rad).sum()))
        med.append(float(np.median(r)))
    best = {rad: float(K_GRID[int(np.argmax(c))]) for rad, c in counts.items()}
    best['median'] = float(K_GRID[int(np.argmin(med))])
    # How well the objective is actually constrained: the width of the k range
    # holding >=95 % of the peak count at 30 mm.  A flat scan is not a
    # measurement, and this is what says so.
    c30 = np.array(counts[FOCUS_RADII_MM[0]], float)
    within = K_GRID[c30 >= 0.95 * c30.max()]
    return dict(best=best, k=float(np.median(list(best.values()))),
                plateau=[float(within.min()), float(within.max())],
                n=int(len(xl)), grid=K_GRID.tolist(),
                counts={str(r): c for r, c in counts.items()},
                median_miss=med)


def estimators(rec: dict) -> dict:
    """The three coincident-sample k values from one arm's imaging record."""
    # NOT k_phys.  run145_target_imaging sets `k_phys = k_track_coincident`
    # verbatim (it is the value it trusts for the QUOTED image, not a separate
    # measurement), so reading it here would count the per-track estimator
    # twice and make any arm look self-consistent.  The genuine focus estimator
    # is k_opt: the k that minimises r_core over the scan.  Its own caveat,
    # from the imaging source, is that the naive scan can rail -- which is
    # precisely the kind of failure the three-way spread is here to catch.
    return {
        'band': rec.get('pointing_x_coincident', {}).get('implied_k'),
        'track': rec.get('k_track_coincident', {}).get('median'),
        'focus': rec.get('_focus_k'),        # filled by build(), see focus_scan
    }


def combine(per_sub: dict) -> dict:
    """One arm's verdict from its per-sub-run estimator sets.

    ``per_sub`` maps sub-run -> {estimator: k}.  Returns the central value, the
    two spreads that decide the verdict, and the verdict itself.  Nothing is
    silently dropped: a missing estimator makes the arm uncertifiable rather
    than shrinking the spread it is judged on.
    """
    subs = sorted(per_sub)
    vals = {e: [per_sub[s].get(e) for s in subs] for e in ESTIMATORS}
    missing = [e for e, v in vals.items() if any(x is None or not np.isfinite(x)
                                                 for x in v)]

    flat = [x for e in ESTIMATORS for x in vals[e]
            if x is not None and np.isfinite(x)]
    if not flat:
        return dict(k=None, verdict='NO DATA', reason='no estimator returned')

    # central value: median over estimators of the per-estimator sub-run mean,
    # so one wild estimator cannot drag it and one wild sub-run cannot either.
    per_est = {e: float(np.mean([x for x in vals[e] if x is not None
                                 and np.isfinite(x)]))
               for e in ESTIMATORS
               if any(x is not None and np.isfinite(x) for x in vals[e])}
    k = float(np.median(list(per_est.values())))

    lo, hi = min(per_est.values()), max(per_est.values())
    spread = hi / lo - 1.0 if lo > 0 else np.inf
    repro = 0.0
    for e in ESTIMATORS:
        v = [x for x in vals[e] if x is not None and np.isfinite(x)]
        if len(v) > 1 and min(v) > 0:
            repro = max(repro, max(v) / min(v) - 1.0)

    if missing:
        verdict, reason = 'NOT CALIBRATED', \
            f'estimator(s) missing: {", ".join(missing)}'
    elif spread > SPREAD_MAX and repro > REPRO_MAX:
        verdict, reason = 'NOT CALIBRATED', \
            (f'estimators disagree by {spread:.0%} (max {SPREAD_MAX:.0%}) AND '
             f'do not reproduce between sub-runs ({repro:.0%})')
    elif spread > SPREAD_MAX:
        verdict, reason = 'PROVISIONAL', \
            (f'estimators disagree by {spread:.0%} (max {SPREAD_MAX:.0%}); '
             f'reproduces between sub-runs to {repro:.0%}')
    elif repro > REPRO_MAX:
        verdict, reason = 'PROVISIONAL', \
            f'does not reproduce between sub-runs ({repro:.0%})'
    else:
        verdict, reason = 'CALIBRATED', \
            (f'three estimators agree to {spread:.0%}, reproduce to {repro:.0%}')

    return dict(k=k, k_lo=lo, k_hi=hi, spread=spread, repro=repro,
                per_estimator=per_est,
                per_subrun={s: per_sub[s] for s in subs},
                verdict=verdict, reason=reason,
                v_insitu=V_BUNDLE_PRIOR / k if k else None)


def source_check(recs: list) -> dict:
    """The scale-free source position per sub-run -- a geometry check, not k."""
    out = {}
    for sub, rec in recs:
        p = rec.get('pointing_x_coincident', {})
        out[sub] = dict(axis=p.get('source_measured_axis'),
                        mm=p.get('source_measured_mm'),
                        err=p.get('zero_crossing_err'),
                        n=p.get('n'))
    return out


def build(run: str, subruns, merged_dir: str) -> dict:
    import json as _json
    from pathlib import Path
    from ntof_tracking.reco import geometry as G

    imgs = {}
    for sub in subruns:
        p = paths.require(paths.out('kcal') / f'{run}_{sub}' / 'imaging_summary.json',
                          f'target imaging for {sub} -- run '
                          f'ntof_tracking.run145_target_imaging first')
        imgs[sub] = {r['arm']: r for r in json.load(open(p))['results']}

    # The focus estimator is re-derived here, per sub-run, on the coincident
    # sample -- see focus_scan for why the imaging's k_opt is not used.
    cfg = _json.loads((Path(str(paths.root('runs'))) / run
                       / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)
    scans = {}
    for sub in subruns:
        for a in ARMS:
            if a not in imgs[sub]:
                continue
            try:
                xl, yl, tx, ty = coincident_tracks(run, sub, a, merged_dir)
            except FileNotFoundError as exc:
                print(f'  [focus] {a}/{sub}: skipped -- {exc}')
                continue
            if len(xl) < 200:
                print(f'  [focus] {a}/{sub}: only {len(xl)} coincident tracks, '
                      f'not scanned')
                continue
            s = focus_scan(xl, yl, tx, ty, trs[f'mx17_{a}'])
            scans[(a, sub)] = s
            imgs[sub][a]['_focus_k'] = s['k']

    arms = {}
    for a in ARMS:
        per_sub = {s: estimators(imgs[s][a]) for s in subruns if a in imgs[s]}
        if not per_sub:
            arms[a] = dict(k=None, verdict='NO DATA',
                           reason='arm absent from every imaging summary')
            continue
        r = combine(per_sub)
        r['source_check'] = source_check(
            [(s, imgs[s][a]) for s in subruns if a in imgs[s]])
        r['n_coincident'] = int(sum(
            imgs[s][a].get('pointing_x_coincident', {}).get('n', 0)
            for s in subruns if a in imgs[s]))
        # The focus scan's plateau: how wide a range of k the data cannot
        # distinguish.  A wide plateau is the honest reason an arm is not
        # certified even when the three point estimates happen to agree.
        pl = [scans[(a, s)]['plateau'] for s in subruns if (a, s) in scans]
        if pl:
            r['focus_plateau'] = [min(p[0] for p in pl), max(p[1] for p in pl)]
            r['focus_n'] = int(sum(scans[(a, s)]['n'] for s in subruns
                                   if (a, s) in scans))
            width = r['focus_plateau'][1] / max(r['focus_plateau'][0], 1e-9) - 1
            if width > SPREAD_MAX and r['verdict'] == 'CALIBRATED':
                r['verdict'] = 'PROVISIONAL'
                r['reason'] += (f'; but the focus scan is flat over '
                                f'k = {r["focus_plateau"][0]:.2f}'
                                f'-{r["focus_plateau"][1]:.2f} ({width:.0%})')
        arms[a] = r

    usable = {a: r['k'] for a, r in arms.items()
              if r.get('verdict') in ('CALIBRATED', 'PROVISIONAL')}
    return dict(schema=SCHEMA, run=run, subruns=list(subruns),
                v_bundle=V_BUNDLE_PRIOR, arms=arms,
                apply=usable,
                criteria=dict(spread_max=SPREAD_MAX, repro_max=REPRO_MAX,
                              sample='pointing-coincident (wall segment AND '
                                     'plastic bar that fired)'),
                caveat='tan_true = k * tan_reco; v_true = v_bundle / k. Arms '
                       'not in `apply` have no angle scale: leave their angles '
                       'null rather than defaulting k to 1.')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns', default='stat090_0000,stat090_0001')
    ap.add_argument('--merged', default=str(paths.out('fullpass') / 'run_145'),
                    help='per-arm merged events_prelim.parquet tree (the full '
                         'pass), for the focus scan')
    a = ap.parse_args()

    subs = [s for s in a.subruns.split(',') if s]
    r = build(a.run, subs, a.merged)
    od = paths.out('kcal')
    out = os.path.join(str(od), f'k_arm_{a.run}.json')
    json.dump(r, open(out, 'w'), indent=1)

    print(f'{"arm":>4} {"k":>6} {"v_insitu":>9} {"band":>7} {"track":>7} '
          f'{"focus":>7} {"spread":>7} {"repro":>7}  verdict')
    for arm in ARMS:
        v = r['arms'][arm]
        if v.get('k') is None:
            print(f'{arm:>4} {"--":>6} {"--":>9} {"":>31}  {v["verdict"]}')
            continue
        pe = v['per_estimator']
        print(f'{arm:>4} {v["k"]:>6.3f} {v["v_insitu"]:>9.1f} '
              f'{pe.get("band", float("nan")):>7.3f} '
              f'{pe.get("track", float("nan")):>7.3f} '
              f'{pe.get("focus", float("nan")):>7.3f} '
              f'{v["spread"]:>6.1%} {v["repro"]:>6.1%}  {v["verdict"]}')
    print()
    for arm in ARMS:
        v = r['arms'][arm]
        print(f'  {arm}: {v["verdict"]} -- {v.get("reason", "")}')
    print()
    print('apply:', ','.join(f'{a_}={k:.4f}' for a_, k in r['apply'].items())
          or '(nothing certified)')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
