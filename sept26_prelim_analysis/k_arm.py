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

A NOTE ON CIRCULARITY, AND WHY IT DOES NOT BITE.  The pointing-coincident
sample is selected by extrapolating each track to the wall and the plastic with
its RAW tan -- i.e. at k = 1 -- so the sample k is measured on is defined before
k is known.  Checked by iterating once (2026-09-07): apply k, re-derive the
sample, re-measure.  The sample shifts by 8-21 %, and the answer does not --

    arm   k in    k out (both sub-runs)
     A    1.27    1.20, 1.20
     C    1.62    1.55, 1.60
     D    1.75    1.75, 1.80

every one within a grid step of the input.  k is a fixed point of the
selection, so the ordering costs nothing.

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
#: The grid runs to 3.5 (v = 12 um/ns), well past any plausible drift velocity
#: for Ar/iso 90/10, so a real optimum is interior.  It used to stop at 2.6,
#: which is close enough to chamber B's apparent peak that a clipped maximum
#: could have passed for a measurement -- `focus_scan` now refuses one outright
#: rather than leaving that to be noticed.
K_GRID = np.round(np.arange(0.60, 3.51, 0.05), 2)
FOCUS_RADII_MM = (30.0, 10.0)

#: Charge window, as percentiles of ``x_q_sum`` within the coincident sample.
#:
#: Both tails are KNOWN failure modes, identified before this window was chosen
#: rather than by scanning for a good answer:
#:
#:   low   the model fits noise.  Chi2/dof is *small* there because there is
#:         nothing to fit, so a chi2 cut selects these rather than removing
#:         them -- in chamber D the lowest-chi2 quarter of the coincident
#:         sample anti-points, corr(lever, tan) = -0.43, while the other three
#:         quarters all point correctly at +0.40 to +0.70.
#:   high  saturation (``sat_adc`` = 3700) and discharges.  D's top charge
#:         quartile degrades to corr +0.13.
#:
#: Banding by charge instead of chi2 separates them cleanly: D runs
#: -0.01 / +0.56 / +0.71 / +0.13 across charge quartiles.
#:
#: **It is not a tuned cut.**  The focus estimator -- the one that decides the
#: verdict -- moves by less than a plateau width as the window is opened from
#: 25-75 all the way to 0-100: A 1.20 -> 1.17, C 1.60 -> 1.52, D 1.68 -> 1.73.
#: What the window fixes is the *band* estimator, which is dilution-sensitive
#: and is the one that failed.  Chamber B is the exception and stays
#: uncertified: its focus k jumps 1.35 -> 2.30 across windows, which is B
#: having no stable optimum rather than the window doing something.
CHARGE_WINDOW = (25.0, 75.0)

#: The pointing band is fitted over this lever-arm range only.  Beyond ~130 mm
#: from the perpendicular foot the drift-window truncation flattens it; inside
#: 30 mm the lever is too short to carry angle information.  Same window
#: ``run145_target_imaging`` uses.
LEVER_WINDOW_MM = (30.0, 130.0)
D_PERP_MM = 234.6


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
    # Both planes carry the same in-plane sign (build_tracks.IN_PLANE_SIGN_Y,
    # measured 2026-09-07).  The focus objective is the miss distance in the XZ
    # projection and is blind to the y sign for these chambers -- which is
    # exactly how the y error survived -- but the frame must still be right.
    from sept26_prelim_analysis.build_tracks import IN_PLANE_SIGN_Y
    q = df['x_q_sum'].to_numpy()
    m = coin & sel & np.isfinite(q) & (q > 0)
    g = df[m]
    qs = q[m]
    lo, hi = np.percentile(qs, CHARGE_WINDOW)
    keep = (qs >= lo) & (qs <= hi)
    return dict(
        xl=TI.local_x(g['x_p0'].to_numpy())[keep],
        yl=(IN_PLANE_SIGN_Y * (g['y_p0'].to_numpy() - TI.STRIP_MAP_HALF))[keep],
        tx=g['x_tan_theta'].to_numpy()[keep],
        ty=g['y_tan_theta'].to_numpy()[keep],
        q=qs[keep], foot_x=TI.PINWHEEL[arm],
        n_coincident=int(m.sum()), q_lo=float(lo), q_hi=float(hi))


def band_k(S: dict) -> float:
    """Angle scale from the slope of the pointing band.

    A point source at ``D_PERP_MM`` forces tan = (u - foot)/d_perp, so the
    expected slope is 1/d_perp and k is the ratio to the fitted one.  Sensitive
    to dilution: any population whose tan does not track position flattens the
    band and inflates k without bound, which is why it is never quoted alone.
    """
    from ntof_tracking import run145_target_imaging as TI
    lev = S['xl'] - S['foot_x']
    m = ((np.abs(lev) > LEVER_WINDOW_MM[0]) & (np.abs(lev) < LEVER_WINDOW_MM[1])
         & (np.abs(S['tx']) > 1e-3))
    if m.sum() < 200:
        return float('nan')
    slope, _ = TI._robust_line(lev[m], S['tx'][m])
    return (1.0 / D_PERP_MM) / slope if abs(slope) > 1e-12 else float('nan')


def track_k(S: dict) -> float:
    """Median of the per-track ratio tan_expected / tan_reco."""
    lev = S['xl'] - S['foot_x']
    m = ((np.abs(lev) > LEVER_WINDOW_MM[0]) & (np.abs(lev) < LEVER_WINDOW_MM[1])
         & (np.abs(S['tx']) > 1e-3))
    if m.sum() < 200:
        return float('nan')
    return float(np.median((lev[m] / D_PERP_MM) / S['tx'][m]))


def focus_scan(S: dict, tr, gap_mm: float = 30.0) -> dict:
    """k that maximises the count of tracks pointing within each fixed radius.

    Returns the per-radius optima and the grid, so a flat scan is visible as a
    flat scan rather than collapsing to whichever bin won by one track.
    """
    from ntof_tracking import run145_target_imaging as TI

    xl, yl, tx, ty = S['xl'], S['yl'], S['tx'], S['ty']
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
    # An objective still climbing at either edge has no optimum inside the
    # grid, and the argmax is then just the last bin.  That is not a
    # measurement, and it must not be reported as one.
    c0 = np.array(counts[FOCUS_RADII_MM[0]], float)
    railed = bool(c0[-1] >= 0.98 * c0.max() or c0[0] >= 0.98 * c0.max())
    # How well the objective is actually constrained: the width of the k range
    # holding >=95 % of the peak count at 30 mm.  A flat scan is not a
    # measurement, and this is what says so.
    c30 = np.array(counts[FOCUS_RADII_MM[0]], float)
    within = K_GRID[c30 >= 0.95 * c30.max()]
    return dict(best=best,
                k=float('nan') if railed else float(np.median(list(best.values()))),
                railed=railed,
                plateau=[float(within.min()), float(within.max())],
                n=int(len(xl)), grid=K_GRID.tolist(),
                counts={str(r): c for r, c in counts.items()},
                median_miss=med)


def estimators(S: dict, scan: dict) -> dict:
    """All three k values, computed HERE on one consistently-defined sample.

    Nothing is read from ``imaging_summary.json`` any more.  Two reasons: the
    imaging's ``k_phys`` is ``k_track_coincident`` verbatim, so reading it as a
    third opinion double-counts one estimator; and its ``k_opt`` minimises a
    median conditioned on a cut that k itself moves.  Computing all three on
    the same rows also means the spread between them measures the estimators
    and not three different samples.
    """
    return {'band': band_k(S), 'track': track_k(S), 'focus': scan['k']}


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


def source_check(run: str, arm: str, samples: dict) -> dict:
    """The scale-free source position per sub-run -- a geometry check, not k.

    COMPUTED HERE, from the same samples the estimators run on.  It used to be
    READ from ``imaging_summary.json``, and on 2026-09-08 that quietly went
    stale: the y in-plane sign fix changed which tracks are pointing-coincident
    (the coincidence predicts a v on the wall and the plastic), the sample
    shrank by 30-45 %, and chamber D's crossing moved from -48 mm to -10 mm
    while the report kept quoting -48. A cached number from a file nothing
    re-derives is a trap; recomputing costs nothing here because the sample is
    already in memory.
    """
    from sept26_prelim_analysis import source_imaging as SI
    out = {}
    for sub, S in sorted(samples.items()):
        c = SI.crossing(S['xl'], S['tx'],
                        dict(foot=S['foot_x'], lo=LEVER_WINDOW_MM[0],
                             hi=LEVER_WINDOW_MM[1]))
        ax, mm = ((None, float('nan')) if not np.isfinite(c['x0'])
                  else SI.to_global(run, arm, c['x0']))
        out[sub] = dict(axis=ax, mm=mm, err=c['err'], n=c['n'])
    return out


def build(run: str, subruns, merged_dir: str) -> dict:
    import json as _json
    from pathlib import Path
    from ntof_tracking.reco import geometry as G

    # The imaging summary is OPTIONAL, and only supplies the scale-free source
    # position reported alongside as a geometry cross-check.  Every estimator
    # is computed here now, so requiring it would make the calibration depend
    # on a step it no longer uses -- which is exactly what broke the chain on
    # the first sub-run reconstructed after the refactor.
    imgs = {}
    for sub in subruns:
        p = paths.out('kcal') / f'{run}_{sub}' / 'imaging_summary.json'
        if os.path.exists(p):
            imgs[sub] = {r['arm']: r for r in json.load(open(p))['results']}
        else:
            imgs[sub] = {}
            print(f'  [k] {sub}: no imaging summary -- the scale-free source '
                  f'position will be absent for it (estimators unaffected)')

    # The focus estimator is re-derived here, per sub-run, on the coincident
    # sample -- see focus_scan for why the imaging's k_opt is not used.
    cfg = _json.loads((Path(str(paths.root('runs'))) / run
                       / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)
    scans, samples, raw = {}, {}, {}
    for sub in subruns:
        for a in ARMS:
            try:
                S = coincident_tracks(run, sub, a, merged_dir)
            except FileNotFoundError as exc:
                print(f'  [k] {a}/{sub}: skipped -- {exc}')
                continue
            if len(S['xl']) < 200:
                print(f'  [k] {a}/{sub}: only {len(S["xl"])} tracks in the '
                      f'charge window, not measured')
                continue
            sc = focus_scan(S, trs[f'mx17_{a}'])
            scans[(a, sub)] = sc
            samples[(a, sub)] = estimators(S, sc)
            raw[(a, sub)] = S

    arms = {}
    for a in ARMS:
        per_sub = {s: samples[(a, s)] for s in subruns if (a, s) in samples}
        if not per_sub:
            arms[a] = dict(k=None, verdict='NO DATA',
                           reason='no sub-run yielded a measurable sample')
            continue
        r = combine(per_sub)
        r['source_check'] = source_check(
            run, a, {s: raw[(a, s)] for s in subruns if (a, s) in raw})
        r['n_coincident'] = int(sum(
            raw[(a, s)]['n_coincident'] for s in subruns if (a, s) in raw))
        # The focus scan's plateau: how wide a range of k the data cannot
        # distinguish.  A wide plateau is the honest reason an arm is not
        # certified even when the three point estimates happen to agree.
        pl = [scans[(a, s)]['plateau'] for s in subruns if (a, s) in scans]
        if pl:
            r['focus_plateau'] = [min(p[0] for p in pl), max(p[1] for p in pl)]
            r['focus_n'] = int(sum(scans[(a, s)]['n'] for s in subruns
                                   if (a, s) in scans))
            # The scan curves themselves, so the figure and the verdict are
            # built from one object and cannot disagree about the plateau.
            r['scan'] = {s: scans[(a, s)] for s in subruns if (a, s) in scans}
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
