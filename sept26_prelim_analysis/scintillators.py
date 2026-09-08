#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scintillators.py -- what the n_TOF scintillators contribute, and what they could.

TWO THINGS, and the split is the point of the module.

**The audit.**  Today the scintillators enter this analysis as a *filter* and
nothing else.  Three roles, all boolean:

  1. stage 1 (`candidate_filter`) -- which arms have a wall+plastic coincidence
     in the accept window.  That is what separates INTER from IMPLIED from NONE.
  2. efficiency (`efficiency.py`) -- "wall AND plastic fired in arm X" is the
     MM-independent denominator.
  3. the funnel (`funnel.py`) -- does the track extrapolate onto the wall
     *segment* and plastic *bar* that actually fired.

No amplitude and no time is read as anything but "in the window".  :func:`audit`
counts what is there so the page can say so with numbers rather than adjectives.

**The measurement that is missing, and is started here.**  The wall is read out
at BOTH ENDS of each bar group -- `detn` = 2*group + {1, 2}, four groups of four
bars, the parity being the two ends.  The bars run along **v = global y**, the
beam axis, so

    dt  = t_1 - t_2         propagation along the bar   ->  y
    lr  = log(A_1 / A_2)     attenuation along the bar   ->  y

both measure the ONE coordinate the target-pointing method cannot reach, because
the He-3 capsule is a 10 mm point in XZ and 80 mm long in y.  Micromegas tracks
give the truth: a track that predicts a crossing on the fired group predicts
where on the bar.

THREE THINGS THIS MODULE REFUSES TO DO.

  * It does not average the two ends into a position without saying what the
    scale is.  ``dt`` and ``lr`` are calibrated separately and their agreement
    is the check.
  * It does not call the residual a resolution.  The truth here is an
    extrapolated MM track, which has its own error, so the residual is an
    UPPER LIMIT on the wall's resolution and is labelled as one.
  * It does not resolve chamber D's sign flip.  D's correlation with y runs
    opposite to A's and C's in *both* estimators, which is either D's two wall
    ends swapped in the readout or D's MM y plane mirrored -- and those two are
    degenerate in this data.  See :func:`sign_check`.

    python -m sept26_prelim_analysis.scintillators --run run_145
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

SCHEMA = 'sept26_prelim/scintillators/1'
ARMS = ('A', 'B', 'C', 'D')
#: `det` code -> (family, arm).  Wall 0-3, plastic 4-7, liquid 8-11, in A B C D
#: order within each family.
FAMILY = {**{i: 'WAL' for i in range(0, 4)},
          **{i: 'PSS' for i in range(4, 8)},
          **{i: 'LIQ' for i in range(8, 12)}}
DET_ARM = {i: ARMS[i % 4] for i in range(12)}
WAL_CODE = {a: i for i, a in enumerate(ARMS)}
DT_WINDOW = (-100.0, 60.0)

#: Half-length of a wall bar along v (= global y), mm.  geometry.SIPM_HALF_V.
BAR_HALF_V = 250.0
#: Instrumented bars, 1..16, in four groups of four -- ASCENDING in u.  Verified
#: two ways on run_145: geometrically, since `run_config` places sipm_X_01..16
#: monotonically along each arm's own u_hat; and empirically, since the
#: ascending map matches 4.7x as many tracks as the descending one.
BARS_PER_GROUP = 4


def read_slim(run: str, subruns) -> pd.DataFrame:
    """Every n_TOF hit in the selected sub-runs, with family and arm attached."""
    import uproot
    cols = ['eventId', 'det', 'detn', 'tof', 'dt_ns', 'amp', 'area_0',
            'satuflag', 'pileup1', 'is_control']
    out = []
    for sub in subruns:
        d = os.path.join(str(paths.root('runs')), run, sub, 'ntof_hits')
        f = sorted(x for x in os.listdir(paths.require(d, 'slim directory'))
                   if x.endswith('.root'))
        if not f:
            raise FileNotFoundError(f'no slim n_TOF file under {d}')
        a = uproot.open(os.path.join(d, f[0]))['hits'].arrays(cols, library='np')
        out.append(pd.DataFrame(a).assign(subrun=sub))
    d = pd.concat(out, ignore_index=True)
    d['family'] = d.det.map(FAMILY)
    d['arm'] = d.det.map(DET_ARM)
    d['in_time'] = ((d.is_control == 0) & (d.dt_ns >= DT_WINDOW[0])
                    & (d.dt_ns <= DT_WINDOW[1]))
    return d


# --------------------------------------------------------------------------- #
# the audit
# --------------------------------------------------------------------------- #
def audit(slim: pd.DataFrame) -> pd.DataFrame:
    """Per (family, arm): hits, in-time hits, channels, saturation, pile-up.

    The channel count is what says how much position information the element
    carries at all -- the wall's 8 are four u groups x two ends, the plastics'
    2 are the two bars, and the liquid's 1 is a single cell with no internal
    structure whatever.
    """
    rows = []
    for (fam, arm), g in slim.groupby(['family', 'arm']):
        it = g[g.in_time]
        rows.append(dict(
            family=fam, arm=arm, det=int(g.det.iloc[0]),
            n_hits=len(g), n_in_time=len(it),
            n_channels=int(g.detn.nunique()),
            n_events_in_time=int(it.eventId.nunique()),
            frac_saturated=float((it.satuflag != 0).mean()) if len(it) else np.nan,
            frac_pileup=float((it.pileup1 != 0).mean()) if len(it) else np.nan,
            median_amp=float(it.amp.median()) if len(it) else np.nan))
    return pd.DataFrame(rows).sort_values(['family', 'arm'], ignore_index=True)


# --------------------------------------------------------------------------- #
# the wall, read at both ends
# --------------------------------------------------------------------------- #
def wall_pairs(slim: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, arm, wall group) that fired at BOTH ends.

    Where a group has more than one hit at an end, the largest is taken -- the
    slim is hit-level and a bar can ring.  Saturated and piled-up ends are kept
    here and cut downstream, so the cut's cost is visible.
    """
    w = slim[(slim.family == 'WAL') & slim.in_time].copy()
    w['grp'] = (w.detn - 1) // 2
    w['end'] = np.where(w.detn % 2 == 1, 1, 2)
    w = (w.sort_values('amp', ascending=False)
          .drop_duplicates(['subrun', 'eventId', 'arm', 'grp', 'end']))
    p = w.pivot_table(index=['subrun', 'eventId', 'arm', 'grp'], columns='end',
                      values=['tof', 'amp', 'area_0', 'satuflag', 'pileup1'],
                      aggfunc='first')
    p.columns = [f'{a}_{b}' for a, b in p.columns]
    p = p.dropna(subset=['tof_1', 'tof_2']).reset_index()
    p['dt_ns_ends'] = p.tof_1 - p.tof_2
    with np.errstate(divide='ignore', invalid='ignore'):
        p['log_ratio'] = np.log(p.amp_1 / p.amp_2)
        p['amp_geom'] = np.sqrt(p.amp_1 * p.amp_2)
    p['clean'] = ((p.satuflag_1 == 0) & (p.satuflag_2 == 0)
                  & np.isfinite(p.log_ratio))
    # A physical window, from the bar and not from taste.  The bar is 500 mm
    # end to end and light cannot cross it faster than c, so |dt| has a
    # ceiling; anything past it joined two different particles.  Generous by
    # a factor ~2 on the measured v_eff, so it removes tails and not signal.
    p['physical'] = (p.clean & (p.dt_ns_ends.abs() < DT_PHYS_MAX_NS)
                     & (p.log_ratio.abs() < LR_MAX))
    return p


#: Ceiling on |dt| between the two ends of a bar, ns.  2 x 250 mm at the
#: slowest effective propagation speed the data supports, x2 headroom.
DT_PHYS_MAX_NS = 12.0
#: Ceiling on |log(A_1/A_2)|.  At lambda ~ 600-900 mm the full bar gives ~0.8.
LR_MAX = 3.0


def self_consistency(pairs: pd.DataFrame) -> pd.DataFrame:
    """Do the two estimators agree with each other, WITHOUT any Micromegas?

    They measure the same coordinate by different physics -- propagation delay
    and attenuation -- so a correlation between them is evidence the wall
    carries position information at all.  It is available for chamber B too,
    which has no tracks to check against.

    NOT a discriminator for which end is which: swapping the two ends of a bar
    flips the sign of both estimators, so their correlation stays negative
    either way.  It says the wall works, not how it is wired.
    """
    rows = []
    for arm, g in pairs[pairs.physical].groupby('arm'):
        # Centre BOTH estimators per bar group before correlating.  The four
        # groups carry their own cable and gain offsets, so pooled they make
        # four parallel bands and the pooled correlation understates what one
        # group actually carries (arm D: -0.36 pooled, -0.83 centred).
        dtc = g.dt_ns_ends - g.groupby('grp').dt_ns_ends.transform('median')
        lrc = g.log_ratio - g.groupby('grp').log_ratio.transform('median')
        rows.append(dict(arm=arm, n=len(g),
                         corr=float(np.corrcoef(dtc, lrc)[0, 1]),
                         corr_pooled=float(np.corrcoef(g.dt_ns_ends,
                                                       g.log_ratio)[0, 1]),
                         dt_median=float(g.dt_ns_ends.median()),
                         dt_iqr=float(g.dt_ns_ends.quantile(.75)
                                      - g.dt_ns_ends.quantile(.25)),
                         lr_median=float(g.log_ratio.median()),
                         lr_iqr=float(g.log_ratio.quantile(.75)
                                      - g.log_ratio.quantile(.25))))
    return pd.DataFrame(rows).sort_values('arm', ignore_index=True)


# --------------------------------------------------------------------------- #
# calibration against Micromegas tracks
# --------------------------------------------------------------------------- #
def track_crossings(run: str, subruns, dca_max: float = 30.0) -> pd.DataFrame:
    """Gated, target-pointing tracks with a predicted wall crossing.

    ``y_wall`` is the global y where the track's line crosses the wall -- the
    truth the two estimators are calibrated against.  The pointing cut is not
    cosmetic: without it the sample is dominated by tracks that have no reason
    to be the particle that lit the bar, and the correlation halves.
    """
    src = paths.out('stage3_fullpass')
    cols = ['event_id', 'arm', 'gated', 'p0_y', 'd_y', 'pred_sipm_bar',
            'pred_sipm_s_mm', 'dca_axis_mm', 'angle_calibrated']
    out = []
    for sub in subruns:
        p = paths.require(src / f'tracks_{run}_{sub}.parquet',
                          f'stage-3 tracks for {sub}')
        out.append(pd.read_parquet(p, columns=cols).assign(subrun=sub))
    t = pd.concat(out, ignore_index=True)
    t = t[t.gated & np.isfinite(t.pred_sipm_bar)
          & (t.dca_axis_mm < dca_max)].copy()
    t['y_wall'] = t.p0_y + t.pred_sipm_s_mm * t.d_y
    t['grp'] = ((t.pred_sipm_bar - 1) // BARS_PER_GROUP).astype(int)
    return t.rename(columns={'event_id': 'eventId'})


def _robust_line(x, y, n_iter=8):
    """IRLS line fit -- the sample is a band on a flat background of pairs
    where the track and the fired bar were different particles."""
    A = np.vstack([x, np.ones_like(x)]).T
    w = np.ones_like(x)
    c = np.array([0.0, 0.0])
    for _ in range(n_iter):
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        r = y - (c[0] * x + c[1])
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * max(s, 1e-6))) ** 2)
    return float(c[0]), float(c[1])


def _robust_line_groups(x, y, grp, n_iter=10):
    """One common slope, one offset per group, IRLS.

    The two ends of a bar group are read out through their own cables, so each
    group carries its own additive offset -- measured here at up to **8 ns
    within a single arm**, which is metres of cable and nothing physical.
    Pooling the groups without it smears the estimator: removing it lifts the
    correlation with the track's y from 0.30 to 0.52 in arm A and 0.36 to 0.60
    in arm C, while moving the slope by under 5 %.  So the offsets are a
    calibration constant and the slope is the measurement.

    Returns (slope, {group: offset}).  The offsets are RELATIVE: the mean
    illumination height is common to the groups and absorbs into all of them
    together, so this fixes the shape and not the zero.  See
    :func:`calibrate` for what that costs.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    g = np.asarray(grp)
    keys = np.unique(g)
    D = np.column_stack([x] + [(g == k).astype(float) for k in keys])
    w = np.ones_like(x)
    c = np.zeros(D.shape[1])
    for _ in range(n_iter):
        c, *_ = np.linalg.lstsq(D * w[:, None], y * w, rcond=None)
        r = y - D @ c
        sc = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * max(sc, 1e-6))) ** 2)
    return float(c[0]), {k: float(v) for k, v in zip(keys, c[1:])}


def calibrate(pairs: pd.DataFrame, trk: pd.DataFrame) -> tuple:
    """Slope of each estimator against the track's y at the wall, per chamber.

    Returns (per-chamber table, the matched rows).  The physics of each slope:

        dt = 2 y / v_eff        ->  v_eff = 2 / slope   [mm/ns]
        lr = -2 y / lambda      ->  lambda = -2 / slope [mm]

    Both are quoted with their sign, because the sign is a result: it says
    which end is which, and in chamber D it disagrees with A and C.
    """
    m = trk.merge(pairs[pairs.physical],
                  on=['subrun', 'eventId', 'arm', 'grp'], how='inner')
    rows = []
    for arm, g in m.groupby('arm'):
        if len(g) < 100:
            rows.append(dict(arm=arm, n=len(g)))
            continue
        y = g.y_wall.to_numpy()
        grp = g.grp.to_numpy()
        rec = dict(arm=arm, n=int(len(g)))
        for est, col in (('dt', 'dt_ns_ends'), ('lr', 'log_ratio')):
            v = g[col].to_numpy()
            sl, offs = _robust_line_groups(y, v, grp)
            off = np.array([offs[k] for k in grp])
            pred = ((v - off) / sl if abs(sl) > 1e-9
                    else np.full_like(v, np.nan))
            res = pred - y
            rec[f'{est}_slope'] = sl
            rec[f'{est}_intercept'] = float(np.mean(list(offs.values())))
            rec[f'{est}_offsets'] = json.dumps({int(k): round(o, 4)
                                                for k, o in offs.items()})
            rec[f'{est}_corr'] = float(np.corrcoef(y, v - off)[0, 1])
            # Robust width of the residual: an UPPER LIMIT on the wall's
            # resolution, because y_wall is itself an extrapolated track.
            rec[f'{est}_resid_mm'] = float(1.4826 * np.median(np.abs(res - np.median(res))))
        rec['v_eff_mm_ns'] = 2.0 / rec['dt_slope'] if rec['dt_slope'] else np.nan
        rec['lambda_mm'] = -2.0 / rec['lr_slope'] if rec['lr_slope'] else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values('arm', ignore_index=True), m


def group_stability(matched: pd.DataFrame, min_n: int = 150) -> pd.DataFrame:
    """The same slope, fitted inside each wall group separately.

    The pooled fit runs across all four u groups at once, so a slope could in
    principle be manufactured by the groups differing from each other rather
    than by position along a bar.  Fixing the group kills that: if the slope
    survives within groups, it is position.
    """
    rows = []
    for arm, g in matched.groupby('arm'):
        pooled_lr, _ = _robust_line(g.y_wall.to_numpy(), g.log_ratio.to_numpy())
        pooled_dt, _ = _robust_line(g.y_wall.to_numpy(), g.dt_ns_ends.to_numpy())
        rows.append(dict(arm=arm, grp='pooled', n=int(len(g)),
                         lr_slope=pooled_lr, dt_slope=pooled_dt))
        for grp, h in g.groupby('grp'):
            if len(h) < min_n:
                continue
            sl, _ = _robust_line(h.y_wall.to_numpy(), h.log_ratio.to_numpy())
            sd, _ = _robust_line(h.y_wall.to_numpy(), h.dt_ns_ends.to_numpy())
            rows.append(dict(arm=arm, grp=str(int(grp)), n=int(len(h)),
                             lr_slope=sl, dt_slope=sd))
    return pd.DataFrame(rows)


def sign_check(cal: pd.DataFrame) -> dict:
    """Which chambers run the same way, and what the odd one out could be.

    A chamber's sign is the product (wall end order) x (MM y plane order).  A
    single chamber disagreeing localises ONE flip to that chamber's chain and
    says nothing about which link it is in.  Recorded, not resolved.
    """
    g = cal.dropna(subset=['lr_slope'])
    if g.empty:
        return dict(resolved=False, reason='no chamber calibrated')
    sgn = {r.arm: int(np.sign(r.lr_slope)) for r in g.itertuples()}
    agree = {s: [a for a, v in sgn.items() if v == s] for s in (-1, 1)}
    odd = min(agree.values(), key=len) if all(agree.values()) else []
    return dict(sign_per_arm=sgn, odd_ones_out=odd,
                consistent=len(odd) == 0,
                # both estimators must flip together for this to be one flip
                # somewhere in the chain rather than two unrelated faults
                both_estimators_agree={
                    r.arm: bool(np.sign(r.lr_slope) != np.sign(r.dt_slope))
                    for r in g.itertuples()},
                degenerate_between=['the chamber’s two wall ends swapped '
                                    'in the readout map',
                                    'the chamber’s MM y strip plane mirrored'],
                what_would_break_it=[
                    'the wall cabling map for that arm (external fact)',
                    'the y-plane strip mapping order for that arm (external fact)',
                    'a y-asymmetric source -- ruled out here: the He-3 gas '
                    'polycone is volume-symmetric in y to a skew of +0.06'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--dca', type=float, default=30.0)
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    slim = read_slim(a.run, subs)
    aud = audit(slim)
    pairs = wall_pairs(slim)
    sc = self_consistency(pairs)
    trk = track_crossings(a.run, subs, a.dca)
    cal, matched = calibrate(pairs, trk)
    stab = group_stability(matched)

    od = paths.out('scint')
    aud.to_csv(od / f'audit_{a.run}.csv', index=False)
    sc.to_csv(od / f'self_consistency_{a.run}.csv', index=False)
    cal.to_csv(od / f'wall_calibration_{a.run}.csv', index=False)
    stab.to_csv(od / f'group_stability_{a.run}.csv', index=False)
    # The corrected estimators travel with the matched rows, so a figure or a
    # downstream y measurement never has to re-derive the offsets.
    for est, col in (('dt', 'dt_ns_ends'), ('lr', 'log_ratio')):
        adj = np.full(len(matched), np.nan)
        for _, r in cal.dropna(subset=[f'{est}_slope']).iterrows():
            offs = {int(k): v for k, v in json.loads(r[f'{est}_offsets']).items()}
            sel = (matched.arm == r.arm).to_numpy()
            adj[sel] = (matched.loc[sel, col].to_numpy()
                        - matched.loc[sel, 'grp'].map(offs).to_numpy())
        matched[f'{col}_adj'] = adj
    matched[['subrun', 'eventId', 'arm', 'grp', 'y_wall', 'dt_ns_ends',
             'dt_ns_ends_adj', 'log_ratio', 'log_ratio_adj',
             'amp_geom', 'dca_axis_mm']].to_parquet(
        od / f'wall_matched_{a.run}.parquet', index=False)
    meta = dict(schema=SCHEMA, run=a.run, subruns=subs, dca_max=a.dca,
                n_slim_hits=int(len(slim)), n_wall_pairs=int(len(pairs)),
                n_wall_pairs_clean=int(pairs.clean.sum()),
                n_wall_pairs_physical=int(pairs.physical.sum()),
                n_matched=int(len(matched)),
                bar_half_v_mm=BAR_HALF_V,
                sign=sign_check(cal))
    json.dump(meta, open(od / f'scint_{a.run}.meta.json', 'w'), indent=1,
              default=str)

    print('AUDIT -- what the slim carries, per element')
    print(aud.to_string(index=False))
    print('\nSELF-CONSISTENCY -- the two estimators against each other, no MM')
    print(sc.to_string(index=False))
    print('\nCALIBRATION -- against MM tracks pointing at the target')
    print(cal.to_string(index=False))
    print('\nGROUP STABILITY -- the slope refitted inside each u group')
    print(stab.to_string(index=False))
    print('\nSIGN', json.dumps(meta['sign'], indent=1, default=str))
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
