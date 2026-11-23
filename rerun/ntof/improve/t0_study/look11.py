# study script: separate the rate fit's constant term a into in-time (correlated) pairs and accidentals whose second
# track comes from a source that is constant in ToF (activation, cosmics, ambient), using the column-time difference.
#   a_out = t C + p X,  a = C + X   ->   X = (a_out - t a) / (p - t)
# t = fraction of truly in-time pairs at |dtc| >= 300 ns: late no-child pairs whose two tracks both have an in-time hit
#     in their own (different) wall groups (look8);
# p = out-of-time share of accidentals: b_out / b of the rate fit (the rho-proportional pairs, which are accidental).
# Sample: pairs without a split child (the child pairs are in time and carry no accidentals: b ~ 0, look10).
import os, sys
import numpy as np, pandas as pd
os.environ.setdefault('X17_PASS1_VIEW', 'stage3_r2_final')
sys.path.insert(0, os.path.expanduser('~/PycharmProjects/nTof_x17'))
from sept26_prelim_analysis import pass1_tracks as P1  # noqa: E402
from sept26_prelim_analysis import same_chamber_compare as SC  # noqa: E402

WD = os.environ.get('WORK_DATA', '/media/ucla/work')
KEY = ['run', 'subrun', 'event_id', 'arm', 'track_id']
d = P1.read(KEY + ['gated', 'x_t0', 'y_t0', 'angle_calibrated', 't_since_flash_ns', 'x_q_u50', 'y_q_u50',
                   'x_split_child', 'y_split_child'], arms=SC.ARMS)
d = d[d.t_since_flash_ns.notna() & ~d.run.isin(P1.PRE_ACCESS_RUNS)]
d = d[d.gated.fillna(False).astype(bool) & (d.x_t0 <= 300) & (d.y_t0 <= 300)
      & d.angle_calibrated.fillna(False).astype(bool)]
T = pd.read_parquet(f'{WD}/ntof_pass1/ana/t0_study/tracks_t0.parquet', columns=KEY + ['x_dt0', 'y_dt0'])
d = d.merge(T, on=KEY, how='left')
d['tc'] = 0.5 * ((d.x_dt0 + d.x_q_u50) + (d.y_dt0 + d.y_q_u50))
d['child'] = d.x_split_child.fillna(False).astype(bool) | d.y_split_child.fillna(False).astype(bool)
# own-group wall hit in time (look4 definition)
W = pd.read_parquet(f'{WD}/ntof_pass1/ana/t0_study/wall_hits.parquet', columns=KEY + ['t_wall', 'grp'])
W = W[W.arm.isin(SC.ARMS)]
pk = W.groupby('arm').t_wall.apply(lambda v: v[(v > -60) & (v < 40)].median())
W['in'] = np.abs(W.t_wall - W.arm.map(pk)) < 30
wi = W.groupby(KEY).agg(win=('in', 'any'), grp=('grp', 'first')).reset_index()
d = d.merge(wi, on=KEY, how='left')
d['win'] = d.win.fillna(False).astype(bool)

g = d.groupby(['arm', 'run', 'subrun', 'event_id'])
E = g.agg(n=('arm', 'size'), tof=('t_since_flash_ns', 'first'), tmin=('tc', 'min'), tmax=('tc', 'max'),
          child=('child', 'any'), nwin=('win', 'sum'), ngrp=('grp', 'nunique')).reset_index()
E['dtc'] = E.tmax - E.tmin
pair = E.n >= 2


def fits(sel):
    F = E.copy()
    F['n'] = np.where(pair & ~sel, 1, F.n)
    return SC.tof_fraction(F)[1].set_index('arm')


CUT = float(os.environ.get('CUT_NS', 300))
nc = pair & ~E.child
base, out = fits(nc), fits(nc & (E.dtc >= CUT))
late = E[(E.tof >= SC.FIT_MS * 1e6) & nc & (E.n == 2)]
rows = []
for arm in SC.ARMS:
    s = late[late.arm == arm]
    w = s[(s.nwin == 2) & (s.ngrp == 2)]
    t = np.mean(w.dtc >= CUT)
    t_e = np.sqrt(t * (1 - t) / len(w))
    a, a_out = base.loc[arm, 'a'], out.loc[arm, 'a']
    p = out.loc[arm, 'b'] / base.loc[arm, 'b']
    X = (a_out - t * a) / (p - t)
    # error: a_out and t dominate (a_out's fit error; t binomial)
    X_e = np.hypot(out.loc[arm, 'a_err'], a * t_e) / (p - t)
    rows.append(dict(arm=arm, n_wall_pairs=len(w), t=t, t_err=t_e, p=p, a=1e3 * a, a_out=1e3 * a_out,
                     X=1e3 * X, X_err=1e3 * X_e, X_over_a=X / a, X_over_a_err=X_e / a,
                     acc_rate_fit=base.loc[arm, 'accidental_share'],
                     acc_total=(base.loc[arm, 'accidental_share'] * base.loc[arm, 'frac'] + X) / base.loc[arm, 'frac']))
R = pd.DataFrame(rows)
print(f'cut |dtc| >= {CUT:.0f} ns. no-child pairs, 25-75 ms; a, a_out, X in pairs per 1000 tracked events')
print(R.round(4).to_string(index=False))
# same with all pairs (child pairs are in time: they add to C only)
allb = fits(pair)
for r in R.itertuples():
    tot = allb.loc[r.arm]
    print(f'{r.arm}: all pairs a = {1e3 * tot.a:.3f}; constant-source accidentals X = {r.X:.3f} +- {r.X_err:.3f} '
          f'= {r.X / (1e3 * tot.a):.3f} of a; accidental share of all pairs: rate fit {tot.accidental_share:.3f}, '
          f'+ X -> {(tot.accidental_share * tot.frac + r.X / 1e3) / tot.frac:.3f}')
