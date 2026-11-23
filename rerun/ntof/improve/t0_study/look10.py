# study script: the same-chamber rate fit (same_chamber_compare.tof_fraction: f = a + b rho over 25-75 ms) split by the
# pair's column-time difference.  Pairs with |dtc| < 300 ns ("in") and >= 300 ns ("out") are fitted separately with
# the same singles denominator.  If the rate fit separates correlated from accidental pairs correctly, the out-of-time
# part is almost all b rho (a_out ~ the in-time tail of correlated pairs, ~7 % A / ~20 % C of a), and b_out / b is the
# out-of-time share of the accidentals.  Same event cuts as same_chamber_compare.events, on the stage3_r2_final view.
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
print(f'tracks {len(d)}; with a pin-relative t0 {d.x_dt0.notna().mean():.4f}')
d['tc'] = 0.5 * ((d.x_dt0 + d.x_q_u50) + (d.y_dt0 + d.y_q_u50))
d['child'] = d.x_split_child.fillna(False).astype(bool) | d.y_split_child.fillna(False).astype(bool)
g = d.groupby(['arm', 'run', 'subrun', 'event_id'])
E = g.agg(n=('arm', 'size'), tof=('t_since_flash_ns', 'first'), tmin=('tc', 'min'), tmax=('tc', 'max'),
          child=('child', 'any')).reset_index()
E['dtc'] = E.tmax - E.tmin
print('events', len(E), ' pairs', int((E.n >= 2).sum()), ' pairs with dtc', int(((E.n >= 2) & E.dtc.notna()).sum()))


def fit(E, sel, label):
    F = E.copy()
    F['n'] = np.where((F.n >= 2) & ~sel, 1, F.n)          # pairs outside the selection count as singles' events
    _R, fits = SC.tof_fraction(F)
    for r in fits.itertuples():
        tot = r.pairs
        print(f'  {label:22s} {r.arm}: pairs {tot:6d}  a = {1e3 * r.a:6.3f} +- {1e3 * r.a_err:5.3f} /1000   '
              f'b<rho>/f = {r.accidental_share:.3f}   chi2 {r.chi2:5.1f}/{r.ndf}   correlated {r.correlated_pairs:7.0f}')
    return fits


pair = E.n >= 2
dt = E.dtc.fillna(0)
print('== rate fit, 25-75 ms (a in pairs per 1000 events)')
base = fit(E, pair, 'all pairs')
fin = fit(E, pair & (dt < 300), '|dtc| < 300')
fout = fit(E, pair & (dt >= 300), '|dtc| >= 300')
fnc = fit(E, pair & ~E.child, 'no split child')
fnco = fit(E, pair & ~E.child & (dt >= 300), 'no child, |dtc| >= 300')
fch = fit(E, pair & E.child, 'with a split child')
for arm in SC.ARMS:
    b0 = base[base.arm == arm].iloc[0]
    o = fout[fout.arm == arm].iloc[0]
    i = fin[fin.arm == arm].iloc[0]
    print(f'{arm}: a_out / a = {o.a / b0.a:.3f} +- {o.a_err / b0.a:.3f};  b_out / b = {o.b / b0.b:.3f};  '
          f'b_in / b = {i.b / b0.b:.3f}')
late = E[(E.tof >= 25e6) & pair]
for arm, s in late.groupby('arm'):
    print(f'{arm}: late pairs {len(s)}, |dtc| >= 300 {np.mean(s.dtc >= 300):.3f}, with child {s.child.mean():.3f}, '
          f'no child |dtc|>=300 {np.mean(s.dtc[~s.child] >= 300):.3f}')
