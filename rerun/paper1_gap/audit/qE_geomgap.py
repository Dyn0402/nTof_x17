"""qE_geomgap.py -- chamber E: is the drift-time gradient along y a real gap gradient?

E's walk-corrected q_u90 falls by about 0.17 in ln across the chamber along y (6-23 and 6-24 alike), while its charge
has no matching slope. If this is the gap, the cathode is tilted by about 2 mm across the chamber along y. The geometric
gap (gap_extent's half-height edges of the X charge of inclined tracks, no timing) tests it: we measure it in y slices
of E 6-24 (g_det4, the only E inclined-track cache) and compare with the q_u90 prediction
dg = G0 / (1 + beta) x d ln q_u90 (beta from Magboltz at E's water and field).
Errors: bootstrap over events (60). -> $WORK/audit/q16/qE_geomgap.json
"""
import json, pickle, sys, os
import numpy as np
sys.path[:0] = ['/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/analysis/cloud_basics',
                '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap']
import gap_extent as GE
import gapmap as GM
from r3_lib import MOUNT, DL

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
ev = list(pickle.load(open('/media/dylan/data/x17/cosmic_bench/cloud_basics/gap/incl_g_det4.pkl', 'rb')).values()) \
    if isinstance(pickle.load(open('/media/dylan/data/x17/cosmic_bench/cloud_basics/gap/incl_g_det4.pkl', 'rb')), dict) \
    else pickle.load(open('/media/dylan/data/x17/cosmic_bench/cloud_basics/gap/incl_g_det4.pkl', 'rb'))
bad = set(GE.bad_channels('g_det4', 'x'))
xm = np.array([e['ref_mesh_x'] + 14 * e['tan_x'] for e in ev])
ym = np.array([e['ref_mesh_y'] + 14 * e['tan_y'] for e in ev])
xin = (xm > np.percentile(xm, 5)) & (xm < np.percentile(xm, 95))
print('events', len(ev), 'y range (5-95 %)', np.percentile(ym, [5, 95]).round(1))


def est(sel, nb=60, seed=1):
    s = [e for e, k in zip(ev, sel) if k]
    r = GM.g_estimate(s, bad, 'x')
    rng = np.random.default_rng(seed)
    B = [GM.g_estimate([s[i] for i in rng.integers(0, len(s), len(s))], bad, 'x')['sym'] for _ in range(nb)]
    return dict(n=len(s), gap=r['sym'], err=float(np.nanstd(B)), pooled=r['pooled'])


res = {'all': est(xin)}
print('all', res['all'])
for nsl in (2, 3):
    e = np.percentile(ym[xin], np.linspace(0, 100, nsl + 1))
    for i in range(nsl):
        m = xin & (ym >= e[i]) & (ym <= e[i + 1])
        r = est(m, seed=10 * nsl + i); r['y_mid'] = float(np.median(ym[m]))
        res[f'{nsl}:{i}'] = r
        print(nsl, i, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

# the q_u90 profile of the same row along y, in the same slices (pass-1 gate table, walk-corrected as R3)
d = DL.load(MOUNT['E624']['rows'][0])
q = np.log(0.5 * (d.x_q_u90.values + d.y_q_u90.values)); Q = np.log(0.5 * (d.x_q_sum.values + d.y_q_sum.values))
ok = np.isfinite(q) & np.isfinite(Q)
walk = np.polyfit(Q[ok], q[ok], 1)[0]
qc = q - walk * (Q - np.median(Q[ok]))
ymd = d.ym.values
res['q90_profile'] = {}
for nsl in (2, 3):
    e = np.percentile(ym[xin], np.linspace(0, 100, nsl + 1))
    for i in range(nsl):
        m = ok & (ymd >= e[i]) & (ymd <= e[i + 1])
        res['q90_profile'][f'{nsl}:{i}'] = dict(y_mid=float(np.median(ymd[m])), lnq=float(np.median(qc[m])), n=int(m.sum()))
print('q90 profile', res['q90_profile'])
json.dump(res, open(f'{OUT}/qE_geomgap.json', 'w'), indent=1, default=float)
