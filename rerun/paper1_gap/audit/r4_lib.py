"""r4_lib: events-table loading, selection, covariate removal and the quadratic shape fit of ln q_u90 (audit R4)."""
import sys, json, os, re
import numpy as np, pandas as pd
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap')
import gapmap as GM
WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
EOS = f'{WORK}/audit/r4/eos'
OUT = f'{WORK}/audit/r4'
G0, BETA = 28.2, 0.68
SC = 'mx17_det3_saturday_scan_6-27-26'
GOLD1 = f'{SC}/long_run_resist_490V_drift_1000V'
GOLD2 = 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check'
R625 = 'mx17_det3_day_6-25-26/long_run'
NAMES = ['gu', 'gw', 'guu', 'gww', 'guw']


def volts(rel):
    m = re.search(r'resist_(\d+)V', rel); d = re.search(r'drift_(\d+)V', rel)
    if rel == R625: return 495, 500
    if 'p2_det1' in rel: return 490, 1000
    return (int(m.group(1)) if m else None, int(d.group(1)) if d else None)


def load_events(rel, sat0=False):
    """chamber-only selection: single track, both views ok and quality, no spark, interior >= 30 mm."""
    d = pd.read_parquet(f'{EOS}/{rel}/events.parquet')
    s = (d.n_tracks == 1) & d.x_ok & d.y_ok & d.x_quality_ok & d.y_quality_ok & ~d.spark
    d = d[s].copy()
    d['xm'] = d.x_p0 + 14 * d.x_tan_theta
    d['ym'] = d.y_p0 + 14 * d.y_tan_theta
    d = d[(d.xm > GM.IN_X[0]) & (d.xm < GM.IN_X[1]) & (d.ym > GM.IN_Y[0]) & (d.ym < GM.IN_Y[1])]
    q = 0.5 * (d.x_q_u90 + d.y_q_u90)
    qs = 0.5 * (d.x_q_sum + d.y_q_sum)
    d = d[(q > 0) & (qs > 0) & np.isfinite(q) & np.isfinite(qs)].copy()
    d['q'] = 0.5 * (d.x_q_u90 + d.y_q_u90)
    d['lq'] = np.log(d.q)
    d['lQ'] = np.log(0.5 * (d.x_q_sum + d.y_q_sum))
    d['U'] = (d.xm - 200) / 100
    d['W'] = (d.ym - 200) / 100
    d['atx'] = d.x_tan_theta.abs(); d['aty'] = d.y_tan_theta.abs()
    if sat0:
        d = d[(d.x_n_sat_strips == 0) & (d.y_n_sat_strips == 0)]
    return d.reset_index(drop=True)


def design(d, cov=('walk', 'tan')):
    U, W = d.U.values, d.W.values
    cols = [np.ones(len(d)), U, W, U * U / 2, W * W / 2, U * W]
    if 'walk' in cov: cols.append(d.lQ.values - np.median(d.lQ.values))
    if 'tan' in cov: cols += [d.atx.values, d.aty.values]
    return np.vstack(cols).T


def rfit(A, y, w=None, clip=3.5, passes=4):
    n = len(y); w = np.ones(n) if w is None else w
    keep = w > 0
    for _ in range(passes):
        ww = w * keep
        c = np.linalg.lstsq(A * np.sqrt(ww)[:, None], y * np.sqrt(ww), rcond=None)[0]
        r = y - A @ c
        rs = 1.4826 * np.median(np.abs(r[w > 0]))
        keep = np.abs(r) < clip * rs
    return c, keep


def fit_shape(d, cov=('walk', 'tan'), nboot=200, seed=1):
    A = design(d, cov); y = d.lq.values
    c, keep = rfit(A, y)
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)
        B.append(rfit(A, y, w)[0])
    B = np.array(B)
    return dict(c=c, err=B.std(0), boot=B, N=len(y), keep=float(keep.mean()), mean_q=float(d.q.mean()),
                med_q=float(d.q.median()))
