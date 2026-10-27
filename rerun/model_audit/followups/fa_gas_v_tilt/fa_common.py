"""fa_common.py -- shared helpers of the FA follow-up (gas v(t), v(x, y), X-Y v, tilt per run)."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
WS = os.path.join(ROOT, 'ws_v_w0')
FA = os.path.join(ROOT, 'followups', 'fa')                     # local pulls of the FA condor products
EOS_FA = '/eos/user/d/dneff/x17rerun/followups/fa'
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
GOLD = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
GOLD_ROW = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
            'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
            'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4',
            'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
            'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
CENTRE = 200.0          # chamber centre in the strip frame [mm]: positions are taken relative to it, not to a median
COL = {'det2': '#4C72B0', 'det3': '#DD8452', 'det4': '#55A868', 'det6': '#C44E52', 'det7': '#8172B3'}
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)


def huber(X, y, k=1.345, it=40):
    """robust multi-linear fit y = X beta (Huber IRLS); returns beta, cov (from the final weighted LSQ), mask used."""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    wt = np.ones_like(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        u = np.abs(r) / (k * s); wt = np.where(u <= 1, 1.0, 1.0 / u)
        sw = np.sqrt(wt)
        nb = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.allclose(nb, beta, atol=1e-9, rtol=1e-7):
            beta = nb
            break
        beta = nb
    r = y - X @ beta
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    A = (X * wt[:, None]).T @ X
    cov = np.linalg.pinv(A) * s ** 2
    return beta, cov, int(m.sum())


def load_table(path):
    t = pd.read_parquet(path)
    if t.index.name == 'event_id':
        t = t.reset_index()
    return t


def golden_table(det):
    return load_table(os.path.join(WS, f'{det}.parquet'))


def tmean(a, cut=0.2):
    """symmetric trimmed mean (cut fraction off each end).  Used for the column duration instead of a median: about half
    of the per-event T sit exactly on the fit's 10 ns grid (the chi2 in T is flat between 60 ns samples), so a median of
    many events is quantised to 10 ns (1.2 %); a trimmed mean is not."""
    a = np.sort(np.asarray(a, float)[np.isfinite(np.asarray(a, float))])
    k = int(cut * len(a))
    return float(np.mean(a[k:len(a) - k])) if len(a) > 2 * k else np.nan


def load_colsum(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    n = len(d['eid'])
    df = pd.DataFrame({k: v for k, v in d.items() if v.ndim == 1 and len(v) == n})
    S = {p: d[f'{p}_S'] for p in 'xy'}
    return df, S, d
