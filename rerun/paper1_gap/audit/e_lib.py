"""e_lib.py -- Task E (audit): drift-line SHAPE from the waveforms.  Extraction of per-sample charge centroids and the
per-region regressions (centroid - ref_mesh) = z_k tan + delta_k.  Import only; writes nothing.

Time frame: t = 60 ns * k - t0_fit (the fit's t0 is pinned to C - 10 ns ftst, so t is a trigger-referenced time).
Per event, view, sample k: centroid of the strips with W > nsig*noise within +-R of the predicted landing position
ref_mesh + tan z_pred(t) (R = r0 + r1 |tan|), z_pred from a first-pass calibration (iterated).
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO = '/home/dylan/PycharmProjects/nTof_x17'
sys.path.insert(0, f'{REPO}/rerun/paper1_gap')
import gapmap as GM  # noqa: E402
import cellobs as CO  # noqa: E402

WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/e'
SNS = 60.0
SAT = 3500.0
GAP = 28.2
JOINT = json.load(open(f'{REPO}/rerun/paper1_gap/results/joint.json'))


def dead_mask(run):
    R = GM.RUNS[run]
    b = json.load(open(f'{WORK}/pass1/{R["rel"]}/calib_bundle/bundle.json'))
    return {'x': set(GM.MASK_X), 'y': set(int(c) for c in b['dead']['y'])}


def load_events(run):
    """merged incl + headon caches matched to the standard-selection table.  -> dict eid -> event, table, match stats"""
    R = GM.RUNS[run]
    t = GM.load_table(WORK, R['rel'])
    ev = {}
    stats = {}
    for nm in ('headon', 'incl'):
        d = pickle.load(open(f'{WORK}/build_{R["key"]}/out/{R["key"]}/{nm}.pkl', 'rb'))
        ok = [k for k in d if k in t.index]
        a = np.array([d[k]['tan_x'] for k in ok]); b = t.loc[ok, 'tan_ref_x'].values
        c = np.array([d[k]['ref_mesh_x'] for k in ok]); e = t.loc[ok, 'ref_mesh_x'].values
        stats[nm] = dict(n_cache=len(d), n_matched=len(ok), frac=len(ok) / len(d),
                         max_dtan=float(np.abs(a - b).max()), max_dref=float(np.abs(c - e).max()))
        for k in ok:
            if k not in ev:
                d[k]['src'] = nm
                ev[k] = d[k]
    stats['n_events'] = len(ev)
    return ev, t, stats


def zpred_default(t):
    return np.clip(0.0372 * (t - 90.0), 0.0, GAP)


def extract(ev, tab, view, bad, zpred=zpred_default, nsig=3.0, r0=3.0, r1=6.0, amin=0.0, sat_event=True):
    """rows (one per event and sample with a usable centroid).  Columns: eid, k, t, tan, ref, cen, A, n, src, sat_ev,
    ftst, xm, ym, drop_mask (a masked strip sits in the window), drop_sat (a saturated strip in the window)."""
    rows = []
    ks = np.arange(32)
    for eid, e in ev.items():
        if view not in e or eid not in tab.index:
            continue
        row = tab.loc[eid]
        t0 = float(row[f'{view}_t0'])
        tan = float(e[f'tan_{view}']); ref = float(e[f'ref_mesh_{view}'])
        W = np.asarray(e[view]['W'], float); pos = np.asarray(e[view]['pos'], float)
        nz = np.asarray(e[view]['noise'], float); ch = np.asarray(e[view]['ch'])
        isbad = np.array([int(c) in bad for c in ch])
        t = SNS * ks - t0
        sel = (t > -150) & (t < 950)
        kk = ks[sel]; tt = t[sel]
        c = ref + tan * zpred(tt)
        R = r0 + r1 * abs(tan)
        inwin = np.abs(pos[:, None] - c[None, :]) <= R           # (n, nk)
        Wk = W[:, kk]
        sig = (Wk > nsig * nz[:, None]) & inwin
        wgt = np.where(sig, Wk, 0.0)
        A = wgt.sum(0)
        nst = sig.sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            cen = (wgt * pos[:, None]).sum(0) / A
        dm = (inwin & isbad[:, None]).any(0)
        ds = (inwin & (Wk >= SAT)).any(0)
        satev = bool((W.max(1) >= SAT).any())
        good = (nst >= 2) & (A > amin)
        for j in np.where(good)[0]:
            rows.append((eid, int(kk[j]), tt[j], tan, ref, cen[j], A[j], nst[j], e['src'], satev,
                         int(e[f'ftst_{view}']), float(row.xm), float(row.ym), bool(dm[j]), bool(ds[j])))
    return pd.DataFrame(rows, columns=['eid', 'k', 't', 'tan', 'ref', 'cen', 'A', 'n', 'src', 'sat_ev', 'ftst', 'xm',
                                      'ym', 'drop_mask', 'drop_sat'])


# ---------------------------------------------------------------- regression per time bin
def clip_flags(y, tan, tb, nb, clip=3.5):
    """robust per-bin line, returns keep mask (3 passes)."""
    keep = np.ones(len(y), bool)
    for _ in range(3):
        S = sums(y, tan, tb, nb, keep.astype(float))
        a, b = solve(S)
        r = y - a[tb] - b[tb] * tan
        for i in range(nb):
            m = (tb == i) & keep
            if m.sum() > 10:
                rs = 1.4826 * np.median(np.abs(r[m] - np.median(r[m])))
                keep[tb == i] = np.abs(r[tb == i]) < clip * max(rs, 1e-6)
    return keep


def sums(y, tan, tb, nb, w):
    S = np.zeros((nb, 5))
    for j, v in enumerate([np.ones_like(y), tan, tan * tan, y, tan * y]):
        S[:, j] = np.bincount(tb, weights=w * v, minlength=nb)[:nb]
    return S


def solve(S):
    """y = a + b tan per bin from the sums (weights folded in)."""
    n, st, stt, sy, sty = S.T
    with np.errstate(invalid='ignore', divide='ignore'):
        det = n * stt - st ** 2
        b = (n * sty - st * sy) / det
        a = (sy - b * st) / n
    return a, b


def line_cov(S, y2):
    """errors of (a, b) from sums and the residual rms (y2 = sum w y^2 per bin)."""
    n, st, stt, sy, sty = S.T
    with np.errstate(invalid='ignore', divide='ignore'):
        det = n * stt - st ** 2
        a, b = solve(S)
        rss = y2 - a * sy - b * sty
        s2 = rss / np.maximum(n - 2, 1)
        va = s2 * stt / det
        vb = s2 * n / det
    return np.sqrt(va), np.sqrt(vb)


TB = np.array([0, 0.03, 0.06, 0.12, 0.2, 0.3, 0.45, 0.6, 0.8, 1.01])


def balance(d):
    """row weights making the +tan and -tan rows equal in number within every |tan| bin (removes the leak of an odd
    nonlinearity tan*|tan| into the intercept when the region's mean tan is not zero)."""
    ib = np.digitize(np.abs(d.tan.values), TB)
    sg = d.tan.values > 0
    wt = np.ones(len(d))
    for i in np.unique(ib):
        m = ib == i
        n1, n0 = (m & sg).sum(), (m & ~sg).sum()
        if n1 == 0 or n0 == 0:
            wt[m] = 0.0
            continue
        wt[m & sg] = (n1 + n0) / (2.0 * n1); wt[m & ~sg] = (n1 + n0) / (2.0 * n0)
    return wt


def profile_region(df, edges, ymin_n=60, keep=None, boot=0, seed=0, ret_all=False):
    """df: rows of one region (cols t, tan, y = cen - ref).  Time bins by edges.  Returns per-bin z, delta, errors,
    n; bootstrap over events (resampling eids) if boot > 0, returning the (boot, nb) arrays too."""
    nb = len(edges) - 1
    tb = np.digitize(df.t.values, edges) - 1
    ok = (tb >= 0) & (tb < nb)
    d = df[ok]; tb = tb[ok]
    y = (d.cen - d.ref).values; tan = d.tan.values
    kp = clip_flags(y, tan, tb, nb) if keep is None else keep[ok]
    w = kp.astype(float) * (d.wt.values if 'wt' in d else 1.0)
    S = sums(y, tan, tb, nb, w)
    a, b = solve(S)
    ea, eb = line_cov(S, np.bincount(tb, weights=w * y * y, minlength=nb)[:nb])
    out = dict(z=b, delta=a, ez=eb, ed=ea, n=S[:, 0], tmid=np.array([d.t.values[tb == i].mean() if (tb == i).any() else np.nan
                                                                     for i in range(nb)]))
    if boot:
        rng = np.random.default_rng(seed)
        eids = d.eid.values
        ue, inv = np.unique(eids, return_inverse=True)
        zb, db = [], []
        for _ in range(boot):
            cnt = np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(float)
            Sb = sums(y, tan, tb, nb, w * cnt[inv])
            aa, bb = solve(Sb)
            zb.append(bb); db.append(aa)
        out['zb'], out['db'] = np.array(zb), np.array(db)
    return out
