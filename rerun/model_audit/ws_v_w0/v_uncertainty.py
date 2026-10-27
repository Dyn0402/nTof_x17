#!/usr/bin/env python3
"""v_uncertainty.py -- a real uncertainty on the reco-ladder v (F8 of the B4 model audit).

The ledger quoted the ladder v (median of (w - c) / tan_ref over 0.10 < |tan_ref| < 0.45, F8) with a purely
statistical error, 1.4826 MAD / sqrt(n) ~ 0.07 um/ns.  This script asks how much the number moves under everything
that should not matter, and assembles a budget per chamber and view:

  stat      event bootstrap of the estimator (B resamples) -- the honest statistical error of a median
  block     bootstrap of contiguous time blocks (catches slow drifts: gas, temperature, pressure)
  splits    consistency of disjoint subsets (time quintiles, chamber quadrants, M3 chi2 terciles, ftst class,
            sign of tan, charge terciles).  Per partition, chi2/ndf of the subset values about their mean; the
            excess scatter sigma_x = sqrt(max(0, var_obs - <var_stat>)) is the part not explained by statistics,
            and the partition's systematic on the GLOBAL value is sigma_x / sqrt(k) ... taken conservatively as
            sigma_x itself (a position- or time-dependent v is a real property that the global number averages
            over a particular population)
  estimator half-range over estimators (median ratio with/without the head-on offset, w0-subtracted, Theil-Sen
            line, 10 % trimmed mean) and windows (0.08-0.30 ... 0.10-0.60) -- the window part is F4's
            nonlinearity: the ladder v is not one number, it depends on the angles averaged
  reference M3 angle scale (z lever arm 1278 mm, +-3 mm; strip pitch +-0.1 %) and the alignment rotation
            (+-0.1 deg, the spread of the rotation scan between rows of one chamber)

Inputs: <root>/ws_v_w0/<det>.parquet from build_table.py.  Output: <root>/ws_v_w0/v_uncertainty.json (the
measurements; the 'budget' block here sums every split -- budget.py separates the population-average uncertainty
from the non-uniformity and the sign asymmetry, and draws the figures).
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import theilslopes

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
RNG = np.random.default_rng(20261027)
B = 300
REF_SCALE_FRAC = np.hypot(3.0 / 1278.0, 0.001)       # M3 lever arm +-3 mm, pitch +-0.1 %
ROT_DEG = 0.1


def est(w, tr, lo=0.10, hi=0.45, kind='median', c=None):
    """the ladder estimators; w [um/ns], tr = tan_ref; c = head-on offset (default: from this sample)."""
    m = (np.abs(tr) > lo) & (np.abs(tr) < hi)
    if m.sum() < 30:
        return np.nan
    hc = np.abs(tr) < 0.05
    if c is None:
        c = np.median(w[hc]) if hc.sum() > 20 else 0.0
    if kind == 'median':                      # production (fleet_tables.slope)
        return float(np.median((w[m] - c) / tr[m]))
    if kind == 'median_noc':
        return float(np.median(w[m] / tr[m]))
    if kind == 'median_w0':                   # offset from the w0 definition (set_w0kw: |tan| < 0.3)
        s = np.abs(tr) < 0.3
        v0 = np.median(w[m] / tr[m])
        w0 = np.median(w[s] - v0 * tr[s])
        return float(np.median((w[m] - w0) / tr[m]))
    if kind == 'theilsen':                    # robust straight line through the whole window incl. its offset
        mm = np.abs(tr) < hi
        idx = np.flatnonzero(mm)
        if len(idx) > 2500:
            idx = RNG.choice(idx, 2500, replace=False)
        return float(theilslopes(w[idx], tr[idx])[0])
    if kind == 'trim':
        r = np.sort((w[m] - c) / tr[m]); k = int(0.1 * len(r))
        return float(np.mean(r[k:len(r) - k]))
    raise ValueError(kind)


def boot(w, tr, n=B, blocks=None, **kw):
    vals = []
    N = len(w)
    for _ in range(n):
        if blocks is None:
            i = RNG.integers(0, N, N)
        else:
            nb = blocks.max() + 1
            pick = RNG.integers(0, nb, nb)
            i = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        vals.append(est(w[i], tr[i], **kw))
    return float(np.nanstd(vals))


def partitions(t, p):
    """disjoint subsets: name -> (labels array, list of label names)."""
    out = {}
    tt = t['evttime'].rank(method='first').to_numpy() if t['evttime'].notna().all() else np.arange(len(t), dtype=float)
    out['time'] = np.minimum((tt / len(t) * 5).astype(int), 4)
    px, py = t['x_p0'].to_numpy(), t['y_p0'].to_numpy()
    out['quadrant'] = (px > np.nanmedian(px)).astype(int) * 2 + (py > np.nanmedian(py)).astype(int)
    c = (t['Chi2X'] + t['Chi2Y']).to_numpy()
    out['m3_chi2'] = np.digitize(c, np.nanpercentile(c, [33.3, 66.7]))
    f = t[f'{p}_ftst'].to_numpy()
    vals, cnt = np.unique(f[np.isfinite(f)], return_counts=True)
    keep = vals[cnt >= 300]
    lab = np.full(len(t), -1)
    for k, v in enumerate(keep):
        lab[f == v] = k
    out['ftst'] = lab
    out['tan_sign'] = (t[f'tan_ref_{p}'].to_numpy() > 0).astype(int)
    q = t[f'{p}_q_sum'].to_numpy()
    out['charge'] = np.digitize(q, np.nanpercentile(q, [33.3, 66.7]))
    return out


def analyse(det):
    t = pd.read_parquet(os.path.join(ROOT, 'ws_v_w0', f'{det}.parquet'))
    res = {}
    for p in 'xy':
        ok = t[f'{p}_ok'].astype(bool).to_numpy() & np.isfinite(t[f'{p}_w']).to_numpy() & np.isfinite(t[f'tan_ref_{p}']).to_numpy()
        s = t[ok].sort_values('evttime') if t['evttime'].notna().all() else t[ok]
        w = s[f'{p}_w'].to_numpy() * 1e3; tr = s[f'tan_ref_{p}'].to_numpy()
        base = est(w, tr)
        m = (np.abs(tr) > 0.10) & (np.abs(tr) < 0.45)
        hc = np.abs(tr) < 0.05
        r = (w[m] - np.median(w[hc])) / tr[m]
        naive = float(1.4826 * np.median(np.abs(r - np.median(r))) / np.sqrt(m.sum()))
        stat = boot(w, tr)
        blocks = np.minimum((np.arange(len(w)) / len(w) * 25).astype(int), 24)
        block = boot(w, tr, blocks=blocks)
        R = dict(n=int(len(w)), n_window=int(m.sum()), v=base, err_naive=naive, err_stat=stat, err_block=block,
                 splits={}, estimators={}, windows={}, tan_bins={})
        # partitions
        for name, lab in partitions(s, p).items():
            labs = [k for k in np.unique(lab) if k >= 0]
            vv, ee, nn = [], [], []
            # a one-signed subset has a one-signed head-on band: its offset must come from the whole sample
            cg = dict(c=float(np.median(w[np.abs(tr) < 0.05]))) if name == 'tan_sign' else {}
            for k in labs:
                i = lab == k
                if i.sum() < 200:
                    continue
                vv.append(est(w[i], tr[i], **cg)); ee.append(boot(w[i], tr[i], n=150, **cg)); nn.append(int(i.sum()))
            vv, ee = np.array(vv), np.array(ee)
            if len(vv) < 2:
                continue
            wm = np.sum(vv / ee ** 2) / np.sum(1 / ee ** 2)
            chi2 = float(np.sum((vv - wm) ** 2 / ee ** 2)); ndf = len(vv) - 1
            sx = float(np.sqrt(max(0.0, np.var(vv, ddof=1) - np.mean(ee ** 2))))
            R['splits'][name] = dict(values=vv.tolist(), errors=ee.tolist(), n=nn, chi2=chi2, ndf=ndf,
                                     p=float(__import__('scipy.stats', fromlist=['chi2']).chi2.sf(chi2, ndf)),
                                     sigma_excess=sx)
        # |tan| bins (F4 nonlinearity, reported separately)
        for lo, hi in [(0.10, 0.20), (0.20, 0.30), (0.30, 0.45)]:
            R['tan_bins'][f'{lo:.2f}-{hi:.2f}'] = dict(v=est(w, tr, lo, hi), err=boot(w, tr, n=150, lo=lo, hi=hi))
        for kind in ['median', 'median_noc', 'median_w0', 'theilsen', 'trim']:
            R['estimators'][kind] = est(w, tr, kind=kind)
        for lo, hi in [(0.08, 0.30), (0.10, 0.30), (0.10, 0.45), (0.15, 0.45), (0.10, 0.60)]:
            R['windows'][f'{lo:.2f}-{hi:.2f}'] = est(w, tr, lo, hi)
        # reference: rotation +-0.1 deg mixes the other view's tangent in
        other = 'y' if p == 'x' else 'x'
        to = s[f'tan_ref_{other}'].to_numpy()
        d = np.radians(ROT_DEG)
        rot = [est(w, np.cos(d) * tr + sg * np.sin(d) * to) for sg in (+1, -1)]
        R['ref_rotation'] = float(np.max(np.abs(np.array(rot) - base)))
        R['ref_scale'] = float(base * REF_SCALE_FRAC)
        # budget
        split_sys = float(np.sqrt(np.sum([v['sigma_excess'] ** 2 for v in R['splits'].values()])))
        est_sys = float(0.5 * (max(R['estimators'].values()) - min(R['estimators'].values())))
        win_sys = float(0.5 * (max(R['windows'].values()) - min(R['windows'].values())))
        R['budget'] = dict(stat=max(stat, block), splits=split_sys, estimator=est_sys, window=win_sys,
                           reference=float(np.hypot(R['ref_rotation'], R['ref_scale'])))
        R['budget']['total_excl_window'] = float(np.sqrt(sum(R['budget'][k] ** 2 for k in ['stat', 'splits', 'estimator', 'reference'])))
        R['budget']['total'] = float(np.hypot(R['budget']['total_excl_window'], win_sys))
        res[p] = R
        print(f"{det} {p}: v {base:.2f}  naive {naive:.3f} stat {stat:.3f} block {block:.3f} | splits {split_sys:.3f} "
              f"est {est_sys:.3f} window {win_sys:.3f} ref {R['budget']['reference']:.3f} | total {R['budget']['total']:.3f} "
              f"(excl window {R['budget']['total_excl_window']:.3f})")
        for name, v in R['splits'].items():
            print(f"     {name:9s} " + ' '.join(f'{a:.2f}±{b:.2f}' for a, b in zip(v['values'], v['errors'])) +
                  f"  chi2/ndf {v['chi2']:.1f}/{v['ndf']}  excess {v['sigma_excess']:.3f}")
        print('     estimators', {k: round(v, 2) for k, v in R['estimators'].items()}, 'windows', {k: round(v, 2) for k, v in R['windows'].items()})
    return res


def main():
    allres = {det: analyse(det) for det in (sys.argv[1:] or DETS)}
    json.dump(allres, open(os.path.join(ROOT, 'ws_v_w0', 'v_uncertainty.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
