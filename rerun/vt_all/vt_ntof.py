#!/usr/bin/env python3
"""vt_ntof.py --stage3 DIR --out F.json <run> [...] -- v(t) feasibility on the n_TOF beam data (VT_PLAN_2026-10-30).

No reference: the drift velocity is read from the source geometry.  The tracks come from the capsule on the beam axis,
D_PERP_MM from each strip plane, so on the pointing-coincident sample tan_true = (u - foot) / D_PERP and the in-situ
scale k = v_assumed / v_true is the median of the per-track ratio (sept26_prelim_analysis.k_arm.track_k, here on the
stage-3 table: x_local, tan_raw_x, coinc_this_arm).  A k that moves with time is a v that moves with time:
v_rel(t) = k_row / k(t).  Beside it, two column observables that need no geometry but carry gain and attachment: the
fraction of gated tracks whose column end rails at the depth-grid edge, and the median 90 %-charge time x_q_u90.

Time: there is no per-event timestamp in stage 3.  A track's time is its acquisition file's: the sub-run's stem time
(tag 'YYMMDD_HHhMM') plus (file index + 0.5) / n_files of the gap to the next sub-run's stem.  Blocks are the sub-runs,
and each sub-run's two halves (by file index) for the within-sub-run test.
"""
import argparse, datetime as dt, glob, json, os

import numpy as np
import pandas as pd

D_PERP_MM = 234.6                       # k_arm.D_PERP_MM
LEVER = (30.0, 130.0)                   # k_arm.LEVER_WINDOW_MM
N_MIN = 200                             # k_arm's own minimum for track_k
T_REF = dt.datetime(2026, 6, 1)
COLS = ['subrun', 'tag', 'arm', 'gated', 'coinc_this_arm', 'x_local', 'tan_raw_x', 'drift_railed', 'x_q_u90',
        'x_quality_ok', 'v_drift_prior_um_ns']


def stem(tag):
    d, hm = tag.split('_')[:2]
    return dt.datetime(2000 + int(d[:2]), int(d[2:4]), int(d[4:6]), int(hm[:2]), int(hm[3:5]))


def k_track(lev, tan):
    m = (np.abs(lev) > LEVER[0]) & (np.abs(lev) < LEVER[1]) & (np.abs(tan) > 1e-3) & np.isfinite(tan)
    if m.sum() < N_MIN:
        return np.nan, np.nan, int(m.sum())
    q = (lev[m] / D_PERP_MM) / tan[m]
    k = float(np.median(q))
    return k, float(1.2533 * 1.4826 * np.median(np.abs(q - k)) / np.sqrt(m.sum())), int(m.sum())


def measure(run, d):
    files = sorted(glob.glob(os.path.join(d, f'tracks_{run}_*.parquet')))
    T = []
    for f in files:
        t = pd.read_parquet(f, columns=COLS)
        m = json.load(open(f[:-8] + '.meta.json'))
        t['foot'] = t.arm.map(m['geometry']['pinwheel'])
        T.append(t)
    T = pd.concat(T, ignore_index=True)
    st = T.groupby('subrun').tag.agg(lambda s: min(s)).map(stem).sort_index()
    # file time: stem + (index + 0.5)/n of the gap to the next sub-run (the last sub-run: its median gap)
    nxt = st.shift(-1)
    gap = (nxt - st).dt.total_seconds()
    gap = gap.fillna(gap.median())
    T['fidx'] = T.tag.str.split('_').str[2].astype(int)
    nf = T.groupby('subrun').fidx.max() + 1
    T['t_h'] = ((T.subrun.map(st) - T_REF).dt.total_seconds()
                + (T.fidx + 0.5) / T.subrun.map(nf) * T.subrun.map(gap)) / 3600
    T['half'] = (T.fidx >= T.subrun.map(nf) / 2).astype(int)
    out = dict(run=run, n_subruns=len(st), t_start=float(T.t_h.min()), t_end=float(T.t_h.max()), arms={})
    for arm, A in T[T.gated].groupby('arm'):
        C = A[A.coinc_this_arm == 1]
        lev = (C.x_local - C.foot).to_numpy()
        k_run, e_run, n_run = k_track(lev, C.tan_raw_x.to_numpy())
        if not np.isfinite(k_run):
            continue
        rows = []
        for (sub, half), B in [((s, None), b) for s, b in C.groupby('subrun')] + list(C.groupby(['subrun', 'half'])):
            k, e, n = k_track((B.x_local - B.foot).to_numpy(), B.tan_raw_x.to_numpy())
            G = A[(A.subrun == sub) & ((A.half == half) if half is not None else True)]
            rows.append(dict(subrun=sub, half=half, t=float(B.t_h.median()), k=k, k_err=e, n=n,
                             v_rel=k_run / k if k else np.nan, v_rel_err=k_run * e / k ** 2 if k else np.nan,
                             railed=float(G.drift_railed.mean()), q_u90=float(G.x_q_u90.median()), n_gated=len(G)))
        R = pd.DataFrame(rows)
        sub = R[R.half.isna() & np.isfinite(R.k)]
        fit = {}
        if len(sub) >= 3:
            w = 1 / sub.v_rel_err ** 2
            tm = float(np.average(sub.t, weights=w))
            A_ = np.column_stack([np.ones(len(sub)), sub.t - tm]) * np.sqrt(w.to_numpy())[:, None]
            c, *_ = np.linalg.lstsq(A_, sub.v_rel.to_numpy() * np.sqrt(w.to_numpy()), rcond=None)
            cov = np.linalg.inv(A_.T @ A_)
            chi2 = float(np.sum((A_ @ c - sub.v_rel.to_numpy() * np.sqrt(w.to_numpy())) ** 2))
            chi2_0 = float(np.sum(w * (sub.v_rel - np.average(sub.v_rel, weights=w)) ** 2))
            dof = len(sub) - 2
            infl = np.sqrt(max(1, chi2 / dof)) if dof > 0 else 1
            fit = dict(slope_per_h=float(c[1]), slope_err=float(np.sqrt(cov[1, 1]) * infl), chi2=chi2, dof=dof,
                       chi2_const=chi2_0, dof_const=len(sub) - 1, rms_sub=float(np.std(sub.v_rel)),
                       median_err_sub=float(np.median(sub.v_rel_err)),
                       dv_total=float(c[1] * (sub.t.max() - sub.t.min())))
        # within-sub-run: the two halves' difference against its error
        # (a sub-run whose later half has no measured k, or a run of one-file sub-runs, has no pair)
        H = R[R.half.notna() & np.isfinite(R.v_rel)].pivot_table(index='subrun', columns='half',
                                                                 values=['v_rel', 'v_rel_err'])
        if len(H) and {0, 1} <= set(H['v_rel'].columns):
            dh = H['v_rel'][1] - H['v_rel'][0]
            eh = np.hypot(H['v_rel_err'][1], H['v_rel_err'][0])
        else:
            dh = eh = pd.Series(dtype=float)
        out['arms'][arm] = dict(k_run=k_run, k_run_err=e_run, n_run=n_run, v_prior=float(A.v_drift_prior_um_ns.median()),
                                v_insitu=float(A.v_drift_prior_um_ns.median()) / k_run, fit=fit,
                                half_diff=dict(mean=float(dh.mean()), rms=float(dh.std()), med_err=float(eh.median()),
                                               chi2=float(np.nansum((dh / eh) ** 2)), n=int(dh.notna().sum())),
                                blocks=R.replace({np.nan: None}).to_dict('records'))
        f = out['arms'][arm]
        print(f"{run} {arm}: k_run {k_run:.3f}±{e_run:.3f} n {n_run}  sub-runs {len(sub)}  "
              + (f"slope {fit['slope_per_h']*100:+.3f}±{fit['slope_err']*100:.3f} %/h, dv {fit['dv_total']*100:+.1f} %, "
                 f"rms {fit['rms_sub']*100:.2f} % vs err {fit['median_err_sub']*100:.2f} %, chi2_0 {fit['chi2_const']:.0f}/"
                 f"{fit['dof_const']}" if fit else '')
              + f"  halves: diff rms {f['half_diff']['rms']*100:.2f} % vs err {f['half_diff']['med_err']*100:.2f} %",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--stage3', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = []
    for r in a.runs:            # one run at a time, written after each: memory stays one run's columns
        res.append(measure(r, a.stage3))
        json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
