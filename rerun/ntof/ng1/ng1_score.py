#!/usr/bin/env python3
"""ng1_score.py -- the N-G1 gate (NTOF_PASS_PLAN_2026-11-02 §3 P5, "N-G1 gate") on pass 1's own output.

Pass 1 (code ed0b1162, frozen bundles 11-07; EOS Analysis_pass1/stage2/) against is2_v2 on the same events:

  eff     run_145 beam, every tag: gated tracks per trigger, all four arms, on one common trigger set per sub-run
          (the union of the pass's four arm event tables and the is2_v2 stage-3 events).  |tan| < 0.6 on both.
  wall    run_145 beam: SiPM-wall confirmation (match minus the same-width off-time control,
          det_a_scint.match_run, unchanged) per arm, per trigger and per gated track.  Beam only: the slim
          exports (wall hits) exist for beam runs, not for the run_149 cosmic run.
  cos     run_149 cosmic, on the in-situ A-C truth events (ntof_cosmics/insitu_calib.py truth.parquet, 14 sub-runs,
          defined from production, independent of both chains): efficiency (both planes ok + quality_ok) and the
          per-view angle residual against the truth line, pass vs is2_v2's cosmic recipe (is2ts_s3 = is2 bundle,
          seeder 3, TAN_MAX 1.0, two-sided: exactly the is2_v2 stage-2 settings), paired on common events.  Raw
          (each chain's own v) and each chain rescaled on the train third (resolution only, insitu_paired.py).
  line    run_149 cosmic, every sub-run: ac_line.score on the pass's A and C (the A-C line s68 per incidence bin, and
          the angle-scale ratio fitted/true per view).  is2_v2 has no run_149 output beyond the truth events, so the
          is2 comparison of the line is the paired one in `cos`.
  srcy    run_145 beam: the late (>= 20 ms) y band crossing (y_guard.late_y_crossing, the stage-3 guard's
          estimator) on A and C, pass vs is2_v2 on the same sub-runs, against -31 (A) / -35 (C).
  focus   run_145 beam, 20-80 ms (HANDOFF_TRACKING §22 population), |tan| < 0.6: the capsule-focus scan (sharpest x
          tan scale s, median |miss| at the axis with the measured source offset dx = -9.5 mm), pass vs is2_v2 on
          the same tracks' events.  G5: s(focus) against the in-beam-muon / A-C line scale.

    ng1_score.py build | eff | wall | cos | line | srcy | focus | all

Work dir $WORK_DATA/ntof_pass1/ng1_pass (pass/ = the EOS stage-2 copy, s3/ = the scoring track tables,
results/ = every number).  The s3/ tables are a SCORING product (build_tracks.pair_rows + local_and_global, k = 1:
the bundle's v is the working v, y up), not stage 3.

D's run_145 beam rows are on the L1 pin (rerun_dbeam_L1, condor 4677439; rescored 2026-11-19).  wall() caches
results/wall_<chain>_<arm>.parquet: delete the file of any arm whose inputs changed before re-scoring.
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(NT, '..', '..'))
sys.path[:0] = [NT, REPO, os.path.join(REPO, 'sept26_prelim_analysis')]
warnings.filterwarnings('ignore')

WD = os.environ.get('WORK_DATA') or '/media/ucla/work'
W = os.path.join(WD, 'ntof_pass1', 'ng1_pass')
PASS, S3, OUT = f'{W}/pass', f'{W}/s3', f'{W}/results'
IS2S3 = '/media/dylan/data/x17/sept26_prelim/stage3_is2_v2'
INSITU = os.path.expanduser('~/scratch/ntof_insitu')
RUNS = '/media/dylan/data/x17/beam_july/runs'
ARMS = 'ABCD'
TAN_MAX = 0.6
REF_Y = {'A': -31.0, 'C': -35.0}
DX_SRC = -9.5                               # source offset in global x (focus_offset.py, imaging -9.3)
SGN_U = {'A': 1.0, 'C': -1.0}
rng = np.random.default_rng(145)


def rc(run):
    return json.load(open(f'{RUNS}/run_{run}/run_config.json'))


def subruns(run):
    return sorted(os.path.basename(p) for p in glob.glob(f'{PASS}/run_{run}/*'))


def acq_files(run, sub, arm):
    return sorted(glob.glob(f'{PASS}/run_{run}/{sub}/{arm}/*_{arm}.parquet'))


def tag_of(f, sub):
    return os.path.basename(f).split(f'{sub}_')[1][:-len('_X.parquet')]


# --------------------------------------------------------------------------------------------------------------- build
def build():
    """s3/tracks_run_<r>_<sub>.parquet (all arms, stage-3-like columns det_a_scint / y_guard / focus read) and
    s3/events_run_<r>_<sub>.parquet (every seeded trigger per arm, with its event-level flags)."""
    import build_tracks as BT
    import trigger_time
    import capsule_point
    from ntof_tracking.reco import geometry as G
    os.makedirs(S3, exist_ok=True)
    for run in (145, 149):
        trs = G.detector_transforms(rc(run))
        for sub in subruns(run):
            dst = f'{S3}/tracks_run_{run}_{sub}.parquet'
            if os.path.exists(dst):
                continue
            T, E = [], []
            for arm in ARMS:
                for f in acq_files(run, sub, arm):
                    tag = tag_of(f, sub)
                    ev = pd.read_parquet(f, columns=['event_id', 'n_tracks', 'x_ok', 'y_ok', 'x_quality_ok',
                                                     'y_quality_ok', 'x_tan_theta', 'y_tan_theta'])
                    E.append(ev.assign(run=f'run_{run}', subrun=sub, tag=tag, arm=arm))
                    c = pd.read_parquet(f.replace('.parquet', '.candidates.parquet'))
                    c['tag'] = tag
                    t = BT.pair_rows(c)
                    if not len(t):
                        continue
                    t = BT.local_and_global(t, trs[f'mx17_{arm}'], k=1.0)
                    T.append(t.assign(arm=arm))
            t = pd.concat(T, ignore_index=True)
            t['run'], t['subrun'] = f'run_{run}', sub
            t['chi2dof_x'] = t.x_chi2 / t.x_dof.clip(lower=1)
            t['chi2dof_y'] = t.y_chi2 / t.y_dof.clip(lower=1)
            t['q_total'] = t.x_q_sum + t.y_q_sum
            t['dca_axis_mm'], t['y_at_axis'] = capsule_point.axis_dca(t)
            t['drift_railed'] = False
            for c in ('pred_sipm_bar', 'pred_plastic', 'pred_ls', 'wall_A', 'plastic_A', 'k_arm'):
                t[c] = np.nan
            t['k_arm'] = 1.0
            if run == 145:
                t, _ = trigger_time.attach(t, f'run_{run}', sub)
            else:
                t['t_since_flash_ns'] = np.nan
            t.to_parquet(dst, index=False)
            pd.concat(E, ignore_index=True).to_parquet(f'{S3}/events_run_{run}_{sub}.parquet', index=False)
            print(f'  run_{run} {sub}: {len(t):,} tracks', flush=True)
    # the stage-3 sidecar y_up() reads: these tables are y up by construction (local_and_global, TAN_SIGN_Y)
    for f in glob.glob(f'{S3}/tracks_*.parquet'):
        m = f.replace('.parquet', '.meta.json')
        if not os.path.exists(m):
            json.dump(dict(y_convention='up', source='rerun/ntof/ng1/ng1_score.py build (scoring table, k=1)'),
                      open(m, 'w'))


def acc(d):
    return d[d.gated.astype(bool) & (d.tanx.abs() < TAN_MAX) & (d.tany.abs() < TAN_MAX)]


def is2(run, sub, cols=None):
    import build_tracks as BT
    f = f'{IS2S3}/tracks_run_{run}_{sub}.parquet'
    d = pd.read_parquet(f, columns=cols)
    return BT.y_up(d, f)


def pas(run, sub, cols=None):
    return pd.read_parquet(f'{S3}/tracks_run_{run}_{sub}.parquet', columns=cols)


# ----------------------------------------------------------------------------------------------------------------- eff
def eff():
    rows = []
    cols = ['event_id', 'arm', 'gated', 'tanx', 'tany']
    for sub in subruns(145):
        P = pas(145, sub, cols)
        I = is2(145, sub, cols)
        E = pd.read_parquet(f'{S3}/events_run_145_{sub}.parquet')
        trig = set(E.event_id) | set(I.event_id)
        for arm in ARMS:
            r = dict(subrun=sub, arm=arm, n_trig=len(trig), pass_gated=len(acc(P[P.arm == arm])),
                     pass_seeded=int((E.arm == arm).sum()))
            if arm in 'AC':
                r['is2_gated'] = len(acc(I[I.arm == arm]))
            rows.append(r)
    R = pd.DataFrame(rows)
    S = R.groupby('arm').sum(numeric_only=True)
    S['n_trig'] = R.groupby('subrun').n_trig.first().sum()
    S['pass_per_trig'] = S.pass_gated / S.n_trig
    S['is2_per_trig'] = S.get('is2_gated', np.nan) / S.n_trig
    S.loc[~S.index.isin(['A', 'C']), ['is2_gated', 'is2_per_trig']] = np.nan
    S['d_pt'] = 100 * (S.pass_per_trig - S.is2_per_trig)
    S.reset_index().to_csv(f'{OUT}/eff_run145.csv', index=False)
    print(S.round(4).to_string())
    return S


# ---------------------------------------------------------------------------------------------------------------- wall
def wall():
    import det_a_scint as DS
    rows, subs = [], subruns(145)
    for chain, src in (('pass', S3), ('is2_v2', IS2S3)):
        for arm in (ARMS if chain == 'pass' else 'AC'):
            f = f'{OUT}/wall_{chain}_{arm}.parquet'
            if os.path.exists(f):
                M = pd.read_parquet(f)
            else:
                M, _ = DS.match_run('run_145', subs, __import__('pathlib').Path(src), None,
                                    DS.layer_geometry('run_145', arm))
                M.to_parquet(f, index=False)
            M = M[(M.tanx.abs() < TAN_MAX) & (M.tany.abs() < TAN_MAX)]
            w = M[M.on_wall.astype(bool)]
            rows.append(dict(chain=chain, arm=arm, gated=len(M), on_wall=len(w),
                             match=int(w.match_wall.sum()), ctrl=int(w.match_wall_ctrl.sum()),
                             confirmed=int(w.match_wall.sum() - w.match_wall_ctrl.sum())))
    R = pd.DataFrame(rows)
    E = eff().reset_index()
    R = R.merge(E[['arm', 'n_trig']], on='arm')
    R['conf_per_trig'] = R.confirmed / R.n_trig
    R['conf_per_gated'] = R.confirmed / R.gated
    R['conf_per_onwall'] = R.confirmed / R.on_wall
    R.to_csv(f'{OUT}/wall_run145.csv', index=False)
    print(R.round(4).to_string(index=False))
    # Composition: the same single-track events in both chains, and the tracks only one chain has.  Paired bootstrap of
    # the net confirmation difference on the common events.
    rows = []
    for arm in 'AC':
        cut = lambda d: d[(d.tanx.abs() < TAN_MAX) & (d.tany.abs() < TAN_MAX) & d.on_wall.astype(bool)]
        P, I = cut(pd.read_parquet(f'{OUT}/wall_pass_{arm}.parquet')), cut(pd.read_parquet(f'{OUT}/wall_is2_v2_{arm}.parquet'))
        key = lambda d: d.set_index(['subrun', 'event_id']).index
        P1, I1 = P[P.n_trk == 1], I[I.n_trk == 1]
        M = P1.merge(I1, on=['subrun', 'event_id'], suffixes=('_p', '_i'))
        po, io = P1[~key(P1).isin(key(I))], I1[~key(I1).isin(key(P))]
        cp = (M.match_wall_p.astype(int) - M.match_wall_ctrl_p.astype(int)).to_numpy()
        ci = (M.match_wall_i.astype(int) - M.match_wall_ctrl_i.astype(int)).to_numpy()
        idx = rng.integers(0, len(M), (1000, len(M)))
        dd = cp[idx].mean(1) - ci[idx].mean(1)
        net = lambda d: float(d.match_wall.mean() - d.match_wall_ctrl.mean())
        rows.append(dict(arm=arm, n_common=len(M), net_common_pass=float(cp.mean()), net_common_is2=float(ci.mean()),
                         d_common_pt=100 * float(cp.mean() - ci.mean()), d_common_err_pt=100 * float(dd.std()),
                         pos_agree_3mm=float(((M.u_mm_p - M.u_mm_i).abs() < 3).mean()),
                         n_pass_only=len(po), net_pass_only=net(po), n_is2_only=len(io), net_is2_only=net(io)))
    S = pd.DataFrame(rows)
    S.to_csv(f'{OUT}/wall_split_run145.csv', index=False)
    print(S.round(4).to_string(index=False))
    return R


# ----------------------------------------------------------------------------------------------------------------- cos
def _evok(d):
    return (d.x_ok == True) & (d.y_ok == True) & (d.x_quality_ok == True) & (d.y_quality_ok == True)


def cos():
    T = pd.read_parquet(f'{INSITU}/truth.parquet')
    rows, bins = [], []
    for arm in 'AC':
        t = T[T.arm == arm]
        P = []
        for sub in sorted(t.subrun.unique()):
            for f in acq_files(149, sub, arm):
                P.append(pd.read_parquet(f).assign(subrun=sub))
        P = pd.concat(P, ignore_index=True)
        I = pd.read_parquet(f'{INSITU}/reco_is2ts_s3_{arm}.parquet')
        M = t.merge(P, on=['subrun', 'event_id'], how='left').merge(I, on=['subrun', 'event_id'], how='left',
                                                                     suffixes=('_p', '_i'))
        ok_p, ok_i = _evok(M.rename(columns=lambda c: c[:-2] if c.endswith('_p') else c)), \
            _evok(M.rename(columns=lambda c: c[:-2] if c.endswith('_i') else c))
        # The quality rule is RELATIVE: chi2/dof <= 5 x the median chi2/dof of the view over the table.  The pass's
        # table is every trigger of the acquisition; is2ts_s3 was run on an allowlist of the truth events only, whose
        # median is 2-3x higher on C, so its cut is that much looser.  The fit-level efficiency (both planes ok) is
        # free of that; the quality-level one is quoted with each chain's threshold.
        fit_p = (M.x_ok_p == True) & (M.y_ok_p == True)
        fit_i = (M.x_ok_i == True) & (M.y_ok_i == True)
        thr = {}
        for ax in 'xy':
            thr[f'thr_pass_{ax}'] = float(np.median([json.load(open(f.replace('.parquet', '.meta.json')))
                                                     ['quality_rule']['chi2dof_median'][ax] for f in
                                                     [g for s in t.subrun.unique() for g in acq_files(149, s, arm)]]) * 5)
            ci = I[f'{ax}_chi2'] / I[f'{ax}_dof'].clip(lower=1)
            thr[f'thr_is2_{ax}'] = float(5 * ci.groupby(I.subrun).median().median())
        rows.append(dict(arm=arm, n_truth=len(t), pass_fit_eff=float(fit_p.mean()), is2_fit_eff=float(fit_i.mean()),
                         pass_eff=float(ok_p.mean()), is2_eff=float(ok_i.mean()),
                         n_pass=int(ok_p.sum()), n_is2=int(ok_i.sum()), n_both=int((ok_p & ok_i).sum()), **thr))
        both = M[ok_p & ok_i]
        for ax in 'xy':
            tt = both[f'tan_{ax}'].to_numpy(float)
            fp, fi = both[f'{ax}_tan_theta_p'].to_numpy(float), both[f'{ax}_tan_theta_i'].to_numpy(float)
            tr = both.train.to_numpy(bool)
            out = dict(arm=arm, view=ax, corr_pass=float(np.corrcoef(fp, tt)[0, 1]),
                       corr_is2=float(np.corrcoef(fi, tt)[0, 1]))
            for lab, f in (('pass', fp), ('is2', fi)):
                w = tr & (np.abs(tt) > 0.15) & (np.abs(tt) < 0.45)
                out[f'ratio_{lab}'] = float(np.median(f[w] / tt[w]))
                for mode, ff in (('raw', f), ('scaled', f / out[f'ratio_{lab}'])):
                    for lo, hi in ((0.0, 0.08), (0.08, 0.6), (0.0, 0.6)):
                        m = ~tr & (np.abs(tt) >= lo) & (np.abs(tt) < hi)
                        r = np.abs(np.degrees(np.arctan(ff[m]) - np.arctan(tt[m])))
                        bs = [np.percentile(r[i], 68) for i in rng.integers(0, len(r), (300, len(r)))]
                        out[f's68_{lab}_{mode}_{lo}_{hi}'] = float(np.percentile(r, 68))
                        out[f's68err_{lab}_{mode}_{lo}_{hi}'] = float(np.std(bs))
                        out[f'n_{lo}_{hi}'] = int(m.sum())
            # paired difference of s68 on the scaled test events, bootstrap together
            m = ~tr & (np.abs(tt) < 0.6)
            rp = np.abs(np.degrees(np.arctan(fp[m] / out['ratio_pass']) - np.arctan(tt[m])))
            ri = np.abs(np.degrees(np.arctan(fi[m] / out['ratio_is2']) - np.arctan(tt[m])))
            idx = rng.integers(0, len(rp), (1000, len(rp)))
            d = np.percentile(rp[idx], 68, axis=1) - np.percentile(ri[idx], 68, axis=1)
            out['d_s68_scaled'] = float(np.percentile(rp, 68) - np.percentile(ri, 68))
            out['d_s68_scaled_err'] = float(np.std(d))
            bins.append(out)
    R, B = pd.DataFrame(rows), pd.DataFrame(bins)
    R.to_csv(f'{OUT}/cos_eff_truth.csv', index=False)
    B.to_csv(f'{OUT}/cos_res_truth.csv', index=False)
    print(R.round(4).to_string(index=False))
    print(B.T.round(3).to_string())
    return R, B


# ---------------------------------------------------------------------------------------------------------------- line
def line():
    import tempfile
    import ac_line
    with tempfile.TemporaryDirectory() as tmp:
        for arm in 'AC':
            parts = []
            for i, sub in enumerate(subruns(149)):
                for f in acq_files(149, sub, arm):
                    c = pd.read_parquet(f.replace('.parquet', '.candidates.parquet'))
                    c['event_id'] = c['event_id'] + 10_000_000 * int(sub[-4:])
                    parts.append(c)
            pd.concat(parts, ignore_index=True).to_parquet(f'{tmp}/{arm}.candidates.parquet')
        o, M = ac_line.score(f'{tmp}/A.parquet', f'{tmp}/C.parquet', f'{RUNS}/run_149/run_config.json',
                             label='pass1')
    out = {k: v for k, v in o.items() if k != 'bins'}
    for s, arm in (('a', 'A'), ('c', 'C')):
        d = M[[f'd_x_{s}', f'd_y_{s}', f'd_z_{s}']].to_numpy(float)
        pa, pc = M[['p0_x_a', 'p0_y_a', 'p0_z_a']].to_numpy(float), M[['p0_x_c', 'p0_y_c', 'p0_z_c']].to_numpy(float)
        L = (pc - pa) / np.linalg.norm(pc - pa, axis=1, keepdims=True)
        r = np.degrees(np.arccos(np.abs(np.einsum('ij,ij->i', d, L)).clip(0, 1)))
        r = r[np.isfinite(r)]
        out[f'{arm}_res68'] = float(np.percentile(r, 68))
        out[f'{arm}_res68_err'] = float(np.std([np.percentile(r[rng.integers(0, len(r), len(r))], 68)
                                                for _ in range(300)]))
    pd.DataFrame(o['bins']).to_csv(f'{OUT}/line_bins_run149.csv', index=False)
    json.dump(out, open(f'{OUT}/line_run149.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print(pd.DataFrame(o['bins']).round(2).to_string(index=False))
    return out


# ---------------------------------------------------------------------------------------------------------------- srcy
def srcy():
    import y_guard
    cols = ['event_id', 'arm', 'gated', 'angle_calibrated', 'tanx', 'tany', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y',
            'd_z', 't_since_flash_ns', 'x_t0', 'y_t0']
    out, D = {}, {}
    for chain, get in (('pass', pas), ('is2_v2', is2)):
        d = pd.concat([get(145, s, cols).assign(subrun=s) for s in subruns(145)], ignore_index=True)
        d = d[d.gated.astype(bool)]           # is2_v2's stored gated is the 0.8 superset; y_guard cuts 0.6
        D[chain] = d
        out[chain] = y_guard.late_y_crossing(d)
    # the same (arm, trigger)s: tracks of events both chains have a gated track in
    key = lambda d: list(zip(d.arm, d.subrun, d.event_id))
    out['common'] = {}
    for chain, other in (('pass', 'is2_v2'), ('is2_v2', 'pass')):
        ids = set(key(D[other]))
        d = D[chain]
        out['common'][chain] = y_guard.late_y_crossing(d[[x in ids for x in key(d)]])
    for arm in 'AC':
        p, i = out['pass'][arm], out['is2_v2'][arm]
        out[f'{arm}_vs_ref_sigma'] = (p['y_cross_mm'] - REF_Y[arm]) / p['err_mm']
        out[f'{arm}_pass_minus_is2'] = p['y_cross_mm'] - i['y_cross_mm']
        out[f'{arm}_common_pass_minus_is2'] = (out['common']['pass'][arm]['y_cross_mm']
                                               - out['common']['is2_v2'][arm]['y_cross_mm'])
    json.dump(out, open(f'{OUT}/srcy_run145.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    return out


# --------------------------------------------------------------------------------------------------------------- focus
def _focus_arrays(d, arm):
    U = {'A': np.array([1., 0., 0.]), 'C': np.array([-1., 0., 0.])}
    Wh = {'A': np.array([0., 0., 1.]), 'C': np.array([0., 0., -1.])}
    P0, D = d[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), d[['d_x', 'd_y', 'd_z']].to_numpy(float)
    pu, pw = P0 @ U[arm] - SGN_U[arm] * DX_SRC, P0 @ Wh[arm]
    with np.errstate(divide='ignore', invalid='ignore'):
        tx = (D @ U[arm]) / (D @ Wh[arm])
    ok = np.isfinite(pu) & np.isfinite(pw) & np.isfinite(tx)
    return pu[ok], pw[ok], tx[ok]


def focus():
    from ntof_cosmics.focus_split import best_s
    cols = ['event_id', 'arm', 'gated', 'tanx', 'tany', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
            't_since_flash_ns']
    rows = []
    for chain, get in (('pass', pas), ('is2_v2', is2)):
        d = pd.concat([get(145, s, cols).assign(subrun=s) for s in subruns(145)], ignore_index=True)
        d = acc(d)
        d = d[(d.t_since_flash_ns >= 20e6) & (d.t_since_flash_ns < 80e6)]
        for arm in 'AC':
            a = d[d.arm == arm]
            pu, pw, tx = _focus_arrays(a, arm)
            for lo, hi in ((0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.1, 0.5)):
                m = (np.abs(tx) >= lo) & (np.abs(tx) < hi)
                s, e, med, n = best_s(pu[m], pw[m], tx[m], nboot=60, rng=rng)
                rows.append(dict(chain=chain, arm=arm, tan_lo=lo, tan_hi=hi, s_best=s, s_err=e, med_miss=med, n=n))
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/focus_run145.csv', index=False)
    print(R.round(3).to_string(index=False))
    return R


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    steps = dict(build=build, eff=eff, wall=wall, cos=cos, line=line, srcy=srcy, focus=focus)
    for k in (steps if step == 'all' else [step]):
        print(f'== {k}', flush=True)
        steps[k]()
