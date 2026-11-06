#!/usr/bin/env python3
"""pf_score.py -- the pre-freeze checks on the candidate pass bundles (plan §6 11-05; condor 4618426-29, code ed0b1162).

    pf_score.py pull              # EOS tests/prefreeze_<code>/{lat,cospin,samples,ng1} -> $WORK_DATA/ntof_pass1/prefreeze
    pf_score.py lat               # 1. latency per cosmic run over the campaign (free t0; A, C, D)
    pf_score.py line              # 2. A-C cosmic line per run: angle scale (fitted/true) and residuals, pinned + free
    pf_score.py beam              # 3. production samples (run_79/81 E1, 116, 162) + run_145: yield and capsule
                                  #    pointing per arm, against is2_v2 on the same acquisitions and events (A, C)
    pf_score.py all

Sets (make_ntof_jobs.py --per-row; bundles make_pf_bundles.sh):
  lat      cosmic runs 80 (E1), 89, 103, 119, 133, 149, 159; 2 sub-runs each; A C D; t0 FREE
  cospin   the same runs minus 149; A B C D; pinned (cosmic latency)
  samples  runs 79, 81 (E1), 116, 162; first sub-run, every tag; A B C D; pinned (beam latency; D its cosmic one)
  ng1      run_145 (all 24 tags) beam + run_149 cos_0000-0011 cosmic; A B C D; pinned per trigger class
  twosided run_149 cos_0000-0011, A C, as ng1 with WFT_DIAG=reco.TWO_SIDED=1 (mirror-basin check R6; condor 4618430)
  resub    the 37 jobs of 4618427-29 lost to EOS-FUSE venv I/O errors, re-run under lockpy (4620473; s6_resub.txt)
  samples_dL1  runs 79, 81 first sub-run, D only, pinned on the L1 latency (D_<era>_beam_L1) vs samples' L2 pin (4620578)
  lat fill cosmic runs 105-125, 2 sub-runs, A C D, t0 free: when did D's latency step (4620474; s7_latfill.txt)
"""
import glob
import json
import os
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..', '..'))
REPO = os.path.abspath(os.path.join(NT, '..', '..'))
sys.path[:0] = [NT, REPO]
CODE = 'ed0b1162'
EOS = f'/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/prefreeze_{CODE}'
EOSR = '/eos/experiment/ntof/data/x17/july_beam/runs'
IS2 = '/eos/experiment/ntof/data/x17/analysis/sept26_fullpass_is2_v2'
WD = os.environ.get('WORK_DATA') or ('/media/ucla/work' if os.path.isdir('/media/ucla/work')
                                     else os.path.expanduser('~/work_data'))   # laptop / lxplus: no ucla
W = os.path.join(WD, 'ntof_pass1', 'prefreeze')
OUT = os.path.join(W, 'results')
os.makedirs(OUT, exist_ok=True)
REG = pd.read_csv(os.path.join(NT, 'ntof_registry.csv'))
rng = np.random.default_rng(11)


def sh(cmd):
    print('+', cmd[:200]); subprocess.run(cmd, shell=True, check=True)


def pull():
    os.makedirs(W, exist_ok=True)
    for s in ('lat', 'cospin', 'samples', 'ng1', 'twosided', 'samples_dL1', 'bundles'):
        sh(f"ssh lxplus 'cd {EOS} && tar -cf - {s}' | tar -x -C {W}")
    runs = sorted({int(p.split('run_')[1].split('/')[0]) for p in glob.glob(f'{W}/*/run_*/')} | {145, 149})
    os.makedirs(f'{W}/rc', exist_ok=True)
    for r in runs:
        if not os.path.exists(f'{W}/rc/run_{r}.json'):
            sh(f'scp -q lxplus:{EOSR}/run_{r}/run_config.json {W}/rc/run_{r}.json')
    # is2_v2 stage-2 outputs of the same acquisitions (A, C only), candidates + events tables
    os.makedirs(f'{W}/is2', exist_ok=True)
    want = set()
    for f in glob.glob(f'{W}/samples/run_*/*/[AC]/*.meta.json') + glob.glob(f'{W}/ng1/run_145/*/[AC]/*.meta.json'):
        b = os.path.basename(f)[:-len('.meta.json')]          # run_<r>_<sub>_<tag>_<arm>
        p = b.split('_')
        run, sub, tag, arm = p[1], f'{p[2]}_{p[3]}', '_'.join(p[4:7]), p[7]
        want.add(f'run_{run}_{sub}_beam_{arm}_{tag}')
    have = {os.path.basename(x) for x in glob.glob(f'{W}/is2/*')}
    need = sorted(w for w in want if w not in have)
    if need:
        lst = ' '.join(f'{w}.tar.gz' for w in need)
        sh(f"ssh lxplus 'cd {IS2} && for f in {lst}; do [ -f $f ] && echo $f; done' > {W}/is2_list.txt")
        for w in open(f'{W}/is2_list.txt').read().split():
            d = f'{W}/is2/{w[:-7]}'
            os.makedirs(d, exist_ok=True)
            sh(f"ssh lxplus 'tar -xzf {IS2}/{w} -O --wildcards \"*events_*.candidates.parquet\"' > {d}/x.candidates.parquet")
            sh(f"ssh lxplus 'tar -xzf {IS2}/{w} -O --wildcards \"*events_*[0-9].parquet\"' > {d}/x.parquet")


def r149_latency(arm):
    """{view: {'C': ns}} of the run_149 cosmic pin: latency_<arm>_r149.json where it exists (desktop), else the
    pinned cos bundle pulled from EOS, whose t0_abs at ftst 0 is that C (make_pf_bundles.sh writes C - 10 ftst)."""
    f = os.path.join(WD, 'ntof_pass1', 'bundles', f'latency_{arm}_r149.json')
    if os.path.exists(f):
        return json.load(open(f))
    b = json.load(open(f'{W}/bundles/{arm}_E2_cos/bundle.json'))
    return {v: {'C': b['t0_abs'][v]['0']} for v in 'xy'}


def files(s, run, arm, subs=None):
    fs = sorted(glob.glob(f'{W}/{s}/run_{run}/*/{arm}/*_{arm}.parquet'))
    return [f for f in fs if subs is None or f.split('/')[-3] in subs]


def load(fs, cand=False):
    """concatenate per-acquisition tables; event ids restart per sub-run, so offset by the sub-run index"""
    parts = []
    for f in fs:
        d = pd.read_parquet(f.replace('.parquet', '.candidates.parquet') if cand else f)
        sub = f.split('/')[-3]
        d['event_id'] = d['event_id'] + 10_000_000 * int(sub[-4:])
        d['subrun'] = sub
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


def _imed(v):
    c = np.median(v)
    for _ in range(5):
        w = v[np.abs(v - c) < 200]
        c = np.median(w)
    return c, w


def latency(d):
    """Iterated median of t0 + 10 ftst within +-200 ns, on t0 dithered by U(-5, 5) ns.

    Why the dither (2026-11-06): 20-65 % of fitted t0 sit exactly on the absolute 10 ns lattice (D y worst).  The
    template is tabulated on a 10 ns grid and evaluated piecewise-linearly, so chi2(t0) has a common kink wherever
    t0 = 0 mod 10 (every prompt column hits a template node at once; samples and depth bins are 60 ns), and t0 at fixed
    geometry is flat enough (dchi2 ~0.02 over 5 ns) for the kink to hold it.  The plain median then snaps to the lattice
    atoms (latencies of exactly 10.0, 20.0, ...).  Geometry is unaffected (a smooth template moves tan by ~2-5e-4), so
    the fix is in the estimator: the dither (2.9 ns rms on an ~80 ns-wide distribution) breaks the atoms.  C_raw is the
    old undithered median, kept for comparison; tmean the trimmed mean of the same window."""
    out = {}
    for p in 'xy':
        s = d[(d[f'{p}_ok'] == True) & (d[f'{p}_quality_ok'] == True) & (d['n_tracks'] == 1) & (d[f'{p}_ftst'] >= 0)]
        t0 = s[f'{p}_t0'].to_numpy(float)
        v = t0 + 10 * s[f'{p}_ftst'].to_numpy(float)
        ok = np.isfinite(v)
        v, t0 = v[ok], t0[ok]
        if len(v) < 30:
            out[p] = dict(C=np.nan, err=np.nan, n=int(len(v)))
            continue
        lat10 = float(np.mean(np.abs((t0 + 5) % 10 - 5) < 1e-4))
        c_raw, _ = _imed(v)
        vd = v + rng.uniform(-5, 5, len(v))
        c, w = _imed(vd)
        bs = np.std([np.median(w[rng.integers(0, len(w), len(w))]) for _ in range(400)])
        out[p] = dict(C=float(c), err=float(bs), n=int(len(w)), C_raw=float(c_raw), tmean=float(w.mean()),
                      tmean_err=float(w.std() / np.sqrt(len(w))), on_lattice=lat10)
    return out


def t_of(run):
    return str(REG[REG.run == run].t_local.min())


def lat():
    rows = []
    # 80, 89, 103, 119, 133, 149, 159 (4618426) + the fill-in 105-125 that dates D's latency step (4620474)
    for run in sorted({int(p.rstrip('/').split('run_')[-1]) for p in glob.glob(f'{W}/lat/run_*/')}):
        for arm in 'ACD':
            fs = files('lat', run, arm)
            if not fs:
                continue
            L = latency(load(fs))
            for p in 'xy':
                rows.append(dict(run=run, t=t_of(run), era=REG[REG.run == run].mask_era.iloc[0][:2], arm=arm, view=p,
                                 n_acq=len(fs), **L[p]))
    R = pd.DataFrame(rows)
    ref = {(r.arm, r.view): r.C for r in R[R.run == 149].itertuples()}
    R['d_vs_149'] = [r.C - ref.get((r.arm, r.view), np.nan) for r in R.itertuples()]
    bund = {a: r149_latency(a) for a in 'ACD'}
    R['d_vs_bundle'] = [r.C - bund[r.arm][r.view]['C'] for r in R.itertuples()]
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/latency_by_run.csv', index=False)
    print(R.round(1).to_string(index=False))
    return R


def line_one(s, run, subs=None):
    import ac_line
    A, C = files(s, run, 'A', subs), files(s, run, 'C', subs)
    if not A or not C:
        return None, None
    with tempfile.TemporaryDirectory() as tmp:
        for arm, fs in (('A', A), ('C', C)):
            load(fs, cand=True).to_parquet(f'{tmp}/{arm}.candidates.parquet')
        o, M = ac_line.score(f'{tmp}/A.parquet', f'{tmp}/C.parquet', f'{W}/rc/run_{run}.json')
    return o, M


def res68_all(M, s):
    d = M[[f'd_x_{s}', f'd_y_{s}', f'd_z_{s}']].to_numpy(float)
    pa, pc = M[['p0_x_a', 'p0_y_a', 'p0_z_a']].to_numpy(float), M[['p0_x_c', 'p0_y_c', 'p0_z_c']].to_numpy(float)
    L = (pc - pa) / np.linalg.norm(pc - pa, axis=1, keepdims=True)
    r = np.degrees(np.arccos(np.abs(np.einsum('ij,ij->i', d, L)).clip(0, 1)))
    r = r[np.isfinite(r)]
    bs = np.std([np.percentile(r[rng.integers(0, len(r), len(r))], 68) for _ in range(400)]) if len(r) > 5 else np.nan
    return float(np.percentile(r, 68)) if len(r) else np.nan, float(bs)


def line():
    rows = []
    jobs = [('cospin', r, None) for r in (80, 89, 103, 119, 133, 159)] + \
           [('lat', r, None) for r in (80, 89, 103, 119, 133, 149, 159)] + [('ng1', 149, None), ('twosided', 149, None)]
    for s, run, subs in jobs:
        o, M = line_one(s, run, subs)
        if o is None:
            continue
        row = dict(set=s, run=run, t=t_of(run), era=REG[REG.run == run].mask_era.iloc[0][:2], n_line=o['n_ac_line'])
        for arm, sx in (('A', 'a'), ('C', 'c')):
            row[f'{arm}_res68'], row[f'{arm}_res68_err'] = res68_all(M, sx)
            for v in 'xy':
                row[f'{arm}{v}_ratio'] = o[f'{arm}{v}']['ratio']
                row[f'{arm}{v}_ratio_err'] = o[f'{arm}{v}']['ratio_err']
                row[f'{arm}{v}_n'] = o[f'{arm}{v}']['n_kw']
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        if run == 149 and s in ('ng1', 'twosided'):        # mirror check (R6): steep-line bins, one- vs two-sided
            pd.DataFrame(o['bins']).assign(set=s).to_csv(f'{OUT}/line_bins_149_{s}.csv', index=False)
            M.to_parquet(f'{OUT}/lines_149_{s}.parquet')
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/line_by_run.csv', index=False)
    return R


def arm_summary(path_glob_or_df, arm, run, label, events=None):
    """yield + capsule pointing for one (run, arm) table; events restricts to the same trigger ids"""
    import capsule_point
    with tempfile.TemporaryDirectory() as tmp:
        c = path_glob_or_df
        if events is not None:
            c = c[c.event_id.isin(events)]
        c.to_parquet(f'{tmp}/t.candidates.parquet')
        s, T = capsule_point.score(f'{tmp}/t.parquet', arm, f'{W}/rc/run_{run}.json', label)
    return s, T


def is2_cands(run, sub, tags, arm):
    parts = []
    for tag in tags:
        f = f'{W}/is2/run_{run}_{sub}_beam_{arm}_{tag}/x.candidates.parquet'
        if os.path.exists(f) and os.path.getsize(f) > 0:
            d = pd.read_parquet(f)
            d['event_id'] = d['event_id'] + 10_000_000 * int(sub[-4:])
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


def beam():
    rows = []
    for s, run in (('samples', 79), ('samples', 81), ('samples', 116), ('samples', 162), ('ng1', 145),
                   ('samples_dL1', 79), ('samples_dL1', 81)):
        for arm in 'ABCD':
            fs = files(s, run, arm)
            if not fs:
                continue
            ev = load(fs)
            cand = load(fs, cand=True)
            n_trig = int(len(ev))
            sN, TN = arm_summary(cand, arm, run, 'pass')
            row = dict(set=s, run=run, era=REG[REG.run == run].mask_era.iloc[0][:2], arm=arm, n_acq=len(fs),
                       n_trig=n_trig, x_ok=float((ev.x_ok == True).mean()), y_ok=float((ev.y_ok == True).mean()),
                       gated_per_trig=sN['n_tracks'] / max(n_trig, 1), n_point_10=sN['n_point_10'],
                       n_point_15=sN['n_point_15'], frac_point_15=sN['frac_point_15'])
            if arm in 'AC':
                subs = sorted({f.split('/')[-3] for f in fs})
                parts = []
                for sub in subs:
                    tags = sorted({'_'.join(os.path.basename(f).split('_')[4:7]) for f in fs if f.split('/')[-3] == sub})
                    c = is2_cands(run, sub, tags, arm)
                    if c is not None:
                        parts.append(c)
                if parts:
                    ci = pd.concat(parts, ignore_index=True)
                    common = set(ci.event_id) & set(ev.event_id)
                    sI, TI = arm_summary(ci, arm, run, 'is2_v2', events=common)
                    sN2, _ = arm_summary(cand, arm, run, 'pass', events=common)
                    row.update(n_common_trig=len(common), is2_gated_per_trig=sI['n_tracks'] / max(len(common), 1),
                               pass_gated_per_trig_common=sN2['n_tracks'] / max(len(common), 1),
                               is2_n_point_15=sI['n_point_15'], pass_n_point_15_common=sN2['n_point_15'],
                               is2_frac_point_15=sI['frac_point_15'], pass_frac_point_15_common=sN2['frac_point_15'],
                               is2_n_point_10=sI['n_point_10'], pass_n_point_10_common=sN2['n_point_10'])
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(f'{OUT}/beam_by_run.csv', index=False)
    return R


if __name__ == '__main__':
    step = sys.argv[1]
    if step == 'all':
        for f in (lat, line, beam):
            f()
    else:
        {'pull': pull, 'lat': lat, 'line': line, 'beam': beam}[step]()
