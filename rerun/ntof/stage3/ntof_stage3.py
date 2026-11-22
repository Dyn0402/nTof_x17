#!/usr/bin/env python3
"""ntof_stage3.py -- stage 3 (the track tables) of n_TOF pass 1, from the ledger's canonical stage-2 outputs.

One table per (run, sub-run), tracks_run_<r>_<sub>.parquet + .meta.json, built by the same code as every earlier
stage 3 (sept26_prelim_analysis.build_tracks.build: every X/Y pairing a row, `gated` a column, y up, the y guard),
with the n_TOF pass's rules (NTOF_PASS_PLAN §3 G2):

  * angles are true tans from the bundle's v: k = 1 for A, C, D (the frozen bundles carry the in-situ v; k is no longer
    a correction, only a check).  B has no angle (no field-shaping ring): k unset, so its angle columns are null and
    its positions are kept.
  * the drift-velocity correction v(t) (P7a, decided 11-17: no rerun) goes in BESIDE the frozen-v columns, never over
    them: `v_t`, `tanx_vt`, `tany_vt`, every angle-derived column as `<col>_vt`, `in_acceptance_vt` (|tan_vt| < 0.5
    true) and `gated_vt` (vt_join.py; vt_smooth.parquet from EOS Analysis_pass1/vt_table/, --vt).
  * acceptance |tan| < 0.6 true (decided 10-08; the reco cut is 0.6 raw = 0.6 true at k = 1).
  * the canonical output per (row, arm) is ntof_ledger.py's (ledger_canon.json: D run_104 from side/run104_D_L1, and
    any re-run set).  Stage 3 never picks a file the ledger would not.
  * context: stage-1 class and n_TOF flags (old stage 1, keyed on event_id per sub-run) and the trigger's time since the
    flash (trigger_time, t_pred_ns) where those products exist.  Beam-off cosmic sub-runs have neither (no flash): the
    columns stay null, and the sidecar says so.

    ntof_stage3.py --stage2 $WORK_DATA/ntof_pass1/stage2 [--side $WORK_DATA/ntof_pass1/run104] [--rerun-dbeam DIR] --out DIR
                   [--vt $WORK_DATA/ntof_pass1/vt/vt_smooth.parquet] [--runs 145,149] [--force]
                   [--subruns run_116/stat090_0027,... | FILE] [--stage1-dir DIR] [--trigtime-dir DIR]

`--subruns` (a comma list or a file of `run_X/sub` or `run_X sub` lines), `--stage1-dir` (default the old stage 1,
/media/dylan/data/x17/sept26_prelim/stage1) and `--trigtime-dir` (sets $X17_TRIGTIME_DIR for trigger_time; default
<X17_SEPT26_OUT>/trigtime) are for rebuilding the sub-runs whose context came from one slim of several (tracking audit
2026-11-22, finding 12).

`--stage2` / `--side` are local copies of EOS Analysis_pass1/stage2 and the canon's other directories (same layout).
A per-sub-run view directory of symlinks (mx17_<arm>/events_<tag>.*) feeds build_tracks unchanged.
"""
import argparse, csv, hashlib, json, os, sys, tempfile, traceback

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(NT, '..', '..'))
sys.path[:0] = [HERE, NT, REPO]

import ntof_ledger as L   # noqa: E402
import vt_join as VT        # noqa: E402

K_ARM = {'A': 1.0, 'C': 1.0, 'D': 1.0}
TAN_MAX_TRUE = 0.6
OLD = Path('/media/dylan/data/x17/sept26_prelim')


def canonical(r, arm, canon, roots):
    """(dir, file stem) of the canonical output of registry row r, arm."""
    u = L.rule_for(canon, r, arm)
    if u and u['dir'].split('/')[0] not in roots:
        raise KeyError(f"canon dir {u['dir']!r} has no local root (add it to roots in main)")
    base = roots[u['dir'].split('/')[0]] if u else roots['stage2']
    sub = '/'.join(u['dir'].split('/')[1:]) if u else ''
    d = Path(base) / sub / f"run_{r['run']}" / r['subrun'] / arm
    return d, f"run_{r['run']}_{r['subrun']}_{r['tag']}_{arm}"


#: arms whose candidate tangents are recomputed from (w, p0) with the bundle's bow block, each pair with its own
#: partner (tracking audit 2026-11-22, finding 21: the reco re-mapped the two plane winners together after the pairs).
#: B has no angle.  BUNDLES: local copy of Analysis_pass1/bundles (pf/<name>).
BOW_ARMS = ('A', 'C', 'D')
BUNDLES = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1' / 'bundles' / 'pf'


def bundle_for(r, arm, canon):
    u = L.rule_for(canon, r, arm)
    return (u or {}).get('bundle') or L.expected_bundle(r, arm)


def bow_patch(src: Path, meta: Path, bundle_dir: Path, dst: Path) -> dict:
    """Write `src` candidates to `dst` with every paired row's tan_theta / theta_deg recomputed from its own pair
    (wft.reco.bow_pairs), plus `bow_a` (the bow tilt term, v-independent) and `bow_h` (mesh height).  Rows with
    track_id < 0 keep the reco's values (bow_a, bow_h null).  Refuses a bundle whose v differs from the one the reco
    used, or one that reproduces fewer than half of the stored tangents (a wrong bundle reproduces none)."""
    import numpy as np
    from wft.calib import CalibrationBundle
    from wft import reco as R
    cal = CalibrationBundle.load(str(bundle_dir))
    mv = json.loads(meta.read_text()).get('bundle', {}).get('v_drift')
    if mv is not None and abs(float(mv) - float(cal.v_drift)) > 1e-6:
        raise ValueError(f'{src.name}: bundle {bundle_dir.name} v {cal.v_drift} != reco v {mv}')
    c = pd.read_parquet(src)
    c['bow_a'], c['bow_h'] = np.nan, np.nan
    t = c[c['track_id'] >= 0]
    key = ['event_id', 'track_id']
    x = t[t['plane'] == 'x'].reset_index().set_index(key)
    y = t[t['plane'] == 'y'].reset_index().set_index(key)
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    v = R.bow_pairs(cal, x['w'], x['p0'], y['w'], y['p0'])
    old = np.r_[x['tan_theta'].to_numpy(float), y['tan_theta'].to_numpy(float)]
    new = np.r_[v['tan_x'], v['tan_y']]
    ok = np.isfinite(old) & np.isfinite(new)
    same = np.abs(old - new)[ok] < 1e-9
    if ok.sum() > 100 and same.mean() < 0.5:
        raise ValueError(f'{src.name}: bundle {bundle_dir.name} reproduces only {same.mean():.1%} of the tangents')
    for p, idx in (('x', x['index'].to_numpy()), ('y', y['index'].to_numpy())):
        c.loc[idx, 'tan_theta'] = v[f'tan_{p}']
        c.loc[idx, 'theta_deg'] = np.degrees(np.arctan(v[f'tan_{p}']))
        c.loc[idx, 'bow_a'] = v[f'a_{p}']
        c.loc[idx, 'bow_h'] = v[f'h_{p}']
    c.to_parquet(dst, index=False)
    return dict(bundle=bundle_dir.name, n_pairs=int(len(common)), frac_changed=float(1 - same.mean()) if ok.any() else 0.0,
                max_abs_dtan=float(np.abs(old - new)[ok].max()) if ok.any() else 0.0)


def one(run, sub, rows, canon, roots, out, force, vt_path, vt_md5, stage1_dir=None):
    from sept26_prelim_analysis import build_tracks as BT
    dst = Path(out) / f'tracks_run_{run}_{sub}.parquet'
    if dst.exists() and not force:
        return run, sub, 'exists', None
    try:
        with tempfile.TemporaryDirectory(dir=out, prefix='.view_') as view:
            missing, bow_log = [], []
            for r in rows:
                for arm in L.ARMS:
                    d, stem = canonical(r, arm, canon, roots)
                    vd = Path(view) / f'mx17_{arm}'
                    vd.mkdir(exist_ok=True)
                    for ext in ('.candidates.parquet', '.meta.json'):
                        src = d / f'{stem}{ext}'
                        if not src.exists():
                            missing.append(str(src))
                            continue
                        dst_c = vd / f"events_{r['tag']}{ext}"
                        if ext == '.candidates.parquet' and arm in BOW_ARMS:
                            bow_log.append(dict(tag=r['tag'], arm=arm, **bow_patch(
                                src, d / f'{stem}.meta.json', BUNDLES / bundle_for(r, arm, canon), dst_c)))
                        else:
                            os.symlink(src, dst_c)
            if missing:
                raise FileNotFoundError(f'{len(missing)} canonical files missing, first {missing[0]}')
            s1 = Path(stage1_dir or OLD / 'stage1') / f'candidates_run_{run}_{sub}.parquet'
            tracks, meta = BT.build(f'run_{run}', sub, Path(view), stage1=s1 if s1.exists() else None, allow=None,
                                    out_dir=None, k_arm=K_ARM, write=False, tan_max_true=TAN_MAX_TRUE)
        tracks['tan_source'] = 'bundle_v'            # k = 1: tanx/tany are true tans at the frozen bundle v
        # B has no angle by design (positions only), not for want of a calibration: build_tracks puts an angle-less arm
        # outside the true-tan acceptance (gated all False), which for B would throw away every position.  B keeps
        # the reco's own gate; in_acceptance is null (not applicable) and tan_source says why.
        b = tracks['arm'] == 'B'
        tracks['in_acceptance'] = tracks['in_acceptance'].astype('boolean')
        tracks.loc[b, 'in_acceptance'] = pd.NA
        tracks.loc[b, 'gated'] = tracks.loc[b, 'gated_reco']
        tracks.loc[b, 'tan_source'] = 'none (B: positions only)'
        meta['n_gated'] = int(tracks['gated'].sum())
        meta['bow_partner_fix'] = dict(rule='candidate tangents of A, C, D recomputed from (w, p0) with each pair\'s own '
                                            'partner (wft.reco.bow_pairs; tracking audit 2026-11-22, finding 21)',
                                       per_tag=bow_log)
        # v(t): beside the frozen-v columns (vt_join.py)
        trig = {VT.TRIGGER_OF_RCLASS[r['rclass']] for r in rows}
        if len(trig) != 1:
            raise ValueError(f'sub-run rows disagree on trigger class {trig}')
        from ntof_tracking.reco import geometry as G
        from sept26_prelim_analysis import paths
        cfg = json.loads((Path(str(paths.root('runs'))) / f'run_{run}' / 'run_config.json').read_text())
        tracks, vinfo = VT.attach_vt(tracks, VT.load_vt(vt_path), trig.pop(), G.detector_transforms(cfg))
        meta['vt'] = VT.meta_block(vinfo, vt_path, vt_md5)
        meta['n_gated_vt'] = vinfo['n_gated_vt']
        if 'B' in meta.get('provenance', {}):
            meta['provenance']['B']['acceptance']['n_gated'] = int(tracks.loc[b, 'gated'].sum())
            meta['provenance']['B']['acceptance']['b_rule'] = 'gated = gated_reco (no angle, no acceptance cut)'
        meta.update(pass_name='n_TOF pass 1', code_stage2=sorted({c for c in [canon['code'][0]]}),
                    canon=canon, k_rule='k = 1 on A, C, D: tan is true at the bundle v (NTOF_PASS_PLAN G2); B no angle',
                    stage1_context=str(s1) if s1.exists() else None,
                    context_note=None if s1.exists() else 'no stage-1 / trigger-time product for this sub-run '
                                                          '(beam-off cosmic: no flash); context columns null',
                    rows=[r['tag'] for r in rows], reco_dir='(view of canonical files; see canon)')
        tracks.to_parquet(dst, index=False)
        Path(str(dst).replace('.parquet', '.meta.json')).write_text(json.dumps(meta, indent=1, default=str))
        return run, sub, f"{meta['n_tracks']} tracks, {meta['n_gated']} gated", None
    except Exception:
        return run, sub, 'FAILED', traceback.format_exc(limit=4)


def parse_subruns(arg):
    """{(run_X, sub)} from a comma list or a file of 'run_X/sub' or 'run_X sub' lines."""
    text = Path(arg).read_text() if os.path.isfile(arg) else arg.replace(',', '\n')
    out = set()
    for line in text.splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        f = line.replace('/', ' ').split()
        if len(f) != 2 or not f[0].startswith('run_'):
            raise SystemExit(f'--subruns: expected run_X/sub, got {line!r}')
        out.add((f[0], f[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    wd = os.environ.get('WORK_DATA', '/media/ucla/work')
    ap.add_argument('--stage2', default=f'{wd}/ntof_pass1/stage2')
    ap.add_argument('--side', default=f'{wd}/ntof_pass1/side', help='local copy of Analysis_pass1/side')
    ap.add_argument('--tests', default=f'{wd}/ntof_pass1/tests', help='local copy of Analysis_pass1/tests')
    ap.add_argument('--rerun-dbeam', default=f'{wd}/ntof_pass1/rerun_dbeam_L1',
                    help='local copy of Analysis_pass1/rerun_dbeam_L1 (D L2-era beam on L1, 11-17)')
    ap.add_argument('--vt', default=f'{wd}/ntof_pass1/vt/vt_smooth.parquet',
                    help='local copy of EOS Analysis_pass1/vt_table/vt_smooth.parquet (check its md5 against EOS)')
    ap.add_argument('--out', default=f'{wd}/ntof_pass1/stage3')
    ap.add_argument('--canon', default=os.path.join(NT, 'ledger_canon.json'))
    ap.add_argument('--runs', default=None)
    ap.add_argument('--subruns', default=None,
                    help='only these sub-runs: comma list or file of run_X/sub (or "run_X sub") lines')
    ap.add_argument('--stage1-dir', default=str(OLD / 'stage1'), help='stage-1 candidates (default: the old stage 1)')
    ap.add_argument('--trigtime-dir', default=None,
                    help='trigger-time products (sets $X17_TRIGTIME_DIR; default <X17_SEPT26_OUT>/trigtime)')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    canon = L.load_canon(a.canon)
    roots = {'stage2': a.stage2, 'side': a.side, 'tests': a.tests, 'rerun_dbeam_L1': a.rerun_dbeam}
    reg = [r for r in csv.DictReader(open(os.path.join(NT, 'ntof_registry.csv'))) if r['excluded'] != 'True']
    if a.runs:
        reg = [r for r in reg if r['run'] in a.runs.split(',')]
    if a.subruns:
        want = parse_subruns(a.subruns)
        known = {(f"run_{r['run']}", r['subrun']) for r in reg}
        if want - known:
            raise SystemExit(f'--subruns: {len(want - known)} not in the (filtered) registry, e.g. {sorted(want - known)[0]}')
        reg = [r for r in reg if (f"run_{r['run']}", r['subrun']) in want]
    if a.trigtime_dir:
        # inherited by the worker processes; trigger_time.product reads it
        os.environ['X17_TRIGTIME_DIR'] = os.path.abspath(a.trigtime_dir)
    print(f"stage 1: {a.stage1_dir}  trigtime: {os.environ.get('X17_TRIGTIME_DIR', '<X17_SEPT26_OUT>/trigtime')}",
          flush=True)
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    os.makedirs(a.out, exist_ok=True)
    vt_md5 = hashlib.md5(open(a.vt, 'rb').read()).hexdigest()
    print(f'v(t): {a.vt} md5 {vt_md5}', flush=True)
    bad = []
    with ProcessPoolExecutor(a.jobs) as ex:
        fs = [ex.submit(one, run, sub, rows, canon, roots, a.out, a.force, a.vt, vt_md5, a.stage1_dir)
              for (run, sub), rows in sorted(groups.items())]
        for i, f in enumerate(as_completed(fs)):
            run, sub, msg, err = f.result()
            print(f'[{i + 1}/{len(fs)}] run_{run} {sub}: {msg}', flush=True)
            if err:
                bad.append((run, sub, err))
                print(err, flush=True)
    print(f'{len(groups) - len(bad)} / {len(groups)} sub-runs built, {len(bad)} failed')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
