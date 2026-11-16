#!/usr/bin/env python3
"""ntof_stage3.py -- stage 3 (the track tables) of n_TOF pass 1, from the ledger's canonical stage-2 outputs.

One table per (run, sub-run), tracks_run_<r>_<sub>.parquet + .meta.json, built by the same code as every earlier
stage 3 (sept26_prelim_analysis.build_tracks.build: every X/Y pairing a row, `gated` a column, y up, the y guard),
with the n_TOF pass's rules (NTOF_PASS_PLAN §3 G2):

  * angles are true tans from the bundle's v: k = 1 for A, C, D (the frozen bundles carry the in-situ v; k is no longer
    a correction, only a check).  B has no angle (no field-shaping ring): k unset, so its angle columns are null and
    its positions are kept.  C's v(t) correction (P7a) goes in as `tan_vt` beside `tan`, not here.
  * acceptance |tan| < 0.6 true (decided 10-08; the reco cut is 0.6 raw = 0.6 true at k = 1).
  * the canonical output per (row, arm) is ntof_ledger.py's (ledger_canon.json: D run_104 from side/run104_D_L1, and
    any re-run set).  Stage 3 never picks a file the ledger would not.
  * context: stage-1 class and n_TOF flags (old stage 1, keyed on event_id per sub-run) and the trigger's time since the
    flash (trigger_time, t_pred_ns) where those products exist.  Beam-off cosmic sub-runs have neither (no flash): the
    columns stay null, and the sidecar says so.

    ntof_stage3.py --stage2 $WORK_DATA/ntof_pass1/stage2 [--side $WORK_DATA/ntof_pass1/side] --out DIR [--jobs 8]
                   [--runs 145,149] [--force]

`--stage2` / `--side` are local copies of EOS Analysis_pass1/stage2 and the canon's other directories (same layout).
A per-sub-run view directory of symlinks (mx17_<arm>/events_<tag>.*) feeds build_tracks unchanged.
"""
import argparse, csv, json, os, sys, tempfile, traceback

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(NT, '..', '..'))
sys.path[:0] = [NT, REPO]

import ntof_ledger as L   # noqa: E402

K_ARM = {'A': 1.0, 'C': 1.0, 'D': 1.0}
TAN_MAX_TRUE = 0.6
OLD = Path('/media/dylan/data/x17/sept26_prelim')


def canonical(r, arm, canon, roots):
    """(dir, file stem) of the canonical output of registry row r, arm."""
    u = L.rule_for(canon, r, arm)
    base = roots.get(u['dir'].split('/')[0], roots['stage2']) if u else roots['stage2']
    sub = '/'.join(u['dir'].split('/')[1:]) if u else ''
    d = Path(base) / sub / f"run_{r['run']}" / r['subrun'] / arm
    return d, f"run_{r['run']}_{r['subrun']}_{r['tag']}_{arm}"


def one(run, sub, rows, canon, roots, out, force):
    from sept26_prelim_analysis import build_tracks as BT
    dst = Path(out) / f'tracks_run_{run}_{sub}.parquet'
    if dst.exists() and not force:
        return run, sub, 'exists', None
    try:
        with tempfile.TemporaryDirectory(dir=out, prefix='.view_') as view:
            missing = []
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
                        os.symlink(src, vd / f"events_{r['tag']}{ext}")
            if missing:
                raise FileNotFoundError(f'{len(missing)} canonical files missing, first {missing[0]}')
            s1 = OLD / 'stage1' / f'candidates_run_{run}_{sub}.parquet'
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    wd = os.environ.get('WORK_DATA', '/media/ucla/work')
    ap.add_argument('--stage2', default=f'{wd}/ntof_pass1/stage2')
    ap.add_argument('--side', default=f'{wd}/ntof_pass1/side', help='local copy of Analysis_pass1/side')
    ap.add_argument('--tests', default=f'{wd}/ntof_pass1/tests', help='local copy of Analysis_pass1/tests')
    ap.add_argument('--out', default=f'{wd}/ntof_pass1/stage3')
    ap.add_argument('--canon', default=os.path.join(NT, 'ledger_canon.json'))
    ap.add_argument('--runs', default=None)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    canon = L.load_canon(a.canon)
    roots = {'stage2': a.stage2, 'side': a.side, 'tests': a.tests}
    reg = [r for r in csv.DictReader(open(os.path.join(NT, 'ntof_registry.csv'))) if r['excluded'] != 'True']
    if a.runs:
        reg = [r for r in reg if r['run'] in a.runs.split(',')]
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    os.makedirs(a.out, exist_ok=True)
    bad = []
    with ProcessPoolExecutor(a.jobs) as ex:
        fs = [ex.submit(one, run, sub, rows, canon, roots, a.out, a.force) for (run, sub), rows in sorted(groups.items())]
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
