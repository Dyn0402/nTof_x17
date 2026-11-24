#!/usr/bin/env python3
"""ntof_stage3_partial.py -- rerun/ntof/stage3/ntof_stage3.py for a phase-4 test output that covers only some
(tag, arm) pairs of each sub-run.

The repo's stage 3 needs the canonical candidates of EVERY registry row and EVERY arm (A, B, C, D) of a sub-run and
raises FileNotFoundError otherwise.  A phase-4 test never runs chamber B, and a sizing cluster covers only some tags,
so the unmodified script builds nothing.  This wrapper imports the repo module unchanged and replaces only its `one`
with a copy that

  * keeps only the registry rows (tags) that have at least one candidates table under --stage2, and
  * skips an arm whose canonical candidates table is missing (build_tracks already skips an arm without files),

and records the (tag, arm) pairs it used in the sidecar (`partial_rows`).  All other behaviour (bow patch, v(t),
context, gate) is the repo's.  Arguments are those of ntof_stage3.py.
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO = '/home/dylan/PycharmProjects/nTof_x17'
sys.path[:0] = [f'{REPO}/rerun/ntof/stage3', f'{REPO}/rerun/ntof', REPO]
import ntof_stage3 as S3  # noqa: E402


def one(run, sub, rows, canon, roots, out, force, vt_path, vt_md5, stage1_dir=None):
    present = []
    for r in rows:
        arms = [arm for arm in S3.L.ARMS
                if (lambda d, s: (d / f'{s}.candidates.parquet').exists())(*S3.canonical(r, arm, canon, roots))]
        if arms:
            present.append((r, arms))
    if not present:
        return run, sub, 'no test output', None
    keep = {(r['tag'], a) for r, arms in present for a in arms}
    canonical = S3.canonical
    # a copy of S3.one's body with the missing (tag, arm) pairs dropped (see the module docstring)
    from sept26_prelim_analysis import build_tracks as BT
    import pandas as pd
    dst = Path(out) / f'tracks_run_{run}_{sub}.parquet'
    if dst.exists() and not force:
        return run, sub, 'exists', None
    rows = [r for r, _ in present]
    try:
        with tempfile.TemporaryDirectory(dir=out, prefix='.view_') as view:
            bow_log = []
            for r in rows:
                for arm in S3.L.ARMS:
                    if (r['tag'], arm) not in keep:
                        continue
                    d, stem = canonical(r, arm, canon, roots)
                    vd = Path(view) / f'mx17_{arm}'
                    vd.mkdir(exist_ok=True)
                    for ext in ('.candidates.parquet', '.meta.json'):
                        src = d / f'{stem}{ext}'
                        dst_c = vd / f"events_{r['tag']}{ext}"
                        if ext == '.candidates.parquet' and arm in S3.BOW_ARMS:
                            bow_log.append(dict(tag=r['tag'], arm=arm, **S3.bow_patch(
                                src, d / f'{stem}.meta.json', S3.BUNDLES / S3.bundle_for(r, arm, canon), dst_c)))
                        else:
                            os.symlink(src, dst_c)
            s1 = Path(stage1_dir or S3.OLD / 'stage1') / f'candidates_run_{run}_{sub}.parquet'
            tracks, meta = BT.build(f'run_{run}', sub, Path(view), stage1=s1 if s1.exists() else None, allow=None,
                                    out_dir=None, k_arm=S3.K_ARM, write=False, tan_max_true=S3.TAN_MAX_TRUE)
        tracks['tan_source'] = 'bundle_v'
        b = tracks['arm'] == 'B'
        tracks['in_acceptance'] = tracks['in_acceptance'].astype('boolean')
        tracks.loc[b, 'in_acceptance'] = pd.NA
        tracks.loc[b, 'gated'] = tracks.loc[b, 'gated_reco']
        tracks.loc[b, 'tan_source'] = 'none (B: positions only)'
        meta['n_gated'] = int(tracks['gated'].sum())
        meta['bow_partner_fix'] = dict(rule='as ntof_stage3.py', per_tag=bow_log)
        trig = {S3.VT.TRIGGER_OF_RCLASS[r['rclass']] for r in rows}
        if len(trig) != 1:
            raise ValueError(f'sub-run rows disagree on trigger class {trig}')
        from ntof_tracking.reco import geometry as G
        from sept26_prelim_analysis import paths
        cfg = json.loads((Path(str(paths.root('runs'))) / f'run_{run}' / 'run_config.json').read_text())
        tracks, vinfo = S3.VT.attach_vt(tracks, S3.VT.load_vt(vt_path), trig.pop(), G.detector_transforms(cfg))
        meta['vt'] = S3.VT.meta_block(vinfo, vt_path, vt_md5)
        meta['n_gated_vt'] = vinfo['n_gated_vt']
        meta.update(pass_name='phase-4 test (partial sub-run)', canon=canon,
                    stage1_context=str(s1) if s1.exists() else None,
                    rows=[r['tag'] for r in rows], partial_rows=sorted(f'{t}/{a}' for t, a in keep),
                    built_by='T2_ref_compare/scripts/ntof_stage3_partial.py')
        tracks.to_parquet(dst, index=False)
        Path(str(dst).replace('.parquet', '.meta.json')).write_text(json.dumps(meta, indent=1, default=str))
        return run, sub, f"{meta['n_tracks']} tracks, {meta['n_gated']} gated ({len(keep)} tag/arm)", None
    except Exception:
        return run, sub, 'FAILED', traceback.format_exc(limit=6)


S3.one = one

if __name__ == '__main__':
    S3.main()
