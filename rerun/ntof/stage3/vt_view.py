#!/usr/bin/env python3
"""vt_view.py -- the analysis view of n_TOF pass-1 stage 3: the v(t) angles under the plain column names.

Stage 3 (EOS Analysis_pass1/stage3, ntof_stage3.py) keeps the frozen-bundle-v columns (`tanx`, `gated`, ...) and puts
the v(t)-corrected ones beside them (`tanx_vt`, `gated_vt`, ...; vt_join.py).  The analysis uses the v(t) ones
(Dylan 2026-11-17: table correction, true |tan_vt| < 0.5).  Every downstream script of the is2_v2 era reads the plain
names, file by file, so this writes one table per sub-run in which

    <col>         = the stage-3 <col>_vt      (for every column that has a _vt twin; gated = gated_vt,
                                               in_acceptance = in_acceptance_vt)
    <col>_frozen  = the stage-3 <col>          (the frozen-v value, kept for checks)

and every other column unchanged.  `tan_raw_x/y` (= tanx/tany in pass 1: no k) also take the v(t) values, so code that
computes `k * tan_raw` with k = k_arm = 1 gets the analysis angle.  The sidecar is the stage-3 one plus a `view` block (source, md5 of the source
table, the mapping).  Nothing new is computed here: the view is regenerable from EOS stage 3 at any time, so it is not
archived.

    vt_view.py [--src $WORK_DATA/ntof_pass1/stage3] [--out $WORK_DATA/ntof_pass1/stage3_vt] [--jobs 8] [--force]
"""
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


VIEW = 'pass1_vt'


#: pass 1 has no k (k_arm = 1, tan_raw_* == tan*), so the "raw" tangent the is2-era code multiplies by its own k is the
#: analysis angle too: tan_raw_{x,y} take tan{x,y}_vt (frozen kept as tan_raw_*_frozen)
RAW = {'tan_raw_x': 'tanx_vt', 'tan_raw_y': 'tany_vt'}


#: Columns a beam-off sub-run (and run_104 stat090_0016, which has no trigger-time product) does not have: the stage-1
#: context and the flash.  The view writes them as nulls so every table has one schema (readers that ask for them get
#: null, not a missing-column error).
PAD = {'t_since_flash_ns': 'float64', 'e_neutron_keV': 'float64', 'is_flash': 'UInt8', 'event_class': 'object',
       'arms_lit': 'object', 'bunch': 'Int32', 'n_coinc_arms': 'Int64', 'coinc_this_arm': 'Int64',
       'no_stage1_class': 'boolean', **{f'{k}_{a}': 'boolean' for k in ('wall', 'plastic', 'coinc') for a in 'ABCD'}}


def mapping(names):
    twins = [c[:-3] for c in names if c.endswith('_vt') and c[:-3] in names]
    return twins


def one(src, out, force):
    src, out = Path(src), Path(out)
    dst = out / src.name
    if dst.exists() and not force:
        return src.name, 'exists'
    import pandas as pd
    t = pd.read_parquet(src)
    names = list(t.columns)
    twins = mapping(names)
    cols = {}
    for c in names:
        if c.endswith('_vt') and c[:-3] in twins:
            continue
        if c in twins:
            cols[c] = t[c + '_vt']
            cols[c + '_frozen'] = t[c]
        elif c in RAW:
            cols[c] = t[RAW[c]]
            cols[c + '_frozen'] = t[c]
        else:
            cols[c] = t[c]
    v = pd.DataFrame(cols)
    padded = [c for c in PAD if c not in v.columns]
    for c in padded:
        v[c] = pd.Series([None] * len(v), dtype=PAD[c]) if PAD[c] != 'float64' else float('nan')
    tmp = dst.with_suffix('.part')
    v.to_parquet(tmp, index=False, compression='snappy')
    os.replace(tmp, dst)
    mj = src.with_suffix('.meta.json')
    meta = json.load(open(mj)) if mj.exists() else {}
    meta['view'] = dict(name=VIEW, source=str(src), source_md5=hashlib.md5(open(src, 'rb').read()).hexdigest(),
                        rule='<col> = stage-3 <col>_vt; <col>_frozen = stage-3 <col> (frozen bundle v); '
                             'tan_raw_{x,y} = tan{x,y}_vt (no k in pass 1)',
                        columns=twins + list(RAW), padded_null=padded, built_by='rerun/ntof/stage3/vt_view.py')
    json.dump(meta, open(out / mj.name, 'w'), indent=1)
    return src.name, f'{len(twins)} columns'


def main():
    wd = os.environ.get('WORK_DATA', '/media/ucla/work')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=f'{wd}/ntof_pass1/stage3')
    ap.add_argument('--out', default=f'{wd}/ntof_pass1/stage3_vt')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fs = sorted(Path(a.src).glob('tracks_run_*.parquet'))
    with ProcessPoolExecutor(a.jobs) as ex:
        jobs = [ex.submit(one, f, a.out, a.force) for f in fs]
        for i, j in enumerate(as_completed(jobs)):
            name, msg = j.result()
            print(f'[{i + 1}/{len(fs)}] {name}: {msg}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
