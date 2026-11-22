#!/usr/bin/env python3
"""evaluate.py -- compare the ringing-corrected reconstructions with pass 1 (README.md).

    python -m rerun.ntof.ringing.evaluate [--subrun stat090_0000] [--modes pass1,deflect_hits,deflect_wave]

Every reconstruction of reco_test.py, and the canonical pass-1 stage 2 of the same tags and events, goes through
build_tracks (frozen bundle v, k = 1; stage-1 context and trigger time attached) on the same footing.  A track is
"gated" when the reco gates it and |tan x|, |tan y| < 0.5.  Per ToF bin we then give, as in the 2026-11-22 charge-cut
study (/home/dylan/.claude/jobs/bcad1a92/tmp/lq/explore6.py):

    n         gated tracks
    kept      n over pass 1's n
    eff       pointing-excess efficiency: n x excess over pass 1's n x excess (real tracks kept)
    purity    the excess over the late (>= 30 ms) excess of pass 1

The excess is the fraction of tracks passing within 15 mm of the source (x of the late band crossing of pass 1) minus
the same fraction after shuffling the displacements.  We also apply the charge cuts (min view charge >= 600 ADC) to
pass 1 for comparison, check that late tracks keep their tangents, and give the efficiency vs |tan| of late
wall-confirmed pass-1 tracks.
Output: $WORK_DATA/ntof_pass1/ana/ringing/eval_<sub>.csv and late_<sub>.csv, angle_<sub>.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
WD = os.environ.get('WORK_DATA', '/media/ucla/work')
OUT = Path(WD) / 'ntof_pass1' / 'ana' / 'ringing'
STAGE2 = Path(WD) / 'ntof_pass1' / 'stage2'
STAGE1 = Path('/media/dylan/data/x17/sept26_prelim/stage1')
EDG = [1, 2, 3, 5, 8, 12, 16, 20, 30, 75]
TB = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def build(run, sub, cands: dict, metas: dict):
    """cands {arm: candidates frame with acq_tag}; metas {arm: meta path} -> stage-3 style tracks (frozen v)."""
    from sept26_prelim_analysis import build_tracks as BT
    with tempfile.TemporaryDirectory(dir=str(OUT), prefix='.view_') as view:
        for arm, c in cands.items():
            vd = Path(view) / f'mx17_{arm}'
            vd.mkdir()
            for tag, g in c.groupby('acq_tag'):
                g.to_parquet(vd / f'events_{tag}.candidates.parquet', index=False)
                shutil.copy(metas[arm], vd / f'events_{tag}.meta.json')
        s1 = STAGE1 / f'candidates_{run}_{sub}.parquet'
        t, _ = BT.build(run, sub, Path(view), stage1=s1 if s1.exists() else None, allow=None, out_dir=None,
                        k_arm={a: 1.0 for a in cands}, write=False, tan_max_true=0.6, allow_y_fail=True)
    t['ms'] = t.t_since_flash_ns / 1e6
    t['g'] = (t.gated_reco.fillna(False).astype(bool) & (t.tanx.abs() < 0.5) & (t.tany.abs() < 0.5))
    tt = -t.p0_z / t.d_z
    t['pos'], t['disp'] = t.p0_x, tt * t.d_x
    t['qmin'] = np.minimum(t.x_q_sum, t.y_q_sum)
    t['wall'] = False
    for a in cands:
        m = t.arm == a
        t.loc[m, 'wall'] = t.loc[m, f'wall_{a}'].astype('boolean').fillna(False).astype(bool)
    return t


def excess(s, u0, rng):
    if len(s) < 30:
        return np.nan
    m = np.abs(s.pos + s.disp - u0).to_numpy()
    sh = np.abs(s.pos.to_numpy() + rng.permutation(s.disp.to_numpy()) - u0)
    return float((m < 15).mean() - (sh < 15).mean())


def main():
    import pandas as pd
    from sept26_prelim_analysis.capsule_tof import crossing
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--arms', default='A,C')
    ap.add_argument('--modes', default='pass1,deflect_hits,deflect_wave')
    ap.add_argument('--tof-lo', type=float, default=1.0, help='start of the early window of the reco_test runs [ms]')
    a = ap.parse_args()
    arms = a.arms.split(',')
    modes = [m for m in a.modes.split(',') if (OUT / 'reco' / m / f'{a.run}_{a.subrun}_{arms[0]}.parquet').exists()]
    T, metas, wanted, tags = {}, {}, {}, {}
    for m in modes:
        cands = {}
        for arm in arms:
            p = OUT / 'reco' / m / f'{a.run}_{a.subrun}_{arm}.parquet'
            cands[arm] = pd.read_parquet(str(p).replace('.parquet', '.candidates.parquet'))
            metas[(m, arm)] = str(p).replace('.parquet', '.meta.json')
            mj = json.load(open(metas[(m, arm)]))
            tags[arm] = mj.get('ntof_pass', mj).get('tags') or sorted(cands[arm].acq_tag.unique())
            wanted.setdefault(arm, set()).update(cands[arm].event_id.unique().tolist())
        T[m] = build(a.run, a.subrun, cands, {k: metas[(m, k)] for k in arms})
    # the canonical pass-1 stage 2, same tags, the same event list (the events any reco_test mode fitted)
    cands, cm = {}, {}
    for arm in arms:
        d = STAGE2 / a.run / a.subrun / arm
        parts = []
        for f in sorted(d.glob(f'{a.run}_{a.subrun}_*_{arm}.candidates.parquet')):
            c = pd.read_parquet(f)
            if c.acq_tag.iloc[0] in tags[arm]:
                parts.append(c)
                cm[arm] = str(f).replace('.candidates.parquet', '.meta.json')
        cands[arm] = pd.concat(parts, ignore_index=True)
    T['canonical'] = build(a.run, a.subrun, cands, cm)
    # the same trigger list as reco_test: every trigger in the early window, plus the late ones a mode fitted (a
    # mode that seeds fewer early events must not shrink pass 1's early sample)
    c = T['canonical']
    late_ids = c.apply(lambda r: r.event_id in wanted[r.arm], axis=1) if len(c) else c.event_id
    T['canonical'] = c[((c.ms >= a.tof_lo) & (c.ms < 20)) | late_ids.astype(bool)]
    rng = np.random.default_rng(0)
    ref = 'canonical'
    rows, late_rows, ang_rows = [], [], []
    for arm in arms:
        R = T[ref][(T[ref].arm == arm) & T[ref].g]
        late = R[R.ms >= 20]
        u0, _, _ = crossing(late.pos.to_numpy(), late.disp.to_numpy())
        exl = excess(R[R.ms >= 30], u0, rng)
        variants = {m: T[m][(T[m].arm == arm) & T[m].g] for m in T}
        variants['canonical + q600'] = R[R.qmin >= 600]
        for m in list(T):
            if m != 'canonical':
                variants[f'{m} + q600'] = variants[m][variants[m].qmin >= 600]
        for m, S in variants.items():
            for lo, hi in zip(EDG[:-1], EDG[1:]):
                r0 = R[(R.ms >= lo) & (R.ms < hi)]
                s = S[(S.ms >= lo) & (S.ms < hi)]
                e0, e1 = excess(r0, u0, rng), excess(s, u0, rng)
                rows.append(dict(arm=arm, mode=m, lo=lo, hi=hi, n=len(s), n_pass1=len(r0), kept=len(s) / max(len(r0), 1),
                                 eff=(len(s) * e1) / (len(r0) * e0) if e0 and e0 > 0.005 else np.nan,
                                 purity=e1 / exl, q_med=float(s.qmin.median()) if len(s) else np.nan,
                                 frac_q_lt600=float((s.qmin < 600).mean()) if len(s) else np.nan))
            # late tracks: same events, best (rank-0 pair) track tangents
            if m in T:
                k = ['tag', 'event_id']
                L0 = R[R.ms >= 30].sort_values('track_id').drop_duplicates(k)
                L1 = S[S.ms >= 30].sort_values('track_id').drop_duplicates(k)
                j = L0.merge(L1, on=k, how='left', suffixes=('', '_n'))
                d = np.maximum(np.abs(j.tanx - j.tanx_n), np.abs(j.tany - j.tany_n))
                late_rows.append(dict(arm=arm, mode=m, n_late=len(L0), found=float(j.tanx_n.notna().mean()),
                                      identical=float((d < 1e-9).mean()), within_0p01=float((d < 0.01).mean())))
                W = L0[L0.wall]
                jw = W.merge(L1, on=k, how='left', suffixes=('', '_n'))
                ok = (np.maximum(np.abs(jw.tanx - jw.tanx_n), np.abs(jw.tany - jw.tany_n)) < 0.01)
                i = np.digitize(jw.tanx.abs(), TB) - 1
                ang_rows.append(dict(arm=arm, mode=m, **{f'tanx_{TB[b]}-{TB[b + 1]}': float(ok[i == b].mean())
                                                         for b in range(len(TB) - 1)}))
    E = pd.DataFrame(rows)
    E.to_csv(OUT / f'eval_{a.subrun}.csv', index=False)
    pd.DataFrame(late_rows).to_csv(OUT / f'late_{a.subrun}.csv', index=False)
    pd.DataFrame(ang_rows).to_csv(OUT / f'angle_{a.subrun}.csv', index=False)
    pd.set_option('display.width', 250)
    for arm in arms:
        e = E[E.arm == arm]
        print(f'--- {arm}: kept / eff / purity per ToF bin')
        print(e.pivot_table(index='mode', columns='lo', values='kept').round(2).to_string())
        print(e.pivot_table(index='mode', columns='lo', values='eff').round(2).to_string())
        print(e.pivot_table(index='mode', columns='lo', values='purity').round(2).to_string())
        print(e.pivot_table(index='mode', columns='lo', values='n').to_string())
    print(pd.DataFrame(late_rows).round(4).to_string())
    print(pd.DataFrame(ang_rows).round(3).to_string())


if __name__ == '__main__':
    main()
