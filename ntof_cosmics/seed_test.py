#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_test.py -- does the beam seeder's 5-strip minimum throw away near-normal
tracks at n_TOF S/N?  HANDOFF_TRACKING_2026-10-06.md §9.

End to end and not circular: every trigger with hit clusters (>= 3 strips) in
both views of BOTH chambers A and C is reconstructed twice with the production
bundle and production fit -- seeder minimum 5 strips (`MIN_STRIPS_BEAM`,
production) and 3 (`wft.seed.MIN_STRIPS`, the bench) -- then built into tracks
and A-C pairs exactly as the cosmic analysis does.  The joined line through
the two chambers' mesh positions is the truth (it uses no fitted angle), so
the yield and the angle quality of near-normal crossings can be compared.

    WFT_BEAM_BASE=<work>/beam/ python ntof_cosmics/seed_test.py --work <work> --subs <file> --min 5
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import cosmic_tracks as CT  # noqa: E402
import insitu_calib as IC   # noqa: E402

RUN = 'run_149'
ARMS = ('A', 'C')


def allowlist(sub: str) -> dict:
    """Triggers with a >= 3-strip seed in x AND y of both A and C."""
    from ntof_tracking import wft_beam as WB
    from wft import io as wio
    have = {}
    for arm in ARMS:
        cfg = WB.beam_config(arm, RUN, sub)
        pos = wio.strip_position_map(cfg)
        for tag in WB.subrun_tags(cfg):
            hp = WB.hits_file_for_tag(cfg, tag)
            if hp is None:
                continue
            h = WB.read_hits_tag(hp, (cfg.MX17_FEU_X, cfg.MX17_FEU_Y))
            s = WB.seeds_from_hits_beam(h, pos, cfg.MX17_FEU_X, cfg.MX17_FEU_Y, min_strips=3)
            ok = {e for e, r in s.items() if r['x'] and r['y']}
            have.setdefault(tag, {})[arm] = ok
    return {tag: set.intersection(*[d.get(a, set()) for a in ARMS]) for tag, d in have.items()}


def run(work: Path, sub: str, min_strips: int, jobs: int) -> pd.DataFrame:
    from ntof_tracking import wft_beam as WB
    from sept26_prelim_analysis import build_tracks as BT
    allow_p = work / 'seedtest' / f'allow_{sub}.json'
    if allow_p.exists():
        allow = {t: set(v) for t, v in json.loads(allow_p.read_text()).items()}
    else:
        allow = allowlist(sub)
        allow_p.parent.mkdir(parents=True, exist_ok=True)
        allow_p.write_text(json.dumps({t: sorted(v) for t, v in allow.items()}))
    orig = WB.seeds_from_hits_beam
    WB.seeds_from_hits_beam = functools.partial(orig, min_strips=min_strips)
    rdir = IC._guard(work / 'seedtest' / f'm{min_strips}' / sub)
    try:
        for arm in ARMS:
            bundle = str(IC.PROD_BUNDLE).format(arm=arm)
            for tag, ids in allow.items():
                if not ids:
                    continue
                cfg = WB.beam_config(arm, RUN, sub)
                cfg.file_tags = [tag]
                out = rdir / f'mx17_{arm}' / f'events_{tag}.parquet'
                if out.exists():
                    continue
                WB.reconstruct_subrun(cfg, bundle, str(out), jobs=jobs,
                                      allow_events={tag: ids}, verbose=False)
    finally:
        WB.seeds_from_hits_beam = orig
    k = CT._k('run_150')
    tracks, _ = BT.build(RUN, sub, rdir, stage1=None, allow=None,
                         out_dir=rdir / 'tracks', k_arm={a: k[a] for a in ARMS})
    P = CT.pairs(tracks)
    P = P[P.pair == 'AC']
    ng = tracks[tracks.gated].groupby(['event_id', 'arm']).size().unstack(fill_value=0)
    ti = tracks.set_index(['event_id', 'arm', 'track_id'])
    a = ti.loc[list(zip(P.event_id, P.arm1, P.track1))].reset_index()
    b = ti.loc[list(zip(P.event_id, P.arm2, P.track2))].reset_index()
    Jz = b.p0_z.to_numpy() - a.p0_z.to_numpy()
    o = pd.DataFrame(dict(subrun=sub, event_id=P.event_id.to_numpy(), sep_mm=P.sep_mm.to_numpy(),
                          jx=(b.p0_x.to_numpy() - a.p0_x.to_numpy()) / Jz,
                          jy=(b.p0_y.to_numpy() - a.p0_y.to_numpy()) / Jz))
    for arm, tr in (('A', a), ('C', b)):
        for ax in 'xy':
            o[f'{arm}_raw_{ax}'] = tr[f'tan_raw_{ax}'].to_numpy()
        o[f'n_{arm}'] = ng.reindex(P.event_id)[arm].fillna(0).astype(int).to_numpy()
    o['n_allowed'] = sum(len(v) for v in allow.values())
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--subs', required=True)
    ap.add_argument('--min', type=int, required=True)
    ap.add_argument('--jobs', type=int, default=15)
    a = ap.parse_args()
    work = IC._guard(a.work)
    os.environ.setdefault('WFT_BEAM_BASE', str(work / 'beam') + '/')
    outs = []
    for sub in Path(a.subs).read_text().split():
        o = run(work, sub, a.min, a.jobs)
        print(f'{sub} min {a.min}: {len(o)} A-C pairs from {o.n_allowed.iloc[0] if len(o) else 0} triggers', flush=True)
        outs.append(o)
    pd.concat(outs, ignore_index=True).to_parquet(work / f'seedtest_m{a.min}.parquet', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
