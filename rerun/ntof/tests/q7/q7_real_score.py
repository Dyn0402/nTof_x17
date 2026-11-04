#!/usr/bin/env python3
"""q7_real_score.py -- score the real-trigger reconstructions of q7_real.py in every configuration.

    q7_real_score.py beam  [--stems r145_A r145_C]          -> $Q7_DATA/real/beam_score.json (+ tracks parquet)
    q7_real_score.py line  [--stem-a r149_A --stem-c r149_C] -> $Q7_DATA/real/line_score.json

beam (run_145): capsule pointing as rerun/ntof/capsule_point.py (distance of each X-Y track's line to the vertical
  capsule axis, x -9.5, z -3.7 mm), per configuration, for gated tracks with and without the stage-3 true-angle
  acceptance (|tan_x|, |tan_y| < 0.6 / 1.0; raw = true at kw ~ 1).  Against the baseline tm0.6_count_gate: events whose
  pairs change, gated tracks gained / lost, and the single-track check -- every event with one candidate per view.
line (run_149, beam-off cosmics): rerun/ntof/ac_line.py's A-C line score per configuration, and per-track movement of
  the line events against the baseline.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))      # rerun/ntof: ac_line, capsule_point
import q7lib as L  # noqa: E402

REAL = os.path.join(L.DATA, 'real')
ACC = (0.6, 1.0)


def tracks_of(cand, arm, run_config):
    from sept26_prelim_analysis import build_tracks as BT
    from ntof_tracking.reco import geometry as G
    trs = G.detector_transforms(json.load(open(run_config)))
    c = cand[cand['rank'] >= 0].copy()
    c['tag'] = c['acq_tag']
    df = BT.pair_rows(c)
    df = BT.local_and_global(df, trs[f'mx17_{arm}'], k=1.0)
    df['arm'] = arm
    return df


def common_acq(C, key=('acq_tag',)):
    """Acquisitions present in every configuration (a missing condor job drops that acquisition from all of them)."""
    have = C.groupby(list(key)).cfg.nunique()
    n = C.cfg.nunique()
    keep = set(have[have == n].index)
    drop = sorted(set(have.index) - keep)
    return keep, drop


def beam(stems):
    import capsule_point as CP
    rc = os.path.join(L.RUNS, 'run_145', 'run_config.json')
    out = {}
    for stem in stems:
        arm = stem.split('_')[-1]
        print(stem, flush=True)
        C = pd.read_parquet(os.path.join(REAL, f'{stem}.candidates.parquet'))
        E = pd.read_parquet(os.path.join(REAL, f'{stem}.events.parquet'))
        keep, drop = common_acq(E)
        C, E = C[C.acq_tag.isin(keep)], E[E.acq_tag.isin(keep)]
        out.setdefault('_dropped', {})[stem] = [str(d) for d in drop]
        allT = []
        res = {}
        base = None
        for cf, g in C.groupby('cfg'):
            T = tracks_of(g, arm, rc)
            T['dca'], T['y_at'] = CP.axis_dca(T)
            T['cfg'] = cf
            allT.append(T)
        T = pd.concat(allT, ignore_index=True)
        T.to_parquet(os.path.join(REAL, f'{stem}.tracks.parquet'), index=False)
        key = ['tag', 'event_id']
        B = T[T.cfg == L.BASELINE]
        bset = set(zip(B.tag, B.event_id, B.x_p0.round(4), B.y_p0.round(4), B.gated))
        Eb = E[E.cfg == L.BASELINE].set_index(['acq_tag', 'event_id'])
        one = Eb[(Eb.x_n_candidates <= 1) & (Eb.y_n_candidates <= 1)].index
        for cf, t in T.groupby('cfg'):
            g_ = t[t.gated]
            tmax = np.maximum(g_.tanx.abs(), g_.tany.abs())
            r = dict(n_events=int(E[E.cfg == cf].shape[0]), n_pairs=int(len(t)), n_gated=int(len(g_)),
                     n_gated_point15=int((g_.dca < 15).sum()), n_gated_nonpoint=int((g_.dca >= 15).sum()))
            for acc in ACC:
                m = tmax < acc
                r[f'acc{acc:g}'] = dict(n=int(m.sum()), point15=int((g_.dca[m] < 15).sum()),
                                        point10=int((g_.dca[m] < 10).sum()),
                                        nonpoint=int((g_.dca[m] >= 15).sum()),
                                        frac_point15=float((g_.dca[m] < 15).mean()) if m.any() else np.nan)
            # per tan bin
            r['bins'] = [dict(lo=lo, hi=hi, n=int(mm.sum()), point15=int((g_.dca[mm] < 15).sum()))
                         for lo, hi in zip((0, 0.2, 0.4, 0.6, 0.8, 1.0), (0.2, 0.4, 0.6, 0.8, 1.0, 1.5))
                         for mm in [(tmax >= lo) & (tmax < hi)]]
            # against the baseline: gated tracks (same x and y candidate) gained / lost
            s = set(zip(t.tag, t.event_id, t.x_p0.round(4), t.y_p0.round(4), t.gated))
            bg = {k[:4] for k in bset if k[4]}
            cg = {k[:4] for k in s if k[4]}
            r['vs_base'] = dict(gated_lost=len(bg - cg), gated_gained=len(cg - bg))
            # shallow gated baseline tracks (|tan| < 0.6 both): lost?  (the single-track cost of a change)
            Bs = B[B.gated & (np.maximum(B.tanx.abs(), B.tany.abs()) < 0.6)]
            bs = set(zip(Bs.tag, Bs.event_id, Bs.x_p0.round(4), Bs.y_p0.round(4)))
            r['vs_base']['shallow_gated_lost'] = len(bs - cg)
            r['vs_base']['shallow_gated_lost_pointing'] = int(
                Bs[[k in (bs - cg) for k in zip(Bs.tag, Bs.event_id, Bs.x_p0.round(4), Bs.y_p0.round(4))]]
                .dca.lt(15).sum())
            Ec = E[E.cfg == cf].set_index(['acq_tag', 'event_id'])
            pb = Eb.n_tracks
            pc = Ec.n_tracks.reindex(pb.index)
            r['vs_base']['events_ntracks_changed'] = int((pb != pc).sum())
            r['vs_base']['ntracks_transitions'] = {f'{o}->{n}': int(c) for (o, n), c in
                                                   pd.DataFrame(dict(o=pb, n=pc))[pb != pc].groupby(['o', 'n'])
                                                   .size().items()}
            # single-candidate events: winner tangents and positions must not move with select / split modes
            w = ['x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta']
            d = (Ec.loc[one, w] - Eb.loc[one, w]).abs().max(axis=1)
            r['one_cand'] = dict(n=int(len(one)), winner_moved=int((d > 1e-9).sum()),
                                 ntracks_changed=int((Ec.loc[one, 'n_tracks'] != Eb.loc[one, 'n_tracks']).sum()))
            # winners (events table) moved at all
            d = (Ec[w] - Eb[w].reindex(Ec.index)).abs().max(axis=1)
            r['winner_moved_events'] = int((d > 1e-9).sum())
            res[cf] = r
        out[stem] = res
        print(stem, 'done', flush=True)
    json.dump(out, open(os.path.join(REAL, 'beam_score.json'), 'w'), indent=1, default=float)


def _cat(stems):
    return pd.concat([pd.read_parquet(os.path.join(REAL, f'{s}.candidates.parquet')).assign(sub_i=i)
                      for i, s in enumerate(stems)], ignore_index=True)


def line(stems_a, stems_c, acc=None):
    """``acc``: the stage-3 acceptance applied before the one-gated-track-per-arm selection (build_tracks
    --tan-max-true: gated = gated_reco & |tan_x|, |tan_y| < acc; raw = true at kw ~ 1)."""
    import ac_line as AL
    if acc is not None and not getattr(AL, '_q7_wrapped', False):
        _orig = AL.tracks

        def _acc_tracks(path, arm, tr):
            df = _orig(path, arm, tr)
            df['gated'] = df['gated'] & (df.tanx.abs() < acc) & (df.tany.abs() < acc)
            return df
        AL.tracks, AL._q7_wrapped = _acc_tracks, True
    rc = os.path.join(L.RUNS, 'run_149', 'run_config.json')
    CA, CC = _cat(stems_a), _cat(stems_c)
    ka, da = common_acq(CA, ('sub_i', 'acq_tag'))
    kc, dc = common_acq(CC, ('sub_i', 'acq_tag'))
    keep = ka & kc
    CA = CA[[k in keep for k in zip(CA.sub_i, CA.acq_tag)]]
    CC = CC[[k in keep for k in zip(CC.sub_i, CC.acq_tag)]]
    dropped = dict(A=[list(map(str, d)) for d in da], C=[list(map(str, d)) for d in dc])
    tmp = os.path.join(REAL, 'line_tmp')
    os.makedirs(tmp, exist_ok=True)
    out = {}
    lines = {}
    for cf in sorted(CA.cfg.unique()):
        pa, pc = os.path.join(tmp, f'{cf}_A.parquet'), os.path.join(tmp, f'{cf}_C.parquet')
        for C_, p in ((CA, pa), (CC, pc)):
            c = C_[(C_.cfg == cf) & (C_['rank'] >= 0)].copy()
            # event ids repeat across run_149 sub-runs: make them unique per (subrun, acq) for the A-C join
            c['event_id'] = c['event_id'].astype(np.int64) + c['sub_i'].astype(np.int64) * 10_000_000
            c.to_parquet(p.replace('.parquet', '.candidates.parquet'), index=False)
        s, M = AL.score(pa, pc, rc, 60.0, cf)
        out[cf] = s
        lines[cf] = M
    B = lines[L.BASELINE].set_index('event_id')
    for cf, M in lines.items():
        M = M.set_index('event_id')
        com = B.index.intersection(M.index)
        mv = {}
        for s_ in ('a', 'c'):
            d = np.hypot(M.loc[com, f'p0_x_{s_}'] - B.loc[com, f'p0_x_{s_}'],
                         M.loc[com, f'p0_y_{s_}'] - B.loc[com, f'p0_y_{s_}'])
            dt = np.maximum((M.loc[com, f'tanx_{s_}'] - B.loc[com, f'tanx_{s_}']).abs(),
                            (M.loc[com, f'tany_{s_}'] - B.loc[com, f'tany_{s_}']).abs())
            mv[s_.upper()] = dict(moved_p0_gt1mm=int((d > 1).sum()), moved_tan_gt1e3=int((dt > 1e-3).sum()))
        out[cf]['vs_base'] = dict(common=int(len(com)), only_base=int(len(B.index.difference(M.index))),
                                  only_cfg=int(len(M.index.difference(B.index))), moved=mv)
    out['_dropped'] = dropped
    out['_acc'] = acc
    json.dump(out, open(os.path.join(REAL, 'line_score.json' if acc is None else f'line_score_acc{acc:g}.json'), 'w'),
              indent=1, default=float)
    print(dropped)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('beam')
    b.add_argument('--stems', nargs='+', default=['r145_A', 'r145_C'])
    l_ = sub.add_parser('line')
    l_.add_argument('--stems-a', nargs='+', required=True)
    l_.add_argument('--stems-c', nargs='+', required=True)
    l_.add_argument('--acc', type=float, default=None)
    a = ap.parse_args()
    if a.cmd == 'beam':
        beam(a.stems)
    else:
        line(a.stems_a, a.stems_c, a.acc)


if __name__ == '__main__':
    main()
