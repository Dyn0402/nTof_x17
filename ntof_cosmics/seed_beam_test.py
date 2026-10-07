#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_beam_test.py -- does the beam seeder's 3-strip minimum (which recovers
head-on cosmics, HANDOFF_TRACKING_2026-10-06.md §9) let beam junk in?

Re-runs the September full pass of one beam sub-run tag by tag, with the full
pass's OWN saved bundle (`reco_fullpass/<run>/<sub>/mx17_<arm>/calib_bundle_
prelim`), every trigger (the full pass had no allowlist), the same env
switches -- changing only `min_strips`.  ``reco`` at ``--min 5`` must reproduce
the full pass bit for bit; that is checked by ``verify``.  Track building and
gating then follow the stage-3 code (`build_tracks.build`) with the stage-3
k_arm, so a min-3 track is gated exactly as a production track would be.

    python ntof_cosmics/seed_beam_test.py reco --min 3 [--arms A C] [--tags 000 001]
    python ntof_cosmics/seed_beam_test.py verify --arm A --tag 000
    python ntof_cosmics/seed_beam_test.py build --min 3      # --min 0 = the production reco
    python ntof_cosmics/seed_beam_test.py compare --min 3
    python ntof_cosmics/seed_beam_test.py scint --min 3 --arms A C D

Outputs go under ``--work`` (default ~/scratch/ntof_insitu/beamseed); nothing
is written to /media/dylan/data.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

FULLPASS = Path('/media/dylan/data/x17/sept26_prelim/reco_fullpass')
STAGE3 = Path('/media/dylan/data/x17/sept26_prelim/stage3_fullpass')
WORK = Path.home() / 'scratch' / 'ntof_insitu' / 'beamseed'
ARMS = ('A', 'B', 'C', 'D')


def _guard(p) -> Path:
    p = Path(p).expanduser().resolve()
    if str(p).startswith('/media/dylan/data'):
        raise SystemExit(f'refusing to write under /media/dylan/data: {p}')
    return p


def prod_dir(run, sub, arm) -> Path:
    return FULLPASS / run / sub / f'mx17_{arm}'


def tags_of(run, sub, arm, want=None) -> list[str]:
    tags = sorted(p.name[len('events_'):-len('.meta.json')]
                  for p in prod_dir(run, sub, arm).glob('events_*.meta.json'))
    if want:
        tags = [t for t in tags if any(t.endswith(w) for w in want)]
    return tags


def reco(work: Path, run: str, sub: str, min_strips: int, arms, want, jobs: int,
         bundles: dict | None = None, label: str | None = None):
    """``bundles`` {arm: path} replaces the full pass's bundle (default: the
    full pass's own); ``label`` names the output dir (default m<min>)."""
    from ntof_tracking import wft_beam as WB
    label = label or f'm{min_strips}'
    for arm in arms:
        bundle = (bundles or {}).get(arm) or str(prod_dir(run, sub, arm) / 'calib_bundle_prelim')
        for tag in tags_of(run, sub, arm, want):
            out = work / label / run / sub / f'mx17_{arm}' / f'events_{tag}.parquet'
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            cfg = WB.beam_config(arm, run, sub)
            cfg.file_tags = [tag]
            orig, orig_fn = WB.MIN_STRIPS_BEAM, WB.seeds_from_hits_beam
            # seeds_from_hits_beam binds its default at definition time (the
            # driver calls it without min_strips), and the sidecar reads the
            # global: patch both, or the meta would lie about the seeder
            WB.MIN_STRIPS_BEAM = min_strips
            WB.seeds_from_hits_beam = functools.partial(orig_fn, min_strips=min_strips)
            t = time.time()
            try:
                WB.reconstruct_subrun(cfg, bundle, str(out), jobs=jobs, verbose=False)
            finally:
                WB.MIN_STRIPS_BEAM, WB.seeds_from_hits_beam = orig, orig_fn
            m = json.loads(out.with_suffix('.meta.json').read_text())
            assert m['selection']['min_strips'] == min_strips, m['selection']
            print(f'{arm} {tag} min {min_strips}: {m["n_seeded"]} seeded in {time.time() - t:.0f} s',
                  flush=True)


def verify(work: Path, run, sub, arm, tag):
    a = pd.read_parquet(prod_dir(run, sub, arm) / f'events_{tag}.parquet')
    b = pd.read_parquet(work / 'm5' / run / sub / f'mx17_{arm}' / f'events_{tag}.parquet')
    print(f'production {len(a)} rows, rerun {len(b)} rows')
    key = [c for c in ('event_id', 'eventId', 'plane') if c in a.columns]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    if len(a) != len(b):
        print('row counts differ')
        return 1
    bad = []
    for c in a.columns:
        if c not in b.columns:
            bad.append((c, 'missing'))
            continue
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == 'f':
            d = np.nanmax(np.abs(x - y)) if len(x) else 0
            same_nan = np.array_equal(np.isnan(x), np.isnan(y))
            if d > 1e-6 or not same_nan:
                bad.append((c, f'max |d| {d:.3g}'))
        elif x.dtype == object:
            if not all(str(p) == str(q) for p, q in zip(x, y)):
                bad.append((c, 'object differs'))
        elif not np.array_equal(x, y):
            bad.append((c, 'differs'))
    print('IDENTICAL' if not bad else f'{len(bad)} columns differ: {bad[:10]}')
    return int(bool(bad))


def build(work: Path, run, sub, min_strips, label: str | None = None, k_one: bool = False):
    """Stage-3 tracks from a reco dir. ``min_strips`` 0 = the production reco
    itself (reco_fullpass), built with today's code so both sides match.
    ``k_one``: an in-situ bundle carries its own v and w0/kw, so its tans need
    no k -- build with k = 1 (B stays uncalibrated either way)."""
    from sept26_prelim_analysis import build_tracks as BT
    meta = json.loads((STAGE3 / f'tracks_{run}_{sub}.meta.json').read_text())
    k = dict(meta['k_arm']['applied'])
    if k_one:
        k = {a: 1.0 for a in k}
    label = label or ('prod' if min_strips == 0 else f'm{min_strips}')
    rdir = FULLPASS / run / sub if min_strips == 0 and label == 'prod' else work / label / run / sub
    odir = work / label / run / sub / 'tracks'
    odir.mkdir(parents=True, exist_ok=True)
    tracks, _ = BT.build(run, sub, rdir, stage1=Path(meta['stage1']), allow=None,
                         out_dir=odir, k_arm=k)
    tracks.to_parquet(odir / 'tracks.parquet', index=False)
    return tracks


MATCH_MM = 2.0
NEAR_NORMAL = 0.08


def _match(P: pd.DataFrame, Q: pd.DataFrame) -> np.ndarray:
    """For each row of P, is there a row of Q in the same (tag, event, arm)
    within MATCH_MM in both planes' p0?"""
    key = ['tag', 'event_id', 'arm']
    m = P.reset_index().merge(Q[key + ['x_p0', 'y_p0']], on=key, how='left', suffixes=('', '_q'))
    ok = ((m.x_p0 - m.x_p0_q).abs() < MATCH_MM) & ((m.y_p0 - m.y_p0_q).abs() < MATCH_MM)
    return m.assign(ok=ok).groupby('index').ok.any().reindex(P.index, fill_value=False).to_numpy()


def describe(T: pd.DataFrame) -> dict:
    n = len(T)
    if n == 0:
        return dict(n=0)
    nmin = np.minimum(T.x_n_strips, T.y_n_strips)
    chi = np.maximum(T.chi2dof_x, T.chi2dof_y)
    nn = (T.tan_raw_x.abs() < NEAR_NORMAL) & (T.tan_raw_y.abs() < NEAR_NORMAL)
    late = np.maximum(T.x_t0, T.y_t0) > 300
    return dict(n=n,
                min_strips_le4=float((nmin <= 4).mean()),
                med_n_strips=float(nmin.median()),
                med_chi2dof=float(chi.median()),
                chi2dof_gt20=float((chi > 20).mean()),
                late_t0=float(late.mean()),
                scint_coinc=float(T.coinc_this_arm.fillna(False).astype(bool).mean()),
                in_bore=float(T.in_bore.fillna(False).astype(bool).mean()),
                med_dca_mm=float(T.dca_axis_mm.median()),
                near_normal=float(nn.mean()),
                med_q_per_len=float(T.q_per_len.median()))


def compare(work: Path, run, sub, m=3):
    base = work / 'prod' / run / sub / 'tracks' / 'tracks.parquet'
    alt = work / f'm{m}' / run / sub / 'tracks' / 'tracks.parquet'
    P, Q = pd.read_parquet(base), pd.read_parquet(alt)
    od = work / 'compare' / run / sub
    od.mkdir(parents=True, exist_ok=True)
    rows, cls = [], []
    for arm in sorted(set(P.arm) | set(Q.arm)):
        Pg = P[(P.arm == arm) & P.gated].reset_index(drop=True)
        Qg = Q[(Q.arm == arm) & Q.gated].reset_index(drop=True)
        lost = ~_match(Pg, Qg)
        gained = ~_match(Qg, Pg)
        nnP = ((Pg.tan_raw_x.abs() < NEAR_NORMAL) & (Pg.tan_raw_y.abs() < NEAR_NORMAL))
        nnQ = ((Qg.tan_raw_x.abs() < NEAR_NORMAL) & (Qg.tan_raw_y.abs() < NEAR_NORMAL))
        sc = lambda T: T.coinc_this_arm.fillna(False).astype(bool)
        ib = lambda T: T.in_bore.fillna(False).astype(bool)
        ev2 = lambda T: int((T.groupby(['tag', 'event_id']).size() >= 2).sum())
        rows.append(dict(arm=arm, events_prod=P[P.arm == arm][['tag', 'event_id']].drop_duplicates().shape[0],
                         events_m3=Q[Q.arm == arm][['tag', 'event_id']].drop_duplicates().shape[0],
                         gated_prod=len(Pg), gated_m3=len(Qg),
                         lost=int(lost.sum()), gained=int(gained.sum()),
                         gained_scint=int((gained & sc(Qg)).sum()),
                         nearnormal_prod=int(nnP.sum()), nearnormal_m3=int(nnQ.sum()),
                         nearnormal_scint_bore_prod=int((nnP & sc(Pg) & ib(Pg)).sum()),
                         nearnormal_scint_bore_m3=int((nnQ & sc(Qg) & ib(Qg)).sum()),
                         events_2gated_prod=ev2(Pg), events_2gated_m3=ev2(Qg)))
        for lab, T in (('prod', Pg), ('matched', Qg[~gained]), ('gained', Qg[gained]),
                       ('lost', Pg[lost])):
            cls.append(dict(arm=arm, sample=lab, **describe(T)))
        Qg[gained].to_parquet(od / f'gained_{arm}.parquet', index=False)
        Pg[lost].to_parquet(od / f'lost_{arm}.parquet', index=False)
    R, C = pd.DataFrame(rows), pd.DataFrame(cls)
    R.to_csv(od / 'summary.csv', index=False)
    C.to_csv(od / 'samples.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(R.to_string(index=False)); print(); print(C.round(3).to_string(index=False))


def _gained_scint(P: pd.DataFrame, Q: pd.DataFrame) -> np.ndarray:
    key = ['subrun', 'event_id']
    m = Q.reset_index().merge(P[key + ['u_mm', 'v_mm']], on=key, how='left', suffixes=('', '_p'))
    ok = ((m.u_mm - m.u_mm_p).abs() < MATCH_MM) & ((m.v_mm - m.v_mm_p).abs() < MATCH_MM)
    return ~ok.groupby(m['index']).any().reindex(Q.index, fill_value=False).to_numpy()


def scint(work: Path, run, sub, m=3, arms=('A', 'C', 'D')):
    """External purity: each gated track extrapolated to its arm's SiPM wall and
    plastic (`det_a_scint.match_run`, measured geometry, same-width off-time
    control window = the accidental floor).  Production vs min-m, split into the
    tracks both seeders find and the ones only min-m finds."""
    from sept26_prelim_analysis import det_a_scint as S
    od = work / 'compare' / run / sub
    rows, pairs = [], []
    for arm in arms:
        geo = S.layer_geometry(run, arm)
        M = {}
        for lab in ('prod', f'm{m}'):
            f = od / f'scint{arm}_{lab}.parquet'
            if f.exists():
                M[lab] = pd.read_parquet(f)
                continue
            src = work / lab / run / sub / 'tracks'
            if not (src / f'tracks_{run}_{sub}.parquet').exists():
                break
            M[lab], _ = S.match_run(run, [sub], src, None, geo)
            M[lab].to_parquet(f, index=False)
        if len(M) < 2:
            continue
        P, Q = M['prod'], M[f'm{m}']
        Q = Q.assign(gained=_gained_scint(P, Q))
        nn = lambda T: (T.tanx.abs() < 0.1) & (T.tany.abs() < 0.1)  # noqa: E731

        def row(T, lab):
            w, p = T[T.on_wall], T[T.on_plas]
            return dict(arm=arm, sample=lab, n=len(T), n_wall=len(w),
                        wall=w.match_wall.mean(), wall_ctrl=w.match_wall_ctrl.mean(),
                        wall_excess_n=int(w.match_wall.sum() - w.match_wall_ctrl.sum()),
                        n_plas=len(p), plas=p.match_plas.mean(), plas_ctrl=p.match_plas_ctrl.mean(),
                        plas_excess_n=int(p.match_plas.sum() - p.match_plas_ctrl.sum()))
        rows += [row(P, 'production'), row(Q, f'min {m}, all'), row(Q[~Q.gained], f'min {m}, shared'),
                 row(Q[Q.gained], f'min {m}, gained'), row(P[nn(P)], 'production, near-normal'),
                 row(Q[nn(Q)], f'min {m}, near-normal')]
        for lab, T in (('production', P), (f'min {m}', Q)):
            two = T[T.n_trk == 2].sort_values(['event_id', 'u_mm'])
            g = two.groupby('event_id')
            a, b = g.nth(0).set_index('event_id'), g.nth(1).set_index('event_id')
            sep = np.hypot(a.u_mm - b.u_mm, a.v_mm - b.v_mm)
            pairs.append(dict(arm=arm, sample=lab, events=len(a), sep_lt12=float((sep < 12).mean()),
                              sep_12_24=float(((sep >= 12) & (sep < 24)).mean()),
                              sep_med_mm=float(np.median(sep)) if len(sep) else np.nan,
                              either_wall=float((a.match_wall | b.match_wall).mean()),
                              both_wall=float((a.match_wall & b.match_wall).mean())))
    R, PR = pd.DataFrame(rows), pd.DataFrame(pairs)
    R.to_csv(od / 'scint_confirmation.csv', index=False)
    PR.to_csv(od / 'pairs.csv', index=False)
    with pd.option_context('display.width', 250):
        print(R.round(3).to_string(index=False)); print(); print(PR.round(3).to_string(index=False))


R_EDGES = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60])


def kbeam(work: Path, run, sub, labels, arms=('A', 'C')):
    """Beam angle response against capsule pointing -- `k_arm`'s sample
    (gated, this arm's scintillators in coincidence, its charge and lever
    windows), true tan = lever / D_PERP, x view.  For each table: raw/true per
    |true tan| bin and k_arm's band and track estimators on the RAW tans.  A
    bundle whose v and w0/kw are right reads 1 everywhere."""
    from ntof_tracking import run145_target_imaging as TI
    from sept26_prelim_analysis import k_arm as K
    rows, ks = [], []
    for lab in labels:
        f = work / lab / run / sub / 'tracks' / 'tracks.parquet'
        if not f.exists():
            print(f'{lab}: no tracks'); continue
        t = pd.read_parquet(f)
        for arm in arms:
            g = t[(t.arm == arm) & t.gated & t.coinc_this_arm.astype(bool) & (t.x_q_sum > 0)
                  & np.isfinite(t.tan_raw_x)].copy()
            if not len(g):
                continue
            lo, hi = np.percentile(g.x_q_sum, K.CHARGE_WINDOW)
            g = g[(g.x_q_sum >= lo) & (g.x_q_sum <= hi)]
            g['lev'] = g.x_local - TI.PINWHEEL[arm]
            g = g[(g.lev.abs() > K.LEVER_WINDOW_MM[0]) & (g.lev.abs() < K.LEVER_WINDOW_MM[1])
                  & (g.tan_raw_x.abs() > 1e-3)]
            te = (g.lev / K.D_PERP_MM).to_numpy()
            a, fr = np.abs(te), g.tan_raw_x.to_numpy() * np.sign(te)
            for lo_, hi_ in zip(R_EDGES[:-1], R_EDGES[1:]):
                m = (a >= lo_) & (a < hi_)
                if m.sum() < 30:
                    continue
                rows.append(dict(label=lab, arm=arm, lo=lo_, hi=hi_, n=int(m.sum()),
                                 ratio_med=float(np.median(fr[m] / a[m])),
                                 sign_ok=float((fr[m] > 0).mean())))
            S = dict(xl=g.x_local.to_numpy(), tx=g.tan_raw_x.to_numpy(),
                     foot_x=float(TI.PINWHEEL[arm]))
            ks.append(dict(label=lab, arm=arm, n=len(g), band=K.band_k(S), track=K.track_k(S)))
    od = work / 'compare' / run / sub
    od.mkdir(parents=True, exist_ok=True)
    R, KK = pd.DataFrame(rows), pd.DataFrame(ks)
    tag = '_'.join(labels)
    R.to_csv(od / f'kbeam_response_{tag}.csv', index=False)
    KK.to_csv(od / f'kbeam_k_{tag}.csv', index=False)
    with pd.option_context('display.width', 250):
        print(KK.round(3).to_string(index=False)); print()
        print(R.pivot_table(index=['arm', 'lo'], columns='label', values='ratio_med').round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['reco', 'verify', 'build', 'compare', 'scint', 'kbeam'])
    ap.add_argument('--work', default=str(WORK))
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--sub', default='stat090_0000')
    ap.add_argument('--min', type=int, default=3)
    ap.add_argument('--arms', nargs='+', default=list(ARMS))
    ap.add_argument('--arm', default='A')
    ap.add_argument('--tags', nargs='*', default=None)
    ap.add_argument('--tag', default='000')
    ap.add_argument('--jobs', type=int, default=15)
    ap.add_argument('--bundles', default=None, help='A=path,C=path: replace the full-pass bundles')
    ap.add_argument('--label', default=None, help='output dir name (default m<min> / prod)')
    ap.add_argument('--k-one', action='store_true', help='build with k = 1 (in-situ bundles)')
    ap.add_argument('--labels', nargs='+', default=['prod', 'm3'])
    a = ap.parse_args()
    bundles = dict(kv.split('=', 1) for kv in a.bundles.split(',')) if a.bundles else None
    work = _guard(a.work)
    if a.step == 'reco':
        reco(work, a.run, a.sub, a.min, a.arms, a.tags, a.jobs, bundles, a.label)
    elif a.step == 'verify':
        tag = tags_of(a.run, a.sub, a.arm, [a.tag])[0]
        return verify(work, a.run, a.sub, a.arm, tag)
    elif a.step == 'build':
        build(work, a.run, a.sub, a.min, a.label, a.k_one)
    elif a.step == 'compare':
        compare(work, a.run, a.sub, a.min)
    elif a.step == 'kbeam':
        kbeam(work, a.run, a.sub, a.labels, tuple(a.arms) if a.arms != list(ARMS) else ('A', 'C'))
    elif a.step == 'scint':
        scint(work, a.run, a.sub, a.min, tuple(a.arms))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
