#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repass_readiness.py -- before the is2_v1 campaign re-pass (A and C), what does
the re-pass actually change, and is anything in it unvalidated?

Compares three chains on the same data, AS DELIVERED (raw tan x the k each
chain applies at stage 3):

    production   full-pass bundle (v 42.6), seeder min 5, capsule k_arm
    m3           production bundle with seeder min 3 (seed_beam_test.py, §10a)
    is2_v1       in-situ bundles is2_A / is2_C, min 3, k = muon norm x gas ratio

Steps (outputs under results/repass_readiness/):

    cosmic   run_149 A-C line truth, held-out (test) split: median delivered
             tan / true tan and MAD resolution per |tan| bin, production vs is2.
             Production = the production bundle at min 3 (reco_prod_s3_*) x
             run_145's k_arm; is2 = reco_is2_s3_* x kcal_is2_v1(run_145).
    beam     run_145 stat090_0000 (the is2_v1 smoke sub-run): gated counts,
             the per-track shift is2/prod by |tan|, the raw-tan plausibility
             cut audit (wft.reco TAN_MAX) and scintillator confirmation
             (det_a_scint.match_run; production and m3 tables are the cached
             ones from seed_beam_test.py scint).

    python ntof_cosmics/repass_readiness.py cosmic
    python ntof_cosmics/repass_readiness.py beam

Why the TAN_MAX audit: `wft.reco._candidate_score` calls a fit plausible only
if |raw tan| < TAN_MAX = 0.6, in the bundle's own raw units.  Raw tan scales
as 1/v, so on the v = 42.6 production bundles 0.6 raw meant 0.6 x k_arm true
(A 0.76, C 0.97); on the in-situ bundles it means ~0.58 true.  Implausible
fits lose the candidate ranking and fail the x/y pairing gate, so the cut is
an angular acceptance, and the re-pass changes it.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

OUT = HERE / 'results' / 'repass_readiness'
INSITU = Path.home() / 'scratch' / 'ntof_insitu'
SP = Path('/media/dylan/data/x17/sept26_prelim')
PROD_S3 = SP / 'stage3_fullpass'
IS2_S3 = SP / 'smoke_is2_v1' / 'stage3'
KCAL = SP / 'kcal_is2_v1'
SEEDCMP = INSITU / 'beamseed' / 'compare'
RUN, SUB = 'run_145', 'stat090_0000'

BINS = [(0.04, 0.08), (0.08, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.60)]
SHIFT_BINS = [0.03, 0.08, 0.15, 0.25, 0.35, 0.5, 0.7]
TAN_MAX = 0.6                       # wft.reco.TAN_MAX, raw units
U_WIN = (250.0, 1100.0)             # wft.reco U_MIN_NS / U_MAX_NS
TAN_MAX_SCAN = (0.6, 0.8, 1.0, 1.2)


def _mad(v) -> float:
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def _k_prod(arm: str) -> float:
    meta = json.loads((PROD_S3 / f'tracks_{RUN}_{SUB}.meta.json').read_text())
    return float(meta['k_arm']['applied'][arm])


def _k_is2(arm: str) -> float:
    return float(json.loads((KCAL / f'k_arm_{RUN}.json').read_text())['apply'][arm])


# --------------------------------------------------------------------------- #
def cosmic() -> pd.DataFrame:
    T = pd.read_parquet(INSITU / 'truth.parquet')
    T = T[~T.train]
    rows = []
    for arm in 'AC':
        for chain, lab, k in (('production', 'prod_s3', _k_prod(arm)), ('is2_v1', 'is2_s3', _k_is2(arm))):
            f = INSITU / f'reco_{lab}_{arm}.parquet'
            if not f.exists():          # C's production-bundle min-3 table is called reco_prod_C
                f = INSITU / f'reco_prod_{arm}.parquet'
            M = T[T.arm == arm].merge(pd.read_parquet(f), on=['subrun', 'event_id'])
            for ax in 'xy':
                t = M[f'tan_{ax}'].to_numpy()
                r = k * M[f'{ax}_tan_theta'].to_numpy()
                ok = np.isfinite(r) & np.isfinite(t)
                for lo, hi in BINS:
                    m = ok & (np.abs(t) >= lo) & (np.abs(t) < hi)
                    rr = r[m] / t[m]
                    rows.append(dict(arm=arm, axis=ax, chain=chain, k=k, lo=lo, hi=hi, n=int(m.sum()),
                                     ratio=float(np.median(rr)) if m.any() else np.nan,
                                     ratio_err=float(1.2533 * _mad(rr) / np.sqrt(m.sum())) if m.sum() > 2 else np.nan,
                                     sigma=_mad(r[m] - t[m]),
                                     tail=float((np.abs(r[m] - t[m]) > 0.15).mean()) if m.any() else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'cosmic_closure.csv', index=False)
    with pd.option_context('display.width', 200):
        print(R.pivot_table(index=['arm', 'axis', 'lo'], columns='chain', values=['ratio', 'sigma']).round(3))
    return R


# --------------------------------------------------------------------------- #
COLS = ['tag', 'event_id', 'arm', 'gated', 'x_p0', 'y_p0', 'tanx', 'tany', 'x_tan_theta', 'y_tan_theta',
        'x_plausible', 'y_plausible', 'x_quality_ok', 'y_quality_ok', 'x_q_uend', 'y_q_uend', 'k_arm',
        'coinc_this_arm', 'x_t0']


def _tracks(src: Path) -> pd.DataFrame:
    return pd.read_parquet(src / f'tracks_{RUN}_{SUB}.parquet', columns=COLS)


def beam():
    P, I = _tracks(PROD_S3), _tracks(IS2_S3)
    shift, gate, scan = [], [], []
    for arm in 'AC':
        # per-track shift: same particle (p0 within 2 mm in both planes)
        Pg, Ig = P[(P.arm == arm) & P.gated], I[(I.arm == arm) & I.gated]
        m = Pg.merge(Ig, on=['tag', 'event_id'], suffixes=('_p', '_i'))
        m = m[((m.x_p0_p - m.x_p0_i).abs() < 2) & ((m.y_p0_p - m.y_p0_i).abs() < 2)]
        for ax in 'xy':
            a, b = m[f'tan{ax}_p'].abs(), m[f'tan{ax}_i'].abs()
            for lo, hi in zip(SHIFT_BINS[:-1], SHIFT_BINS[1:]):
                s = (a >= lo) & (a < hi)
                shift.append(dict(arm=arm, axis=ax, lo=lo, hi=hi, n=int(s.sum()),
                                  is2_over_prod=float(np.median(b[s] / a[s])) if s.any() else np.nan))
        # the raw-tan plausibility cut
        for chain, T in (('production', P), ('is2_v1', I)):
            g = T[T.arm == arm]
            k = float(g.k_arm.median())
            base = (g.x_quality_ok & g.y_quality_ok & g.x_q_uend.between(*U_WIN) & g.y_q_uend.between(*U_WIN))
            r = np.maximum(g.x_tan_theta.abs(), g.y_tan_theta.abs())
            over = base & (r >= TAN_MAX)
            under = base & (r < TAN_MAX) & (r >= 0.3)
            gate.append(dict(arm=arm, chain=chain, k=k, true_tan_reach=TAN_MAX * k,
                             rows=len(g), gated=int(g.gated.sum()),
                             quality_ok=int(base.sum()), fail_tan_only=int(over.sum()),
                             fail_tan_frac=float(over.sum() / base.sum()),
                             arm_scint_over=float(g[over].coinc_this_arm.fillna(False).astype(bool).mean()),
                             arm_scint_03_06=float(g[under].coinc_this_arm.fillna(False).astype(bool).mean()),
                             late_over=float((g[over].x_t0 > 300).mean())))
            for tm in TAN_MAX_SCAN:
                scan.append(dict(arm=arm, chain=chain, tan_max_raw=tm, true_tan_reach=tm * k,
                                 candidates=int((base & (r < tm)).sum())))
    S, G, C = pd.DataFrame(shift), pd.DataFrame(gate), pd.DataFrame(scan)
    S.to_csv(OUT / 'beam_shift.csv', index=False)
    G.to_csv(OUT / 'gate_tanmax.csv', index=False)
    C.to_csv(OUT / 'gate_tanmax_scan.csv', index=False)
    with pd.option_context('display.width', 220):
        print(S.pivot_table(index=['arm', 'axis'], columns='lo', values='is2_over_prod').round(3), '\n')
        print(G.round(3).to_string(index=False), '\n')
        print(C.pivot_table(index=['arm', 'chain'], columns='tan_max_raw', values='candidates'), '\n')
    scint()


def scint():
    from sept26_prelim_analysis import det_a_scint as DS
    od = SEEDCMP / RUN / SUB
    rows = []
    for arm in 'AC':
        f = od / f'scint{arm}_is2v1.parquet'
        if f.exists():
            Q = pd.read_parquet(f)
        else:
            Q, _ = DS.match_run(RUN, [SUB], IS2_S3, None, DS.layer_geometry(RUN, arm))
            Q.to_parquet(f, index=False)
        for chain, T in (('production', pd.read_parquet(od / f'scint{arm}_prod.parquet')),
                         ('m3', pd.read_parquet(od / f'scint{arm}_m3.parquet')),
                         ('is2_v1', Q)):
            w, p = T[T.on_wall], T[T.on_plas]
            rows.append(dict(arm=arm, chain=chain, gated=len(T), n_wall=len(w),
                             wall=float(w.match_wall.mean()), wall_ctrl=float(w.match_wall_ctrl.mean()),
                             wall_excess=int(w.match_wall.sum() - w.match_wall_ctrl.sum()),
                             n_plas=len(p), plas=float(p.match_plas.mean()),
                             plas_excess=int(p.match_plas.sum() - p.match_plas_ctrl.sum())))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'scint_confirmation.csv', index=False)
    print(R.round(3).to_string(index=False))


def main() -> int:
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('step', choices=['cosmic', 'beam', 'all'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step in ('cosmic', 'all'):
        cosmic()
    if a.step in ('beam', 'all'):
        beam()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
