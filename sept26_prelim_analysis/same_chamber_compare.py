#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
same_chamber_compare.py -- same-chamber pairs, data against Geant4: how many
e+e- pairs we should have seen, how many we see, and the shapes that do not
depend much on efficiency.

Inputs
  data   ~/x17_local/same_chamber/events_tof.parquet   (per event: tracks in the
         arm after the cuts, time since the flash -- built by this module's
         ``events`` step from stage3_is2_v2_cf)
         ~/x17_local/same_chamber_late/{cf,is2}_{A,C}  det_a_intra, tracks >= 12 ms
  sim    <g4>/g4_pairs.parquet, g4_singles.parquet, g4_same_arm.json
         (same_chamber_g4.py over MX17_Full_Geant same_arm_pairs.py parts)

The measurements, in the order the report reads them:

1.  RATE-INDEPENDENT PAIR FRACTION.  The fraction of tracked events that hold a
    second track in the same chamber, against time since the flash.  Pairs from
    one capture come with their own trigger and are a constant fraction; two
    unrelated captures in one readout window scale with the instantaneous rate.
    A fit  f = a + b * rho  over 12-75 ms (rho = tracked events per ms) gives the
    correlated fraction ``a`` -- no vertex test, no mixing, and the per-track
    efficiency enters once.
2.  WHAT GEANT4 SAYS THOSE PAIRS ARE (class breakdown, per trigger).
3.  EXPECTED vs SEEN: a_G4 = (G4 pairs in acceptance) / (G4 singles in
    acceptance) x eps x f_two, with the per-track efficiency eps bracketed
    (wall-confirmed singles per trigger ... the efficiency map's centre) and
    f_two the overlay bench's both-tracks recovery for a separated pair.
4.  SHAPES: the opening angle of the correlated pairs (late real pairs minus the
    accidental share x the late event-mixed shape) against Geant4's
    reconstructed e+e- angle; the vertex position along the beam; and the
    wall-reach double ratio P(both legs confirmed | pair) / P(confirmed | single)^2,
    which carries the lepton energies and in which the per-leg confirmation
    efficiency cancels to first order.

    python -m sept26_prelim_analysis.same_chamber_compare events
    python -m sept26_prelim_analysis.same_chamber_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import paths

LOCAL = Path.home() / 'x17_local'
G4 = LOCAL / 'g4_same_arm' / 'out'
LATE = LOCAL / 'same_chamber_late'
EVENTS = LOCAL / 'same_chamber' / 'events_tof.parquet'
CF_STAGE3 = LOCAL / 'stage3_is2_v2_cf' / 'tracks_campaign.parquet'
ARMS = ('A', 'C')
TOF_EDGES_MS = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50,
                         55, 60, 67, 75], float)
LATE_MS = 12.0
#: per-track efficiency bracket: (wall-confirmed singles per trigger, data / G4;
#: scintillator-tagged map centre) -- filled at run time from the inputs
#: overlay-bench recovery of both tracks of a pair >= 24 mm apart (production
#: chain, run_145; STATUS 2026-09-14): A 47 %, C 39 %
F_TWO = {'A': 0.47, 'C': 0.39}
OPEN_BINS = np.arange(0.0, 90.1, 10.0)
#: the data's capsule relative to the CAD frame Geant4 uses.  It was +32.2
#: until 2026-10-13: that was the stage-3 y mirror (COORDINATES.md).
from sept26_prelim_analysis.same_chamber_physics import CAPSULE_SHIFT_MM  # noqa: E402
HALF_MIP_EV = 0.5 * 0.4751e6
#: The reconstruction's own response to Geant4 events (ntof_cosmics/g4_digi,
#: g4_<arm>_is2_full: nose-campaign steps digitised into quiet run_145 triggers,
#: reconstructed with the is2 bundles; one-sided search, before is2_v2):
#: P(>= 1 track | 1 true lepton crossing >= 20 mm of the gap) and
#: P(2 tracks | 2 such leptons).  Measured 2026-10-13 on 15.4k (A) / 15.3k (C)
#: events, 420 / 466 of them with two leptons.
DIGI = {'A': dict(p1=0.537, p2=0.163, n2=420), 'C': dict(p1=0.451, p2=0.142, n2=466)}


def events(src: Path = CF_STAGE3) -> pd.DataFrame:
    """Per (arm, event): tracks after the analysis cuts, and time since the flash."""
    import pyarrow.parquet as pq
    from sept26_prelim_analysis.campaign_imaging import PRE_ACCESS_RUNS
    d = pq.read_table(src, columns=['run', 'subrun', 'event_id', 'arm', 'gated_reco', 'tanx',
                                    'tany', 'x_t0', 'y_t0', 'angle_calibrated',
                                    't_since_flash_ns']).to_pandas()
    d = d[~d.run.isin(PRE_ACCESS_RUNS)]
    ok = (d.gated_reco.fillna(False).astype(bool) & (d.tanx.abs() < 0.6) & (d.tany.abs() < 0.6)
          & (d.x_t0 <= 300) & (d.y_t0 <= 300) & d.angle_calibrated.fillna(False).astype(bool))
    d = d[ok & d.arm.isin(ARMS)]
    E = (d.groupby(['arm', 'run', 'subrun', 'event_id'])
         .agg(n=('arm', 'size'), tof=('t_since_flash_ns', 'first')).reset_index())
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    E.to_parquet(EVENTS, index=False)
    return E


def tof_fraction(E: pd.DataFrame):
    rows, fits = [], []
    for arm in ARMS:
        g = E[E.arm == arm]
        ms = g.tof.to_numpy() / 1e6
        i = np.digitize(ms, TOF_EDGES_MS) - 1
        for k in range(len(TOF_EDGES_MS) - 1):
            s = g[i == k]
            n1, n2 = len(s), int((s.n >= 2).sum())
            w = TOF_EDGES_MS[k + 1] - TOF_EDGES_MS[k]
            rows.append(dict(arm=arm, lo=TOF_EDGES_MS[k], hi=TOF_EDGES_MS[k + 1], n1=n1, n2=n2,
                             frac=n2 / max(n1, 1), err=np.sqrt(max(n2, 1)) / max(n1, 1),
                             rho=n1 / w))
        R = pd.DataFrame([r for r in rows if r['arm'] == arm])
        L = R[R.lo >= LATE_MS]
        w = 1 / L.err ** 2
        X = np.c_[np.ones(len(L)), L.rho]
        cov = np.linalg.inv(X.T @ (X * w.to_numpy()[:, None]))
        b = cov @ X.T @ (w * L.frac)
        chi = float(((L.frac - X @ b) ** 2 * w).sum())
        n1, n2 = L.n1.sum(), L.n2.sum()
        acc_share = float(b[1] * np.average(L.rho, weights=L.n1) / (n2 / n1))
        fits.append(dict(arm=arm, late_ms=LATE_MS, events=int(n1), pairs=int(n2),
                         frac=n2 / n1, a=float(b[0]), a_err=float(np.sqrt(cov[0, 0])),
                         b=float(b[1]), b_err=float(np.sqrt(cov[1, 1])), chi2=chi,
                         ndf=len(L) - 2, accidental_share=acc_share,
                         correlated_pairs=float(b[0] * n1)))
    return pd.DataFrame(rows), pd.DataFrame(fits)


def g4_tables():
    P = pd.read_parquet(G4 / 'g4_pairs.parquet')
    S = pd.read_parquet(G4 / 'g4_singles.parquet')
    J = json.load(open(G4 / 'g4_same_arm.json'))
    n_trig = J['counts']['trig.events']
    rows = []
    for arm in ARMS:
        p = P[(P.arm == arm) & P.trig]
        s = S[(S.arm == arm) & S.trig & S.acc]
        for c, g in p.groupby('cls'):
            rows.append(dict(arm=arm, cls=c, n_true=len(g), acc=g.p_acc.sum(),
                             vtx=g.p_vtx.sum(), per_trigger_acc=g.p_acc.sum() / n_trig,
                             open_true_med=g.open_true.median(),
                             both_reach=float(np.average(
                                 (g.wall_eV1 >= HALF_MIP_EV) & (g.wall_eV2 >= HALF_MIP_EV),
                                 weights=g.p_acc)) if g.p_acc.sum() > 0 else np.nan))
        rows.append(dict(arm=arm, cls='(singles in acceptance)', n_true=len(s), acc=float(len(s)),
                         per_trigger_acc=len(s) / n_trig,
                         both_reach=float((s.wall_eV >= HALF_MIP_EV).mean())))
    T = pd.DataFrame(rows)
    return P, S, T, n_trig


def data_headline(chain: str = 'cf', late: bool = True) -> dict:
    out = {}
    base = LATE if late else LOCAL / 'same_chamber'
    for arm in ARMS:
        H = pd.read_csv(base / f'{chain}_{arm}' / 'headline_per_run.csv')
        out[arm] = dict(triggers=int(H.n_triggers.sum()), tracks=int(H.n_tracks.sum()),
                        wall=int(H.n_trk_wall.sum()), wallctrl=int(H.n_trk_wallctrl.sum()),
                        onwall=int(H.n_trk_onwall.sum()), map_eff=float(np.nan))
        M = pd.read_csv(base / f'{chain}_{arm}' / 'eff_map_2d.csv')
        c = M[(M.u_mid.abs() < 120) & (M.v_mid.abs() < 100)]
        out[arm]['map_eff'] = float(c.eff_track.median())
        out[arm]['tag_eff'] = float(np.average(H.efficiency, weights=H.n_tagged))
    return out


def expectation(T: pd.DataFrame, F: pd.DataFrame, n_trig: int, DH_all: dict,
                late_vtx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        t = T[T.arm == arm]
        s = t[t.cls == '(singles in acceptance)'].iloc[0]
        pr = t[t.cls != '(singles in acceptance)']
        ratio = pr.acc.sum() / s.acc
        ratio_ee = pr[pr.cls == 'e+e- (one gamma)'].acc.sum() / s.acc
        ratio_vtx = pr.vtx.sum() / s.acc
        reach = float(s.both_reach)
        h = DH_all[arm]
        g4_single_per_trig = s.acc / n_trig
        eps = {'wall-confirmed singles': (h['wall'] - h['wallctrl']) / h['triggers']
               / (g4_single_per_trig * reach),
               'all singles': h['tracks'] / h['triggers'] / g4_single_per_trig,
               'tagged efficiency map, centre': h['map_eff']}
        f = F[F.arm == arm].iloc[0]
        v = late_vtx[late_vtx.arm == arm]
        # the direct route: Geant4 events per tracked event with >= 2 leptons in
        # the gap (|tan| < 0.6), times the reconstruction's measured pair/single
        # response -- no efficiency bracket
        Sg = pd.read_parquet(G4 / 'g4_singles.parquet')
        sa = Sg[(Sg.arm == arm) & Sg.trig & Sg.acc].groupby(['file', 'event']).size()
        r2 = float((sa >= 2).sum() / len(sa))
        dg = DIGI[arm]
        pred = r2 * dg['p2'] / dg['p1']
        rows.append(dict(arm=arm, eps_source='digitised Geant4 through the reconstruction',
                         eps=dg['p1'], two_track='measured (digitised)', f_two=dg['p2'] / dg['p1'] ** 2,
                         g4_pairs_per_single=r2, g4_ee_share=ratio_ee / ratio,
                         predicted_frac=pred, data_frac=f.a, data_err=f.a_err,
                         data_over_g4=f.a / pred, predicted_vertexed=np.nan,
                         data_vertex_excess=float(v.excess.iloc[0]) if len(v) else np.nan,
                         data_vertex_err=float(v.err.iloc[0]) if len(v) else np.nan))
        for name, e in eps.items():
            for two, ft in (('no two-track loss', 1.0), ('bench two-track recovery', F_TWO[arm])):
                pred = ratio * e * ft
                rows.append(dict(arm=arm, eps_source=name, eps=e, two_track=two, f_two=ft,
                                 g4_pairs_per_single=ratio, g4_ee_share=ratio_ee / ratio,
                                 predicted_frac=pred, data_frac=f.a, data_err=f.a_err,
                                 data_over_g4=f.a / pred,
                                 predicted_vertexed=ratio_vtx * e * ft * f.events,
                                 data_vertex_excess=float(v.excess.iloc[0]) if len(v) else np.nan,
                                 data_vertex_err=float(v.err.iloc[0]) if len(v) else np.nan))
    return pd.DataFrame(rows)


def late_vertex(chain: str = 'cf') -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        V = pd.read_csv(LATE / f'{chain}_{arm}' / 'vertex_excess.csv').set_index('selection')
        r = V.loc['all']
        rows.append(dict(arm=arm, chain=chain, k_real=int(r.k_real), n_real=int(r.n_real),
                         expected=float(r.rate_mixed * r.n_real),
                         excess=float(r.k_real - r.rate_mixed * r.n_real),
                         err=float(np.sqrt(max(r.k_real, 1))), ratio=float(r.ratio),
                         ratio_err=float(r.err)))
    return pd.DataFrame(rows)


def late_pairs(chain: str, arm: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    f = LATE / f'{chain}_{arm}' / 'pairs.parquet'
    if 'y_up' not in pq.read_schema(f).names:
        raise ValueError(f'{f} predates the 2026-10-13 y fix (vy_mm mirrored); rebuild it')
    return pd.read_parquet(f,
                           columns=['mixed', 'vertex', 'open_deg', 'vy_mm', 'n_legs_wall',
                                    'wall', 'wall2', 'onwall', 'prompt', 'sep_plane_mm'])


def opening(P: pd.DataFrame, F: pd.DataFrame, chain: str = 'cf') -> pd.DataFrame:
    """Correlated-pair opening angle: late real minus accidental share x mixed."""
    rows = []
    mid = 0.5 * (OPEN_BINS[:-1] + OPEN_BINS[1:])
    for arm in ARMS:
        D = late_pairs(chain, arm)
        R, M = D[~D.mixed], D[D.mixed]
        share = float(F[F.arm == arm].accidental_share.iloc[0])
        hr, _ = np.histogram(R.open_deg, OPEN_BINS)
        hm, _ = np.histogram(M.open_deg, OPEN_BINS)
        acc = share * len(R) * hm / max(hm.sum(), 1)
        corr = hr - acc
        err = np.sqrt(hr + (share * len(R)) ** 2 * hm / max(hm.sum(), 1) ** 2)
        g = P[(P.arm == arm) & P.trig]
        ge, _ = np.histogram(g[g.cls == 'e+e- (one gamma)'].open_reco, OPEN_BINS,
                             weights=g[g.cls == 'e+e- (one gamma)'].p_acc)
        ge2, _ = np.histogram(g[g.cls == 'e+e- (one gamma)'].open_reco, OPEN_BINS,
                              weights=g[g.cls == 'e+e- (one gamma)'].p_acc ** 2)
        ga, _ = np.histogram(g.open_reco, OPEN_BINS, weights=g.p_acc)
        ga2, _ = np.histogram(g.open_reco, OPEN_BINS, weights=g.p_acc ** 2)
        # high-statistics cross-check: Geant4's scattered conversion angle (all
        # 71k Al pairs) thrown through the toy acceptance, no vertex test
        T = pd.read_csv(paths.out('same_chamber') / 'physics_folded_models.csv')
        tc = T[(T.arm == arm) & (T.model == 'Al conversion')].set_index('mid').accepted
        tc = tc.reindex(mid).fillna(0).to_numpy()
        for k in range(len(mid)):
            rows.append(dict(arm=arm, chain=chain, mid=mid[k], real=int(hr[k]), mixed=int(hm[k]),
                             accidental=acc[k], correlated=corr[k], err=err[k],
                             g4_ee=ge[k], g4_ee_var=ge2[k], g4_all=ga[k], g4_all_var=ga2[k],
                             toy_conv=tc[k], toy_conv_var=0.0, mixed_var=float(hm[k]),
                             real_var=float(hr[k]), accidental_share=share))
    return pd.DataFrame(rows)


def shape_chi2(O: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, chain), o in O.groupby(['arm', 'chain']):
        live = o.real.to_numpy() > 0
        k, s = o.correlated.to_numpy(), o.err.to_numpy()
        for model in ('g4_ee', 'g4_all', 'toy_conv', 'mixed'):
            m = o[model].to_numpy(float)
            if m.sum() <= 0:
                continue
            scale = k.sum() / m.sum()
            pred = m * scale
            var = s ** 2 + o[model + '_var'].to_numpy(float) * scale ** 2
            chi = float(((k - pred) ** 2 / np.where(var > 0, var, 1))[live].sum())
            rows.append(dict(arm=arm, chain=chain, model=model, chi2=chi, ndf=int(live.sum()) - 1,
                             median_corr=float(np.interp(0.5, np.cumsum(np.clip(k, 0, None))
                                                         / max(np.clip(k, 0, None).sum(), 1e-9),
                                                         o.mid)),
                             median_model=float(np.interp(0.5, np.cumsum(m) / m.sum(), o.mid))))
    return pd.DataFrame(rows)


def wall_double_ratio(P: pd.DataFrame, T: pd.DataFrame, F: pd.DataFrame, DH: dict,
                      chain: str = 'cf') -> pd.DataFrame:
    """P(both legs confirmed | correlated pair) / P(confirmed | single)^2."""
    rows = []
    for arm in ARMS:
        D = late_pairs(chain, arm)
        R, M = D[~D.mixed], D[D.mixed]
        share = float(F[F.arm == arm].accidental_share.iloc[0])
        both_r = R.wall.mean()
        both_m = M.wall.mean()
        both_corr = (both_r - share * both_m) / (1 - share)
        h = DH[arm]
        p1 = h['wall'] / h['tracks']
        t = T[T.arm == arm]
        g_ee = t[t.cls == 'e+e- (one gamma)'].iloc[0]
        g_s = t[t.cls == '(singles in acceptance)'].iloc[0]
        rows.append(dict(arm=arm, chain=chain, data_both_real=both_r, data_both_mixed=both_m,
                         data_both_correlated=both_corr, data_single=p1,
                         data_double_ratio=both_corr / p1 ** 2,
                         g4_both_ee=g_ee.both_reach, g4_single=g_s.both_reach,
                         g4_double_ratio=g_ee.both_reach / g_s.both_reach ** 2))
    return pd.DataFrame(rows)


def vertex_y(P: pd.DataFrame, chain: str = 'cf') -> pd.DataFrame:
    edges = np.arange(-150.0, 151.0, 25.0)
    mid = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for arm in ARMS:
        D = late_pairs(chain, arm)
        R, M = D[~D.mixed], D[D.mixed]
        s = len(R) / max(len(M), 1)
        hr, _ = np.histogram(R[R.vertex].vy_mm, edges)
        hm, _ = np.histogram(M[M.vertex].vy_mm, edges)
        g = P[(P.arm == arm) & P.trig & P.vy_med.notna()]
        hg, _ = np.histogram(g.vy_med + CAPSULE_SHIFT_MM, edges, weights=g.p_vtx)
        for k in range(len(mid)):
            rows.append(dict(arm=arm, mid=mid[k], real=int(hr[k]), expected=hm[k] * s,
                             excess=hr[k] - hm[k] * s, g4=hg[k]))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('step', nargs='?', default='all', choices=['events', 'all'])
    ap.add_argument('--out', default=str(paths.out('same_chamber')))
    a = ap.parse_args()
    if a.step == 'events':
        E = events()
        print(E.groupby('arm').n.value_counts().unstack().iloc[:, :4])
        return 0
    od = Path(a.out)
    E = pd.read_parquet(EVENTS)
    R, F = tof_fraction(E)
    P, S, T, n_trig = g4_tables()
    DH_all = data_headline('cf', late=False)
    DH_late = data_headline('cf', late=True)
    V = pd.concat([late_vertex('cf'), late_vertex('is2')], ignore_index=True)
    X = expectation(T, F, n_trig, DH_all, V[V.chain == 'cf'])
    O = pd.concat([opening(P, F, 'cf'), opening(P, F, 'is2')], ignore_index=True)
    C = shape_chi2(O)
    W = pd.concat([wall_double_ratio(P, T, F, DH_late, c) for c in ('cf', 'is2')],
                  ignore_index=True)
    Y = vertex_y(P, 'cf')
    for name, df in (('cmp_tof_fraction', R), ('cmp_tof_fit', F), ('cmp_g4_classes', T),
                     ('cmp_expectation', X), ('cmp_late_vertex', V), ('cmp_opening', O),
                     ('cmp_opening_chi2', C), ('cmp_wall_double_ratio', W),
                     ('cmp_vertex_y', Y)):
        df.to_csv(od / f'{name}.csv', index=False)
    json.dump(dict(n_trig_g4=n_trig, data_all=DH_all, data_late=DH_late, f_two=F_TWO,
                   late_ms=LATE_MS), open(od / 'cmp.meta.json', 'w'), indent=1)
    pd.set_option('display.width', 220)
    for name, df in (('ToF fit', F), ('G4 classes', T), ('expectation', X), ('late vertex', V),
                     ('opening chi2', C), ('wall double ratio', W)):
        print(f'\n== {name}\n{df.round(4).to_string(index=False)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
