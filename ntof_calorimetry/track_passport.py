#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_passport.py -- everything the n_TOF set-up measures about ONE track,
put together: the tracking (direction, position, pointing, drift), the MM gap
deposit (dE/dx in the gas), the wall (dE/dx in 3 mm PVT), the plastic (20 mm
PVT, a calorimeter for the stopping electrons), the liquid (behind the plastic:
did it get through?) and the time since the flash (the neutron's energy).

The point is a demonstrator of the detection capability, not X17 physics.
Four questions, each answered in simulation (Geant4 neutron-beam events, the
same sample `mm_edep_combo` uses, with the measured resolutions applied) and,
where the data can answer, on run_145 (5 Aug, >= 12 ms, the arm's own trigger,
single gated track):

P1 PATH LENGTH.  The tracking predicts the path in every layer: 30 mm x sec
   in the gap, 3 mm x sec in the wall, 20 mm x sec in the plastic for a
   particle that gets through it (the liquid fired).  Each deposit should scale
   as sec^1.  A test that the tracks and the counters see the same particle.
P2 TWO PARTICLES.  The MM and the wall each measure dE/dx independently.  A
   single lepton's two Landau fluctuations are independent, so any correlation
   between them comes from tracks that carry more than one particle (a pair,
   a delta, a Compton electron with its partner).  Geant4 says how much; the
   data say whether the two detectors agree on which tracks those are.
P3 ENERGY.  The kinetic energy at the gap from the plastic (+ wall, + the
   track angle, + the MM), kNN regression on Geant4, cross-validated.  Shows
   what the tracking adds to the calorimetry (the plastic path length).
P4 ORIGIN.  Capsule (He-3 capture in the capsule's Al / CFRP) vs elsewhere,
   and Compton vs pair conversion, from the geometry and the energies.

Then a "passport" per track: the numbers above with their uncertainties and
the classification, for a handful of example data tracks.

MM calibration on beam (the open item of MM_EDEP.md): the window curve and k
come from the BEAM simulation (generated at the beam gain), and the data/sim
scale from the forward fold on the beam data -- not the cosmic k.

    python -m ntof_calorimetry.track_passport          # -> <calo>/passport/
    python -m ntof_calorimetry.make_passport_report    # -> <calo>/passport/report.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry.mip_sample import OUT  # noqa: E402
from ntof_calorimetry import mm_edep_ana as MA  # noqa: E402
from ntof_calorimetry import mm_edep_combo as MC  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402

PP = OUT / 'passport'
EDEP = OUT / 'mm_edep'
RES = EDEP / 'results'
RUN = 'run_145'
ARMS = ('A', 'C')
LATE_NS = 12e6
FLIGHT_M = 19.5                     # capsule_tof.FLIGHT_M
M_N_KG, EV = 1.674927e-27, 1.602177e-19
WALL_MM, PLAS_MM, GAP_MM = 3.0, 20.0, 30.0
#: stack along the chamber normal (scint_stack geometry.csv, run_84): mm from the mesh
Z_WALL, Z_PLAS, Z_LIQ = 97.4, 190.0, 250.0
SAT_ADC = 3700
CAPSULE_DCA_MM = 25.0               # "points at the capsule": DCA to its axis
SEED = 17
T_THROUGH = 6.0                     # MeV at the gap: gets through 20 mm of plastic


# ------------------------------------------------------------------ helpers
def neutron_energy_ev(t_ns: np.ndarray) -> np.ndarray:
    v = FLIGHT_M / (np.asarray(t_ns, float) * 1e-9)
    return 0.5 * M_N_KG * v ** 2 / EV


def auc(score: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def slope_loglog(x: np.ndarray, y: np.ndarray, edges) -> dict:
    """Median of y in bins of x, and the power-law slope of the medians
    (weighted by bin counts).  Also the bootstrap error of the slope."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.digitize(x, edges)
    xm, ym, n = [], [], []
    for i in range(1, len(edges)):
        m = b == i
        if m.sum() < 30:
            continue
        xm.append(np.median(x[m]))
        ym.append(np.median(y[m]))
        n.append(m.sum())
    xm, ym, n = map(np.asarray, (xm, ym, n))
    if len(xm) < 3:
        return dict(slope=np.nan, err=np.nan, x=xm, y=ym, n=n)
    p = np.polyfit(np.log(xm), np.log(ym), 1, w=np.sqrt(n))
    rng = np.random.default_rng(3)
    bs = []
    for _ in range(100):
        k = rng.integers(0, len(x), len(x))
        xx, yy, bb = x[k], y[k], b[k]
        xs, ys, ns = [], [], []
        for i in range(1, len(edges)):
            m = bb == i
            if m.sum() < 30:
                continue
            xs.append(np.median(xx[m]))
            ys.append(np.median(yy[m]))
            ns.append(m.sum())
        if len(xs) >= 3:
            bs.append(np.polyfit(np.log(xs), np.log(ys), 1, w=np.sqrt(ns))[0])
    return dict(slope=float(p[0]), err=float(np.std(bs)), norm=float(np.exp(p[1])), x=xm, y=ym, n=n)


# --------------------------------------------------------- MM, beam scale
def mm_beam_calibration(arm: str) -> dict:
    """k and window curve from the BEAM simulation, and the data/sim scale
    from the forward fold on the late beam data (with the self gain map).
    E_hat_data [keV of gap ionisation] = Q / (k_b c_b(f)) / s."""
    Sb = MA.load_sim_prof('beam', arm)
    Wb = MA.window_curve(Sb)
    kb = Wb.attrs['k']
    contained = arm == 'A'
    Sbq = MA._q(Sb, Wb, kb, contained)
    lr = np.log(Sbq.E_hat / Sbq.E_true)
    B = MA.with_plateau(MA.load_beam(arm, late=True))
    Bq = MA._q(B, Wb, kb, contained)
    g = pd.read_parquet(RES / f'beam_{arm}.parquet')[['subrun', 'event_id', 'track_id', 'q', 'q_c']]
    g = g.assign(lg=np.log(g.q / g.q_c))[['subrun', 'event_id', 'track_id', 'lg']]
    Bq = Bq.merge(g, on=['subrun', 'event_id', 'track_id'], how='left')
    Bq['lg'] = Bq.lg.fillna(0.0)
    f = MA.fold_fit((Bq.q / np.exp(Bq.lg)).to_numpy(), Sbq.q.to_numpy(), n_boot=30)
    return dict(arm=arm, k_beam=float(kb), window=Wb, sim_rsd=MA.rsd(lr), sim_bias=float(np.median(lr)),
                sim_w68=MA.w68(lr), scale=f['scale'], scale_err=f.get('scale_err', np.nan),
                sigma_x=f['sigma_x'], sigma_x_err=f.get('sigma_x_err', np.nan), n_data=len(Bq),
                sigma_total=float(np.hypot(MA.rsd(lr), f['sigma_x'])),
                mip_keV=float(np.median(Sbq.E_true / Sbq.sec)))


# ------------------------------------------------------------------- data
def _scale_fits() -> dict:
    """Data/Geant4 scale per arm: wall (all times), plastic per bar."""
    S = pd.read_csv(OUT / 'wp' / 'scale_fits.csv')
    out = {}
    for arm in ARMS:
        s = S[S.arm == arm]
        out[arm] = dict(wall=float(s[(s['var'] == 'e_wall_fit') & s.subset.isna()].scale.iloc[0]),
                        bar1=float(s[s.subset == 'bar 1'].scale.iloc[0]),
                        bar2=float(s[s.subset == 'bar 2'].scale.iloc[0]))
    return out


def data(cal: dict) -> pd.DataFrame:
    """Late run_145 A/C single tracks on their own arm's trigger, with every
    layer.  Energies put on the Geant4 scale (wall_plastic scale fits) so the
    simulation-trained estimators apply."""
    B = pd.read_pickle(EDEP / f'beam_{RUN}.pkl')
    B = B[B.arm.isin(ARMS) & (B.t_since_flash_ns >= LATE_NS)].copy()
    own = pd.Series(np.where(B.arm == 'A', B.hw_A, B.hw_C), index=B.index)
    B = B[own.fillna(False).astype(bool)]
    B = B.dropna(subset=['tanx', 'tany', 'e_wall', 'e_plas'])
    # liquid and the per-group wall pulses from the scint-stack table
    St = pd.read_parquet(paths.spell('scint', 'tracks', f'stack_{RUN}.parquet'),
                         columns=['subrun', 'event_id', 'arm', 'track_id', 'l_amp_on', 'l_sat_on', 'l_dt_on',
                                  'p1_dt_on', 'p2_dt_on', 'u_mm', 'v_mm', 'drift_railed', 'q_total'])
    St['arm'] = St.arm.astype(str)
    St['subrun'] = St.subrun.astype(str)
    B['subrun'] = B.subrun.astype(str)
    B = B.merge(St, on=['subrun', 'event_id', 'arm', 'track_id'], how='left')
    lq = pd.read_csv(OUT / 'c4' / 'mip_scale.csv').set_index('arm')
    B['liq_fired'] = B.l_amp_on.fillna(0) > B.arm.map(lq.zs_mv)
    B['e_liq'] = B.l_amp_on.fillna(0) / B.arm.map(lq.mv_per_mev_mip)
    sf = _scale_fits()
    B['e_wall_g4'] = B.e_wall / B.arm.map(lambda a: sf[a]['wall'])
    B['e_plas_g4'] = B.e_plas / np.where(B.bar == 1, B.arm.map(lambda a: sf[a]['bar1']),
                                         B.arm.map(lambda a: sf[a]['bar2']))
    # tracking
    B['sec'] = np.sqrt(1 + B.tanx ** 2 + B.tany ** 2)
    B['theta_deg'] = np.degrees(np.arccos(1 / B.sec))
    B['phi_deg'] = np.degrees(np.arctan2(B.tany, B.tanx))
    B['E_n_meV'] = neutron_energy_ev(B.t_since_flash_ns) * 1e3
    B['to_capsule'] = B.dca_axis_mm < CAPSULE_DCA_MM
    # MM: beam-calibrated gap deposit
    B['t0'] = B[['x_t0', 'y_t0']].mean(1)
    B['v'] = B.v_drift_um_ns
    B['Q'] = B.x_Q + B.y_Q
    B['f_obs'] = (MA.WIN_NS - B.t0) / (MA.GAP_UM / B.v)
    B['mm_sat'] = (B.x_raw_max > SAT_ADC) | (B.y_raw_max > SAT_ADC)
    B['E_mm'] = np.nan
    for arm in ARMS:
        c = cal[arm]
        m = (B.arm == arm).to_numpy()
        B.loc[m, 'E_mm'] = MA.apply_window(B[m], c['window'], c['k_beam']).to_numpy() / c['scale']
    g = pd.concat([pd.read_parquet(RES / f'beam_{a}.parquet').assign(arm=a) for a in ARMS])
    g = g.assign(lg=np.log(g.q / g.q_c), subrun=g.subrun.astype(str))[['subrun', 'event_id', 'track_id', 'arm', 'lg']]
    B = B.merge(g, on=['subrun', 'event_id', 'track_id', 'arm'], how='left')
    B['E_mm'] = B.E_mm / np.exp(B.lg.fillna(0.0))
    B['mm_ok'] = (np.isfinite(B.E_mm) & (B.E_mm > 0) & ~B.mm_sat & B.lg.notna()
                  & np.where(B.arm == 'A', B.f_obs > 1.35, True))
    B['q_mm'] = B.E_mm / B.sec                     # keV per 30 mm of depth
    B['sigma_mm'] = B.arm.map(lambda a: cal[a]['sigma_total'])
    return B.reset_index(drop=True)


# -------------------------------------------------------------- simulation
def sim(cal: dict, seed: int = SEED) -> pd.DataFrame:
    """Geant4 neutron-beam events through each arm's gap, with the measured
    resolutions; 'trig' is the plastic-fired category (the data's trigger)."""
    rng = np.random.default_rng(seed)
    G = MC.g4_sample()
    G = MC.smear_scint(G, MC.scint_cal(), rng)
    q = G.edep_gap.to_numpy(float) / G.sec.to_numpy()
    sig = G.armL.map(lambda a: cal[a]['sigma_total']).to_numpy()
    G['q_mm'] = q * np.exp(rng.normal(0, 1, len(G)) * sig)
    G['q_true'] = q
    G['trig'] = G.cat == 'plastic'
    G['multi'] = G.dom_share < 0.8
    G['conv'] = G.lead_op == 'conv'
    G['capsule'] = G.lead_ov.astype(str).str.startswith('He3Cap')
    G['E_n_meV'] = G.neutron_E_eV * 1e3
    # no liquid in this Geant4 reduction: 'through' the 20 mm plastic is
    # taken from the kinetic energy at the gap -- ~1.2 MeV is lost before the
    # plastic (C2) and 20 mm PVT stops ~4 MeV, so T > 6 MeV gets through
    G['through'] = G.lead_ke0 > T_THROUGH
    return G.reset_index(drop=True)


def _rel(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.asarray(v, float) / np.median(ref)


# ------------------------------------------------------------- P1 path
SEC_EDGES = np.array([1.0, 1.02, 1.05, 1.09, 1.14, 1.2, 1.3, 1.45, 1.7, 2.2])


def path_length(D: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    rows = []
    Gt = G[G.trig]
    for arm in ARMS:
        d = D[D.arm == arm]
        g = Gt[Gt.armL == arm]
        dm = d[d.mm_ok]
        tests = [
            ('MM gap', 'data', dm.sec, dm.E_mm),
            ('MM gap', 'Geant4', g.sec, g.q_mm * g.sec),
            ('wall', 'data', d.sec, d.e_wall_g4),
            ('wall', 'Geant4', g.sec, g.e_wall),
            ('plastic, liquid fired', 'data', d[d.liq_fired].sec, d[d.liq_fired].e_plas_g4),
            ('plastic, liquid fired', 'Geant4 (T > 6 MeV)', g[g.through].sec, g[g.through].e_plas),
            ('plastic, all', 'data', d.sec, d.e_plas_g4),
            ('plastic, all', 'Geant4', g.sec, g.e_plas),
        ]
        for layer, src, x, y in tests:
            m = np.isfinite(x) & np.isfinite(y) & (y > 0)
            r = slope_loglog(x[m], y[m], SEC_EDGES)
            rows.append(dict(arm=arm, layer=layer, source=src, n=int(m.sum()), slope=r['slope'], err=r['err'],
                             sec_med=json.dumps(np.round(r['x'], 4).tolist()),
                             dep_med=json.dumps(np.round(r['y'], 4).tolist()), nbin=json.dumps(r['n'].tolist())))
    return pd.DataFrame(rows)


def path_length_cosmic() -> pd.DataFrame:
    """The clean version of P1: cosmic muons (MIPs, no scattering) in run_149.
    MM: the gap deposit E_hat (gain-map corrected) against sec, with Heed's
    own angle dependence (3 GeV muon, 0/30/45 deg) as the expectation.  Wall:
    `wall_plastic`'s MPV-vs-path fit, against Bichsel."""
    rows = []
    ph = pd.read_csv(RES / 'physics.csv')
    mu = ph[ph.particle.str.startswith('mu')].sort_values('theta')
    sec_h = 1 / np.cos(np.radians(mu.theta.to_numpy()))
    k_exp = float(np.polyfit(np.log(sec_h), np.log(mu.median_keV.to_numpy()), 1)[0])
    for arm in ARMS:
        c = pd.read_parquet(RES / f'cosmic_{arm}.parquet')
        E = (c.q_c * c.sec).to_numpy()
        r = slope_loglog(c.sec.to_numpy(), E, SEC_EDGES)
        rows.append(dict(arm=arm, layer='MM gap', source='cosmic data', n=len(c), slope=r['slope'], err=r['err'],
                         expected=k_exp, sec_med=json.dumps(np.round(r['x'], 4).tolist()),
                         dep_med=json.dumps(np.round(r['y'] / r['y'][0], 4).tolist())))
    W = pd.read_csv(OUT / 'wp' / 'wall_path.csv')
    for arm, w in W[W.arm.isin(ARMS)].groupby('arm'):
        sec = w.path_mm.to_numpy() / WALL_MM
        p = np.polyfit(np.log(sec), np.log(w.mpv_mv.to_numpy()), 1)
        pe = np.polyfit(np.log(sec), np.log(w.exp_rel.to_numpy()), 1)
        rel = w.mpv_mv.to_numpy() / w.mpv_mv.to_numpy()[0]
        rows.append(dict(arm=arm, layer='wall', source='cosmic data', n=int(w.n.sum()), slope=float(p[0]),
                         err=float(np.mean(w.err / w.mpv_mv) * np.sqrt(2) / np.log(sec.max() / sec.min())),
                         expected=float(pe[0]), sec_med=json.dumps(np.round(sec, 4).tolist()),
                         dep_med=json.dumps(np.round(rel, 4).tolist())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------- P2 two particles
def two_particles(D: pd.DataFrame, G: pd.DataFrame) -> tuple:
    """MM vs wall dE/dx (both normalised to their medians, per arm, and the
    wall corrected to normal incidence).  Spearman r and the 'both high'
    excess; in Geant4 the same, split by the truth (one lepton / several)."""
    from scipy.stats import spearmanr
    rows, pts = [], []
    for arm in ARMS:
        d = D[(D.arm == arm) & D.mm_ok]
        g = G[(G.armL == arm) & G.trig]
        for src, mm, wall, truth in (('data', d.q_mm, d.e_wall_g4 / d.sec, None),
                                     ('Geant4', g.q_mm, g.e_wall / g.sec, g.multi.to_numpy())):
            a, b = _rel(mm, mm), _rel(wall, wall)
            ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
            a, b = a[ok], b[ok]
            hi_a, hi_b = a > 1.6, b > 1.6
            row = dict(arm=arm, source=src, n=int(ok.sum()), spearman=float(spearmanr(a, b)[0]),
                       p_mm_hi=float(hi_a.mean()), p_wall_hi=float(hi_b.mean()),
                       p_both_hi=float((hi_a & hi_b).mean()),
                       lift=float((hi_a & hi_b).mean() / max(hi_a.mean() * hi_b.mean(), 1e-9)),
                       p_wall_hi_given_mm_hi=float(hi_b[hi_a].mean()) if hi_a.any() else np.nan)
            if truth is not None:
                t = truth[ok]
                row.update(frac_multi=float(t.mean()),
                           frac_multi_both_hi=float(t[hi_a & hi_b].mean()) if (hi_a & hi_b).any() else np.nan,
                           spearman_single=float(spearmanr(a[~t], b[~t])[0]),
                           auc_multi_mm=auc(a, t), auc_multi_wall=auc(b, t), auc_multi_sum=auc(np.log(a) + np.log(b), t))
            rows.append(row)
            pts.append(pd.DataFrame(dict(arm=arm, source=src, mm=a, wall=b,
                                         multi=truth[ok] if truth is not None else np.nan)))
    return pd.DataFrame(rows), pd.concat(pts, ignore_index=True)


# --------------------------------------------------------------- P3 energy
FEATURES = {
    'plastic': lambda X: np.c_[np.log1p(X.e_plas)],
    'plastic+wall': lambda X: np.c_[np.log1p(X.e_plas), np.log1p(X.e_wall)],
    'plastic+wall+angle': lambda X: np.c_[np.log1p(X.e_plas), np.log1p(X.e_wall), np.log(X.sec)],
    'plastic+wall+angle+MM': lambda X: np.c_[np.log1p(X.e_plas), np.log1p(X.e_wall), np.log(X.sec),
                                             np.log(np.clip(X.q_rel, 1e-3, None))],
}


def _feat_frame(X: pd.DataFrame, e_plas, e_wall, q, sec) -> pd.DataFrame:
    return pd.DataFrame(dict(e_plas=np.asarray(e_plas, float), e_wall=np.asarray(e_wall, float),
                             q_rel=np.asarray(q, float), sec=np.asarray(sec, float)), index=X.index)


def energy(G: pd.DataFrame) -> tuple:
    """Kinetic energy of the leading lepton at the gap, triggered tracks."""
    rows, preds = [], {}
    g = G[G.trig].reset_index(drop=True)
    F = _feat_frame(g, g.e_plas, g.e_wall, _rel(g.q_mm, g.q_mm), g.sec)
    y = g.lead_ke0.to_numpy(float)
    for name, fn in FEATURES.items():
        p = MC.knn_cv(fn(F), y, k=40)
        preds[name] = p
        for lab, m in (('all', np.ones(len(y), bool)), ('sec < 1.1', (g.sec < 1.1).to_numpy()),
                       ('sec >= 1.1', (g.sec >= 1.1).to_numpy()), ('T < 3 MeV', y < 3),
                       ('T 3-5 MeV', (y >= 3) & (y < 5)), ('T 5-7 MeV', (y >= 5) & (y < 7)),
                       ('T >= 7 MeV', y >= 7), ('one lepton', ~g.multi.to_numpy()),
                       ('several', g.multi.to_numpy())):
            if m.sum() < 50:
                continue
            res = p[m] - y[m]
            rows.append(dict(features=name, subset=lab, n=int(m.sum()), w68=MC.width68(res),
                             rel_w68=MC.width68(res / y[m].clip(0.3)), bias=float(np.median(res))))
    P = pd.DataFrame(dict(armL=g.armL, T=y, sec=g.sec, through=g.through, multi=g.multi,
                          **{f'T_{k}': v for k, v in preds.items()}))
    return pd.DataFrame(rows), P


# --------------------------------------------------------------- P4 origin
ORIGIN_FEATURES = {
    'energies (plastic, wall, MM)': lambda X: np.c_[np.log1p(X.e_plas), np.log1p(X.e_wall),
                                                    np.log(np.clip(X.q_rel, 1e-3, None))],
    'tracking (position, direction)': lambda X: np.c_[X.u0 / 100, X.v0 / 100, X.tu, X.tv],
    'tracking + energies': lambda X: np.c_[X.u0 / 100, X.v0 / 100, X.tu, X.tv, np.log1p(X.e_plas),
                                           np.log1p(X.e_wall), np.log(np.clip(X.q_rel, 1e-3, None))],
}


def origin(G: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for samp, g in (('triggered', G[G.trig]), ('all gap tracks', G)):
        g = g.reset_index(drop=True)
        F = pd.DataFrame(dict(e_plas=g.e_plas, e_wall=g.e_wall, q_rel=_rel(g.q_mm, g.q_mm),
                              u0=g.lead_u0, v0=g.lead_v0, tu=g.tan_gap_u, tv=g.tan_gap_v))
        for target in ('capsule', 'conv', 'multi'):
            y = g[target].to_numpy(bool)
            for name, fn in ORIGIN_FEATURES.items():
                p = MC.knn_cv(fn(F), y.astype(float), k=60)
                rows.append(dict(sample=samp, target=target, features=name, n=len(y), base_rate=float(y.mean()),
                                 auc=auc(p, y)))
    return pd.DataFrame(rows)


# --------------------------------------------------------- per-track apply
def apply_to_data(D: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    """Geant4-trained estimators on every data track: T at the gap (best
    feature set the track has), P(several particles in the gap), and the
    measured per-layer numbers -- the passport table."""
    from scipy.spatial import cKDTree
    g = G[G.trig].reset_index(drop=True)
    out = D.copy()
    out['q_rel'] = np.nan
    for arm in ARMS:
        m = (out.arm == arm) & out.mm_ok
        out.loc[m, 'q_rel'] = _rel(out.loc[m, 'q_mm'], out.loc[m, 'q_mm'])
    gq = _rel(g.q_mm, g.q_mm)
    Fg = _feat_frame(g, g.e_plas, g.e_wall, gq, g.sec)
    Fd = _feat_frame(out, out.e_plas_g4, out.e_wall_g4, out.q_rel, out.sec)

    def knn_apply(name, y, k=40, mask=None):
        Xg = FEATURES[name](Fg)
        mu, sd = Xg.mean(0), Xg.std(0)
        t = cKDTree((Xg - mu) / sd)
        res = np.full(len(out), np.nan)
        res_sd = np.full(len(out), np.nan)
        m = np.ones(len(out), bool) if mask is None else mask
        Xd = FEATURES[name](Fd[m])
        _, nn = t.query((Xd - mu) / sd, k=k)
        res[m] = y[nn].mean(1)
        res_sd[m] = y[nn].std(1)
        return res, res_sd

    T = g.lead_ke0.to_numpy(float)
    mm = out.mm_ok.to_numpy()
    t3, s3 = knn_apply('plastic+wall+angle', T)
    t4, s4 = knn_apply('plastic+wall+angle+MM', T, mask=mm)
    out['T_hat'] = np.where(mm, t4, t3)
    out['T_sd'] = np.where(mm, s4, s3)
    out['T_features'] = np.where(mm, 'plastic+wall+angle+MM', 'plastic+wall+angle')
    pm, _ = knn_apply('plastic+wall+angle+MM', g.multi.to_numpy(float), k=60, mask=mm)
    out['p_multi'] = pm
    pc, _ = knn_apply('plastic+wall+angle+MM', g.conv.to_numpy(float), k=60, mask=mm)
    out['p_conv'] = pc
    # simple classes, read off the measurements
    mip_p = 3.40 * out.sec
    out['cls'] = np.select(
        [out.mm_ok & (out.p_multi > 0.5),
         out.liq_fired,
         out.e_plas_g4 < 0.8 * mip_p],
        ['several particles', 'through the plastic', 'stopped in plastic (low T)'],
        'stopped in plastic')
    return out


def sim_predictions(G: pd.DataFrame) -> pd.DataFrame:
    """The same estimators on the simulation (cross-validated), so the data's
    prediction distributions can be compared like for like."""
    g = G[G.trig].reset_index(drop=True)
    F = _feat_frame(g, g.e_plas, g.e_wall, _rel(g.q_mm, g.q_mm), g.sec)
    X = FEATURES['plastic+wall+angle+MM'](F)
    return pd.DataFrame(dict(armL=g.armL, T=g.lead_ke0, sec=g.sec, multi=g.multi,
                             T_hat=MC.knn_cv(X, g.lead_ke0.to_numpy(float), k=40),
                             p_multi=MC.knn_cv(X, g.multi.to_numpy(float), k=60)))


def examples(P: pd.DataFrame, n_per: int = 2, seed: int = 5) -> pd.DataFrame:
    """Example tracks for the passport cards: per class, typical members
    (centre of the chamber, MM measured, clean drift)."""
    rng = np.random.default_rng(seed)
    base = P[P.mm_ok & (P.x_local.abs() < 120) & (P.y_local.abs() < 120)]
    picks = []
    want = [('through the plastic', base.liq_fired & (base.sec < 1.1)),
            ('stopped in plastic', (base.cls == 'stopped in plastic') & (base.sec < 1.15)),
            ('stopped in plastic (low T)', base.cls == 'stopped in plastic (low T)'),
            ('several particles', base.cls == 'several particles'),
            ('steep track', (base.sec > 1.5) & (base.cls != 'several particles')),
            ('off the capsule', (base.dca_axis_mm > 80) & (base.cls != 'several particles'))]
    for lab, m in want:
        c = base[m]
        if not len(c):
            continue
        k = rng.choice(len(c), min(n_per, len(c)), replace=False)
        picks.append(c.iloc[k].assign(example=lab))
    return pd.concat(picks, ignore_index=True)


# ------------------------------------------------------------------- main
CARD_COLS = ['example', 'arm', 'subrun', 'event_id', 'track_id', 't_since_flash_ns', 'E_n_meV', 'x_local', 'y_local',
             'tanx', 'tany', 'theta_deg', 'phi_deg', 'sec', 'dca_axis_mm', 'to_capsule', 't0', 'f_obs',
             'E_mm', 'q_mm', 'q_rel', 'sigma_mm', 'e_wall_g4', 'wgrp', 'e_plas_g4', 'bar', 'p_sat', 'liq_fired',
             'e_liq', 'T_hat', 'T_sd', 'T_features', 'p_multi', 'p_conv', 'cls', 'chi2dof_x', 'chi2dof_y']


def main() -> int:
    PP.mkdir(parents=True, exist_ok=True)
    cal = {a: mm_beam_calibration(a) for a in ARMS}
    calj = {a: {k: v for k, v in c.items() if k != 'window'} for a, c in cal.items()}
    for a in ARMS:
        cal[a]['window'].to_csv(PP / f'window_beam_{a}.csv', index=False)
    print('MM beam calibration', json.dumps(calj, indent=1, default=float))
    D = data(cal)
    G = sim(cal)
    print(f'data {len(D)} tracks ({D.mm_ok.sum()} with MM); Geant4 {len(G)} gap tracks, {G.trig.sum()} triggered')
    PL = path_length(D, G)
    PC = path_length_cosmic()
    PC.to_csv(PP / 'path_length_cosmic.csv', index=False)
    print(PC[['arm', 'layer', 'n', 'slope', 'err', 'expected']].round(3).to_string(index=False))
    TP, pts = two_particles(D, G)
    EN, EP = energy(G)
    OR = origin(G)
    P = apply_to_data(D, G)
    SP = sim_predictions(G)
    EX = examples(P)
    PL.to_csv(PP / 'path_length.csv', index=False)
    TP.to_csv(PP / 'two_particles.csv', index=False)
    pts.to_parquet(PP / 'two_particles_points.parquet')
    EN.to_csv(PP / 'energy.csv', index=False)
    EP.to_parquet(PP / 'energy_sim_pred.parquet')
    OR.to_csv(PP / 'origin.csv', index=False)
    keep = [c for c in CARD_COLS if c in P.columns and c != 'example']
    P[keep + ['mm_ok', 'x_Q', 'y_Q']].to_parquet(PP / 'passport_data.parquet')
    SP.to_parquet(PP / 'passport_sim.parquet')
    EX[[c for c in CARD_COLS if c in EX.columns] + ['x_prof', 'y_prof']].to_pickle(PP / 'examples.pkl')
    G[['armL', 'trig', 'cat', 'lead_ke0', 'e_wall', 'e_plas', 'q_mm', 'sec', 'multi', 'conv', 'capsule', 'through',
       'n_lep', 'dom_share', 'E_n_meV', 'lead_op', 'lead_ov']].to_parquet(PP / 'g4.parquet')
    summ = dict(cal=calj, n_data=int(len(D)), n_data_mm=int(D.mm_ok.sum()), n_g4=int(len(G)),
                n_g4_trig=int(G.trig.sum()),
                classes=P.groupby('arm').cls.value_counts(normalize=True).round(4).unstack().to_dict(orient='index'),
                liq_fired=P.groupby('arm').liq_fired.mean().round(4).to_dict(),
                to_capsule=P.groupby('arm').to_capsule.mean().round(4).to_dict())
    (PP / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        print(PL[['arm', 'layer', 'source', 'n', 'slope', 'err']].round(3).to_string(index=False))
        print(TP.round(3).to_string(index=False))
        print(EN[EN.subset.isin(['all', 'sec >= 1.1', 'several', 'T < 3 MeV', 'T >= 7 MeV'])].round(3).to_string(index=False))
        print(OR.round(3).to_string(index=False))
        print(json.dumps(summ['classes'], indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
