#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm_edep_sim.py -- how accurately the MM measures the energy a track deposits
in its 30 mm gap (MM_EDEP.md, stage 2): known deposits through the real
measurement chain, measured against their truth.

TRUTH.  Heed tracks (`heed_edep.py`, Ar/iC4H10 90/10): energy lost in the gap,
conduction electrons, and their depth profile (0.5 mm bins).

CHAIN, per track (ntof_cosmics/g4_digi/digitise.py's pieces, unchanged):
  geometry  copied from a REAL data track: its calibrated slopes, position,
            t0 and drift velocity, so window truncation, edge effects and the
            angular mix are the data's own; the Heed track is the one of the
            pool whose angle is nearest
  drift     transverse diffusion of the bundle; Polya gain per electron
  response  the bundle's impulse template, resistive kernel, per-channel gain
  readout   added to a real quiet run_145 trigger's raw ADC, clipped at the rail
  reco      the production seeder + forward-model fit (for the road)
  charge    `mm_charge`'s estimator exactly: pedestal, common mode from the
            block's channels OUTSIDE the road, road sum over all samples

Stored per track: the truth at each stage (e_dep -> n_e -> sum of avalanche
gains -> noise-free road signal -> measured), so the measurement error can be
split into ionisation statistics, avalanche, window/road containment and
noise/common mode.

    python -m ntof_calorimetry.mm_edep_sim cosmic --arm A      # run_149 geometry, mu
    python -m ntof_calorimetry.mm_edep_sim beam --arm A        # run_145 geometry, e-
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry.mip_sample import OUT  # noqa: E402
from ntof_calorimetry import mm_charge as MC  # noqa: E402
from ntof_cosmics.g4_digi import digitise as DG  # noqa: E402

HEED = OUT / 'heed'
EDEP = OUT / 'mm_edep'
BUNDLES = Path.home() / 'scratch' / 'ntof_insitu' / 'bundles'
HALF = 199.29
BIN_MM = 0.5
GAP = 30.0
#: x/y split of the avalanche (data: median Qx/Qy 0.976 A, 0.964 C on cosmics)
XY_SPLIT = {'A': 0.494, 'C': 0.491}
#: kinetic energy of the beam electrons AT THE GAP (Geant4 neutron beam,
#: g4_digi g4_A_is2_full dom_ke_gap: 5-95 % 1.1-6.5 MeV, median 2.9), binned
#: onto the Heed pools: (<0.75] 0.5, (0.75-1.5] 1, (1.5-3] 2, (3-6] 4, (>6] 8
BEAM_MIX = {0.5: 0.014, 1.0: 0.136, 2.0: 0.375, 4.0: 0.385, 8.0: 0.09}


# ------------------------------------------------------------------ inputs
def heed_pool(particle: str, T: float) -> dict:
    z = np.load(HEED / f'{particle}_{T:g}_thc.npz')
    return {k: z[k] for k in ('e_dep', 'n_e', 'n_clu', 'prof', 'theta_trk')}


def geometry_cosmic(arm: str) -> pd.DataFrame:
    """run_149 single gated tracks (mm_charge's table: same selection as the data)."""
    D = pd.read_pickle(MC.M2 / 'charge_run149.pkl')
    D = D[D.arm == arm].dropna(subset=['tanx', 'tany', 'x_t0', 'y_t0', 'v_drift_um_ns'])
    return D[['subrun', 'event_id', 'tanx', 'tany', 'x_local', 'y_local', 'x_t0', 'y_t0',
              'v_drift_um_ns']].reset_index(drop=True)


def geometry_beam(arm: str) -> pd.DataFrame:
    from ntof_calorimetry import mm_edep_data as MD
    D = MD.beam_tracks(arm)
    return D[['subrun', 'event_id', 'tanx', 'tany', 'x_local', 'y_local', 'x_t0', 'y_t0',
              'v_drift_um_ns']].dropna().reset_index(drop=True)


# ------------------------------------------------------------------ worker
def _electrons(prof: np.ndarray, rng, tan_u: float, tan_v: float, u_m: float, v_m: float,
               v_um_ns: float, hyper: dict) -> dict:
    """Heed's per-depth electron counts -> electrons at the mesh.  Depth z from
    the entry (drift cathode) side: drift length to the mesh = GAP - z."""
    b = np.repeat(np.arange(len(prof)), prof.astype(int))
    z = (b + rng.uniform(0, 1, len(b))) * BIN_MM
    d = np.clip(GAP - z, 0, None)                      # distance to the mesh
    u = u_m + tan_u * d
    v = v_m + tan_v * d
    ta = d / (v_um_ns * 1e-3)
    sig = np.sqrt(hyper['sigma_p0'] ** 2 + hyper['Dp'] ** 2 * ta)
    u = u + rng.normal(0, 1, len(u)) * sig
    v = v + rng.normal(0, 1, len(v)) * sig
    g = rng.gamma(DG.POLYA_K, 1.0 / DG.POLYA_K, len(u))
    return dict(u=u, v=v, t=ta, g=g)


def _charge(raw: np.ndarray, ped: np.ndarray, pm: np.ndarray, p0: float, tn: float,
            sig: np.ndarray | None = None) -> dict:
    """mm_charge's estimator on one plane's (ns, 512) raw ADC."""
    road = MC._road(pm, p0, tn)
    ctrl = MC._road(pm, p0, tn, MC.CTRL_SHIFT_MM if p0 < 200 else -MC.CTRL_SHIFT_MM)
    if len(road) < 3:
        return {}
    w = MC._baseline(raw.copy(), ped, np.union1d(road, ctrl))
    prof = w[:, road].sum(1)
    out = dict(Q=float(prof.sum()), prof=prof.astype(np.float32), n_road=int(len(road)),
               Q_ctrl=float(w[:, ctrl].sum()) if len(ctrl) else np.nan,
               raw_max=float(raw[:, road].max()))
    if sig is not None:
        out['S_road'] = float(sig[:, road].sum())     # noise-free signal in the road
        out['S_all'] = float(sig.sum())
    return out


def run_one(job: dict) -> dict:
    from wft import model as wm
    rng = np.random.default_rng(job['seed'])
    out = dict(job['truth'])
    with DG._SimModel():
        hyper = dict(wm.HYPER)
        el = _electrons(job['prof'], rng, job['tan_u'], job['tan_v'], job['u_m'], job['v_m'],
                        job['v_true'], hyper)
        out['G_sum'] = float(el['g'].sum())
        ov = job['ov']
        W, RAW, SIG = {}, {}, {}
        for p in 'xy':
            coord = HALF - el['u'] if p == 'x' else HALF + DG.STRIP_V_SIGN * el['v']
            share = job['xy_split'] if p == 'x' else 1.0 - job['xy_split']
            t0p = job['t0'][p]
            s = DG.plane_waveform(p, coord, el['t'], el['g'] * job['adc_per_e'] * share,
                                  DG._ST['pos_maps'][p], t0p, hyper)
            s *= wm.GAIN[p][:, None]
            raw = np.clip(ov[p] + s.T, None, DG.ADC_RAIL)
            RAW[p], SIG[p] = raw, s.T
            W[p] = DG._cns(raw - DG._ST['ped'][p][None, :]).T
    try:
        r = DG.reco_event(job['eid'], W, dict(x=ov['ftst_x'], y=ov['ftst_y'])) or {}
    except Exception as err:                                            # noqa: BLE001
        r = dict(error=repr(err)[:200])
    for k in ('x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'x_t0', 'y_t0', 'x_ok', 'y_ok',
              'v_drift_um_ns', 'x_n_strips', 'y_n_strips'):
        if k in r:
            out[f'reco_{k}'] = r[k]
    # truth road: strip coordinate at the mesh and d(strip)/d(depth)
    true_road = dict(x=(HALF - job['u_m'], -job['tan_u']), y=(HALF + DG.STRIP_V_SIGN * job['v_m'],
                                                               DG.STRIP_V_SIGN * job['tan_v']))
    for p in 'xy':
        pm = DG._ST['pos_maps'][p]
        ped = DG._ST['ped'][p]
        c = _charge(RAW[p], ped, pm, *true_road[p], sig=SIG[p])
        for k, v in c.items():
            out[f'{p}_{k}_true'] = v
        p0, tn = r.get(f'{p}_p0'), r.get(f'{p}_tan_theta')
        if p0 is not None and np.isfinite(p0) and np.isfinite(tn):
            c = _charge(RAW[p], ped, pm, p0, tn, sig=SIG[p])
            for k, v in c.items():
                out[f'{p}_{k}'] = v
    return out


# --------------------------------------------------------------- driver
def setup_overlay(arm: str):
    from ntof_tracking import wft_beam as WB
    run, sub = 'run_145', 'stat090_0000'
    tag = WB.subrun_tags(WB.beam_config(arm, run, sub))[0]
    ov = DG.Overlay(arm, run, sub, tag)
    return ov


def make_jobs(G: pd.DataFrame, pool: dict, ov, arm: str, adc_per_e: float, n: int,
              seed: int, kind: str, T: float) -> list:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(G), size=min(n, len(G)), replace=False)
    th_pool = np.tan(np.radians(pool['theta_trk']))
    order = np.argsort(th_pool)
    ths = th_pool[order]
    used = np.zeros(len(order), bool)
    jobs = []
    for i, gi in enumerate(idx):
        g = G.iloc[gi]
        tp = float(np.hypot(g.tanx, g.tany))
        # the pool track with the nearest polar angle, not reused
        k = int(np.clip(np.searchsorted(ths, tp), 0, len(ths) - 1))
        lo, hi = k, k
        while used[lo] and used[hi]:
            lo, hi = max(lo - 1, 0), min(hi + 1, len(ths) - 1)
        k = lo if not used[lo] else hi
        used[k] = True
        j = order[k]
        o = ov.events[i % len(ov.events)]
        dt_xy = float(g.x_t0 - g.y_t0) if np.isfinite(g.x_t0 - g.y_t0) else 0.0
        jobs.append(dict(
            eid=int(o['eid']), ov=o, prof=pool['prof'][j], tan_u=float(g.tanx), tan_v=float(g.tany),
            u_m=float(g.x_local), v_m=float(g.y_local), v_true=float(g.v_drift_um_ns),
            t0=dict(x=float(g.x_t0), y=float(g.x_t0) - dt_xy), adc_per_e=adc_per_e,
            xy_split=XY_SPLIT[arm], seed=int(rng.integers(1 << 31)),
            truth=dict(arm=arm, kind=kind, T=T, i=i, src_subrun=g.subrun, src_event=int(g.event_id),
                       tanx=float(g.tanx), tany=float(g.tany), x_local=float(g.x_local),
                       y_local=float(g.y_local), t0=float(g.x_t0), v=float(g.v_drift_um_ns),
                       theta_heed=float(pool['theta_trk'][j]), e_dep=float(pool['e_dep'][j]),
                       n_e=int(pool['n_e'][j]), n_clu=int(pool['n_clu'][j]))))
    return jobs


def run(jobs: list, bundle: Path, state: dict, n_jobs: int) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=DG.worker_init,
                             initargs=(str(bundle), state, 3, None)) as pool:
        for i, r in enumerate(pool.map(run_one, jobs, chunksize=8)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f'  {i + 1}/{len(jobs)}', flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['cosmic', 'beam'])
    ap.add_argument('--arm', default='A')
    ap.add_argument('--n', type=int, default=6000)
    ap.add_argument('--T', type=float, nargs='*', default=None, help='beam: e- energies (MeV)')
    ap.add_argument('--w', type=float, nargs='*', default=None,
                    help='beam: weight of each --T (default: the Geant4 neutron-beam spectrum at the gap)')
    ap.add_argument('--adc-per-e', type=float, default=13.0)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    EDEP.mkdir(parents=True, exist_ok=True)
    ov = setup_overlay(a.arm)
    state = ov.state()
    bundle = BUNDLES / f'is2_{a.arm}'
    if a.mode == 'cosmic':
        G = geometry_cosmic(a.arm)
        sets = [('mu-', 3000.0, 1.0)]
    else:
        G = geometry_beam(a.arm)
        Ts = a.T or list(BEAM_MIX)
        ws = a.w or [BEAM_MIX.get(T, 1.0) for T in Ts]
        sets = [('e-', T, w / sum(ws)) for T, w in zip(Ts, ws)]
    out = []
    for p, T, w in sets:
        pool = heed_pool(p, T)
        jobs = make_jobs(G, pool, ov, a.arm, a.adc_per_e, int(round(a.n * w)), seed=int(T * 1000) + ord(a.arm),
                         kind=p, T=T)
        print(f'{a.mode} {a.arm} {p} {T:g} MeV: {len(jobs)} tracks, adc/e {a.adc_per_e}', flush=True)
        out.append(run(jobs, bundle, state, a.jobs))
    R = pd.concat(out, ignore_index=True)
    f = EDEP / f'sim_{a.mode}_{a.arm}{a.label}.parquet'
    R.drop(columns=[c for c in R.columns if c.endswith('prof') or c.endswith('prof_true')]).to_parquet(f)
    # keep the per-sample road profiles separately (plateau estimator)
    P = R[['i', 'T'] + [c for c in R.columns if c.endswith('prof') or c.endswith('prof_true')]]
    P.to_pickle(str(f).replace('.parquet', '_prof.pkl'))
    print(f'-> {f} ({len(R)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
