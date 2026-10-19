#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pair_sensitivity.py -- PLAN.md C3: does the plastic energy add anything to
the opening angle for X17 against the IPC continuum?

Input: the pair sim `pairs_thermal_trig_2cm_nose` (10^7 events, X17 + IPC
50/50, thermal capture vertices, the production stack), reduced per event x
arm by `condor/reduce_edep.py` (prompt deposits, MeV).

THE IPC IS REWEIGHTED.  The sim's IPC generator is the 1/M + isotropic ansatz
that CLAUDE.md retires.  Each IPC event gets w = p_target / p_sim in
(true opening angle, kinetic-energy split y = (T+ - T-)/(T+ + T-)), 5 deg x
0.1 bins, with p_target from `ipc_born.sample` for pure M1 and pure E0 -- the
two physical channels below 2 eV, whose mix is unknown, so they bracket it.

EVENT MODEL (stated, not measured):
  pair seen   the two leptons point into two different arms (dominant
              momentum component), and both arms' drift gas took > GAS_MIN;
  trigger     some arm has wall >= WALL_MIN and plastic >= PLAS_TRIG (the
              measured 2.1-2.9 MeVee, `c1/calib_plastic_e.json`);
  energy      each leg's plastic deposit (both bars of its arm), smeared by
              sigma = RES x sqrt(E x 3.41 MeV) (RES = the cosmic MIP
              sigma/MPV, median 0.18);
  angle       true opening angle smeared by SIG_THETA per pair.

FIGURE OF MERIT.  For S << B the Asimov Z^2 = sum s_i^2 / b_i, so the GAIN
Z^2(theta x E_low) / Z^2(theta) does not depend on the X17 or IPC
normalisation.  The plan's kill: gain < 1.10 -> stop at the threshold cut.
E_low = the lower of the two legs' plastic deposits (the soft leg at wide
angle, which stops in the plastic and so measures its own energy).

    python -m ntof_calorimetry.pair_sensitivity pull   # EOS -> OUT/c3
    python -m ntof_calorimetry.pair_sensitivity
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry.mip_sample import OUT  # noqa: E402
from sept26_prelim_analysis import ipc_born as IB  # noqa: E402

C3 = OUT / 'c3'
EOS = '/eos/experiment/ntof/data/x17/full_sim/calorimetry/c3_pairs_edep/pairs_thermal_trig_2cm_nose'
M_E = 0.51099895
GAS_MIN = 0.0005          # MeV in the drift gap: a track
WALL_MIN = 0.3
PLAS_TRIG = 2.5
RES = 0.18
SIG_THETA = 4.0
THETA_MIN = 100.0
ARM_N = {0: (1, 0, 0), 1: (-1, 0, 0), 2: (0, 0, 1), 3: (0, 0, -1)}   # sim armID: 0=+X 1=-X 2=+Z 3=-Z
TB = np.arange(0, 181, 5.0)
YB = np.linspace(-1, 1, 21)


def pull() -> None:
    C3.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', f'lxplus:{EOS}/', f'{C3}/raw/'], check=True)


def load() -> tuple:
    E, H = [], []
    for i, f in enumerate(sorted((C3 / 'raw').glob('*_events.csv.gz'))):
        e = pd.read_csv(f)
        h = pd.read_csv(str(f).replace('_events.csv.gz', '.csv.gz'))
        e['ev'] = i * 10_000_000 + e.eventID
        h['ev'] = i * 10_000_000 + h.eventID
        E.append(e)
        H.append(h)
    return pd.concat(E, ignore_index=True), pd.concat(H, ignore_index=True)


def leg_arm(px, py, pz) -> np.ndarray:
    P = np.stack([px, py, pz], 1)
    N = np.array([ARM_N[a] for a in range(4)], float)
    c = P @ N.T
    a = np.argmax(c, 1)
    a[np.max(c, 1) < np.cos(np.radians(60))] = -1
    return a


def ipc_weights(E: pd.DataFrame) -> dict:
    """Per IPC event: w_M1, w_E0 (sim ansatz -> ipc_born multipole)."""
    ipc = E.event_type == 1
    th = E.openingAngle[ipc].to_numpy()
    y = ((E.ep_ke - E.em_ke) / (E.ep_ke + E.em_ke))[ipc].to_numpy()
    hs, _, _ = np.histogram2d(th, y, [TB, YB])
    hs = hs / hs.sum()
    out = {}
    for kind in ('M1', 'E0'):
        s = IB.sample(kind)
        tp, tm = s.e_plus - M_E, s.e_minus - M_E
        ht, _, _ = np.histogram2d(s.theta_deg, (tp - tm) / (tp + tm), [TB, YB], weights=s.weight)
        ht = ht / ht.sum()
        r = np.where(hs > 0, ht / np.where(hs > 0, hs, 1), 0.0)
        it = np.clip(np.digitize(th, TB) - 1, 0, len(TB) - 2)
        iy = np.clip(np.digitize(y, YB) - 1, 0, len(YB) - 2)
        w = np.zeros(len(E))
        w[ipc.to_numpy()] = r[it, iy]
        out[kind] = w
        # coverage: target mass in bins the sim never populated
        out[f'{kind}_uncovered'] = float(ht[hs == 0].sum())
    return out


def build(E: pd.DataFrame, H: pd.DataFrame, rng) -> pd.DataFrame:
    E = E.copy()
    E['arm_m'] = leg_arm(E.em_px, E.em_py, E.em_pz)
    E['arm_p'] = leg_arm(E.ep_px, E.ep_py, E.ep_pz)
    H = H.assign(e_plas=H.e_plas_L + H.e_plas_R)
    idx = H.set_index(['ev', 'armID'])
    trig = (H.e_wall >= WALL_MIN) & (H.e_plas >= PLAS_TRIG)
    E['trig'] = E.ev.isin(H.ev[trig])
    for leg in ('m', 'p'):
        k = pd.MultiIndex.from_arrays([E.ev, E[f'arm_{leg}']])
        for c in ('e_gas', 'e_plas', 'e_wall', 'e_liq'):
            E[f'{c}_{leg}'] = idx[c].reindex(k).fillna(0.0).to_numpy()
        true = E[f'e_plas_{leg}'].to_numpy()
        E[f'obs_{leg}'] = np.clip(true + rng.normal(0, 1, len(E)) * RES * np.sqrt(np.clip(true, 0, None) * 3.41), 0, None)
    E['seen'] = ((E.arm_m >= 0) & (E.arm_p >= 0) & (E.arm_m != E.arm_p)
                 & (E.e_gas_m > GAS_MIN) & (E.e_gas_p > GAS_MIN))
    E['theta_obs'] = E.openingAngle + rng.normal(0, SIG_THETA, len(E))
    E['e_low'] = np.minimum(E.obs_m, E.obs_p)
    E['e_low_true'] = np.minimum(E.e_plas_m, E.e_plas_p)
    # wall + plastic: what a calibrated SiPM wall would add (legs stopping in it)
    E['e_low_wp_true'] = np.minimum(E.e_plas_m + E.e_wall_m, E.e_plas_p + E.e_wall_p)
    E['both_plas'] = (E.e_plas_m > 0.3) & (E.e_plas_p > 0.3)
    E['t_soft'] = np.minimum(E.em_ke, E.ep_ke)
    return E


#: coarse bins: the IPC MC has ~30k effective events above 100 deg, so a fine
#: 2D grid turns MC sparsity into fake gain (a first version found the
#: SMEARED energy worth more than the true one -- impossible)
TB2 = np.arange(THETA_MIN, 181, 5.0)
EB2 = np.array([0, 0.3, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7, 20.0])
MIN_MC = 10


def _z2(s, b, nb) -> float:
    """sum s^2/b with every bin holding < MIN_MC background MC events pooled
    into one bin (so a sparse corner cannot carry the answer)."""
    s, b, nb = s.ravel(), b.ravel(), nb.ravel()
    ok = nb >= MIN_MC
    z = np.sum(s[ok] ** 2 / b[ok])
    if (~ok).any() and b[~ok].sum() > 0:
        z += s[~ok].sum() ** 2 / b[~ok].sum()
    return float(z)


def z2_gain(S: pd.DataFrame, B: pd.DataFrame, wb: np.ndarray, ecol: str) -> dict:
    s1, _ = np.histogram(S.theta_obs, TB2)
    b1, _ = np.histogram(B.theta_obs, TB2, weights=wb)
    n1, _ = np.histogram(B.theta_obs, TB2)
    s2, _, _ = np.histogram2d(S.theta_obs, S[ecol], [TB2, EB2])
    b2, _, _ = np.histogram2d(B.theta_obs, B[ecol], [TB2, EB2], weights=wb)
    n2, _, _ = np.histogram2d(B.theta_obs, B[ecol], [TB2, EB2])
    s1, b1, s2, b2 = s1 / s1.sum(), b1 / b1.sum(), s2 / s2.sum(), b2 / b2.sum()
    z1, z2 = _z2(s1, b1, n1), _z2(s2, b2, n2)
    # stability: the same on each half of the background MC
    h = np.arange(len(B)) % 2 == 0
    half = []
    for m in (h, ~h):
        bb, _, _ = np.histogram2d(B.theta_obs[m], B[ecol][m], [TB2, EB2], weights=wb[m])
        nn, _, _ = np.histogram2d(B.theta_obs[m], B[ecol][m], [TB2, EB2])
        b1h, _ = np.histogram(B.theta_obs[m], TB2, weights=wb[m])
        n1h, _ = np.histogram(B.theta_obs[m], TB2)
        half.append(_z2(s2, bb / bb.sum(), nn) / _z2(s1, b1h / b1h.sum(), n1h))
    return dict(z2_theta=z1, z2_theta_e=z2, gain=z2 / z1, gain_half_a=half[0], gain_half_b=half[1])


def thresholds(S, B, wb) -> pd.DataFrame:
    rows = []
    for ec in np.arange(0, 6.01, 0.5):
        ks, kb = S.e_low >= ec, B.e_low >= ec
        rows.append(dict(e_cut=ec, eff_x17=float(ks.mean()), eff_ipc=float(np.sum(wb[kb.to_numpy()]) / wb.sum())))
    return pd.DataFrame(rows)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == 'pull':
        pull()
        return 0
    rng = np.random.default_rng(5)
    E, H = load()
    W = ipc_weights(E)
    D = build(E, H, rng)
    sel = D.seen & D.trig & (D.theta_obs >= THETA_MIN)
    S = D[sel & (D.event_type == 0)]
    out, thr = {}, []
    for kind in ('M1', 'E0'):
        mb = sel & (D.event_type == 1)
        B, wb = D[mb], W[kind][mb.to_numpy()]
        out[kind] = dict(n_sig=int(len(S)), n_bkg=int(len(B)), w_eff=float(wb.sum() ** 2 / np.sum(wb ** 2)),
                         uncovered=W[f'{kind}_uncovered'],
                         smeared=z2_gain(S, B, wb, 'e_low'), true_deposit=z2_gain(S, B, wb, 'e_low_true'),
                         true_wall_plus_plastic=z2_gain(S, B, wb, 'e_low_wp_true'),
                         true_soft_T=z2_gain(S, B, wb, 't_soft'),
                         frac_both_plastic_x17=float(S.both_plas.mean()),
                         frac_both_plastic_ipc=float(np.sum(wb[B.both_plas.to_numpy()]) / wb.sum()))
        thr.append(thresholds(S, B, wb).assign(ipc=kind))
    cols = ['ev', 'event_type', 'openingAngle', 'theta_obs', 'em_ke', 'ep_ke', 'e_plas_m', 'e_plas_p',
            'obs_m', 'obs_p', 'e_low', 'e_liq_m', 'e_liq_p', 'seen', 'trig']
    D[sel][cols].assign(w_M1=W['M1'][sel.to_numpy()], w_E0=W['E0'][sel.to_numpy()]).to_parquet(C3 / 'selected.parquet')
    pd.concat(thr).to_csv(C3 / 'thresholds.csv', index=False)
    summ = dict(n_events=int(len(D)), frac_seen=float(D.seen.mean()), frac_trig=float((D.seen & D.trig).mean()),
                model=dict(GAS_MIN=GAS_MIN, WALL_MIN=WALL_MIN, PLAS_TRIG=PLAS_TRIG, RES=RES,
                           SIG_THETA=SIG_THETA, THETA_MIN=THETA_MIN), results=out)
    (C3 / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(pd.concat(thr).round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
