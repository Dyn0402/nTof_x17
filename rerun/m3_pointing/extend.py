#!/usr/bin/env python3
"""extend.py -- chamber A Run 1 with the chi2 < 1 recipe lifted: every single-track, four-cluster M3 ray.

Input: the production reconstruction of every such event (reco_subset.py on the staged rr1 inputs, $WORK/reco/).
  1. closure: on the recipe events the new table must reproduce the pass-1 table (p0 per view).
  2. the rays: M3 at the alignment z in the raw strip frame (budget.rays_table's transform), the fiducial region
     (common.mx17_fiducial), the any-candidate pick (budget.cand_arms' 'near': the sane candidate nearest the ray).
  3. the per-track width: the M3 term of fit.json (never refitted here: chi2 > 1 is an extrapolation of a model
     fitted at chi2 < 1) and A Run 1's chamber term from validate.py.
  4. pull closure in bins of chi2 out to 50, and the efficiency per chi2 bin with R = 5 mm and with d <= n.
Writes $WORK/extend/{rays.parquet, closure.json, eff.parquet, pulls.parquet}.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO]
from m3common import WORK, core_sigma, s68, pointing_gauss  # noqa: E402
from rows import INPUTS, RAYS  # noqa: E402
from fit import add_m3  # noqa: E402
from model import sigma_track  # noqa: E402
from wft import compat  # noqa: E402
from wft.seed import SPARK_VETO_HITS  # noqa: E402
from common.mx17_fiducial import ray_regions, good_x  # noqa: E402

REL = 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V'
RECO = os.path.join(WORK, 'reco')
OUT = os.path.join(WORK, 'extend')
CHI2_EDGES = [0, 0.1, 0.3, 0.6, 1, 2, 4, 8, 16, 50, 1e9]
N_GRID = [3, 4, 5, 6, 8]


def rays(al):
    R = pd.read_parquet(os.path.join(WORK, 'tables', 'A_run1.relaxed.parquet'))
    R['ref_x'] = al['ref_x_sign'] * R.m3_X_at_x; R['ref_y'] = R.m3_Y_at_y
    tx = al['ref_x_sign'] * R.m3_sx; ty = R.m3_sy           # tan(sign * atan(s)) = sign * s
    th = np.deg2rad(al['theta_deg']); c, s = np.cos(th), np.sin(th); cx, cy = al['centre_x'], al['centre_y']
    u = R.ref_x - cx - al['x_offset']; v = R.ref_y - cy - al['y_offset']
    R['mesh_x'] = cx + c * u + s * v; R['mesh_y'] = cy - s * u + c * v
    R['tan_x'] = c * tx + s * ty; R['tan_y'] = -s * tx + c * ty
    F = ray_regions(R.mesh_x.to_numpy(), R.mesh_y.to_numpy(), R.tan_x.to_numpy(), R.tan_y.to_numpy(), 'mx17_3')
    R['region'] = F['region']; R['lo_x'] = F['lo_x']
    R['interior_good'] = (R.region == 'interior') & good_x('mx17_3', R.lo_x.to_numpy())
    R['z_x'] = al['z_x']; R['z_y'] = al['z_y']
    return R


def near_pick(R, cand_path):
    C = pd.read_parquet(cand_path)
    C = C[(C['rank'] >= 0) & C.event_id.isin(set(R.event_id))].drop_duplicates(['event_id', 'plane', 'rank'])
    C = C.join(R.set_index('event_id')[['mesh_x', 'mesh_y']], on='event_id')
    C['res'] = np.where(C.plane == 'x', C.p0 - C.mesh_x, C.p0 - C.mesh_y); C['ares'] = C.res.abs()
    C = C[np.isfinite(C.tan_theta) & (C.tan_theta.abs() <= compat.TAN_SANE) & np.isfinite(C.p0)
          & C.p0.between(*compat.P0_SANE_MM)]
    P = C.sort_values(['event_id', 'plane', 'ares', 'rank']).groupby(['event_id', 'plane']).head(1)
    P = P.set_index(['event_id', 'plane'])
    for v in 'xy':
        R[f'near_res_{v}'] = R.event_id.map(P.xs(v, level='plane').res)
    return R


def main():
    os.makedirs(OUT, exist_ok=True)
    al = json.load(open(os.path.join(INPUTS, REL, 'mx17_3', 'alignment.json')))
    new = compat.load_table(os.path.join(RECO, 'events.parquet'), max_dropped=None)
    old = pd.read_parquet('/media/ucla/work/claude-jobs/17c92c46/Analysis_pass1/' + REL + '/mx17_3/wft/events.parquet')
    # 1. closure on the recipe events
    J = old.merge(new, on='event_id', suffixes=('_p1', '_new'))
    clo = {'n_pass1': len(old), 'n_common': len(J)}
    for v in 'xy':
        both = J[f'{v}_ok_p1'] & J[f'{v}_ok_new']
        d = (J[f'{v}_p0_new'] - J[f'{v}_p0_p1'])[both]
        clo[v] = dict(n_both_ok=int(both.sum()), ok_agree=float(np.mean(J[f'{v}_ok_p1'] == J[f'{v}_ok_new'])),
                      frac_p0_within_1um=float(np.mean(d.abs() < 1e-3)), frac_p0_within_50um=float(np.mean(d.abs() < 0.05)),
                      p99_abs_dp0_mm=float(np.percentile(d.abs(), 99)))
    print('closure', json.dumps(clo, indent=1))
    json.dump(clo, open(os.path.join(OUT, 'closure.json'), 'w'), indent=1)

    # 2. rays + residuals
    R = rays(al)
    old_rays = pd.read_parquet(f'{RAYS}/{REL}/mx17_3/rays_det3.parquet')
    chk = R.merge(old_rays[['event_id', 'mesh_x', 'mesh_y']], on='event_id', suffixes=('', '_b'))
    print('ray transform closure: max |d mesh|', float(np.max(np.abs(chk.mesh_x - chk.mesh_x_b))),
          float(np.max(np.abs(chk.mesh_y - chk.mesh_y_b))), 'on', len(chk))
    T = new[['event_id', 'x_ok', 'y_ok', 'x_p0', 'y_p0', 'n_hits', 'spark']]
    R = R[R.event_id <= new.event_id.max()].merge(T, on='event_id', how='left')
    R['in_table'] = R.x_ok.notna()
    for v in 'xy':
        R[f'{v}_ok'] = R[f'{v}_ok'].astype('boolean').fillna(False).astype(bool)
        R[f'res_{v}'] = R[f'{v}_p0'] - R[f'mesh_{v}']
    R = near_pick(R, os.path.join(RECO, 'events.candidates.parquet'))
    # 3. per-track width (the model of fit.json, the chamber term of validate.py)
    par = json.load(open(os.path.join(WORK, 'decompose.json')))
    fr = json.load(open(os.path.join(WORK, 'fit.json')))
    ch = json.load(open(os.path.join(WORK, 'validate', 'chamber.json')))['A_run1']
    pmf = os.path.join(WORK, 'posmap.json')
    pm = json.load(open(pmf)) if os.path.exists(pmf) and not os.environ.get('M3P_NO_POSMAP') else None
    R = add_m3(R, par)
    for v in 'xy':
        R[f'sig_{v}'], R[f'vm3_{v}'] = sigma_track(R, fr[v]['glob'], ch[v]['model'], v, pm=pm)
        R[f'sig0_{v}'], _ = sigma_track(R, fr[v]['glob'], ch[v]['const'], v, const=True)
        R[f'bias_{v}'] = ch[v]['model']['m']; R[f'bias0_{v}'] = ch[v]['const']['m']
    R['chi2_max'] = np.maximum(R.chi2_X, R.chi2_Y)
    R.to_parquet(os.path.join(OUT, 'rays.parquet'))

    # 4a. pull closure vs the chi2 of the matched coordinate
    S = R[R.interior_good & R.x_ok & R.y_ok & (R.n_hits <= SPARK_VETO_HITS) & (R.res_x.abs() < 15)
          & (R.res_y.abs() < 15)]
    pulls = []
    for v, c in (('x', 'Y'), ('y', 'X')):
        for lo, hi in zip(CHI2_EDGES[:-1], CHI2_EDGES[1:]):
            s = S[(S[f'chi2_{c}'] >= lo) & (S[f'chi2_{c}'] < hi)]
            if len(s) < 60:
                continue
            p1 = (s[f'res_{v}'] - s[f'bias_{v}']) / s[f'sig_{v}']; p0 = (s[f'res_{v}'] - s[f'bias0_{v}']) / s[f'sig0_{v}']
            pulls.append(dict(view=v, lo=lo, hi=hi, n=len(s), chi2=float(s[f'chi2_{c}'].median()),
                              res_core=core_sigma(s[f'res_{v}'])[0], res_s68=s68(s[f'res_{v}']),
                              sig_med=float(s[f'sig_{v}'].median()), pm3_med=float(np.sqrt(s[f'vm3_{v}']).median()),
                              core1=core_sigma(p1)[0], core0=core_sigma(p0)[0], s68_1=s68(p1), s68_0=s68(p0),
                              f3_1=float(np.mean(np.abs(p1) < 3)), f3_0=float(np.mean(np.abs(p0) < 3))))
    P = pd.DataFrame(pulls); P.to_parquet(os.path.join(OUT, 'pulls.parquet'))
    print(P.round(3).to_string(index=False))

    # 4b. efficiency per chi2 bin (worst coordinate): interior_good denominator
    E = R[R.interior_good].sort_values('event_id').reset_index(drop=True)
    # accidentals by event mixing (as Paper I): the production fit of event i+s against the ray of event i
    px, py = E.x_p0.to_numpy(), E.y_p0.to_numpy(); okf = (E.x_ok & E.y_ok).to_numpy()
    acc_R = {r: np.zeros(len(E)) for r in (5, 10, 20)}; acc_n = {n: np.zeros(len(E)) for n in N_GRID}
    for sft in range(1, 21):
        j = np.roll(np.arange(len(E)), -sft)
        mx, my = px[j] - E.mesh_x.to_numpy(), py[j] - E.mesh_y.to_numpy()
        dRm = np.where(okf[j], np.nan_to_num(np.hypot(mx, my), nan=1e9), 1e9)
        dnm = np.where(okf[j], np.nan_to_num(np.hypot((mx - E.bias_x) / E.sig_x, (my - E.bias_y) / E.sig_y), nan=1e9), 1e9)
        for r in acc_R:
            acc_R[r] += (dRm <= r) / 20
        for n in N_GRID:
            acc_n[n] += (dnm <= n) / 20
    rows = []
    for lo, hi in zip(CHI2_EDGES[:-1], CHI2_EDGES[1:]):
        m = ((E.chi2_max >= lo) & (E.chi2_max < hi)).to_numpy()
        s = E[m]
        if len(s) < 60:
            continue
        rx, ry = s.near_res_x.to_numpy(), s.near_res_y.to_numpy()
        dR = np.nan_to_num(np.hypot(rx, ry), nan=1e9)
        dn = np.nan_to_num(np.hypot((rx - s.bias_x) / s.sig_x, (ry - s.bias_y) / s.sig_y).to_numpy(), nan=1e9)
        o = dict(lo=lo, hi=hi, n=len(s), any_fit=float(np.mean(np.isfinite(rx) & np.isfinite(ry))),
                 eff_R5=float(np.mean(dR <= 5)), eff_R10=float(np.mean(dR <= 10)), eff_R20=float(np.mean(dR <= 20)),
                 sig_x_med=float(s.sig_x.median()), sig_y_med=float(s.sig_y.median()))
        for r in acc_R:
            o[f'acc_R{r}'] = float(np.mean(acc_R[r][m]))
        for n in N_GRID:
            o[f'eff_n{n}'] = float(np.mean(dn <= n)); o[f'acc_n{n}'] = float(np.mean(acc_n[n][m]))
            o[f'win_n{n}_med_mm'] = float(np.nanmedian(n * np.hypot(s.sig_x, s.sig_y) / np.sqrt(2)))
        rows.append(o)
    Ef = pd.DataFrame(rows); Ef.to_parquet(os.path.join(OUT, 'eff.parquet'))
    print(Ef.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
