#!/usr/bin/env python3
"""boot_analyze.py --raw R --data D --ids I [--out results/bootstrap.json] -- the residual bootstrap's verdict
(BOOTSTRAP_2026-11-01.md).  Run where the job outputs are (lxplus: .../x17rerun/floor_boot).

Per chamber and view:
  toy truths, pooled over the 3 / 9 / 16 deg points with the data's angle mix (pass1_basis.s68_w):
    L8      the direct toy (the toy fit against the generated angle)
    T       toy residuals moved onto toy truths: the method's calibration, eps = s68(T) / s68(L8)
    N       Gaussian noise only
    D, D<x> data residuals (and with one structure removed)
    U_boot  = q(s68(D), s68(T)): the angle the data's residuals carry beyond the toy's own, as moved by the bootstrap;
              U_boot / eps the same scaled by the calibration; share = (U_boot / eps)^2 / U_cond^2 against the data's
              clean-event remainder U_cond (results/floor_budget.json, the 'topo' step)
    abl<x>  the part of the excess one structure carries: (s68(D)^2 - s68(D<x>)^2) / (s68(D)^2 - s68(T)^2)
  data truths (FD's s68): dD, dN, dD<x>; against the clean data s68 measured against M3 (ids json)
  chi2/dof: median of the refits in each arm, against the clean data planes' own (the pickles)
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DETS = dict(det2='o22_long_det2', det3='sat_det3', det4='g_det4', det7='g_det7_long')
ANG = (3, 9, 16)


def s68(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) <= 5)]
    if len(a) < 20:
        return np.nan
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def wq(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def s68_w(parts, wts):
    """parts: {angle: d array}; each angle point weighted by the data's mix, divided by its count."""
    d, w = [], []
    for a, x in parts.items():
        x = np.asarray(x, float)
        x = x[np.isfinite(x) & (np.abs(x) <= 5)]
        if len(x):
            d.append(x)
            w.append(np.full(len(x), wts[a] / len(x)))
    if not d:
        return np.nan
    d, w = np.concatenate(d), np.concatenate(w)
    med = wq(d, w, 0.5)
    return wq(np.abs(d - med), w, 0.68)


def boot_w(parts, wts, n=200, seed=5):
    rs = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(s68_w({a: (lambda x: x[rs.integers(0, len(x), len(x))])(np.asarray(x)) for a, x in parts.items()
                          if len(x)}, wts))
    return float(np.nanstd(out))


def q(a, b):
    return float(np.sqrt(max(a * a - b * b, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ids', required=True)
    ap.add_argument('--inputs', required=True, help='toy_inputs_pass1.json (the data angle mix)')
    ap.add_argument('--budget', default=os.path.join(HERE, 'results', 'floor_budget.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'bootstrap.json'))
    a = ap.parse_args()
    INP = json.load(open(a.inputs))
    B = json.load(open(a.budget))
    res = {}
    for det, key in DETS.items():
        ids = json.load(open(os.path.join(a.ids, f'{key}.json')))
        D = pickle.load(open(os.path.join(a.data, f'{key}.pkl'), 'rb'))
        wts = dict(zip(ANG, INP[det]['ang_w']['ig']))
        for view in ('x', 'y'):
            r = dict(det=det, view=view)
            arms = {}
            for f in sorted(glob.glob(os.path.join(a.raw, f'TOY_{det}_{view}_*.npz'))):
                z = np.load(f)
                ang = int(z['angle'])
                arms.setdefault('L8', {}).setdefault(ang, z['L8'])
                for k in z.files:
                    if k in ('L8', 'angle', 'det', 'view', 'n_pool'):
                        continue
                    arms.setdefault(k, {})[ang] = z[k][0]
                    arms.setdefault(k + '_chi', {})[ang] = z[k][1]
                r.setdefault('n_pool', {})[ang] = int(z['n_pool'])
            toy = {}
            for k, parts in arms.items():
                if k.endswith('_chi'):
                    toy[k] = float(np.nanmedian(np.concatenate([np.asarray(v) for v in parts.values()])))
                else:
                    toy[k] = dict(s68=s68_w(parts, wts), err=boot_w(parts, wts), n=int(sum(len(v) for v in parts.values())),
                                  per_angle={a_: s68(v) for a_, v in parts.items()})
            r['toy'] = toy
            vb = B[f'{det}|calib']['views'][view]
            st = {L['step']: L for L in vb['ladder']}
            r['data_clean_s68'] = st['topo']['s68']
            r['U_cond'] = vb['U_cond']
            r['toy_all_budget'] = vb['toy_all']
            if all(k in toy for k in ('L8', 'T', 'D')):
                L8, T, Dd = toy['L8']['s68'], toy['T']['s68'], toy['D']['s68']
                eps = T / L8 if L8 > 0 else np.nan
                ub = q(Dd, T)
                r['eps'] = eps
                r['U_boot'] = ub
                r['U_boot_cal'] = ub / eps if eps > 0 else np.nan
                r['share_raw'] = ub ** 2 / vb['U_cond'] ** 2
                r['share_cal'] = (ub / eps) ** 2 / vb['U_cond'] ** 2 if eps > 0 else np.nan
                ex = Dd ** 2 - T ** 2
                r['ablation'] = {k: dict(s68=toy[k]['s68'], carried=float((Dd ** 2 - toy[k]['s68'] ** 2) / ex)
                                         if ex > 0 else np.nan)
                                 for k in ('Dcm', 'Dmean', 'Dtshape', 'Danti', 'Dtshuf', 'Dsshuf') if k in toy}
                # like with like: the share of each residual's own angle effect a structure carries, data vs toy
                r['own_share'] = {s_: dict(data=float(1 - toy['D' + s_]['s68'] ** 2 / Dd ** 2) if 'D' + s_ in toy else None,
                                           toy=float(1 - toy['T' + s_]['s68'] ** 2 / T ** 2) if 'T' + s_ in toy else None)
                                  for s_ in ('cm', 'tshape', 'anti', 'tshuf', 'sshuf', 'mean')}
            # data truths
            dd = {}
            for f in sorted(glob.glob(os.path.join(a.raw, f'DATA_{det}_{view}_*.npz'))):
                z = np.load(f)
                for k in z.files:
                    if k in ('theta', 'det', 'view'):
                        continue
                    dd[k] = dict(s68=s68(z[k][0]), n=int(np.isfinite(z[k][0]).sum()),
                                 chi2dof=float(np.nanmedian(z[k][1])))
            r['data_truth'] = dd
            if 'dD' in dd and dd['dD']['s68'] > 0:
                r['data_truth_share'] = {k[2:]: float(1 - v['s68'] ** 2 / dd['dD']['s68'] ** 2) for k, v in dd.items()
                                         if k.startswith('dD') and k != 'dD'}
                r['data_truth_over_measured'] = float(dd['dD']['s68'] / r['data_clean_s68'])
            pl = [p for p in D['planes'] if p['view'] == view and p['w_table'] is not None
                  and np.isfinite(p['w_table'])]
            r['n_planes'] = len(pl)
            r['replay_closure'] = float(np.mean([abs(p['w'] - p['w_table']) < 1e-4 for p in pl])) if pl else np.nan
            r['data_chi2dof'] = float(np.median([p['chi2'] / max(p['dof'], 1) for p in pl])) if pl else np.nan
            res[f'{det}|{view}'] = r
            t = r['toy']
            print(f"{det} {view}: clean {r['data_clean_s68']:.3f} U_cond {r['U_cond']:.3f} | toy L8 "
                  f"{t.get('L8', {}).get('s68', np.nan):.3f} T {t.get('T', {}).get('s68', np.nan):.3f} N "
                  f"{t.get('N', {}).get('s68', np.nan):.3f} D {t.get('D', {}).get('s68', np.nan):.3f} eps "
                  f"{r.get('eps', np.nan):.2f} U_boot {r.get('U_boot', np.nan):.3f} (cal {r.get('U_boot_cal', np.nan):.3f})"
                  f" share raw/cal {r.get('share_raw', np.nan):.2f}/{r.get('share_cal', np.nan):.2f} | data truth "
                  + ' '.join(f"{k} {v['s68']:.3f}" for k, v in dd.items())
                  + f" | chi2/dof data {r['data_chi2dof']:.1f} D {t.get('D_chi', np.nan):.1f} T {t.get('T_chi', np.nan):.1f}"
                  f" | replay {r['replay_closure']:.3f}")
            if 'ablation' in r:
                print('    ablation carried: ' + ' '.join(f"{k} {v['carried']:+.2f}" for k, v in r['ablation'].items()))
                print('    own share data/toy: ' + ' '.join(
                    f"{k} {v['data'] if v['data'] is not None else np.nan:+.2f}/{v['toy'] if v['toy'] is not None else np.nan:+.2f}"
                    for k, v in r['own_share'].items()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
