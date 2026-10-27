#!/usr/bin/env python3
"""budget.py -- FD (10b): the X-view angle-resolution budget per chamber, in quadrature, from the realism ladder
(results/ladder_<det>[_alone].json), the measured resolution and chi2 (results/measured_chi2.json) and the
cross-validated 2D tilt map (results/tilt_map_xval.json).

  floor           L0: perfect model, start at truth (ws_floor)
  each term       alone on top of the production fit: sqrt(s68_alone^2 - s68_L1^2) (0 if negative)
  measured terms  all of them together: sqrt(s68_L8^2 - s68_L0^2)
  position        sqrt(measured^2 - after the cross-validated 2D (40 mm) tilt map^2)
  M3              0.15 deg (upper bound)
  unexplained     what is left of the measured s68 in quadrature

    budget.py   -> results/budget.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
TERMS = ['L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']


def q(a, b):
    return float(np.sqrt(max(a * a - b * b, 0.0)))


def main():
    meas = json.load(open(os.path.join(R, 'measured_chi2.json')))
    tilt = json.load(open(os.path.join(R, 'tilt_map_xval.json')))
    out = {}
    for det in DETS:
        cum = json.load(open(os.path.join(R, f'ladder_{det}.json')))['rungs']
        alo = json.load(open(os.path.join(R, f'ladder_{det}_alone.json')))['rungs']
        m = meas[f'{det}_x']
        s_meas = m['all_s68']
        L0, L1, L8 = cum['L0']['all']['s68'], cum['L1']['all']['s68'], cum['L8']['all']['s68']
        terms = {t: q(alo[t]['all']['s68'], L1) for t in TERMS if t in alo}
        hyp = {t: q(alo[t]['all']['s68'], L1) for t in ('H1', 'H2', 'H3', 'H4') if t in alo}
        tm = tilt[f'{det}_x']
        pos = q(tm['s68'], tm['2d_40'])
        m3 = 0.15
        une = float(np.sqrt(max(s_meas ** 2 - L8 ** 2 - pos ** 2 - m3 ** 2, 0)))
        out[det] = dict(measured=s_meas, chi2_data=m['chi2dof'], floor=L0, prod_fit=L1, toy_all=L8,
                        chi2_toy_L1=cum['L1']['all']['chi2dof'], chi2_toy_L8=cum['L8']['all']['chi2dof'],
                        prodfit_term=q(L1, L0), measured_terms=q(L8, L0), terms=terms, hyp=hyp, position=pos,
                        tilt_1d40=tm['1d_40'], tilt_2d40=tm['2d_40'], m3=m3, unexplained=une,
                        s68_err=cum['L8']['all']['s68_err'],
                        cum={k: v['all']['s68'] for k, v in cum.items()},
                        cum_chi2={k: v['all']['chi2dof'] for k, v in cum.items()})
        print(f"{det}: meas {s_meas:.2f} (chi2 {m['chi2dof']:.0f}) floor {L0:.2f} toy {L8:.2f} (chi2 {cum['L8']['all']['chi2dof']:.0f})"
              f" | terms " + ' '.join(f'{t} {v:.2f}' for t, v in terms.items()) +
              f" | pos {pos:.2f} M3 {m3} unexplained {une:.2f}" + (f" | hyp {hyp}" if hyp else ''))
    json.dump(out, open(os.path.join(R, 'budget.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
