#!/usr/bin/env python3
"""k1_verdict.py [--out results/k1_verdict.json] -- K1 against the data remainder (K1_TEST_2026-11-01.md).

Inputs: results/k1_test.json (k1_analyze.py), results/floor_budget.json (the data ladder: clean-event s68 and U_cond
on the calibration rows), results/chi2_scaling.json (the data chi2/dof against q_sum).

Per chamber and view, for each K1 variant:
  amp_cal    the size at which the toy's centre-strip residual scatter (rc_sd) equals the data's (T16: det3 X 0.063,
             Y 0.058; the other chambers 0.06, T16: 'det2 / det4 base agree'); linear interpolation over the three sizes,
             'unreached' when even the largest size stays below
  d2         the angle variance the variant adds over the full toy (L8), s68_w^2(K1) - s68_w^2(L8), at amp_cal (linear in
             amp^2 between the sizes), or at the largest size if unreached
  share      d2 / U_cond^2: the fraction of the data's clean-event remainder the variant explains
  amp_need   the size the variant would need to explain all of U_cond^2 (d2 extrapolated as k amp^2 from the largest size)
plus the chi2 level (toy cd at q50 against the data's) and the K1-aware fit (L8w, K1<v>w against L8, K1<v>).
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
RC_DATA = {('det3', 'x'): 0.063, ('det3', 'y'): 0.058}
RC_DEFAULT = 0.06
VARS = dict(G='per-strip gain jitter', T='per-strip time jitter [ns]', W='per-event pulse-width stretch',
            S='per-sample proportional noise', L='per-event footprint width')
FD_INPUTS = json.load(open(os.path.join(HERE, '..', 'model_audit', 'followups', 'fd_floor_realism', 'results',
                                        'toy_inputs_pass1.json')))


def data_cd_at(dec, q):
    qs, cs = np.array([d[0] for d in dec]), np.array([d[1] for d in dec])
    return float(np.exp(np.interp(np.log(q), np.log(qs), np.log(cs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(R, 'k1_verdict.json'))
    a = ap.parse_args()
    T = json.load(open(os.path.join(R, 'k1_test.json')))
    FB = json.load(open(os.path.join(R, 'floor_budget.json')))
    CS = json.load(open(os.path.join(R, 'chi2_scaling.json')))
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        for view in ('x', 'y'):
            dv = FB[f'{det}|calib']['views'][view]
            L = {s['step']: s for s in dv['ladder']}
            Uc = dv['U_cond']
            base = T.get(f'{det}|{view}|L8||')
            basew = T.get(f'{det}|{view}|L8w|0|0.06')
            if base is None:
                continue
            q50 = FD_INPUTS[det]['q50']['ig']
            cs = CS.get(f'{det}|{view}')
            rec = dict(data=dict(clean_s68=L['topo']['s68'], U_cond=Uc, all_s68=dv['measured'],
                                 chi2_select_share=dv['chi2_select_share_data'],
                                 cd_q50=data_cd_at(cs['deciles'], q50) if cs else None,
                                 cd_slope=cs['slope_log_excess'] if cs else None,
                                 rc_sd=RC_DATA.get((det, view), RC_DEFAULT),
                                 rc_source='T16' if (det, view) in RC_DATA else 'T16 (det3), assumed'),
                       toy_L8=dict(s68_w=base['s68_w'], a=base['a'], b=base['b'], rc_sd=base['rc_sd'],
                                   cd_q50=base['cd_q50'], cd_slope=base['cd_slope'], sel_share=base['sel_share']),
                       weights_L8w=dict(s68_w=basew['s68_w'] if basew else None,
                                        d_s68=(basew['s68_w'] - base['s68_w']) if basew else None),
                       variants={})
            target = rec['data']['rc_sd']
            for v, name in VARS.items():
                pts = sorted((s['amp'], s) for k, s in T.items()
                             if s['det'] == det and s['view'] == view and s['rung'] == f'K1{v}')
                if not pts:
                    continue
                amps = np.array([p[0] for p in pts])
                rcs = np.array([p[1]['rc_sd'] for p in pts])
                d2s = np.array([p[1]['s68_w'] ** 2 - base['s68_w'] ** 2 for p in pts])
                reached = rcs.max() >= target
                if reached:
                    amp_cal = float(np.interp(target, np.r_[0, rcs if np.all(np.diff(rcs) > 0) else np.sort(rcs)],
                                              np.r_[0, amps if np.all(np.diff(rcs) > 0) else amps[np.argsort(rcs)]]))
                    d2 = float(np.interp(amp_cal ** 2, np.r_[0, amps ** 2], np.r_[0, d2s]))
                else:
                    amp_cal, d2 = None, float(d2s[-1])
                k = d2s[-1] / amps[-1] ** 2 if d2s[-1] > 0 else np.nan
                amp_need = float(np.sqrt(Uc ** 2 / k)) if np.isfinite(k) and k > 0 else None
                w = T.get(f'{det}|{view}|K1{v}w|{pts[1][1]["amp"]:g}|0.06')
                mid = pts[1][1]
                rec['variants'][v] = dict(
                    name=name, amps=amps.tolist(), rc_sd=rcs.tolist(), d2=d2s.tolist(),
                    a=[p[1]['a'] for p in pts], b=[p[1]['b'] for p in pts],
                    cd_q50=[p[1]['cd_q50'] for p in pts], cd_slope=[p[1]['cd_slope'] for p in pts],
                    sel_share=[p[1]['sel_share'] for p in pts],
                    amp_cal=amp_cal, reached=bool(reached), d2_cal=d2, share=d2 / Uc ** 2,
                    amp_need=amp_need,
                    k1aware=dict(amp=mid['amp'], s68_w=w['s68_w'] if w else None,
                                 d_s68=(w['s68_w'] - mid['s68_w']) if w else None))
            out[f'{det}|{view}'] = rec
            line = ' '.join(f"{v}:{(s['amp_cal'] if s['reached'] else 'unr'):>5} share {s['share']:+.2f}"
                            .replace("'", '') for v, s in rec['variants'].items())
            print(f"{det} {view} U_cond {Uc:.2f} data cd {rec['data']['cd_q50'] or 0:.0f} toy cd {base['cd_q50']:.0f} | {line}")
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
