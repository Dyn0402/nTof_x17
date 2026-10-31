#!/usr/bin/env python3
"""budget_pass1.py -- the FD angle budget (budget.py) re-stated on pass 1, both arms (2026-10-31).

What changes against budget.py (10-27, audit tables):
  measured   s68 re-measured on the pass-1 golden rows (fd_pass1.py; FD's convention, |d| <= 5 deg, median-subtracted),
             both arms; the 03_angles headline (no clip) is in rerun/paper1_res, see the report's convention note
  L4 gain    withdrawn (GA 10-28: the sigma 0.31 input was the stripe structure; a measured map moves s68 by 0 +- 0.03)
  L8 v drift pass arm, row with a v(t) map: MEASURED, q(s68_pass, s68_vt) on the same events (what the correction
             removes); row below the v(t) gate: the toy term rescaled from its v_rms (2.9 %) to the row's
             |dv|/sqrt(12).  vt arm: 0 where the row has a map, the pass value where it has none.  The toy term
             rescaled by the map's rms is kept as L8_toy_scaled: it over-predicts the measured gain (det6 0.71 against
             0.31 deg), so the toy's v-drift rung is not used where data exist
  position   what is left of the cross-validated 2D (40 mm) tilt AFTER the pass's bow block: q(slope, 2d40) on the
             pass table (was q(s68, 2d40) on audit tables with no bow)
  M3         0.05 deg (GB 10-28, measured <= 0.11) instead of the 0.15 bound
Unchanged (toy rungs on the AUDIT bundles; not re-run on the pass bundles): floor L0, the production-fit term
q(L1, L0), L2 footprint, L3 X copy, L5 pulse, L6 det3 diffusion, L7 dead + CM.  The toy is X view only, so the Y
budget is measured / position / M3 / rest only.
toy_all (all toy terms together) = sqrt(L8cum^2 - T_L4^2 - T_L8,toy^2 + T_L8,row^2): the cumulative rung with the
withdrawn and the rescaled terms swapped in quadrature (the cumulative rung is not exactly additive; approximation).

    budget_pass1.py   -> results/budget_pass1.json, prints the table
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
CALIB = dict(det2='o22_long_det2', det3='sat_det3', det4='g_det4', det6='g_det6_long', det7='g_det7_long')
HEAD = dict(det2='c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2',
            det3='c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3')
M3 = 0.05
TOY_VRMS = 0.02886751345948129


def q(a, b):
    return float(np.sqrt(max(a * a - b * b, 0.0)))


def main():
    old = json.load(open(os.path.join(R, 'budget.json')))
    fd = json.load(open(os.path.join(R, 'fd_pass1.json')))
    vt = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'vt_all', 'results', 'bench_vt.csv')))}
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        o = old[det]
        for role, key in (('calib', CALIB[det]), ('headline', HEAD.get(det))):
            if key is None:
                continue
            f = fd[key]
            rms_pass = f['v_rms'] if f['vt_map'] else abs(float(vt[key]['dv_pct'] or 0)) / 100 / np.sqrt(12)
            t8_toy = o['terms']['L8'] * rms_pass / TOY_VRMS
            for arm in ('pass_', 'vt'):
                for view in ('x', 'y'):
                    for smp in ('full', 'ig'):
                        m = f['views'][view][smp][arm]
                        pos = q(m['slope'], m['2d40'])
                        if view == 'x':
                            if f['vt_map']:     # measured: what the v(t) correction removes on the same events
                                t8 = q(f['views'][view][smp]['pass_']['s68'], f['views'][view][smp]['vt']['s68']) \
                                    if arm == 'pass_' else 0.0
                            else:
                                t8 = t8_toy
                            terms = dict(o['terms'])
                            terms['L4'] = 0.0
                            terms['L8'] = t8
                            toy_all = float(np.sqrt(max(o['toy_all'] ** 2 - o['terms']['L4'] ** 2 - o['terms']['L8'] ** 2
                                                        + t8 ** 2, o['floor'] ** 2)))
                            une = float(np.sqrt(max(m['s68'] ** 2 - toy_all ** 2 - pos ** 2 - M3 ** 2, 0)))
                            rec = dict(measured=m['s68'], floor=o['floor'], prodfit_term=o['prodfit_term'], toy_all=toy_all,
                                       terms=terms, L8_toy_scaled=t8_toy, position=pos, m3=M3, unexplained=une,
                                       v_rms=rms_pass, n=m['n'], tilt_1d40=m['1d40'], tilt_2d40=m['2d40'],
                                       slope=m['slope'])
                        else:
                            rest = float(np.sqrt(max(m['s68'] ** 2 - pos ** 2 - M3 ** 2, 0)))
                            rec = dict(measured=m['s68'], position=pos, m3=M3, rest_incl_model=rest, n=m['n'],
                                       tilt_1d40=m['1d40'], tilt_2d40=m['2d40'], slope=m['slope'])
                        out[f"{det}|{role}|{arm.rstrip('_')}|{view}|{smp}"] = dict(key=key, **rec)
    json.dump(dict(old={d: dict(measured=old[d]['measured'], position=old[d]['position'], m3=old[d]['m3'],
                                unexplained=old[d]['unexplained'], toy_all=old[d]['toy_all'], floor=old[d]['floor'],
                                terms=old[d]['terms']) for d in old}, new=out),
              open(os.path.join(R, 'budget_pass1.json'), 'w'), indent=1)
    print('X view, interior_good, calib row: det | measured pass / vt | toy_all pass / vt | L8 pass [toy scaled] | '
          'position pass / vt | unexplained pass / vt')
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        p = out[f'{det}|calib|pass|x|ig']; v = out[f'{det}|calib|vt|x|ig']
        print(f"{det} | {p['measured']:.2f} / {v['measured']:.2f} | {p['toy_all']:.2f} / {v['toy_all']:.2f} | "
              f"{p['terms']['L8']:.2f} [{p['L8_toy_scaled']:.2f}] | {p['position']:.2f} / {v['position']:.2f} | "
              f"{p['unexplained']:.2f} / {v['unexplained']:.2f}")
    print('X view, full sample (FD convention), calib row: det | measured old -> pass / vt | toy_all | position old -> '
          'pass / vt | M3 | unexplained old -> pass / vt')
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        p = out[f'{det}|calib|pass|x|full']; v = out[f'{det}|calib|vt|x|full']; o = old[det]
        print(f"{det} | {o['measured']:.2f} -> {p['measured']:.2f} / {v['measured']:.2f} | {o['toy_all']:.2f} -> "
              f"{p['toy_all']:.2f} / {v['toy_all']:.2f} | {o['position']:.2f} -> {p['position']:.2f} / "
              f"{v['position']:.2f} | {M3} | {o['unexplained']:.2f} -> {p['unexplained']:.2f} / {v['unexplained']:.2f}")
    print('Y view (no toy), full, calib row: det | measured pass / vt | position pass / vt | rest pass / vt')
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        p = out[f'{det}|calib|pass|y|full']; v = out[f'{det}|calib|vt|y|full']
        print(f"{det} | {p['measured']:.2f} / {v['measured']:.2f} | {p['position']:.2f} / {v['position']:.2f} | "
              f"{p['rest_incl_model']:.2f} / {v['rest_incl_model']:.2f}")


if __name__ == '__main__':
    main()
