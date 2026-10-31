#!/usr/bin/env python3
"""budget_pass1.py -- the FD angle budget (budget.py) re-stated on pass 1, both arms (2026-10-31).

What changes against budget.py (10-27, audit tables, toy on the audit bundles):
  measured   s68 re-measured on the pass-1 golden rows (fd_pass1.py; FD's convention, |d| <= 5 deg, median-subtracted),
             both arms; the 03_angles headline (no clip) is in rerun/paper1_res, see the report's convention note
  toy        the realism ladder RE-RUN on the pass-1 calibration bundles (pass1_basis.py, FD_BASIS=pass; condor DAG
             condor/make_jobs_pass1.py, 334 jobs): pass template / sigma_p0 / dead mask / t0 prior, the row's q50, the
             dead strips the pass mask still misses (det7 none on interior_good), interior placement, each row's own
             per-event r(t) in the v-drift rung.  Every toy s68 is the DATA-WEIGHTED one ('all_w': the toy's 3 / 9 / 16 deg
             points weighted by the row's |theta_ref| mix); the 10-27 equal-weight pooling is kept as *_eq for comparison
             sample ig -> ladder_pass1_ig_<det>; sample full -> the same, except det7 -> ladder_pass1_full_det7 (its
             dead-strip term lives below ch 256, outside interior_good)
  L4 gain    withdrawn (GA 10-28: the sigma 0.31 input was the stripe structure; a measured map moves s68 by 0 +- 0.03)
  L8 v drift pass arm, row with a v(t) map: MEASURED, q(s68_pass, s68_vt) on the same events (what the correction
             removes); the toy's own L8 on the row's r(t) is kept as L8_toy.  Row with no map: the toy's L8 (its r(t) is
             the survey's uniform |dv|).  vt arm: 0 where the row has a map, the pass value where it has none
  position   what is left of the cross-validated 2D (40 mm) tilt AFTER the pass's bow block: q(slope, 2d40) on the
             pass table (was q(s68, 2d40) on audit tables with no bow)
  M3         0.05 deg (GB 10-28, measured <= 0.11) instead of the 0.15 bound
The toy is X view only, so the Y budget is measured / position / M3 / rest only.
toy_all (all toy terms together) = the quadrature sum of the floor, the production-fit term and every rung alone (L4
0, L8 as above).  The 10-27 form, the cumulative rung with L4/L8 swapped in quadrature, is kept as toy_all_cumswap: it
agrees within 0.02 deg except on det4, where the withdrawn gain-map rung (0.65 deg) does not come out in quadrature.
Toy statistics: 1500 events per angle point (results/ladder_pass1_*_n1500_*; the 150-event run is ladder_pass1_*_<det>),
bootstrap errors per term in term_err, term_sig = the rung's s68 above the production fit's by > 2 sigma.

    budget_pass1.py   -> results/budget_pass1.json, prints the table
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
CALIB = dict(det2='o22_long_det2', det3='sat_det3', det4='g_det4', det6='g_det6_long', det7='g_det7_long')
HEAD = dict(det2='c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2',
            det3='c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3')
M3 = 0.05
TERMS = ['L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']


def q(a, b):
    return float(np.sqrt(max(a * a - b * b, 0.0)))


def toy(det, smp, stat, suf=None):
    """the pass-basis ladder for this chamber and sample: floor, production-fit term, per-rung terms, cumulative.
    suf: '_n1500' (1500 events per angle point; the default when present) or '' (150, the 10-27 statistics)."""
    tag = 'pass1_full' if (smp == 'full' and det == 'det7') else 'pass1_ig'
    if suf is None:
        suf = '_n1500' if os.path.exists(os.path.join(R, f'ladder_{tag}_n1500_{det}.json')) else ''
    tag += suf
    cum = json.load(open(os.path.join(R, f'ladder_{tag}_{det}.json')))['rungs']
    alo = json.load(open(os.path.join(R, f'ladder_{tag}_{det}_alone.json')))['rungs']
    L0, L1, L8 = (cum[k][stat]['s68'] for k in ('L0', 'L1', 'L8'))
    e1 = cum['L1'][stat]['s68_err']
    return dict(tag=tag, floor=L0, prod_fit=L1, prodfit_term=q(L1, L0), cum8=L8,
                terms={t: q(alo[t][stat]['s68'], L1) for t in TERMS if t in alo},
                term_err={t: qerr(alo[t][stat]['s68'], alo[t][stat]['s68_err'], L1, e1) for t in TERMS if t in alo},
                # resolved: the rung's s68 exceeds the production fit's by more than 2 sigma (bootstrap)
                term_sig={t: bool(alo[t][stat]['s68'] - L1 > 2 * np.hypot(alo[t][stat]['s68_err'], e1))
                          for t in TERMS if t in alo},
                cum={k: v[stat]['s68'] for k, v in cum.items()}, err8=cum['L8'][stat]['s68_err'])


def qerr(a, da, b, db):
    """1 sigma of q(a, b) from the bootstrap errors of a and b; near q = 0 the resolvable size sqrt(2 a hypot(da, db))
    (a term below it is the toy's statistics, 150 events per angle point)"""
    qq = q(a, b)
    thr = float(np.sqrt(2 * a * np.hypot(da, db)))
    return thr if qq < thr else float(np.hypot(a * da, b * db) / qq)


def main():
    old = json.load(open(os.path.join(R, 'budget.json')))
    fd = json.load(open(os.path.join(R, 'fd_pass1.json')))
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        for role, key in (('calib', CALIB[det]), ('headline', HEAD.get(det))):
            if key is None:
                continue
            f = fd[key]
            for smp in ('full', 'ig'):
                T = toy(det, smp, 'all_w')
                Teq = toy(det, smp, 'all')
                t8_toy = T['terms']['L8']
                if role == 'headline' and not f['vt_map']:   # the calib row's r(t) is not this row's: rescale
                    vr = fd[CALIB[det]]['v_rms'] or json.load(open(os.path.join(
                        R, f"ladder_{T['tag']}_{det}.json")))['basis']['inputs']['v_rms']
                    t8_toy = t8_toy * (f['v_rms'] / vr if vr else 0.0)
                for arm in ('pass_', 'vt'):
                    for view in ('x', 'y'):
                        m = f['views'][view][smp][arm]
                        pos = q(m['slope'], m['2d40'])
                        if view == 'x':
                            e8 = T['term_err']['L8']
                            toy_sig = T['term_sig']['L8']
                            if f['vt_map']:
                                # measured: what the v(t) correction removes on the same events; where the toy's
                                # whole drift cost is significantly larger (det6: the linear model misses the knee),
                                # the pass arm carries the toy and the v(t) arm the part not removed
                                t8m = q(f['views'][view][smp]['pass_']['s68'], f['views'][view][smp]['vt']['s68'])
                                full8 = t8_toy if (t8_toy - t8m > 2 * e8) else t8m
                                t8 = full8 if arm == 'pass_' else q(full8, t8m)
                            else:
                                t8m = None
                                t8 = t8_toy if toy_sig else 0.0    # below the toy's resolution: 0
                            terms = dict(T['terms'])
                            terms['L4'] = 0.0
                            terms['L8'] = t8
                            # all toy terms in quadrature (floor, production fit, every rung alone, L4 out, L8 as
                            # chosen); the cumulative rung with L4/L8 swapped is kept as a check -- it fails on det4,
                            # whose withdrawn gain-map rung (0.65 deg) does not come out in quadrature
                            toy_all = float(np.sqrt(T['floor'] ** 2 + T['prodfit_term'] ** 2
                                                    + sum(v ** 2 for v in terms.values())))
                            toy_cum = float(np.sqrt(max(T['cum8'] ** 2 - T['terms']['L4'] ** 2
                                                        - T['terms']['L8'] ** 2 + t8 ** 2, T['floor'] ** 2)))
                            une = float(np.sqrt(max(m['s68'] ** 2 - toy_all ** 2 - pos ** 2 - M3 ** 2, 0)))
                            rec = dict(measured=m['s68'], floor=T['floor'], prodfit_term=T['prodfit_term'],
                                       toy_all=toy_all, toy_all_cumswap=toy_cum, terms=terms, term_err=T['term_err'],
                                       term_sig=T['term_sig'],
                                       L8_toy=t8_toy,
                                       L8_meas=t8m, L4_toy=T['terms']['L4'],
                                       toy_cum=T['cum'], toy_tag=T['tag'], toy_cum8_err=T['err8'],
                                       eq=dict(floor=Teq['floor'], prodfit_term=Teq['prodfit_term'], terms=Teq['terms'],
                                               cum8=Teq['cum8']),
                                       position=pos, m3=M3, unexplained=une, v_rms=f['v_rms'], n=m['n'],
                                       tilt_1d40=m['1d40'], tilt_2d40=m['2d40'], slope=m['slope'])
                        else:
                            rest = float(np.sqrt(max(m['s68'] ** 2 - pos ** 2 - M3 ** 2, 0)))
                            rec = dict(measured=m['s68'], position=pos, m3=M3, rest_incl_model=rest, n=m['n'],
                                       tilt_1d40=m['1d40'], tilt_2d40=m['2d40'], slope=m['slope'])
                        out[f"{det}|{role}|{arm.rstrip('_')}|{view}|{smp}"] = dict(key=key, **rec)
    json.dump(dict(old={d: dict(measured=old[d]['measured'], position=old[d]['position'], m3=old[d]['m3'],
                                unexplained=old[d]['unexplained'], toy_all=old[d]['toy_all'], floor=old[d]['floor'],
                                prodfit_term=old[d]['prodfit_term'], terms=old[d]['terms']) for d in old}, new=out),
              open(os.path.join(R, 'budget_pass1.json'), 'w'), indent=1)
    print('X view, interior_good, calib row (toy on the pass bundles, data-weighted): det | floor | prodfit | '
          'L2 L3 L5 L6 L7 | L4(withdrawn) | L8 meas [toy] | toy_all pass / vt | position pass / vt | '
          'measured pass / vt | unexplained pass / vt')
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        p = out[f'{det}|calib|pass|x|ig']; v = out[f'{det}|calib|vt|x|ig']; t = p['terms']
        print(f"{det} | {p['floor']:.2f} | {p['prodfit_term']:.2f} | " +
              ' '.join(f"{t[k]:.2f}" for k in ('L2', 'L3', 'L5', 'L6', 'L7')) +
              f" | {p['L4_toy']:.2f} | {t['L8']:.2f} [{p['L8_toy']:.2f}] | {p['toy_all']:.2f} / {v['toy_all']:.2f} | "
              f"{p['position']:.2f} / {v['position']:.2f} | {p['measured']:.2f} / {v['measured']:.2f} | "
              f"{p['unexplained']:.2f} / {v['unexplained']:.2f}")
    print('same, full sample (FD convention): det | toy_all pass / vt | unexplained old -> pass / vt')
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        p = out[f'{det}|calib|pass|x|full']; v = out[f'{det}|calib|vt|x|full']; o = old[det]
        print(f"{det} | {p['toy_all']:.2f} / {v['toy_all']:.2f} | {o['unexplained']:.2f} -> {p['unexplained']:.2f} / "
              f"{v['unexplained']:.2f}  (L7 {p['terms']['L7']:.2f}, toy {p['toy_tag']})")
    print('equal-weight toy (10-27 pooling) vs data-weighted, ig calib: det | floor eq / w | L8 toy eq / w | cum8 eq / w')
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        p = out[f'{det}|calib|pass|x|ig']
        print(f"{det} | {p['eq']['floor']:.2f} / {p['floor']:.2f} | {p['eq']['terms']['L8']:.2f} / {p['L8_toy']:.2f} | "
              f"{p['eq']['cum8']:.2f} / {p['toy_cum']['L8']:.2f}")


if __name__ == '__main__':
    main()
