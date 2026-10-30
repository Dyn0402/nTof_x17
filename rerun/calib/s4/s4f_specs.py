#!/usr/bin/env python3
"""s4f_specs.py <dir> -- write the arm grids of the S4 follow-up (2026-10-30) for s4f_fit.py.

<dir> holds, per chamber (det4 = g_det4, det7 = g_det7_long):
    rr1_<det>/headon.pkl          s4_stacks.py's head-on cache (S4, rr1 decode)
    prod_<det>/headon.pkl         s4f_build.py --gen prod (pre-rerun decode), and prod_<det>/selections.json
    pass_<det>/arrays.npz         the pass bundle (Analysis_pass1/<row>/wft/calib_bundle)
    old_<det>/arrays.npz          the bundle the pre-pass input used (det4 calib_bundle_lp, det7 calib_bundle_r06)
Then, for each spec_*.json:  [XW_MIN_EV=50] s4f_fit.py --spec <dir>/spec_X.json --out <dir>/X.json --nboot 20
(XW_MIN_EV=50 for the amp2/ch grids, whose matched subsets fall below the default 120 events).
"""
import json
import sys

D = sys.argv[1]
V = {'det4': 33.0, 'det7': 36.8}


def arm(det, dec='rr1', sel=None, tm='pass', sat=3500, **kw):
    s = None if sel is None else [f'{D}/prod_{det}/selections.json', sel]
    return dict(cache=f'{D}/{dec}_{det}/headon.pkl', sel=s, template=f'{D}/{tm}_{det}/arrays.npz', sat=sat, **kw)


def dump(name, det, arms):
    json.dump(dict(T0=30 / (V[det] * 1e-3), arms=arms), open(f'{D}/spec_{name}.json', 'w'), indent=1)


for det in V:
    # 2 x 3 x 2: decode x selection x template, at the default cut
    dump(det, det, {f'{det}_{dec}_{sel}_{tm}': arm(det, dec, None if sel == 'all' else sel, tm)
                    for dec in ('rr1', 'prod') for sel in ('all', 'hits_prod', 'hits_rr1') for tm in ('pass', 'old')})
    # amplitude bins (X window maximum), all / hits-chain / complement
    a = {f'{det}_notHits': arm(det, sel='hits_prod', sel_not=True)}
    for lo, hi in ((0, 1000), (1000, 2000), (2000, 3500), (0, 2000), (1000, 3500)):
        a[f'{det}_all_amp{lo}-{hi}'] = arm(det, sat=hi, amp_lo=lo)
        a[f'{det}_hits_amp{lo}-{hi}'] = arm(det, sel='hits_prod', sat=hi, amp_lo=lo)
        a[f'{det}_notHits_amp{lo}-{hi}'] = arm(det, sel='hits_prod', sel_not=True, sat=hi, amp_lo=lo)
    dump(f'amp_{det}', det, a)
    # matched amplitude (1000-3500 ADC), both templates
    dump(f'amp2_{det}', det, {f'{det}_{lab}_amp1000-3500_{tm}': arm(det, sel=s, tm=tm, amp_lo=1000, **x)
                              for tm in ('pass', 'old')
                              for lab, s, x in (('all', None, {}), ('hits', 'hits_prod', {}),
                                                ('notHits', 'hits_prod', {'sel_not': True}))})
# det7 saturation-cut scan, S4 configuration and the pre-pass one
s = {}
for sat in (1000, 1500, 2000, 2500, 3000, 3500, 4000, None):
    s[f'det7_rr1_all_pass@sat{sat}'] = arm('det7', sat=sat)
    s[f'det7_prod_hits_prod_old@sat{sat}'] = arm('det7', 'prod', 'hits_prod', 'old', sat=sat)
dump('det7sat', 'det7', s)
# det7 on X ch 256-511 only (det7 X readout faults: quote ch 256-511)
c = {}
for tm in ('pass', 'old'):
    for sat in (2000, 2500, 3000, 3500):
        c[f'det7_ch256_all_{tm}@sat{sat}'] = arm('det7', tm=tm, sat=sat, ch_min=256)
    c[f'det7_ch256_all_amp1000-3500_{tm}'] = arm('det7', tm=tm, amp_lo=1000, ch_min=256)
    c[f'det7_ch256_hits_{tm}'] = arm('det7', sel='hits_prod', tm=tm, ch_min=256)
    c[f'det7_ch256_notHits_{tm}'] = arm('det7', sel='hits_prod', sel_not=True, tm=tm, ch_min=256)
dump('ch_det7', 'det7', c)
