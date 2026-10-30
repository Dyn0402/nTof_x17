#!/usr/bin/env python3
"""s4r_specs.py <dir> -- write the arm grids of the S4 re-check of det2, det3 and det6 (2026-10-30) for s4f_fit.py.

The S4 follow-up found the head-on stack needs (a) each chamber's masks and (b) an amplitude floor (det4's T is 60 ns
longer below 1000 ADC; det7's Dp moved with its faulty X region).  This applies both to the three chambers S4 left as
'holding'.  <dir> holds, per chamber (det2 = o22_long_det2, det3 = sat_det3, det6 = g_det6_long):
    rr1_<det>/headon.pkl      S4's head-on cache (s4_stacks.py: rr1 decode, ref-only through the pass alignment)
    pass_<det>/{arrays.npz, bundle.json}   the pass bundle (Analysis_pass1/<row>/wft/calib_bundle)
    old_<det>/arrays.npz      the bundle the pre-pass input used (bench_attach.BUNDLE: det2/det3 calib_bundle_r06,
                              det6 calib_bundle_lp)
The X mask is the pass bundle's `dead` list (census dead + shorts, censored in the fit) united with strip_gain_audit's
flagged strips (head-on dead/weak + inclined-track holes; annotate_bad, the stack's own mask, FINDINGS §35).
Then, for each spec:  s4f_fit.py --spec <dir>/spec_r_<det>.json --out <dir>/r_<det>.json --nboot 20
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CB = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'sps_beam_test_26', 'analysis', 'cloud_basics'))
sys.path.insert(0, CB)
from strip_gain_audit import bad_set  # noqa: E402

D = sys.argv[1]
V = {'det2': 39.3, 'det3': 35.8, 'det6': 24.17}          # physics_inputs.json timing v (the fit's starting T only)
KEY = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det6': 'g_det6_long'}
AUDIT = json.load(open(os.path.join(CB, 'results', 'strip_gain_audit.json')))


def masks(det):
    dead = set(json.load(open(f'{D}/pass_{det}/bundle.json'))['dead']['x'])
    audit = bad_set(AUDIT[KEY[det]], 'x')
    return dict(dead=sorted(dead), audit=sorted(audit), union=sorted(dead | audit))


def arm(det, tm='pass', sat=3500, amp_lo=0.0, mask=None, guard=None):
    a = dict(cache=f'{D}/rr1_{det}/headon.pkl', sel=None, template=f'{D}/{tm}_{det}/arrays.npz', sat=sat, amp_lo=amp_lo)
    if mask is not None:
        a['mask_x'] = mask
    if guard is not None:
        a['guard'] = guard
    return a


for det in V:
    M = masks(det)
    json.dump(M, open(f'{D}/mask_{det}.json', 'w'), indent=1)
    u = M['union']
    a = {}
    for tm in ('pass', 'old'):
        a[f'{det}_none_{tm}'] = arm(det, tm)                                   # S4 as run (closure)
        a[f'{det}_mask_{tm}'] = arm(det, tm, mask=u)
        a[f'{det}_none_amp1000-3500_{tm}'] = arm(det, tm, amp_lo=1000)
        a[f'{det}_mask_amp1000-3500_{tm}'] = arm(det, tm, amp_lo=1000, mask=u)
        for lo, hi in ((0, 1000), (1000, 2000), (2000, 3500)):
            a[f'{det}_mask_amp{lo}-{hi}_{tm}'] = arm(det, tm, sat=hi, amp_lo=lo, mask=u)
    # exclusion instead of censoring: no masked strip within +-2 of the X peak
    a[f'{det}_maskg2_pass'] = arm(det, mask=u, guard=2)
    a[f'{det}_maskg2_amp1000-3500_pass'] = arm(det, amp_lo=1000, mask=u, guard=2)
    a[f'{det}_maskg2_amp0-1000_pass'] = arm(det, sat=1000, mask=u, guard=2)
    json.dump(dict(T0=30 / (V[det] * 1e-3), arms=a), open(f'{D}/spec_r_{det}.json', 'w'), indent=1)
    print(det, 'mask X:', len(M['dead']), 'bundle dead,', len(M['audit']), 'audit,', len(u), 'union;', len(a), 'arms')

# where do the low-amplitude events sit?  amplitude bin x (M3 reference >= / < 30 mm from the active-area edge), masked,
# pass template; det4 from the S4 follow-up's directory (argv[2], s4f: rr1_det4/headon.pkl, pass_det4/)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402
DIRS = {d: D for d in V}
if len(sys.argv) > 2:
    DIRS['det4'], V['det4'], KEY['det4'] = sys.argv[2], 33.0, 'g_det4'
for det, dd in DIRS.items():
    dead = set(json.load(open(f'{dd}/pass_{det}/bundle.json'))['dead']['x'])
    u = sorted(dead | bad_set(AUDIT[KEY[det]], 'x'))
    box = TRUE_ACTIVE_BY_DET['mx17_' + det[-1]]
    a = {}
    w = lambda lo, hi, side, tm, mk=True, d=30.0: dict(
        cache=f'{dd}/rr1_{det}/headon.pkl', sel=None, template=f'{dd}/{tm}_{det}/arrays.npz', sat=hi, amp_lo=lo,
        edge=[box, d, side], **(dict(mask_x=u) if mk else {}))
    for lo, hi in ((0, 1000), (1000, 3500)):
        for side in ('in', 'out'):
            a[f'{det}_mask_amp{lo}-{hi}_{side}30_pass'] = w(lo, hi, side, 'pass')
    # the interior estimator at every unsaturated amplitude: both templates, masked and not; 20 / 40 mm edge distance
    for tm in ('pass', 'old'):
        a[f'{det}_mask_in30_{tm}'] = w(0, 3500, 'in', tm)
    a[f'{det}_none_in30_pass'] = w(0, 3500, 'in', 'pass', mk=False)
    a[f'{det}_mask_out30_pass'] = w(0, 3500, 'out', 'pass')
    for d in (20.0, 40.0):
        a[f'{det}_mask_in{d:.0f}_pass'] = w(0, 3500, 'in', 'pass', d=d)
    # mask x amplitude over the whole area (d = -1e9: every event), for the attribution of the low-amplitude T
    for mk in (False, True):
        for lo, hi in ((0, 1000), (1000, 3500)):
            a[f'{det}_{"mask" if mk else "none"}_amp{lo}-{hi}_all_pass'] = w(lo, hi, 'in', 'pass', mk=mk, d=-1e9)
            a[f'{det}_{"mask" if mk else "none"}_amp{lo}-{hi}_in30_pass'] = w(lo, hi, 'in', 'pass', mk=mk)
    json.dump(dict(T0=30 / (V[det] * 1e-3), arms=a), open(f'{D}/spec_where_{det}.json', 'w'), indent=1)
