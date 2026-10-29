#!/usr/bin/env python3
"""g1_gate.py [--pass-root A] [--audit-root B] [--out g1.md] -- gate G1 of the from-scratch pass (PASS_PLAN §3).

Per golden row: the pass's final calibration reconstruction (calib phase CAL4: <A>/<row>/wft/calib/r3/{budget, angles,
events.meta.json}) against the audit's fe_lm arm (<B>/<row>/wft/fe_lm/: the same algorithm on the audit's fe bundles,
the LM gate of 10-28), read by model_audit/gate_compare.read.  Plus each chamber's CALIB_CONVERGENCE.json.

Pass rule (stated): no chamber worse than the audit by more than ~3x the arm-to-arm reproducibility the LM gate measured
(detection +-0.16 pt, far +-0.09 pt, s68 +-0.009 deg) on the same rows and event counts:
    interior X+Y detected at 5 mm   pass - audit >= -0.5 pt
    far at 5 mm                     pass - audit <= +0.3 pt
    s68 per view                    pass - audit <= +0.03 deg
and converged (CALIB_CONVERGENCE.json).  A failure is either fixed or its cost accepted in writing (PASS_PLAN §3).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit'))
from gate_compare import ROWS, read  # noqa: E402

PASS = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
AUDIT = '/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe'
TOL = dict(det5=-0.5, far5=0.3, s68=0.03)


def pct(x):
    return None if x is None else 100.0 * x if x <= 1.0 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pass-root', default=PASS); ap.add_argument('--audit-root', default=AUDIT)
    ap.add_argument('--arm', default='calib/r3'); ap.add_argument('--base', default='fe_lm')
    ap.add_argument('--out', default=None); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    L = ['| chamber | n (pass/audit) | det 5 mm pass / audit (Δ pt) | far 5 mm (Δ pt) | s68 x (Δ°) | s68 y (Δ°) | converged | G1 |',
         '|---|---|---|---|---|---|---|---|']
    res = {}
    for det in ROWS:
        P, B = read(a.pass_root, det, a.arm), read(a.audit_root, det, a.base)
        conv_p = os.path.join(a.pass_root, ROWS[det], 'wft', 'calib', 'CALIB_CONVERGENCE.json')
        conv = json.load(open(conv_p)) if os.path.exists(conv_p) else None
        if not P or not B:
            res[det] = dict(missing=('pass' if not P else 'audit'))
            L.append(f'| {det} | missing {res[det]["missing"]} | | | | | | – |')
            continue
        d = {}
        d['det5'] = pct(P.get('det5')) - pct(B.get('det5')) if P.get('det5') is not None and B.get('det5') is not None else None
        d['far5'] = pct(P.get('far5')) - pct(B.get('far5')) if P.get('far5') is not None and B.get('far5') is not None else None
        for v in 'xy':
            d[f's68_{v}'] = (P[f's68_{v}'] - B[f's68_{v}']) if P.get(f's68_{v}') is not None and B.get(f's68_{v}') is not None else None
        fails = [k for k, t in (('det5', TOL['det5']),) if d[k] is not None and d[k] < t]
        fails += [k for k in ('far5',) if d[k] is not None and d[k] > TOL['far5']]
        fails += [k for k in ('s68_x', 's68_y') if d[k] is not None and d[k] > TOL['s68']]
        missing = [k for k, x in d.items() if x is None]
        ok_conv = bool(conv and conv.get('converged'))
        verdict = 'PASS' if not fails and not missing and ok_conv else 'FAIL'
        res[det] = dict(pass_=P, audit=B, delta=d, fails=fails, missing=missing,
                        converged=ok_conv, convergence_failed=(conv or {}).get('failed'), verdict=verdict)
        f = lambda x, n=2: '–' if x is None else f'{x:.{n}f}'
        L.append(f"| {det} | {P.get('n_events')}/{B.get('n_events')} | {f(pct(P.get('det5')))} / {f(pct(B.get('det5')))} "
                 f"({f(d['det5'], 2)}) | {f(pct(P.get('far5')))} ({f(d['far5'], 2)}) | {f(P.get('s68_x'), 3)} ({f(d['s68_x'], 3)}) "
                 f"| {f(P.get('s68_y'), 3)} ({f(d['s68_y'], 3)}) | {'yes' if ok_conv else 'NO ' + str((conv or {}).get('failed'))} "
                 f"| **{verdict}**{' ' + ','.join(fails + missing) if fails or missing else ''} |")
    txt = '\n'.join(L) + f"\n\nRule: {TOL} (pass - audit; ~3x the LM gate's arm-to-arm reproducibility), plus converged.\n"
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
