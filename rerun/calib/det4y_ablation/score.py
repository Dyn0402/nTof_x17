#!/usr/bin/env python3
"""score.py [--root R] [--json out] -- the det4 Y ablation table (DET4Y_ABLATION_2026-10-30.md).

Per arm: s68 / bias / n per view under both fixed alignments (angles_pass, angles_prod), the interior X+Y detection
and far at 5 mm (budget, pass alignment), n events, and the bundle's w0/kw.  Plus the two G1 references as they stand
on EOS: the pass's calib/r3 and the audit's fe_lm (each on its own alignment, as G1 read them).
"""
import argparse
import json
import os

ROW = 'mx17_det4_day_6-24-26/long_run/mx17_4/wft'
R = '/eos/user/d/dneff/x17rerun/det4y_ablation/Analysis'
PASS = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
AUDIT = '/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe'
ARMS = {'a': 'audit fe bundle as is', 'b': 'fe + pass template', 'c': 'fe + pass bow (w0/kw re-measured)',
        'e': 'fe + pass template + pass bow (w0/kw re-measured)', 'd': 'pass bundle (c2, the G1 r3 bundle)'}


def ang(d):
    p = os.path.join(d, 'angular_resolution.json')
    if not os.path.exists(p):
        return None
    a = json.load(open(p))['planes']
    return {v: dict(s68=a[v]['s68_deg'], bias=a[v]['bias_deg'], n=a[v]['n']) for v in 'xy'}


def bud(d):
    p = os.path.join(d, 'budget', 'budget_det4.json')
    if not os.path.exists(p):
        return None
    b = json.load(open(p))['budget'].get('interior|xy|5', {})
    return dict(det5=b.get('detected'), far5=b.get('far'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=R)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    w = os.path.join(a.root, ROW)
    res = {}
    for x, what in ARMS.items():
        r = dict(what=what, pass_=ang(os.path.join(w, f'arm_{x}', 'angles_pass')),
                 prod=ang(os.path.join(w, f'arm_{x}', 'angles_prod')), budget=bud(os.path.join(w, f'arm_{x}')))
        bj = os.path.join(w, f'bundle_{x}', 'bundle.json')
        if os.path.exists(bj):
            b = json.load(open(bj))
            r['w0'], r['kw'] = b.get('w0'), b.get('kw')
            r['bow_tanform'] = bool((b.get('bow') or {}).get('a_y'))
        mp = os.path.join(w, f'arm_{x}', 'events.meta.json')
        if os.path.exists(mp):
            r['n_events'] = json.load(open(mp))['n_events']
        res[x] = r
    res['G1_pass_r3'] = dict(what='G1 pass (calib/r3, own r3 alignment)',
                             pass_=ang(os.path.join(PASS, ROW, 'calib', 'r3', 'angles')),
                             budget=bud(os.path.join(PASS, ROW, 'calib', 'r3')))
    res['G1_audit_fe_lm'] = dict(what='G1 audit (fe_lm, prod alignment, code 5d4f5e4)',
                                 prod=ang(os.path.join(AUDIT, ROW, 'fe_lm', 'angles')),
                                 budget=bud(os.path.join(AUDIT, ROW, 'fe_lm')))
    f = lambda v, n=3: '–' if v is None else f'{v:.{n}f}'
    L = ['| arm | what | n ev | s68 y pass-al | s68 y prod-al | s68 x pass-al | bias y | kw y | w0 y | det 5 mm | far 5 mm |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for x, r in res.items():
        P, Q, B = r.get('pass_') or {}, r.get('prod') or {}, r.get('budget') or {}
        L.append(f"| {x} | {r['what']} | {r.get('n_events', '')} | {f((P.get('y') or {}).get('s68'))} | "
                 f"{f((Q.get('y') or {}).get('s68'))} | {f((P.get('x') or {}).get('s68'))} | "
                 f"{f(((P or Q).get('y') or {}).get('bias'), 3)} | {f((r.get('kw') or {}).get('y'), 4)} | "
                 f"{f((r.get('w0') or {}).get('y'), 4)} | {f(B.get('det5'), 2)} | "
                 f"{f(B.get('far5'), 2)} |")
    print('\n'.join(L))
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
