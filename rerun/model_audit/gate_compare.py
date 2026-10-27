#!/usr/bin/env python3
"""gate_compare.py --root DIR [--arms band headon twotrack all] [--base prod] [--json out] -- the fleet gate table.

DIR mirrors Analysis_audit: <root>/<row>/wft/<arm>/{budget/budget_<det>.json, angles/angular_resolution.json,
events.meta.json}.  Per chamber and arm: interior X+Y detected / rejected_cut / not_seeded / fit_failed / far at
5 mm and 3 mm, per-view far, s68 and bias per view, n events, n_tracks >= 2, splits, rescued; arm − base in points.
"""
import argparse, glob, json, os

ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2', 'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}


def read(root, det, arm):
    d = os.path.join(root, ROWS[det], 'wft', arm)
    out = {}
    bp = os.path.join(d, 'budget', f'budget_{det}.json')
    if os.path.exists(bp):
        b = json.load(open(bp))['budget']
        for r in (5, 3):
            k = b.get(f'interior|xy|{r}', {})
            out[f'det{r}'] = k.get('detected'); out[f'far{r}'] = k.get('far'); out[f'rej{r}'] = k.get('rejected_cut')
            out[f'nseed{r}'] = k.get('not_seeded'); out[f'fitf{r}'] = k.get('fit_failed'); out[f'n{r}'] = k.get('n')
            out[f'sum{r}'] = (k.get('detected', 0) or 0) + (k.get('far', 0) or 0) + (k.get('rejected_cut', 0) or 0)
        for v in 'xy':
            k = b.get(f'interior|{v}|5', {}); out[f'far5_{v}'] = k.get('far'); out[f'det5_{v}'] = k.get('detected')
    ap = os.path.join(d, 'angles', 'angular_resolution.json')
    if os.path.exists(ap):
        a = json.load(open(ap))['planes']
        for v in 'xy':
            out[f's68_{v}'] = a[v]['s68_deg']; out[f'bias_{v}'] = a[v]['bias_deg']; out[f'nang_{v}'] = a[v]['n']
            out[f'vspread_{v}'] = a[v]['implied_v_spread']
    mp = os.path.join(d, 'events.meta.json')
    if os.path.exists(mp):
        m = json.load(open(mp)); out['n_events'] = m['n_events']; out['n_multi'] = m['multi_track'].get('n_events_multitrack')
        out['n_splits'] = m['multi_track'].get('n_splits'); out['sel'] = m['selection']
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--base', default='prod')
    ap.add_argument('--arms', nargs='*', default=['band', 'headon', 'twotrack', 'all']); ap.add_argument('--json', default=None)
    a = ap.parse_args(); res = {}; L = []
    L += ['| det arm | n events | interior det 5 mm | Δdet | far 5 | Δfar | rej | not seeded | fit failed | det 3 mm | Δdet3 | s68 x / y | Δs68 x / y | bias x / y | multi | splits |',
          '|' + '---|' * 16]
    for det in ROWS:
        B = read(a.root, det, a.base)
        if not B:
            continue
        res[f'{det}|{a.base}'] = B
        f = lambda x, fmt='.2f': ('—' if x is None else format(x, fmt))
        L.append(f"| {det} {a.base} | {B.get('n_events')} | {f(B.get('det5'))} | | {f(B.get('far5'))} | | {f(B.get('rej5'))} | {f(B.get('nseed5'))} | {f(B.get('fitf5'))} | "
                 f"{f(B.get('det3'))} | | {f(B.get('s68_x'),'.3f')} / {f(B.get('s68_y'),'.3f')} | | {f(B.get('bias_x'),'+.3f')} / {f(B.get('bias_y'),'+.3f')} | {B.get('n_multi')} | {B.get('n_splits')} |")
        for arm in a.arms:
            T = read(a.root, det, arm)
            if not T:
                continue
            res[f'{det}|{arm}'] = T
            dd = lambda k, fmt='+.2f': ('—' if T.get(k) is None or B.get(k) is None else format(T[k] - B[k], fmt))
            L.append(f"| {det} {arm} | {T.get('n_events')} | {f(T.get('det5'))} | {dd('det5')} | {f(T.get('far5'))} | {dd('far5')} | {f(T.get('rej5'))} | {f(T.get('nseed5'))} | {f(T.get('fitf5'))} | "
                     f"{f(T.get('det3'))} | {dd('det3')} | {f(T.get('s68_x'),'.3f')} / {f(T.get('s68_y'),'.3f')} | {dd('s68_x','+.3f')} / {dd('s68_y','+.3f')} | "
                     f"{f(T.get('bias_x'),'+.3f')} / {f(T.get('bias_y'),'+.3f')} | {T.get('n_multi')} | {T.get('n_splits')} |")
    md = '\n'.join(L) + '\n'; print(md)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1); open(a.json.replace('.json', '.md'), 'w').write(md)


if __name__ == '__main__':
    main()
