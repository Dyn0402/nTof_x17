#!/usr/bin/env python3
"""dveto_summary.py --runs DIR --budget DIR [--out results/discharge_veto.json]
-- one table per chamber from dveto_score.py / dveto_gate.py outputs (DISCHARGE_VETO_2026-11-02.md §4).

Per row: the vetoed M3-tagged events of the sample (and of the row), what pass 1 already fits among them (band re-seed),
the rescue attempts under no gate / the rescue gate / the gate + the production n_dropped rule, real (<= 5 mm) and
fake (> 5 mm) against M3, by A1 class (interior); M3-less vetoed triggers with an accepted rescue; and the ordinary
reference from the row's full pass-1 budget rays (interior, under the veto, both views fitted: fraction > 5 mm).
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROWS = {'sat_det3': ('mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3', 'det3'),
        'g_det3_wknd': ('mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3', 'det3'),
        'o22_long_det2': ('mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2', 'det2'),
        'g_det4': ('mx17_det4_day_6-24-26/long_run/mx17_4', 'det4'),
        'g_det6_long': ('mx17_det6_det7_overnight_6-26-26/long_run/mx17_6', 'det6'),
        'g_det7_long': ('mx17_det6_det7_overnight_6-26-26/long_run/mx17_7', 'det7')}


def ordinary_ref(rays):
    R = pd.read_parquet(rays)
    R = R[(R.region == 'interior') & (R.mult <= 50) & R.x_ok.astype(bool) & R.y_ok.astype(bool)]
    far = (R.r > 5).to_numpy()
    return dict(n=int(len(R)), fake_frac=float(far.mean()), purity=float(1 - far.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--budget', required=True)
    ap.add_argument('--sets', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = {}
    for key, (rel, det) in ROWS.items():
        g = os.path.join(a.runs, key, 'gate_on.json')
        s = os.path.join(a.runs, key, 'score.json')
        if not os.path.exists(g):
            continue
        G, S = json.load(open(g)), json.load(open(s))
        full = pd.read_parquet(os.path.join(a.sets, key, 'sets_full.parquet'))
        o = dict(det=det, n_veto_m3_row=int((full.set == 'veto_m3').sum()), n_sets=G['n_sets'],
                 ordinary=ordinary_ref(os.path.join(a.budget, rel, f'rays_{det}.parquet')),
                 pass1_on_sample=S['variants'].get('pass1', {}).get('veto_m3'),
                 on_sample=S['variants']['on'].get('veto_m3'), on_nom3=S['variants']['on'].get('veto_nom3'),
                 ord_nom3=S['variants']['on'].get('ord_nom3'), identity=S.get('identity'),
                 gates=G['rescue_attempts'], rescued_quality=G.get('rescued_quality'),
                 rescue=G.get('rescue'))
        out[key] = o
        ga = G['rescue_attempts']
        print(f"{key:14s} veto_m3 {G['n_sets'].get('veto_m3')}/{o['n_veto_m3_row']}  pass1 real "
              f"{(o['pass1_on_sample'] or {}).get('real')}  ungated {ga['none']['real']}/{ga['none']['fake_m3']}  "
              f"gate {ga['full']['real']}/{ga['full']['fake_m3']}  nom3 {ga['full']['fake_nom3']}/{G['n_sets'].get('veto_nom3')}  "
              f"ordinary purity {o['ordinary']['purity']:.3f}")
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
