#!/usr/bin/env python3
"""band_check.py reco|score -- FE item 13: are tracks under a noise band recovered by the band censoring?

The det3 band sample of rerun/satstudy (ids_band.csv: the 252 interior M3 events with a noise band in det3, then 400
controls; golden row sat_det3, 490 V) through the ONE reconstruction (wft.reco.reconstruct_run, current code) on the
production phys bundle (bundle_det3_phys, as every satstudy arm):
  reco --arm cens    the algorithm as it now stands (band strips censored in the fit, flat hits out of the seeding)
  reco --arm nocens  the same with the band censoring off (wft.diag model.BAND_NSIG=0): the 10-27 unified algorithm
  score              detection against M3 (effbudget rule: both views, |r| <= 5 mm, n_dropped <= 2) for these arms and
                     the satstudy references (production 'prod' = the final_phys table, 'reco_band_flat' = flat-hit drop
                     on the 10-26 code) -> results/band_check.json
Run with X17_BENCH_ROOT=~/x17/cosmic_bench/outliers/stage (rr1_cm0 hits, decoded_root, M3 v2 of the golden row).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis'),
                os.path.join(REPO, 'rerun', 'satstudy')]
S = os.path.expanduser('~/x17/cosmic_bench/satstudy')
OUT = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/band_check')
BUNDLE = os.path.join(S, 'bundle_det3_phys')
N_BAND = 252


def reco(arm, jobs):
    if arm == 'nocens':
        os.environ['WFT_DIAG'] = 'model.BAND_NSIG=0'
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run
    ids = set(pd.read_csv(os.path.join(S, 'ids_band.csv')).event_id.astype(int))
    out = os.path.join(OUT, arm, 'events.parquet'); os.makedirs(os.path.dirname(out), exist_ok=True)
    reconstruct_run(get_config('sat_det3'), CalibrationBundle.load(BUNDLE), out, event_filter=ids, jobs=jobs,
                    bundle_path=BUNDLE)


def score():
    from summarize import rays, detected
    R = rays()
    ids = pd.read_csv(os.path.join(S, 'ids_band.csv')).event_id.astype(int).tolist()
    band, ctl = ids[:N_BAND], ids[N_BAND:]
    tabs = {'prod (final_phys)': None, 'flat-hit drop 10-26 (reco_band_flat)': os.path.join(S, 'reco_band_flat', 'events.parquet'),
            'unified, no censoring': os.path.join(OUT, 'nocens', 'events.parquet'),
            'unified + band censoring': os.path.join(OUT, 'cens', 'events.parquet')}
    res = {}
    for name, p in tabs.items():
        if p is None:
            u = list(dict.fromkeys(band + ctl))
            D = pd.DataFrame(dict(detected=R.reindex(u).cat_xy_5.eq('detected').to_numpy()), index=u)
        else:
            if not os.path.exists(p):
                continue
            E = pd.read_parquet(p)
            D = detected(E, R)
        b = D.reindex(band).detected.fillna(False).astype(bool)
        c = D.reindex(ctl).detected.fillna(False).astype(bool)
        r = dict(band_detected=int(b.sum()), n_band=len(band), band_eff=float(b.mean()), ctl_detected=int(c.sum()),
                 n_ctl=len(ctl), ctl_eff=float(c.mean()))
        if p is not None:
            E = pd.read_parquet(p).drop_duplicates('event_id').set_index('event_id')
            if 'x_n_band_strips' in E:
                eb = E.reindex(band)
                r['band_events_with_censored_strip_in_window'] = int(((eb.x_n_band_strips.fillna(0) > 0) |
                                                                      (eb.y_n_band_strips.fillna(0) > 0)).sum())
                ec = E.reindex(ctl)
                r['ctl_events_with_censored_strip_in_window'] = int(((ec.x_n_band_strips.fillna(0) > 0) |
                                                                     (ec.y_n_band_strips.fillna(0) > 0)).sum())
        res[name] = r
        print(f'{name:42s} band {r["band_detected"]}/{r["n_band"]} ({100 * r["band_eff"]:.1f} %)  '
              f'controls {r["ctl_detected"]}/{r["n_ctl"]} ({100 * r["ctl_eff"]:.1f} %)')
    # controls: did anything move between the two unified arms?
    pa, pb = (os.path.join(OUT, a, 'events.parquet') for a in ('nocens', 'cens'))
    if os.path.exists(pa) and os.path.exists(pb):
        A = pd.read_parquet(pa).drop_duplicates('event_id').set_index('event_id')
        B = pd.read_parquet(pb).drop_duplicates('event_id').set_index('event_id')
        c = sorted(set(ctl) & set(A.index) & set(B.index))
        res['controls_views_moved'] = int(sum(((A.loc[c, f'{v}_p0'] - B.loc[c, f'{v}_p0']).abs() > 0.05).sum() for v in 'xy'))
        bb = sorted(set(band) & set(A.index) & set(B.index))
        res['band_views_moved'] = int(sum(((A.loc[bb, f'{v}_p0'] - B.loc[bb, f'{v}_p0']).abs() > 0.05).sum() for v in 'xy'))
        print(f'views moved by the censoring: controls {res["controls_views_moved"]}, band events {res["band_views_moved"]}')
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'band_check.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('reco', 'score'))
    ap.add_argument('--arm', choices=('cens', 'nocens', 'reseed')); ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    reco(a.arm, a.jobs) if a.cmd == 'reco' else score()


if __name__ == '__main__':
    main()
