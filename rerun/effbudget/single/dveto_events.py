#!/usr/bin/env python3
"""dveto_events.py <run_key> --bench ROOT --rays rays_<det>.parquet --out DIR [--n-veto-nom3 1500] [--n-ord 600]
[--n-ord-nom3 1500] -- the event sets of the discharge-aware veto study (DISCHARGE_VETO_2026-11-02.md §3).

From the staged seeding hits (X17_BENCH_ROOT layout, rr1 generation), the veto count of every trigger exactly as
wft.seed.seeds_from_hits makes it (reco._load_hits: DROP_FLAT_HITS, then the 10 % relative floor), and from the M3
rays file every trigger with any M3 ray at all.  Four sets, written to <out>/events_<set>.txt and <out>/sets.parquet:
  veto_m3     over the veto, one M3 recipe ray (the budget's rays table: chi2 < 1, NClus = 4, single ray), all
  veto_nom3   over the veto, no M3 ray of any quality (truth: no particle M3 saw), a random sample
  ord_m3      under the veto, one recipe ray, a random sample (bit-identity, ordinary fake rate)
  ord_nom3    under the veto, no M3 ray, a random sample (ordinary fake rate on M3-less triggers)
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bench', required=True)
    ap.add_argument('--rays', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-veto-nom3', type=int, default=1500)
    ap.add_argument('--n-ord', type=int, default=600)
    ap.add_argument('--n-ord-nom3', type=int, default=1500)
    a = ap.parse_args()
    os.environ['X17_BENCH_ROOT'] = a.bench
    sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
    from qa_config import get_config, setup_paths
    setup_paths()
    import uproot
    from wft import seed as ws, reco as wr
    cfg = get_config(a.run_key)
    H = wr._load_hits(cfg)
    n = ws.apply_significance_floor(H[H.feu.isin(cfg.MX17_FEUS)]).groupby('eventId').size()
    m3_any = set()
    for f in sorted(glob.glob(os.path.join(cfg.m3_tracking_dir, '*_rays.root'))):
        t = uproot.open(f)['T'].arrays(['evn', 'rayN'], library='np')
        m3_any |= set(t['evn'][t['rayN'] >= 1].astype(int).tolist())
    R = pd.read_parquet(a.rays)
    recipe = set(R.event_id.astype(int))
    rng = np.random.default_rng(20261102)
    veto = set(n.index[n > ws.SPARK_VETO_HITS].astype(int))
    ordi = set(n.index[n <= ws.SPARK_VETO_HITS].astype(int))

    def pick(s, k):
        s = sorted(s)
        return set(rng.choice(s, min(k, len(s)), replace=False).tolist()) if k and s else set(s)
    sets = dict(veto_m3=veto & recipe, veto_nom3=pick(veto - m3_any, a.n_veto_nom3),
                ord_m3=pick(ordi & recipe, a.n_ord), ord_nom3=pick(ordi - m3_any, a.n_ord_nom3))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for k, s in sets.items():
        open(os.path.join(a.out, f'events_{k}.txt'), 'w').write('\n'.join(str(e) for e in sorted(s)))
        rows += [dict(event_id=e, set=k, n_hits=int(n.get(e, 0))) for e in sorted(s)]
    pd.DataFrame(rows).to_parquet(os.path.join(a.out, 'sets.parquet'))
    open(os.path.join(a.out, 'events_all.txt'), 'w').write('\n'.join(str(e) for e in sorted(set().union(*sets.values()))))
    print(a.run_key, 'triggers with hits', len(n), 'over the veto', len(veto), 'M3 any', len(m3_any), 'recipe', len(recipe),
          {k: len(s) for k, s in sets.items()}, 'veto & no M3 (all)', len(veto - m3_any))


if __name__ == '__main__':
    main()
