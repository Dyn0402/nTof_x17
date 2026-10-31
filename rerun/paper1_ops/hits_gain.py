#!/usr/bin/env python3
"""hits_gain.py --key K --out-dir D -- an M3-free charge proxy per bench row, to tell a chamber's flat gain curve from a
broken reference on one night (Paper I §4; rerun/paper1_charge/README.md excluded the 6-23 det3/det4 night as
"M3 degraded", and det4 is flat at SPS too: rerun/paper1_ops/README.md).

Per trigger, from the pass's hits (combined_hits_root_rr1_cm0; QA, not geometry): per view the strips over the per-plane
significance floor (wft.seed.SIG_REL_FLOOR), their summed amplitude and the lead (largest) amplitude.  MIP-like
triggers: 3 <= strips <= 30 in BOTH views (no discharge, no shower, no near-empty view).  Per row the median and the
25th / 75th percentiles of the lead amplitude and of the summed amplitude, with saturated leads (>= SAT) ranked above.
No M3, no fit: the proxy is only for the SHAPE of a gain curve (d ln Q / dV), never an absolute gain.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO]
from wft.seed import SIG_REL_FLOOR  # noqa: E402

BENCH = '/eos/experiment/ntof/data/x17/cosmic_bench/june_tests'
PASS = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
NMIN, NMAX, SAT = 3, 30, 3550.0


def cq(q, sat, p):
    n = len(q)
    if not n:
        return float('nan')
    k = int(np.floor(p * (n - 1)))
    qu = np.sort(q[~sat])
    return float(qu[k]) if k < len(qu) else float('nan')


def main():
    import uproot
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--registry', default=os.path.join(REPO, 'rerun', 'pass_registry.csv'))
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    r = {x['key']: x for x in csv.DictReader(open(a.registry))}[a.key]
    rel = f"{r['run']}/{r['subrun']}/{r['det']}"
    meta = json.load(open(f'{PASS}/{rel}/wft/events.meta.json'))
    fx, fy = int(meta['run']['feu_x']), int(meta['run']['feu_y'])
    fs = sorted(f for f in glob.glob(f"{BENCH}/{r['run']}/{r['subrun']}/combined_hits_root_rr1_cm0/*.root") if '_datrun_' in f)
    P = []
    for f in fs:
        for d in uproot.iterate(f'{f}:hits', ['eventId', 'feu', 'amplitude', 'significance'], step_size='400 MB',
                                library='pd'):
            d = d[d.feu.isin([fx, fy])]
            thr = d.groupby(['eventId', 'feu']).significance.transform('max') * SIG_REL_FLOOR
            d = d[d.significance >= thr]
            g = d.groupby(['eventId', 'feu']).amplitude.agg(['size', 'sum', 'max']).unstack('feu')
            P.append(g)
    G = pd.concat(P)
    G = G.groupby(level=0).sum() if G.index.duplicated().any() else G
    nx, ny = G[('size', fx)].fillna(0), G[('size', fy)].fillna(0)
    m = (nx >= NMIN) & (nx <= NMAX) & (ny >= NMIN) & (ny <= NMAX)
    G = G[m]
    out = dict(key=a.key, det=r['det'], run=r['run'], subrun=r['subrun'], cls=r['cls'], resist_V=float(r['resist_V']),
               drift_V=float(r['drift_V']), n_mip=int(m.sum()), n_trig=int(len(m)))
    for v, fe in (('x', fx), ('y', fy)):
        lead, tot = G[('max', fe)].to_numpy(float), G[('sum', fe)].to_numpy(float)
        sat = lead >= SAT
        out[v] = dict(f_sat=float(sat.mean()), lead_p25=cq(lead, sat, .25), lead_p50=cq(lead, sat, .5),
                      lead_p75=cq(lead, sat, .75), sum_p25=cq(tot, sat, .25), sum_p50=cq(tot, sat, .5),
                      n_strips_med=float(np.median(G[('size', fe)])))
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, a.key + '.json'), 'w'), indent=1)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
