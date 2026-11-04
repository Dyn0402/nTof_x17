#!/usr/bin/env python3
"""make_masks.py -- the n_TOF channel layer: one mask per (chamber, mask era), from the census of census.py (P2).

    make_masks.py [--persist 0.25]

Inputs: $WORK_DATA/ntof_pass1/channel_layer/{sample.csv, acq/*.json, acq/*.npz} (census.py), the bench lists
(rerun/channel_layer/<det>_<key>.json census dead + shorts; the pass-1 golden bundles' dead lists, rule
rerun/model_audit/followups/fe_unify2/masks/bench_<det>.json), the F12 chamber-side / electronics classes
(rerun/model_audit/ws_dead_charge/f12_table.json).

Mask eras (named here; the registry does not carry them yet: see README "Proposed registry column"):
  E1_pre_access   access_epoch pre_access_27jul (run_79, run_80, run_81 up to the 27 July access): A-x connector 8
                  (FEU 3 ch 448-511) electrically disconnected, A common mode x3.3
  E2_post_access  every later acquisition of the pass-1 scope (run_83 ... run_162)
An acquisition in which a whole connector is silent that is live in the rest of its era is listed as a dropout
episode (dropouts.json); it would become its own era if any were found inside the pass-1 scope.

Per (arm, era) the mask is built under the F12 rule (chamber-side faults by position + this dataset's own faults from
its own hits; electronics faults of other datasets are not carried) plus the SPS hot rules:
  chamber_side   F12 class 'detector' (a fault seen through >= 2 different electronics), carried by position, minus
                 positions (not F12 shorts) that this census judges healthy (the same release as bench_carried)
  bench_carried  the bench lists, minus F12 'electronics' channels (they follow the bench FEU channel, not the strip),
                 minus bench-dead (not short) channels the n_TOF census judges healthy (flank f >= 0.6 on >= 20 flank
                 events, not silent); bench shorts are not disproved by a flank test and carry
  ntof_dead      flank dead (f < 0.25) on the era's summed flank counts
  ntof_silent    occupancy < 5 % of the neighbourhood on the era's summed occupancy
  ntof_shorts    short groups on the era's summed amplitude products
  hot            SPS seed probe (> 3 % of a 1000-event block, > 10x the live +-4 neighbours) hot in >= --persist of the
                 era's sampled acquisitions; near oscillators the same
  coupled        SPS coupled channels, persistent pairs (listed; see README for how the reco would use them)
Output: rerun/ntof/channel_layer/masks/ntof_<arm>_<era>.json (bundle-ready 'dead' and 'hot' lists plus every
component, its evidence and the comparison with the bench lists), masks/summary.json, and the per-acquisition table
$WORK_DATA/ntof_pass1/channel_layer/acq_table.csv.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
import rules  # noqa: E402
from census import WORK, POS, FEU_ARM, BENCH_DET, chamber_side_prior  # noqa: E402

ERAS = {'pre_access_27jul': 'E1_pre_access', 'post_access_27jul': 'E2_post_access'}
ARM_FEU = {(a, v): f for f, (a, v) in FEU_ARM.items()}
BENCH_CENSUS = {'A': 'det3_sat_det3.json', 'B': 'det2_o22_long_det2.json', 'C': 'det6_g_det6_long.json',
                'D': 'det7_g_det7_long.json'}
HEALTHY_F, HEALTHY_N = 0.6, 20               # build_masks.py / mask_audit.py flank-healthy rule
SHORT_REPLICATE = 0.5
#: beam-occupancy hot (sept26_prelim_analysis/candidate_filter.py HOT_FLOOR / HOT_K, the old run_145 102-channel mask):
#: occupancy on the era's BEAM acquisitions > 8 % and > 8x the view median.  It catches the contiguous blocks the SPS
#: seed rule cannot (that rule compares a channel with its neighbours): D-x 92-112 / 451-464, D-y 42-63 read 10-21x the
#: median on beam and 1-2x on beam-off cosmics, so they are beam-induced pickup, not dead strips.  They go to the
#: bundle's ``hot`` wildcard (never seeded, noise x HOT_NOISE_INFLATION; HANDOFF_D_NOISY_CHANNELS.md: do not kill the
#: tracks that cross them), not to ``dead``.
OCC_HOT_FLOOR, OCC_HOT_K = 0.08, 8.0
CONN_DROP = 32                               # >= half a connector silent in one acquisition = a connector dropout
OUT = os.path.join(HERE, 'masks')
L = lambda b: [int(c) for c in np.flatnonzero(b)]


def bench_lists():
    """{arm: {view: dict(census=set, bundle=set, electronics=set)}}"""
    T = json.load(open(os.path.join(REPO, 'rerun', 'model_audit', 'ws_dead_charge', 'f12_table.json')))
    out = {}
    for arm, det in BENCH_DET.items():
        cen = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', BENCH_CENSUS[arm])))
        fe = json.load(open(os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fe_unify2', 'masks', f'bench_{det}.json')))
        out[arm] = {}
        for v in 'xy':
            el, cs_short = set(), set()
            for r in T[f'{det}_{v}']['positions']:
                if r['cls'] == 'detector' and any('S' in lab for lab in r['labels'].values()):
                    j = int(np.argmin(np.abs(POS - r['pos'])))
                    if abs(POS[j] - r['pos']) < 0.3:
                        cs_short.add(j)
                if r['cls'] == 'electronics':
                    j = int(np.argmin(np.abs(POS - r['pos'])))
                    if abs(POS[j] - r['pos']) < 0.3:
                        el.add(j)
            out[arm][v] = dict(census=set(cen[v]['dead']) | set(cen[v]['shorts']), census_shorts=set(cen[v]['shorts']),
                               bundle=set(fe['planes'][v]['dead']), electronics=el, chamber_side_short=cs_short)
    return out


def load():
    rows = {r['key']: r for r in csv.DictReader(open(os.path.join(WORK, 'sample.csv')))}
    recs = []
    for k, r in rows.items():
        js = os.path.join(WORK, 'acq', k + '.json')
        if os.path.exists(js):
            d = json.load(open(js)); d['access_epoch'] = r['access_epoch']; d['era'] = ERAS[r['access_epoch']]
            recs.append(d)
    recs.sort(key=lambda d: (d['t_local'], d['key']))
    return recs


_Z = {}


def npz(key):
    if key not in _Z:
        z = np.load(os.path.join(WORK, 'acq', key + '.npz'))
        _Z[key] = {k: z[k] for k in z.files if not k.startswith('S_')}
        _Z[key]['_has_S'] = {k for k in z.files if k.startswith('S_')}
    return _Z[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--persist', type=float, default=0.25, help='hot in >= this fraction of an era\'s acquisitions')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    recs = load()
    prior = chamber_side_prior()
    bench = bench_lists()
    print(f'{len(recs)} probed acquisitions')

    # ---- connector dropouts: a connector >= half silent in an acquisition (silent re-decided from the stored
    # occupancy with the statistical floor rules.SILENT_MIN_EXPECTED)
    drops = defaultdict(list)
    for d in recs:
        z = npz(d['key'])
        for f, r in d['feu'].items():
            sil = rules.flank_decide(dict(n_events=int(z['n_events']), occ=z[f'occ_{f}'], n1=z[f'n1_{f}'], f1=z[f'f1_{f}'],
                                          n2=z[f'n2_{f}'], f2=z[f'f2_{f}']), POS)['silent']
            r['conn_silent'] = [int(sil[64 * k:64 * k + 64].sum()) for k in range(8)]
            for k, n in enumerate(r['conn_silent']):
                if n >= CONN_DROP:
                    drops[(r['arm'], r['view'], k + 1)].append(dict(key=d['key'], run=d['run'], era=d['era'],
                                                                    t_local=d['t_local'], n_silent=n))
    # ---- per-acquisition table
    tab = []
    for d in recs:
        for f, r in d['feu'].items():
            tab.append(dict(key=d['key'], run=d['run'], subrun=d['subrun'], tag=d['tag'], rclass=d['rclass'],
                            t_local=d['t_local'], era=d['era'], arm=r['arm'], view=r['view'], n_events=d['n_events'],
                            median_occ=r['median_occ'], busy_frac=r['busy_frac'], n_silent=len(r['silent']),
                            n_flank_dead=len(r['flank_dead']), n_hot=len(r['hot']), hot=' '.join(map(str, r['hot'])),
                            near_osc=' '.join(map(str, r['near_osc'])),
                            coupled=';'.join(f'{o}:{",".join(map(str, c))}' for o, c in r['coupled'].items()),
                            conn_silent=' '.join(map(str, r['conn_silent'])),
                            conn_occ=' '.join(f'{x:.4f}' for x in r['conn_occ'])))
    with open(os.path.join(WORK, 'acq_table.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(tab[0])); w.writeheader(); w.writerows(tab)

    n_era = Counter(d['era'] for d in recs)
    dropouts = []
    for (arm, v, conn), lst in sorted(drops.items()):
        by_era = Counter(x['era'] for x in lst)
        for era, n in by_era.items():
            dropouts.append(dict(arm=arm, view=v, connector=conn, channels=[64 * (conn - 1), 64 * conn - 1], era=era,
                                 n_acq=n, n_acq_era=n_era[era], whole_era=n >= 0.9 * n_era[era],
                                 runs=sorted({x['run'] for x in lst if x['era'] == era}, key=int),
                                 first=min(x['t_local'] for x in lst if x['era'] == era),
                                 last=max(x['t_local'] for x in lst if x['era'] == era)))
    json.dump(dropouts, open(os.path.join(OUT, 'dropouts.json'), 'w'), indent=1)
    for x in dropouts:
        print(f"connector dropout {x['arm']}-{x['view']} conn {x['connector']} ({x['channels']}): {x['era']} "
              f"{x['n_acq']}/{x['n_acq_era']} acquisitions, runs {x['runs'][:8]}{'...' if len(x['runs']) > 8 else ''}")

    # ---- per (arm, era)
    summary = dict(persist=a.persist, n_acq=dict(n_era), masks={})
    for era in sorted(n_era):
        E = [d for d in recs if d['era'] == era]
        for arm in 'ABCD':
            rec = dict(arm=arm, bench_chamber=BENCH_DET[arm], mask_era=era, n_acquisitions=len(E),
                       runs=sorted({d['run'] for d in E}, key=int),
                       n_events=int(sum(d['n_events'] for d in E)),
                       channel_numbering=f'n_TOF FEU channel (x FEU {ARM_FEU[(arm, "x")]}, y FEU {ARM_FEU[(arm, "y")]}); '
                                         'identical to the bench channel numbering of the chamber (cabling_check.json)',
                       planes={})
            for v in 'xy':
                feu = ARM_FEU[(arm, v)]
                acc = dict(n_events=0, occ=0, n1=0, f1=0, n2=0, f2=0)
                occ_cls = {'production': [0, 0], 'cosmic': [0, 0]}
                S = N = None; n_corr = 0; short_votes = Counter()
                hotc, nearc, cplc = Counter(), Counter(), Counter()
                n_vote = 0
                for d in E:
                    z = npz(d['key'])
                    acc['n_events'] += int(z['n_events'])
                    oc = occ_cls[d['rclass']]; oc[0] = oc[0] + z[f'occ_{feu}']; oc[1] += int(z['n_events'])
                    for k in ('occ', 'n1', 'f1', 'n2', 'f2'):
                        acc[k] = acc[k] + z[f'{k}_{feu}'].astype(np.int64)
                    if f'S_{feu}' in z['_has_S']:
                        Sz = np.load(os.path.join(WORK, 'acq', d['key'] + '.npz'))[f'S_{feu}'].astype(float)
                        short_votes.update(L(rules.short_decide(Sz, z[f'N_{feu}'], POS)['shorts']))
                        S = Sz if S is None else S + Sz
                        N = z[f'N_{feu}'] if N is None else N + z[f'N_{feu}']
                        n_corr += 1
                    r = d['feu'][str(feu)]
                    if d['n_events'] < rules.HOT_MIN_EVENTS:
                        continue
                    n_vote += 1
                    hotc.update(r['hot']); nearc.update(r['near_osc'])
                    for o, cs in r['coupled'].items():
                        cplc.update((int(o), int(c)) for c in cs)
                dec = rules.flank_decide(acc, POS)
                np.save(os.path.join(WORK, f'era_occ_{arm}_{v}_{era}.npy'), dec['occ'])
                sh = rules.short_decide(S, N, POS) if S is not None else dict(shorts=np.zeros(512, bool), groups=[], bands=[])
                # a short is a fixed connection: it must replicate in >= SHORT_REPLICATE of the era's per-run acquisitions
                # (each judged alone), not only in the sum (coherent noise, e.g. A's x3.3 common mode in E1, makes
                # one-off correlated pairs)
                rep = np.zeros(512, bool)
                rep[[c for c, k in short_votes.items() if k >= SHORT_REPLICATE * max(n_corr, 1)]] = True
                sh_raw = L(sh['shorts'])
                sh['shorts'] = sh['shorts'] & rep
                # a connector that is >= half silent on the era sum is dead as a whole (its live-looking channels are
                # the few that pick up noise: A-x connector 8 in E1 is 63/64 silent)
                sil = dec['silent'].copy()
                conn_dead = [k + 1 for k in range(8) if sil[64 * k:64 * k + 64].sum() >= CONN_DROP]
                for k in conn_dead:
                    sil[64 * (k - 1):64 * k] = True
                dec['silent'] = sil
                n = max(n_vote, 1)
                hot = np.zeros(512, bool); hot[[c for c, k in hotc.items() if k >= a.persist * n]] = True
                near = np.zeros(512, bool); near[[c for c, k in nearc.items() if k >= a.persist * n]] = True
                coupled = sorted([o, c, k] for (o, c), k in cplc.items() if k >= a.persist * n)
                cs = prior[arm][v]
                b = bench[arm][v]
                blist = b['census'] | b['bundle']
                healthy = dec['judged'] & (np.nan_to_num(dec['f'], nan=0) >= HEALTHY_F) & ~dec['silent'] & ~sh['shorts']
                ok_n = (np.maximum(acc['n1'], acc['n2']) >= HEALTHY_N)
                healthy &= ok_n
                # a short fires (it carries its group's charge), so the flank test cannot disprove a bench short:
                # bench shorts carry unless F12 puts them on the bench electronics
                disproved = lambda c: healthy[c] and c not in b['census_shorts']
                carried = sorted(c for c in blist if c not in b['electronics'] and not disproved(c))
                dropped_el = sorted(c for c in blist if c in b['electronics'])
                dropped_healthy = sorted(c for c in blist if c not in b['electronics'] and disproved(c))
                # the same release applies to the F12 chamber-side prior: a position F12 flagged dead/silent (not short)
                # that this census, on the era's whole sample, sees firing beside bright neighbours is live at n_TOF
                cs_released = [c for c in L(cs) if healthy[c] and c not in b['chamber_side_short']]
                cs = cs.copy(); cs[cs_released] = False
                dead = cs.copy()
                dead[carried] = True
                dead |= dec['dead'] | dec['silent'] | sh['shorts'] | hot | near
                ob = np.asarray(occ_cls['production'][0], float) / max(occ_cls['production'][1], 1)
                oco = np.asarray(occ_cls['cosmic'][0], float) / max(occ_cls['cosmic'][1], 1)
                occ_hot = (ob > max(OCC_HOT_FLOOR, OCC_HOT_K * np.median(ob))) if occ_cls['production'][1] else np.zeros(512, bool)
                occ_hot &= ~dead
                cpl_ch = sorted(({c for _o, c, _k in coupled} | set(L(occ_hot))) - set(L(dead)))
                rec['planes'][v] = dict(
                    feu=feu,
                    dead=L(dead), hot=cpl_ch,
                    n_dead=int(dead.sum()),
                    components=dict(chamber_side=L(cs), chamber_side_released=cs_released, bench_carried=carried, ntof_flank_dead=L(dec['dead']),
                                    ntof_silent=L(dec['silent']), ntof_shorts=L(sh['shorts']),
                                    ntof_short_groups=sh['groups'], ntof_shorts_not_replicated=sorted(set(sh_raw) - set(L(sh['shorts']))),
                                    dead_connectors=conn_dead, ntof_bands=sh['bands'],
                                    hot=L(hot), near_osc=L(near), coupled=coupled, beam_occupancy_hot=L(occ_hot), weak_not_masked=L(dec['weak'])),
                    beam_occupancy_hot_evidence={str(c): dict(beam_over_median=round(float(ob[c] / max(np.median(ob), 1e-9)), 1),
                                                              cosmic_over_median=round(float(oco[c] / max(np.median(oco), 1e-9)), 1))
                                                 for c in L(occ_hot)},
                    hot_frequency={str(c): round(k / n, 3) for c, k in sorted(hotc.items()) if k >= 2},
                    bench=dict(census_dead_shorts=sorted(b['census']), pass1_bundle_dead=sorted(b['bundle']),
                               dropped_electronics=dropped_el, dropped_healthy_at_ntof=dropped_healthy,
                               f_at_ntof={str(c): (None if not np.isfinite(dec['f'][c]) else round(float(dec['f'][c]), 3))
                                          for c in sorted(blist)}),
                    stats=dict(n_events=acc['n_events'], n_hot_votes=n_vote, n_short_acq=n_corr, n_judged=int(dec['judged'].sum()),
                               median_occ=round(float(np.median(dec['occ'])), 5)))
            rec['rule'] = __doc__.split('\n\n')[3].strip()
            rec['source'] = 'rerun/ntof/channel_layer/make_masks.py'
            json.dump(rec, open(os.path.join(OUT, f'ntof_{arm}_{era}.json'), 'w'), indent=1)
            summary['masks'][f'{arm}_{era}'] = {v: dict(n_dead=rec['planes'][v]['n_dead'], n_hot_list=len(rec['planes'][v]['hot']),
                                                       **{k: len(x) for k, x in rec['planes'][v]['components'].items()
                                                          if isinstance(x, list) and k not in ('ntof_short_groups', 'ntof_bands', 'dead_connectors')})
                                                for v in 'xy'}
            print(f"{arm} {era}: x dead {rec['planes']['x']['n_dead']} hot {len(rec['planes']['x']['hot'])} | "
                  f"y dead {rec['planes']['y']['n_dead']} hot {len(rec['planes']['y']['hot'])}")
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
