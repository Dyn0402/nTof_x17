#!/usr/bin/env python3
"""make_pass_dag.py <phase> <code_dir> [--golden det=it_dir ...] -- DAGs for the re-run pass (RERUN_PLAN §4).

Phases (each writes <phase>.dag in the current dir; submit with condor_submit_dag from ~/x17rerun/pass):

finalize  golden rows (S1 close).  Bundle = the converged W12 fit (w12/<it>/fit_pinned/bundle, --golden det=itN).
          RECO (8000 matched events with it) -> ALIGN (01) -> ANGLES (03) -> W0KW (W6: w0/kw measured under this
          bundle) -> <row>/wft/calib_bundle, the bundle every row of the chamber uses.
vrefit    one per condition group with refit=v (S1b, W7).  RECO with the chamber's golden calib_bundle (8000) ->
          ALIGN -> CACHE (ref from this row's waveform alignment, the group's K) -> FITV (v only, kernel pinned)
          -> RECO2 (8000, refitted bundle) -> ALIGN2 -> ANGLES2 -> W0KW -> <row>/wft/calib_bundle.
pass      every non-excluded row (S2 + S3).  RECO (all M3-matched events, the row's bundle: its own calib_bundle if
          it is a calibration row, else its group's) -> [calibration rows] ALIGN -> EFF (02), ANGLES (03), MAPS (04)
          with the group's waveform alignment.
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'


def rel(r):
    return f"{r['run']}/{r['subrun']}/{r['det']}"


EXTRA = {'eos': '', 'limit': ''}


def node(L, name, sub, args, cpus, mem, code, parents=()):
    if sub == 'stage1.sub':
        k, rest = args.split(' ', 1)
        args = f"{k}{EXTRA['eos']} {rest}"
        if ' reco ' in rest and EXTRA['limit'] and '--limit' not in rest:
            args += EXTRA['limit']
    L.append(f'JOB {name} {sub}')
    L.append(f'VARS {name} code="{code}" args="{args}" cpus="{cpus}" mem="{mem}" node="{name}"')
    L.append(f'RETRY {name} 1')
    for p in parents:
        L.append(f'PARENT {p} CHILD {name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['finalize', 'vrefit', 'pass'])
    ap.add_argument('code')
    ap.add_argument('--golden', nargs='*', default=[], help='det=itN (finalize)')
    ap.add_argument('--bundle-sub', default='fit_pinned/bundle',
                    help='finalize: bundle path under w12/<it>/ (arm D: timeD/bundle)')
    ap.add_argument('--out-tag', default='',
                    help='finalize: write wft/final_<tag>/ and wft/calib_bundle_<tag> (an arm next to the canonical one)')
    ap.add_argument('--registry', default=os.path.join(HERE, 'pass_registry.csv'))
    ap.add_argument('--only', default=None, help='comma list of keys (tests)')
    ap.add_argument('--analysis-root', default=None, help='tests: another Analysis tree (e.g. the dry-run one)')
    ap.add_argument('--limit', type=int, default=None, help='tests: events per reco')
    a = ap.parse_args()
    global A
    if a.analysis_root:
        A = a.analysis_root
    if a.analysis_root:
        EXTRA['eos'] = f' --eos-analysis {a.analysis_root}'
    if a.limit:
        EXTRA['limit'] = f' --limit {a.limit}'
    rows = list(csv.DictReader(open(a.registry)))
    by_key = {r['key']: r for r in rows}
    only = set(a.only.split(',')) if a.only else None
    L = []
    if a.phase == 'finalize':
        its = dict(g.split('=') for g in a.golden)            # det2=it1 ...
        for r in rows:
            d = r['det'].replace('mx17_', 'det')
            if r['refit'] != 'golden' or d not in its or (only and r['key'] not in only):
                continue
            k, it = r['key'], its[d]
            b = f'{{out}}/wft/w12/{it}/{a.bundle_sub}'
            sfx = f'_{a.out_tag}' if a.out_tag else ''
            fin = f'{{out}}/wft/final{sfx}'
            t = f'{fin}/events.parquet'
            node(L, f'RECO_{d}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {b} --out {t} '
                 f'--matched-only --limit 8000 --jobs 16', 16, 24000, a.code)
            node(L, f'ALIGN_{d}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py '
                 f'-- --table {t} --out {fin}/alignment', 1, 8000, a.code, [f'RECO_{d}'])
            node(L, f'ANG_{d}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- '
                 f'--table {t} --alignment {fin}/alignment/alignment.json --out {fin}/angles',
                 1, 8000, a.code, [f'ALIGN_{d}'])
            node(L, f'W0KW_{d}', 'stage1.sub', f'{k} --parts run_config --script rerun/w12/set_w0kw.py -- '
                 f'--bundle {b} --angles {fin}/angles --out {{out}}/wft/calib_bundle{sfx}', 1, 4000, a.code,
                 [f'ANG_{d}'])
    elif a.phase == 'vrefit':
        gold = {r['det']: r for r in rows if r['refit'] == 'golden'}
        for r in rows:
            if r['refit'] != 'v' or (only and r['key'] not in only):
                continue
            k, n = r['key'], r['key'].replace('.', '_')
            g = f"{A}/{rel(gold[r['det']])}/wft/calib_bundle"
            t1, t2 = '{out}/wft/vrefit/events0.parquet', '{out}/wft/vrefit/events1.parquet'
            node(L, f'RECO_{n}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {g} --out {t1} '
                 f'--matched-only --limit 8000 --jobs 16', 16, 24000, a.code)
            node(L, f'ALIGN_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py '
                 f'-- --table {t1} --out {{out}}/wft/vrefit/alignment0', 1, 8000, a.code, [f'RECO_{n}'])
            node(L, f'HITS_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --require cache/event_results_veto50.pkl '
                 f'--script mx_june_cosmic_qa/03_alignment_and_tpc.py', 4, 12000, a.code)
            node(L, f'CACHE_{n}', 'stage1.sub', f'{k} --script rerun/w12/build_cache_wft.py -- '
                 f'--align {{out}}/wft/vrefit/alignment0/alignment.json --out {{out}}/wft/vrefit/cache.pkl --cm 0',
                 2, 16000, a.code, [f'ALIGN_{n}', f'HITS_{n}'])
            node(L, f'FITV_{n}', 'fitv1.sub', f'{rel(r)} {g} {r["K"]} vrefit/fit {A}', 8, 12000, a.code, [f'CACHE_{n}'])
            node(L, f'RECO2_{n}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {{out}}/wft/vrefit/fit/bundle '
                 f'--out {t2} --matched-only --limit 8000 --jobs 16', 16, 24000, a.code, [f'FITV_{n}'])
            node(L, f'ALIGN2_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py '
                 f'-- --table {t2} --out {{out}}/wft/vrefit/alignment1', 1, 8000, a.code, [f'RECO2_{n}'])
            node(L, f'ANG_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- '
                 f'--table {t2} --alignment {{out}}/wft/vrefit/alignment1/alignment.json --out {{out}}/wft/vrefit/angles',
                 1, 8000, a.code, [f'ALIGN2_{n}'])
            node(L, f'W0KW_{n}', 'stage1.sub', f'{k} --parts run_config --script rerun/w12/set_w0kw.py -- '
                 f'--bundle {{out}}/wft/vrefit/fit/bundle --angles {{out}}/wft/vrefit/angles --out {{out}}/wft/calib_bundle',
                 1, 4000, a.code, [f'ANG_{n}'])
    else:
        for r in rows:
            if r['cls'] == 'excluded' or (only and r['key'] not in only):
                continue
            k, n = r['key'], r['key'].replace('.', '_')
            ck = by_key[r['calib_key']]
            own = r['key'] == r['calib_key']
            b = '{out}/wft/calib_bundle' if own else f'{A}/{rel(ck)}/wft/calib_bundle'
            al = '{out}/wft/alignment/alignment.json' if own else f'{A}/{rel(ck)}/wft/alignment/alignment.json'
            node(L, f'RECO_{n}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {b} --matched-only '
                 f'--jobs 16', 16, 24000, a.code)
            after = [f'RECO_{n}']
            if own:
                node(L, f'ALIGN_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py',
                     1, 8000, a.code, after)
                after = [f'ALIGN_{n}']
            for s, sc in (('EFF', '02_efficiency'), ('ANG', '03_angles'), ('MAPS', '04_maps')):
                node(L, f'{s}_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/{sc}.py -- '
                     f'--alignment {al}', 1, 8000, a.code, after)
        # a group's other rows read the group's alignment: they wait for it
        for r in rows:
            if r['cls'] == 'excluded' or (only and r['key'] not in only) or r['key'] == r['calib_key']:
                continue
            ck = by_key[r['calib_key']]
            if only and ck['key'] not in only:
                continue
            n, cn = r['key'].replace('.', '_'), ck['key'].replace('.', '_')
            for s in ('EFF', 'ANG', 'MAPS'):
                L.append(f'PARENT ALIGN_{cn} CHILD {s}_{n}')
    open(f'{a.phase}.dag', 'w').write('\n'.join(L) + '\n')
    print(f'wrote {a.phase}.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
