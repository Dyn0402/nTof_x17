#!/usr/bin/env python3
"""make_dag.py <code_dir> [--only k1,k2] [--what budget,angles,bands] -- the Paper I efficiency / loss DAG on the
pass-1 bench tables (rerun/effbudget/PASS1_RESULT_2026-10-31.md). Writes p1eff.dag in the current directory; copy
rerun/condor/stage1.sub and rerun/effbudget/pass1/census.sub next to it and make log/.

Per non-excluded row of rerun/pass_registry.csv (READ-ONLY on Analysis_pass1; every output under OUTROOT):
  BUD_<k>  rerun/effbudget/budget.py on the row's pass table (wft/events.parquet), its group's waveform alignment and
           its candidates sidecar (trigger-track rule arms) -> OUTROOT/budget/<rel>/
  ANG_<k>  mx_june_wft/03_angles.py --fiducial interior_good on the same table / alignment -> OUTROOT/angles/<rel>/
  BND_<k>  band_census.py (decoded files read from EOS, no staging) -> OUTROOT/bands/<rel>/   (one per run/subrun/det)
stage_job runs with --no-sync and an empty --eos-analysis, so it neither pulls nor pushes any Analysis file.
"""
import argparse, csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
OUTROOT = '/eos/user/d/dneff/x17rerun/paper1_eff'
EMPTY = f'{OUTROOT}/empty_analysis'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}


def rel(r):
    return f"{r['run']}/{r['subrun']}/{r['det']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code'); ap.add_argument('--only', default=None)
    ap.add_argument('--what', default='budget,angles,bands')
    ap.add_argument('--registry', default=os.path.join(HERE, '..', '..', 'pass_registry.csv'))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.registry)) if r['cls'] != 'excluded']
    by = {r['key']: r for r in rows}
    only = set(a.only.split(',')) if a.only else None
    what = set(a.what.split(','))
    L = []

    def node(name, sub, args, cpus, mem):
        L.append(f'JOB {name} {sub}')
        L.append(f'VARS {name} code="{a.code}" args="{args}" cpus="{cpus}" mem="{mem}" node="{name}"')
        L.append(f'RETRY {name} 3')

    for r in rows:
        if only and r['key'] not in only:
            continue
        k, n = r['key'], r['key'].replace('.', '_')
        ak = r.get('align_key') or r['calib_key']
        t = f'{A}/{rel(r)}/wft/events.parquet'
        al = f'{A}/{rel(by[ak])}/wft/alignment/alignment.json'
        sj = f'{k} --parts run_config,m3,hits --no-sync --eos-analysis {EMPTY}'
        if 'budget' in what:
            node(f'BUD_{n}', 'stage1.sub', f'{sj} --script rerun/effbudget/budget.py -- {SHORT[r["det"]]} --table {t} '
                 f'--alignment {al} --cand {t.replace(".parquet", ".candidates.parquet")} '
                 f'--out {OUTROOT}/budget/{rel(r)}', 1, 8000)
        if 'angles' in what:
            node(f'ANG_{n}', 'stage1.sub', f'{sj} --script mx_june_wft/03_angles.py -- --table {t} --alignment {al} '
                 f'--fiducial interior_good --out {OUTROOT}/angles/{rel(r)}', 1, 8000)
        if 'bands' in what:
            node(f'BND_{n}', 'census.sub', f'{k} --out {OUTROOT}/bands', 1, 3000)
    open('p1eff.dag', 'w').write('\n'.join(L) + '\n')
    print(f'wrote p1eff.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
