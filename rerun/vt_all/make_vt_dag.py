#!/usr/bin/env python3
"""make_vt_dag.py <code dir> --out-root R [--dag vt.dag] -- the v(t) DAG of VT_PLAN_2026-10-30: one bench node per run
(vt_bench.py on every non-excluded pass-1 row of the run) and one SPS node per row (vt_sps.py).  Run the DAG from a
directory holding a copy of condor/vt.sub and a log/ dir: condor_submit_dag vt.dag."""
import argparse, csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
#: SPS rows with a defined column (vbox rule, rerun/sps_followup) and >= 6 min of plateau; blocks per row (equal
#: wall-clock windows).  run60 ov07 is left out: beam off for 27 of its 30 minutes (sps_followup iii).
SPS_ROWS = {**{f'run60_flush__ov{i:02d}': 3 for i in (1, 8, 9, 10, 11, 12, 13, 14)},
            'run61_op25__op25': 4, 'run63_flat__flat700': 4, 'run71_raw__raw700': 3, 'run62_rot25_ladder__d700': 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code'); ap.add_argument('--out-root', required=True); ap.add_argument('--dag', default='vt.dag')
    a = ap.parse_args()
    reg = [r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv'))) if r['cls'] != 'excluded']
    runs = {}
    for r in reg:
        runs.setdefault(r['run'], []).append(r['key'])
    L = []
    for i, (run, keys) in enumerate(sorted(runs.items())):
        n = f'bench{i:02d}'
        L += [f'JOB {n} vt.sub', f'VARS {n} node="{n}" code="{a.code}" mem="4000" '
              f'args="vt_bench.py {" ".join(keys)} --out-dir {a.out_root}/bench"', f'RETRY {n} 3']
    for key, nb in SPS_ROWS.items():
        n = 'sps_' + key.replace('__', '_')
        L += [f'JOB {n} vt.sub', f'VARS {n} node="{n}" code="{a.code}" mem="8000" '
              f'args="vt_sps.py {key} --blocks {nb} --per-block 1000 --out-dir {a.out_root}/sps/{key}"', f'RETRY {n} 3']
    open(a.dag, 'w').write('\n'.join(L) + '\n')
    print(f'{a.dag}: {len(runs)} bench nodes ({len(reg)} rows), {len(SPS_ROWS)} SPS nodes')


if __name__ == '__main__':
    main()
