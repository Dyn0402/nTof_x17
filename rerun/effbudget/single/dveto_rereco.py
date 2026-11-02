#!/usr/bin/env python3
"""dveto_rereco.py <run_key> --bench ROOT --bundle B --events ids.txt --out DIR [--variants V] [--jobs N]
-- re-reconstruct a set of events once per variant of the discharge-aware veto (DISCHARGE_VETO_2026-11-02.md §3).

The production reconstruction (wft.reco.reconstruct_run, as `wft.cli reco` in the pass) on the listed events, from
staged rr1 inputs (X17_BENCH_ROOT layout).  Each variant is a set of wft.diag overrides and runs in its own process
(WFT_DIAG), writing <out>/<variant>/events.parquet (+ .candidates / .spark_rescue / .meta).
--variants: 'name=overrides;name=overrides', e.g. 'off=seed.SPARK_RESCUE=0;on=seed.SPARK_RESCUE=1' (an empty override = the code's own
algorithm).  The pass-1 bundle is used as it is.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def run_one(a):
    os.environ['X17_BENCH_ROOT'] = a.bench
    sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run
    cfg = get_config(a.run_key)
    ids = {int(x) for x in open(a.events).read().split()}
    reconstruct_run(cfg, CalibrationBundle.load(a.bundle), os.path.join(a.out, 'events.parquet'), event_filter=ids,
                    jobs=a.jobs, bundle_path=a.bundle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bench', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--variants', default='off=seed.SPARK_RESCUE=0;on=seed.SPARK_RESCUE=1')
    ap.add_argument('--one', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        return run_one(a)
    for item in filter(None, a.variants.split(';')):
        name, _, ov = item.partition('=')
        env = dict(os.environ, WFT_DIAG=ov.replace('|', ','))
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), a.run_key, '--bench', a.bench, '--bundle', a.bundle,
               '--events', a.events, '--out', d, '--jobs', str(a.jobs), '--one']
        print(f'[dveto] {a.run_key} {name}: WFT_DIAG={env["WFT_DIAG"]!r}', flush=True)
        with open(os.path.join(d, 'log.txt'), 'w') as log:
            r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f'[dveto] {a.run_key} {name}: rc {r.returncode}', flush=True)


if __name__ == '__main__':
    main()
