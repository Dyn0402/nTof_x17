#!/usr/bin/env python3
"""bowgate_batch.py -- the bow gate tables (bow_node.py --table-out) for a list of bench rows, with xrootd I/O.

Replaces `make_pass_dag.py bowgate` (one stage_job.sh job per row) for the re-finalization of 2026-11-24
(HANDOFF "Chamber A gap ..."; Dylan: every row gets a gate table). stage_job.py stages the row's whole input set and
Analysis subtree through the /eos FUSE mount, which a condor job must not do (claude/lxplus/EOS_JOB_IO.md). This
driver copies only what bow_node.py reads, over xrootd with an adler32 check, and handles several rows per job
Measured 2026-11-24 on the desktop, all 142 rows without a table: 3.2 s per row (max 10 s), 1.1 MB in (max 21),
peak 0.5 GB; 7.5 core-minutes and 150 MB in total, 1 min with 8 in parallel. That is far below one worthwhile condor job
(>= 30 min), so it runs on the desktop; on lxplus the same command uses eosio. The output reproduces the
make_pass_dag.py bowgate table of det7 6-26 short_run to 6e-17.

Per row (registry rerun/pass_registry.csv; the same inputs make_pass_dag.py bowgate gave bow_node.py):
  in : Analysis_pass1/<row>/wft/events.parquet
       Analysis_pass1/<align row>/wft/alignment/alignment.json
       Analysis_pass1/<calib row>/wft/calib_bundle/{bundle.json, arrays.npz, w0kw.json}
       <eos_run_dir>/<subrun>/m3_tracking_root_v2/*_rays.root   (less inputs.EXCLUDED acquisitions, as inputs.plan)
  out: Analysis_pass1/<row>/wft/bow_gate_table.parquet, then bow_gate_block_row.json (the block is written last, so
       its presence on EOS means the table is there and verified). Rows that already have a table are skipped unless
       --force.

    bowgate_batch.py --keys K1,K2,... --work DIR [--push | --local-out DIR] [--force]
    bowgate_batch.py --list-missing [--min-events N]      (prints the registry rows without a table; uses xrootd)

Copies: `eosio` (lxplus; xrdcp --cksum adler32:source, retries) when it exists, else the XRootD Python bindings of
$XRD_PYTHON (desktop: ~/.venvs/xrd/bin/python). Exits non-zero if any row fails, after trying all of them.
"""
import argparse
import csv
import json
import os
import resource
import runpy
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun')]
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
HOST = 'root://eospublic.cern.ch/'
EOSIO = '/afs/cern.ch/user/d/dneff/bin/eosio'
XRD_PY = os.environ.get('XRD_PYTHON', os.path.expanduser('~/.venvs/xrd/bin/python'))

_XRD_SNIPPET = r'''
import sys, json
from XRootD import client
from XRootD.client.flags import DirListFlags, StatInfoFlags
op = sys.argv[1]
if op == 'ls':
    st, l = client.FileSystem("root://eospublic.cern.ch").dirlist(sys.argv[2], DirListFlags.STAT)
    if not st.ok:
        sys.exit(st.message)
    print(json.dumps([e.name for e in l]))
elif op == 'cp':
    cp = client.CopyProcess()
    for i in range(2, len(sys.argv), 2):
        cp.add_job(sys.argv[i], sys.argv[i + 1], force=True, checksummode='end',
                   checksumtype='adler32')
    cp.prepare()
    st, res = cp.run()
    bad = [r for r in res if not r['status'].ok]
    if not st.ok or bad:
        sys.exit(f'copy failed: {st.message} {[str(r["status"]) for r in bad]}')
'''


def on_lxplus():
    return os.path.exists(EOSIO) and shutil.which('xrdfs') is not None


def ls(d):
    if on_lxplus():
        r = subprocess.run(['xrdfs', HOST, 'ls', d], capture_output=True, text=True)
        if r.returncode:
            raise OSError(f'xrdfs ls {d}: {r.stderr.strip()}')
        return [os.path.basename(x) for x in r.stdout.split()]
    r = subprocess.run([XRD_PY, '-c', _XRD_SNIPPET, 'ls', d], capture_output=True, text=True)
    if r.returncode:
        raise OSError(f'ls {d}: {r.stderr.strip()}')
    return json.loads(r.stdout)


def get(pairs):
    """[(eos path, local path)] -> copied and checksum-verified, or OSError."""
    if not pairs:
        return
    if on_lxplus():
        for src, dst in pairs:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            r = subprocess.run([EOSIO, 'in', src, dst], capture_output=True, text=True)
            if r.returncode:
                raise OSError(f'eosio in {src}: {r.stdout[-400:]} {r.stderr[-400:]}')
        return
    args = []
    for src, dst in pairs:
        args += [HOST + src, os.path.abspath(dst)]
    r = subprocess.run([XRD_PY, '-c', _XRD_SNIPPET, 'cp', *args], capture_output=True, text=True)
    if r.returncode:
        raise OSError(r.stderr.strip()[-800:])


def put(src, dst):
    """local -> EOS, verified (eosio out deletes the local copy)."""
    if on_lxplus():
        r = subprocess.run([EOSIO, 'out', src, dst], capture_output=True, text=True)
        if r.returncode:
            raise OSError(f'eosio out {dst}: {r.stdout[-400:]} {r.stderr[-400:]}')
        return
    r = subprocess.run([XRD_PY, '-c', _XRD_SNIPPET, 'cp', os.path.abspath(src), HOST + dst], capture_output=True,
                       text=True)
    if r.returncode:
        raise OSError(r.stderr.strip()[-800:])


def registry():
    rows = [r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv'))) if r['cls'] != 'excluded']
    return rows, {r['key']: r for r in rows}


def rel(r):
    return f"{r['run']}/{r['subrun']}/{r['det']}"


def register(key, row, bench):
    """qa_config entry for the row (build_table calls get_config(key)), as stage_job.py does."""
    import qa_config
    if key not in qa_config.RUNS:
        qa_config.RUNS[key] = qa_config._Config(
            key, row['run'], row['subrun'], feus=[int(row['feu_x']), int(row['feu_y'])],
            det_z=float(row['det_z']), det_name=row['det'], zero_suppressed=False,
            base_path=f"{bench}/{row['tree']}/")


def do_row(key, by_key, work, push, local_out, force):
    import inputs
    r = by_key[key]
    ak = r.get('align_key') or r['calib_key']
    m = inputs.row_for(key)
    w = os.path.join(work, 'row')
    shutil.rmtree(w, ignore_errors=True)
    os.makedirs(w)
    dst = f'{A}/{rel(r)}/wft'
    if push and not force and 'bow_gate_block_row.json' in ls(dst):
        print(f'[bowgate] {key}: table exists, skipped', flush=True)
        return dict(key=key, status='exists')
    t0 = time.time()
    m3src = f"{m['eos_run_dir']}/{r['subrun']}/{inputs.GENERATIONS['rr1']['m3_tracking_root_v2']}"
    excl = inputs.EXCLUDED.get((r['run'], r['subrun']), set())
    m3 = sorted(n for n in ls(m3src) if n.endswith('_rays.root') and inputs.acq(n) and inputs.acq(n)[0] not in excl)
    if not m3:
        raise OSError(f'no M3 rays in {m3src}')
    cb = f"{A}/{rel(by_key[r['calib_key']])}/wft/calib_bundle"
    pairs = [(f'{A}/{rel(r)}/wft/events.parquet', f'{w}/events.parquet'),
             (f'{A}/{rel(by_key[ak])}/wft/alignment/alignment.json', f'{w}/alignment/alignment.json')]
    pairs += [(f'{cb}/{f}', f'{w}/calib_bundle/{f}') for f in ('bundle.json', 'arrays.npz', 'w0kw.json')]
    pairs += [(f'{m3src}/{n}', f'{w}/m3/{n}') for n in m3]
    get(pairs)
    nbytes = sum(os.path.getsize(d) for _, d in pairs)
    t1 = time.time()
    register(key, m, os.path.join(work, 'bench'))
    tab, blk = f'{w}/out/bow_gate_table.parquet', f'{w}/out/bow_gate_block_row.json'
    argv0 = sys.argv
    sys.argv = [os.path.join(HERE, 'bow_node.py'), key, f"det{r['det'][-1]}", '--events', f'{w}/events.parquet',
                '--alignment', f'{w}/alignment/alignment.json', '--m3dir', f'{w}/m3/', '--bundle',
                f'{w}/calib_bundle', '--out', blk, '--table-out', tab]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sys.argv = argv0
    t2 = time.time()
    # provenance: the EOS inputs, not the scratch copies
    b = json.load(open(blk))
    b['provenance'].update(events=pairs[0][0], alignment=pairs[1][0], m3dir=m3src, bundle=cb,
                           driver='rerun/calib/bowgate_batch.py', code=os.environ.get('RERUN_CODE_COMMIT', '?'),
                           m3_files=len(m3))
    json.dump(b, open(blk, 'w'), indent=1)
    import pandas as pd
    n = len(pd.read_parquet(tab, columns=['tan_ref_x']))
    if push:
        put(tab, f'{dst}/bow_gate_table.parquet')
        put(blk, f'{dst}/bow_gate_block_row.json')          # last: its presence implies a verified table
    elif local_out:
        o = os.path.join(local_out, rel(r))
        os.makedirs(o, exist_ok=True)
        shutil.copy(tab, o)
        shutil.copy(blk, o)
    shutil.rmtree(w, ignore_errors=True)
    rec = dict(key=key, status='ok', n_table=n, m3_files=len(m3), in_mb=round(nbytes / 1e6, 1),
               t_stage=round(t1 - t0, 1), t_run=round(t2 - t1, 1), t_total=round(time.time() - t0, 1),
               maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print('[bowgate] ' + json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default='')
    ap.add_argument('--work', default=os.environ.get('_CONDOR_SCRATCH_DIR', '.'))
    ap.add_argument('--push', action='store_true', help='write the outputs to Analysis_pass1 on EOS')
    ap.add_argument('--local-out', default=None, help='or copy them here (test)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--list-missing', action='store_true')
    a = ap.parse_args()
    rows, by_key = registry()
    if a.list_missing:
        for r in rows:
            if 'bow_gate_block_row.json' not in ls(f'{A}/{rel(r)}/wft'):
                print(r['key'])
        return 0
    work = os.path.join(os.path.abspath(a.work), 'bowgate')
    os.makedirs(work, exist_ok=True)
    os.environ['X17_BENCH_ROOT'] = os.path.join(work, 'bench')        # before qa_config is imported
    os.environ['X17_ANALYSIS_ROOT'] = os.path.join(work, 'Analysis')
    sys.path[:0] = [os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
    out, fail = [], []
    for k in [k for k in a.keys.split(',') if k]:
        try:
            out.append(do_row(k, by_key, work, a.push, a.local_out, a.force))
        except Exception:                                   # noqa: BLE001  (try every row, fail the job at the end)
            traceback.print_exc()
            print(f'[bowgate] {k}: FAILED', flush=True)
            fail.append(k)
    shutil.rmtree(work, ignore_errors=True)                 # condor copies scratch files back to AFS
    print(f'[bowgate] done: {len(out)} ok/skipped, {len(fail)} failed {fail}', flush=True)
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
