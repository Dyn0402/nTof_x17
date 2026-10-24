#!/usr/bin/env python3
"""rerun/stage_job.py -- run one stage of the re-run for one manifest row, anywhere (RERUN_PLAN W10).

    stage_job.py <key> [--gen rr1] [--parts run_config,m3,decoded,hits] [--work DIR] [--no-sync]
                 (--module wft.calibrate | --script mx_june_wft/01_alignment.py) [-- stage args...]

The stage gets [key, *args], or args alone with '{key}' substituted when they contain it (wft.cli reco {key});
'{out}' in an argument becomes the row's Analysis dir (cfg.OUT_BASE) inside the job.

1. stage the row's inputs from EOS into <work>/bench (rerun/inputs.py, canonical directory names, INPUTS.json);
2. pull the row's Analysis subtree <EOS_ANALYSIS_ROOT>/<run>/<subrun>/<det>/ into <work>/Analysis (earlier stages'
   outputs: alignment, caches, bundles, tables);
3. set X17_BENCH_ROOT / X17_ANALYSIS_ROOT, register the row in qa_config.RUNS (its own entry if the key is a
   registered run, else one synthesized from the manifest row, as run_reco_job.py does);
4. run the stage in-process: a module's main(argv) (``--module``; used for anything that pickles workers, e.g.
   wft.calibrate) or a script via runpy with argv [script, key, *args];
5. push the Analysis subtree back to EOS (new or changed files only) with a STAGES.log line.

The stage code is not edited for any of this: it sees the desktop layout.
"""
import argparse
import datetime
import filecmp
import importlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]
import inputs  # noqa: E402


def snapshot(top):
    """{relative path: (size, mtime_ns)} of every file under top."""
    out = {}
    for root, _, files in os.walk(top):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out[os.path.relpath(p, top)] = (st.st_size, st.st_mtime_ns)
    return out


def sync(src, dst, only=None):
    """Copy files under src to dst when missing or different; returns the number copied.  only: restrict to these
    relative paths (the push sends only what the stage wrote, so parallel stages of one row never overwrite each
    other's outputs with the stale copies they pulled)."""
    n = 0
    if not os.path.isdir(src):
        return 0
    for root, _, files in os.walk(src):
        for f in files:
            s = os.path.join(root, f)
            if only is not None and os.path.relpath(s, src) not in only:
                continue
            d = os.path.join(dst, os.path.relpath(s, src))
            if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s) and filecmp.cmp(s, d, shallow=False):
                continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            for k in range(4):
                try:
                    shutil.copyfile(s, d + '.part')
                    os.replace(d + '.part', d)
                    break
                except OSError:
                    time.sleep(3 + 5 * k)
            else:
                raise OSError(f'sync failed: {s}')
            n += 1
    return n


def main():
    argv = sys.argv[1:]
    rest = []
    if '--' in argv:
        i = argv.index('--')
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--gen', default='rr1')
    ap.add_argument('--parts', default='run_config,m3,decoded,hits')
    ap.add_argument('--work', default=os.environ.get('_CONDOR_SCRATCH_DIR', os.getcwd()))
    ap.add_argument('--no-sync', action='store_true', help='do not push outputs back to EOS')
    ap.add_argument('--eos-analysis', default=inputs.EOS_ANALYSIS_ROOT)
    ap.add_argument('--require', default=None,
                    help="product (relative to the row's Analysis dir) that defines success: the stage counts as "
                         "done if it exists, whatever its exit code (e.g. the hits event cache of a short scan "
                         "subrun whose hits alignment fails)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--module')
    g.add_argument('--script')
    a = ap.parse_args(argv)

    bench = os.path.join(a.work, 'bench')
    ana = os.path.join(a.work, 'Analysis')
    # before anything imports qa_config: its registry is rebased when the module loads
    os.environ['X17_BENCH_ROOT'] = bench
    os.environ['X17_ANALYSIS_ROOT'] = ana
    row = inputs.row_for(a.key)
    t0 = time.time()
    rec = inputs.stage(row, bench, a.gen, tuple(p for p in a.parts.split(',') if p))
    print(f'[stage] staged {a.key} ({a.gen}): '
          f'{ {k: len(v) for k, v in rec["files"].items()} } in {time.time() - t0:.0f} s', flush=True)
    rel = os.path.join(row['run'], row['subrun'], row['det'])
    n_in = sync(os.path.join(a.eos_analysis, rel), os.path.join(ana, rel))
    print(f'[stage] pulled {n_in} Analysis files for {rel}', flush=True)
    before = snapshot(os.path.join(ana, rel))

    sys.path[:0] = [os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
    import qa_config
    qa_config.setup_paths()
    if a.key not in qa_config.RUNS:
        qa_config.RUNS[a.key] = qa_config._Config(
            a.key, row['run'], row['subrun'], feus=[int(row['feu_x']), int(row['feu_y'])],
            det_z=float(row['det_z']), det_name=row['det'], zero_suppressed=False,
            base_path=f"{qa_config.DESKTOP_BENCH_ROOT}/{row['tree']}/")
    cfg = qa_config.RUNS[a.key]
    for path in (cfg.BASE_PATH, cfg.OUT_BASE):
        if not (path.startswith(bench) or path.startswith(ana)):
            sys.exit(f'FATAL: {a.key} resolves outside the staged tree: {path}')
    staged = os.path.join(bench, row['tree'], row['run'], row['subrun'])
    if os.path.normpath(os.path.join(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN)) != os.path.normpath(staged):
        sys.exit(f'FATAL: {a.key} reads {cfg.BASE_PATH}{cfg.RUN}/{cfg.SUB_RUN}, inputs were staged at {staged}')

    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                            text=True).stdout.strip() or os.environ.get('RERUN_CODE_COMMIT', '?')
    what = a.module or a.script
    print(f'[stage] running {what} {a.key} {" ".join(rest)}', flush=True)
    t1 = time.time()
    rc = 0
    try:
        # the row key goes first, unless the arguments place it themselves ('{key}', e.g. wft.cli reco {key})
        # '{out}' = the row's Analysis dir in this job (cfg.OUT_BASE), for paths to earlier stages' products
        rest = [x.replace('{out}', cfg.OUT_BASE) for x in rest]
        margs = [x.replace('{key}', a.key) for x in rest] if any('{key}' in x for x in rest) else [a.key, *rest]
        if a.module:
            r = importlib.import_module(a.module).main(margs)
            rc = int(r or 0)
        else:
            script = os.path.join(REPO, a.script)
            sys.argv = [script, *margs]
            sys.path.insert(0, os.path.dirname(script))
            runpy.run_path(script, run_name='__main__')
    except SystemExit as e:
        rc = int(e.code or 0) if not isinstance(e.code, str) else 1
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
    dt = time.time() - t1
    if a.require:
        have = os.path.exists(os.path.join(cfg.OUT_BASE, a.require))
        print(f'[stage] required product {a.require}: {"present" if have else "MISSING"} (stage rc {rc})', flush=True)
        rc = 0 if have else (rc or 1)
    # one record per stage run (a shared appended log loses lines when stages of one row run in parallel)
    now = datetime.datetime.now()
    logd = os.path.join(ana, rel, 'STAGES')
    os.makedirs(logd, exist_ok=True)
    tag = f"{now:%Y%m%dT%H%M%S}_{os.getpid()}_{os.path.basename(what).replace('.py', '')}"
    after = snapshot(os.path.join(ana, rel))
    written = sorted(k for k, v in after.items() if before.get(k) != v)
    json.dump(dict(when=now.isoformat(timespec='seconds'), stage=what, args=rest, rc=rc, seconds=round(dt),
                   code=commit, generation=a.gen, inputs=os.path.join(row['eos_run_dir'], row['subrun']),
                   env={k: v for k, v in os.environ.items() if k.startswith('WFT_')},
                   wrote=written), open(os.path.join(logd, tag + '.json'), 'w'), indent=1)
    if not a.no_sync:
        changed = {k for k, v in snapshot(os.path.join(ana, rel)).items() if before.get(k) != v}
        n_out = sync(os.path.join(ana, rel), os.path.join(a.eos_analysis, rel), only=changed)
        print(f'[stage] pushed {n_out} Analysis files to {os.path.join(a.eos_analysis, rel)}', flush=True)
    print(f'[stage] {what} rc={rc} in {dt:.0f} s', flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
