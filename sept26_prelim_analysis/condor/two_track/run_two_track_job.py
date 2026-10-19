#!/usr/bin/env python3
"""
run_two_track_job.py -- one condor job of the two-track validation.

    run_two_track_job.py <kind> <arm> <tag> <outname> [intra_bench args...]

kind = bench (intra_bench build --only-tag) or splitab (intra_bench split-ab
--only-tag). Fetches into the scratch dir, from EOS:
  * the shipped inputs tarball (reco_fullpass products and bundles of run_145
    stat090_0000, the stage-3 tracks, the x/y pairing calibrations), unpacked
    into an ``out/`` tree laid out like the laptop's sept26_prelim;
  * run_config.json, the tag's combined hits and its two decoded FEU files.
Then points sept26_prelim_analysis.paths (X17_SEPT26_OUT) and wft_beam
(WFT_BEAM_BASE) at them -- env, BEFORE any import, for the reason given in
ntof_tracking/condor/run_beam_job.py -- runs intra_bench, and tars the variant's
output directory as <outname>.tar.gz for the wrapper to push to EOS.
"""
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.abspath(os.getcwd())
CODE = os.path.join(HERE, 'code')
RUN, SUBRUN = 'run_145', 'stat090_0000'
EOS_PUBLIC = os.environ.get('EOS_URL', 'root://eospublic.cern.ch/')
EOS_USER = 'root://eosuser.cern.ch/'
EOS_BASE = os.environ.get('EOS_BEAM_BASE', '/eos/experiment/ntof/data/x17/july_beam/runs')
INPUTS = os.environ.get('TT_INPUTS', '/eos/user/d/dneff/x17/two_track_limit/inputs.tar.gz')


def sh(cmd):
    print('[job]', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fetch(eos_path, dest, url=EOS_PUBLIC):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(eos_path):          # fuse mount, or a local test
        shutil.copy2(eos_path, dest)
        return
    sh(['xrdcp', '-s', '-f', url + eos_path, dest])


def main():
    kind, arm, tag, outname = sys.argv[1:5]
    extra = sys.argv[5:]
    if kind not in ('bench', 'splitab'):
        sys.exit(f'FATAL: kind must be bench or splitab: {kind!r}')

    out = os.path.join(HERE, 'out')
    data = os.path.join(HERE, 'data')
    fetch(INPUTS, os.path.join(HERE, 'inputs.tar.gz'), url=EOS_USER)
    with tarfile.open(os.path.join(HERE, 'inputs.tar.gz')) as t:
        t.extractall(HERE)                # -> out/
    os.environ['X17_ROOT'] = HERE
    os.environ['X17_SEPT26_OUT'] = out
    os.environ['WFT_BEAM_BASE'] = data + '/'
    os.environ['WFT_BEAM_ANALYSIS'] = os.path.join(HERE, 'beam_analysis') + '/'
    os.environ['PYTHONPATH'] = CODE + os.pathsep + os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, CODE)

    from ntof_tracking import wft_beam as wb      # noqa: E402  (after the env)
    sub = f'{EOS_BASE}/{RUN}/{SUBRUN}'
    loc = os.path.join(data, RUN, SUBRUN)
    fetch(f'{EOS_BASE}/{RUN}/run_config.json', os.path.join(data, RUN, 'run_config.json'))
    hits = f'Mx17_{SUBRUN}_datrun_{tag}_feu-combined_hits.root'
    fetch(f'{sub}/combined_hits_root/{hits}', os.path.join(loc, 'combined_hits_root', hits))
    for feu in (wb.BEAM_DETS[arm]['feu_x'], wb.BEAM_DETS[arm]['feu_y']):
        f = f'Mx17_{SUBRUN}_datrun_{tag}_{feu:02d}.root'
        fetch(f'{sub}/decoded_root/{f}', os.path.join(loc, 'decoded_root', f))

    jobs = os.environ.get('RECO_JOBS', '2')
    cmd = 'build' if kind == 'bench' else 'split-ab'
    sh([sys.executable, '-m', 'sept26_prelim_analysis.intra_bench', cmd, '--arms', arm,
        '--jobs', jobs, '--only-tag', tag] + extra)

    variant = extra[extra.index('--variant') + 1]
    res = os.path.join(out, 'intra_bench', variant if kind == 'bench' else f'split_ab_{variant}')
    with tarfile.open(os.path.join(HERE, f'{outname}.tar.gz'), 'w:gz') as t:
        t.add(res, arcname=outname)
    print(f'[job] wrote {outname}.tar.gz from {res}', flush=True)


if __name__ == '__main__':
    main()
