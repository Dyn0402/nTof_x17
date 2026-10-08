#!/usr/bin/env python3
"""
submit_c3_reduce.py -- PLAN.md C3 input: reduce every file of the pair sim
(X17 + IPC 50/50, thermal capture vertices, the production stack) to
per-event x arm scintillator deposits with `reduce_edep.py`.  Run ON LXPLUS:

    python3 submit_c3_reduce.py --src /eos/experiment/ntof/data/x17/full_sim/<pairs dir> [--dry-run]

One job per ROOT file; outputs (~MB each) go to EOS from inside the job.
"""
import argparse
import glob
import os
import stat
import textwrap
from pathlib import Path

SIM_REPO = Path('/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant')
HERE = Path(__file__).resolve().parent
EOS = Path('/eos/experiment/ntof/data/x17/full_sim/calorimetry/c3_pairs_edep')
JOBS = Path('/afs/cern.ch/user/d/dneff/condor/calorimetry/c3_reduce')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--flavour', default='longlunch')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, '*.root')))
    print(f'{len(files)} files in {a.src} -> {EOS}')
    if a.dry_run or not files:
        print('\n'.join(files[:5]))
        return
    out = EOS / Path(a.src.rstrip('/')).name
    out.mkdir(parents=True, exist_ok=True)
    (JOBS / 'logs').mkdir(parents=True, exist_ok=True)
    w = JOBS / 'run.sh'
    w.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -eo pipefail
        set +u; source "{SIM_REPO}/scripts/setup_lxplus.sh" > /dev/null; set -u
        python3 "{HERE}/reduce_edep.py" "$1" "$2"
    """))
    w.chmod(w.stat().st_mode | stat.S_IEXEC)
    sub = JOBS / 'jobs.sub'
    rows = [(f, str(out / Path(f).stem), Path(f).stem) for f in files]
    sub.write_text('\n'.join([
        f'executable = {w}', f'output = {JOBS}/logs/$(tag).out', f'error = {JOBS}/logs/$(tag).err',
        f'log = {JOBS}/logs/condor.log', f'+JobFlavour = "{a.flavour}"', 'request_cpus = 1',
        'request_memory = 4096', 'requirements = (OpSysAndVer =?= "AlmaLinux9")',
        'should_transfer_files = NO', 'arguments = $(src) $(out)',
        'queue src,out,tag from ('] + [f'  {f}, {o}, {t}' for f, o, t in rows] + [')']) + '\n')
    os.system(f'condor_submit {sub}')


if __name__ == '__main__':
    main()
