#!/usr/bin/env python3
"""
submit_c2.py -- PLAN.md C2: single e-/e+ through one arm's full stack, on a
kinetic-energy grid, from the capsule centre (the sim's --single vertex is the
origin, inside the He-3), normal to arm 0 and 20 deg off it.  Run ON LXPLUS:

    python3 submit_c2.py [--dry-run]

Each job: simulate -> reduce_edep.py -> delete the ROOT file.  Results go to
EOS from inside the job; nothing large lands in the scratch dir or on AFS.
"""
import argparse
import os
import random
import stat
import textwrap
from pathlib import Path

SIM_REPO = Path('/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant')
EXE = SIM_REPO / 'build' / 'mx17_full_sim'
HERE = Path(__file__).resolve().parent
EOS = Path('/eos/experiment/ntof/data/x17/full_sim/calorimetry/c2_singles')
JOBS = Path('/afs/cern.ch/user/d/dneff/condor/calorimetry/c2_singles')
ENERGIES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 12, 14, 16]
PARTICLES = ['e-', 'e+']
ANGLES = [(90, 0), (90, 20)]       # (theta from beam, phi from arm 0 = +X)
GAS = 'ArIso'                       # as pairs_thermal_trig_2cm_nose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nevents', type=int, default=20000)
    ap.add_argument('--flavour', default='longlunch')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not EXE.is_file():
        raise SystemExit(f'build the sim first: {EXE}')
    rng = random.Random(20261008)
    rows = []
    for p in PARTICLES:
        for E in ENERGIES:
            for th, ph in ANGLES:
                tag = f'{p.replace("-", "m").replace("+", "p")}_{E:g}MeV_th{th}_ph{ph}'
                rows.append((tag, p, E, th, ph, rng.randint(1, 2**31 - 1)))
    print(f'{len(rows)} jobs x {a.nevents} -> {EOS}')
    if a.dry_run:
        for r in rows[:5]:
            print(' ', r)
        return
    EOS.mkdir(parents=True, exist_ok=True)
    (JOBS / 'logs').mkdir(parents=True, exist_ok=True)
    w = JOBS / 'run.sh'
    w.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        set -eo pipefail
        set +u; source "{SIM_REPO}/scripts/setup_lxplus.sh" > /dev/null; set -u
        TAG="$1"; P="$2"; E="$3"; TH="$4"; PH="$5"; SEED="$6"
        cd "${{_CONDOR_SCRATCH_DIR:-/tmp}}"
        "{EXE}" -t 1 -n {a.nevents} -g {GAS} -s "$SEED" --single "$P" "$E" "$TH" "$PH" -o "$TAG" > "$TAG.log" 2>&1
        python3 "{HERE}/reduce_edep.py" "${{TAG}}_t0.root" "{EOS}/$TAG"
        cp "$TAG.log" "{EOS}/$TAG.log"
        rm -f "${{TAG}}"_t*.root
        echo "done $(date)"
    """))
    w.chmod(w.stat().st_mode | stat.S_IEXEC)
    sub = JOBS / 'jobs.sub'
    sub.write_text('\n'.join([
        f'executable = {w}', f'output = {JOBS}/logs/$(tag).out', f'error = {JOBS}/logs/$(tag).err',
        f'log = {JOBS}/logs/condor.log', f'+JobFlavour = "{a.flavour}"', 'request_cpus = 1',
        'request_memory = 2048', 'requirements = (OpSysAndVer =?= "AlmaLinux9")',
        'should_transfer_files = NO', 'arguments = $(tag) $(p) $(e) $(th) $(ph) $(seed)',
        'queue tag,p,e,th,ph,seed from ('] + [f'  {t}, {p}, {E:g}, {th}, {ph}, {s}' for t, p, E, th, ph, s in rows] + [')']) + '\n')
    os.system(f'condor_submit {sub}')


if __name__ == '__main__':
    main()
