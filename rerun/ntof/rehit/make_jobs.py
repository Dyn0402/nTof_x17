#!/usr/bin/env python3
"""make_jobs.py <subruns.txt> <jobs.txt> -- n_TOF hits on the fa80c06 decodes with the bench rr1 recipe (Q4 study).

One job per acquisition, in the format of rerun/rehit/job.sh (the bench W9 job, reused unchanged): all eight FEUs,
analyzer flags ``--tps 60 --mf 5`` (60 ns sampling, matched filter 5; the bench production recipe), both common-mode
generations (rr1_cm0 = plain block median, the pass's; rr1_cm5 kept for comparison).  Outputs go next to the data:
<subrun>/hits_root_rr1_cm{0,5}/ and <subrun>/combined_hits_root_rr1_cm{0,5}/.

Pedestal per FEU: the sub-run's own ``*_pedthr_*_<feu>.fdf`` (one per sub-run at n_TOF: "pedestals latest" in the
run_config); a sub-run without its own takes the latest earlier one among the run's sub-runs, else the latest earlier
acquisition in july_beam/pedestals.  <subruns.txt>: lines "<run number> <subrun>".  Runs on lxplus (EOS paths).
"""
import glob
import os
import re
import sys

RUNS = '/eos/experiment/ntof/data/x17/july_beam/runs'
PEDS = '/eos/experiment/ntof/data/x17/july_beam/pedestals'
FLAGS = '--tps_60_--mf_5'
DAT = re.compile(r'_datrun_(\d{6}_\d\dH\d\d_\d{3})_(\d{2})\.root$')
PED = re.compile(r'_pedthr_(\d{6}_\d\dH\d\d)_\d{3}_(\d{2})\.fdf$')


def peds_in(d):
    out = {}
    for f in glob.glob(os.path.join(d, '*_pedthr_*.fdf')):
        m = PED.search(f)
        if m:
            out.setdefault(m.group(2), []).append((m.group(1), f))
    return out


def pick(cands, stamp):
    """latest pedestal not after the acquisition's date_time stamp"""
    earlier = sorted((s, f) for s, f in cands if s <= stamp)
    return earlier[-1][1] if earlier else None


def main():
    pairs = [l.split() for l in open(sys.argv[1]) if l.strip()]
    jobs, problems = [], []
    for run, sub in pairs:
        rd = os.path.join(RUNS, f'run_{run}')
        sd = os.path.join(rd, sub)
        own = peds_in(os.path.join(sd, 'raw_daq_data'))
        sib = {}
        for s2 in glob.glob(os.path.join(rd, '*', 'raw_daq_data')):
            for feu, l in peds_in(s2).items():
                sib.setdefault(feu, []).extend(l)
        glob_peds = {}
        for f in glob.glob(os.path.join(PEDS, '**', '*_pedthr_*.fdf'), recursive=True):
            m = PED.search(f)
            if m:
                glob_peds.setdefault(m.group(2), []).append((m.group(1), f))
        by = {}
        for d in sorted(glob.glob(os.path.join(sd, 'decoded_root_fa80c06', '*_datrun_*.root'))):
            m = DAT.search(d)
            if m:
                by.setdefault(m.group(1), {})[m.group(2)] = d
        if not by:
            problems.append((run, sub, 'no fa80c06 decodes'))
            continue
        for fnum, feus in sorted(by.items()):
            stamp = fnum.rsplit('_', 1)[0]
            spec = []
            for feu, d in sorted(feus.items()):
                ped = pick(own.get(feu, []), stamp) or pick(sib.get(feu, []), stamp) or pick(glob_peds.get(feu, []), stamp)
                if ped is None:
                    problems.append((run, sub, fnum, feu, 'pedestal unresolved'))
                    continue
                spec.append(f'{feu}={d}={ped}={FLAGS}')
            if len(spec) == 8:
                jobs.append(f'{sd} {fnum} {";".join(spec)}')
            else:
                problems.append((run, sub, fnum, f'{len(spec)}/8 FEUs'))
    open(sys.argv[2], 'w').write('\n'.join(jobs) + '\n')
    print(f'{len(jobs)} jobs, {len(problems)} problems')
    for p in problems[:30]:
        print('  ', p)


if __name__ == '__main__':
    main()
