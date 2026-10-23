#!/usr/bin/env python3
"""make_jobs.py <fdfs.txt> <jobs.txt> -- W9 hits regeneration on the fa80c06 decodes, one condor job per (subrun, file).

Inputs per job: the new decodes (<subrun>/decoded_root_fa80c06/*_datrun_*_<fnum>_<feu>.root, MX17 FEUs only, as the
7-24 production), and per FEU the pedestal .fdf, resolved as mx_june_cosmic_qa/reprocess_cosmic_bench.py did
(own subrun with matching run prefix > unique in own subrun > unique timestamp among sibling subruns of the run);
the job decodes it with the same pinned decoder.  Analyzer flags are the production recipe (flags.json, computed
from each run's Dream cfg on the laptop: --tps 60 --mf 5 for every June row).  Two generations, so the hits
common-mode choice can be decided on data: rr1_cm0 (plain block median = production) and rr1_cm5 (masked ±5).
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = json.load(open(os.path.join(HERE, 'flags.json')))
DAT = re.compile(r'_datrun_(\d{6}_\d\dH\d\d_\d{3})_(\d{2})\.root$')     # (date_time_fnum, feu)
PED = re.compile(r'_pedthr_.*_(\d{3})_(\d{2})\.fdf$')
MX17 = {'03', '04', '06', '07', '08'}


def prefix(f):
    for t in ('_datrun_', '_pedthr_'):
        if t in f:
            return f.split(t)[0]
    return ''


def peds(subdir):
    out = defaultdict(list)
    for f in glob.glob(os.path.join(subdir, 'raw_daq_data', '*_pedthr_*.fdf')):
        m = PED.search(f)
        if m:
            out[m.group(2)].append(f)
    return out


def main():
    fdfs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    subs = sorted({os.path.dirname(os.path.dirname(f)) for f in fdfs})
    jobs, problems = [], []
    for sd in subs:
        run_dir = os.path.dirname(sd)
        key = f'{os.path.basename(run_dir)}/{os.path.basename(sd)}'
        own = peds(sd)
        sib = defaultdict(list)
        for s2 in glob.glob(os.path.join(run_dir, '*')):
            if os.path.isdir(s2) and s2 != sd:
                for feu, ps in peds(s2).items():
                    sib[feu].extend(ps)
        dats = sorted(glob.glob(os.path.join(sd, 'decoded_root_fa80c06', '*_datrun_*.root')))
        by = defaultdict(dict)
        for d in dats:
            m = DAT.search(d)
            if m and m.group(2) in MX17:
                by[m.group(1)][m.group(2)] = d
        if not by:
            problems.append((key, 'no new decodes'))
            continue
        pre = prefix(os.path.basename(dats[0]))
        for fnum, feus in sorted(by.items()):     # fnum = date_time_fnum: one subrun can hold several acquisitions
            spec = []
            for feu, d in sorted(feus.items()):
                c = own.get(feu, [])
                ped = None
                if len(c) > 1:
                    # one pedestal per acquisition (det3_test 6-22 short_run: 15H51 and 15H55): same date_time stamp
                    stamp = re.search(r'_datrun_(\d{6}_\d\dH\d\d)_', os.path.basename(d)).group(1)
                    st = [p for p in c if f'_pedthr_{stamp}_' in os.path.basename(p)]
                    ex = [p for p in c if prefix(os.path.basename(p)) == pre]
                    ped = st[0] if len(st) == 1 else (ex[0] if len(ex) == 1 else None)
                elif len(c) == 1:
                    ped = c[0]
                else:
                    stamps = {os.path.basename(p).split('_pedthr_')[1] for p in sib.get(feu, [])}
                    if len(stamps) == 1:
                        ped = sorted(sib[feu])[0]
                if ped is None:
                    problems.append((key, fnum, feu, 'pedestal unresolved'))
                    continue
                spec.append(f'{feu}={d}={ped}={FLAGS[key][feu].replace(" ", "_")}')
            if spec:
                jobs.append(f'{sd} {fnum} {";".join(spec)}')
    open(sys.argv[2], 'w').write('\n'.join(jobs) + '\n')
    print(f'{len(jobs)} jobs, {len(problems)} problems')
    for p in problems:
        print('  ', p)


if __name__ == '__main__':
    main()
