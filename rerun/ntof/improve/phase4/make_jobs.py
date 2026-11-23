#!/usr/bin/env python3
"""make_jobs.py -- job lists of the phase-4 test re-reconstruction (AUDIT_FIX_ROADMAP.md phase 4, README.md here).

    python rerun/ntof/improve/phase4/make_jobs.py --code e86028b5 [--outdir jobs/]

One file per configuration, jobs_<cfg>.txt, in ntof.sub's format (<run> <subrun> <tag> <arm> <bundle dir> <out dir>
[ntof_reco.py args]); the WFT_DIAG string of each configuration is written to diag_<cfg>.txt (the submit macro).
Rows: the five test sub-runs, per-row bundles as pass 1 (make_ntof_jobs.py --per-row --unplaced skip).
"""
import argparse, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..', '..'))
EOS = '/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1'
SUBRUNS = [('145', 'stat090_0000'), ('145', 'stat090_0001'), ('145', 'stat090_0002'),
           ('104', 'stat090_0003'), ('139', 'stat090_0003')]
FLOOR, PRIOR, RANK, PROF = 'seed.SIG_FLOOR_LOCAL_MM=30', 'reco.TWO_TRACK_T0_PRIOR=1', 'seed.SEED_RANK=charge', \
    'reco.STORE_PROFILES=1'
#: cfg -> (WFT_DIAG, extra ntof_reco args for A/C, extra for D (None = D not run), D bundles 'pf' or 'osc')
CFGS = {
    'ref':      ('', '', '', 'pf'),
    'all':      (','.join([FLOOR, PRIOR, RANK, PROF]), '--headon-rescue --ringing deflect_hits', '--headon-rescue', 'osc'),
    'floor30':  (FLOOR, '', '', 'pf'),
    't0prior':  (PRIOR, '', '', 'pf'),
    'seedrank': (RANK, '', '', 'pf'),
    'headon':   ('', '--headon-rescue', '--headon-rescue', 'pf'),
    'ringing':  ('', '--ringing deflect_hits', None, 'pf'),
    'osc':      ('', None, '', 'osc'),
}
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--code', required=True)
ap.add_argument('--outdir', default='jobs')
ap.add_argument('--cfgs', default=','.join(CFGS))
a = ap.parse_args()
base = f'{EOS}/tests/phase4_{a.code}'
lines = subprocess.check_output(
    [sys.executable, os.path.join(NT, 'condor', 'make_ntof_jobs.py'), '--bundles', f'{EOS}/bundles', '--out', 'OUT',
     '--arms', 'ACD', '--runs', '104,139,145', '--per-row', '--unplaced', 'skip'], text=True).splitlines()
want = set(SUBRUNS)
rows = [l.split() for l in lines if l.strip() and (l.split()[0], l.split()[1]) in want]
# the bundle the ledger's canon names for the row (D beam rows of the L2 era run on the L1 pin, D_E2_beam_L1, since
# 2026-11-17); make_ntof_jobs --per-row does not know the canon, and ref must be pass 1 exactly
import csv
sys.path.insert(0, NT)
import ntof_ledger as L  # noqa: E402
canon = L.load_canon(os.path.join(NT, 'ledger_canon.json'))
reg = {(r['run'], r['subrun'], r['tag']): r for r in csv.DictReader(open(os.path.join(NT, 'ntof_registry.csv')))}
for r in rows:
    u = L.rule_for(canon, reg[(r[0], r[1], r[2])], r[3])
    if u and u.get('bundle'):
        r[4] = f'{EOS}/bundles/{u["bundle"]}'
os.makedirs(a.outdir, exist_ok=True)
for cfg in a.cfgs.split(','):
    diag, ex_ac, ex_d, bset = CFGS[cfg]
    out = []
    for r in rows:
        run, sub, tag, arm, bundle = r[:5]
        ex = ex_d if arm == 'D' else ex_ac
        if ex is None:
            continue
        if arm == 'D' and bset == 'osc':
            bundle = f'{base}/bundles_osc/{os.path.basename(bundle)}'
        od = f'{base}/out/{cfg}/run_{run}/{sub}/{arm}'
        out.append(' '.join([run, sub, tag, arm, bundle, od] + ([ex] if ex else [])))
    open(f'{a.outdir}/jobs_{cfg}.txt', 'w').write('\n'.join(out) + '\n')
    open(f'{a.outdir}/diag_{cfg}.txt', 'w').write(diag + '\n')
    print(cfg, len(out), 'jobs', 'diag:', diag or '-')
