#!/usr/bin/env python3
"""make_jobs.py -- job lists of the phase-4 test re-reconstruction (AUDIT_FIX_ROADMAP.md phase 4 and 6, README.md here).

    python rerun/ntof/improve/phase4/make_jobs.py --code <commit> [--outdir jobs/] [--cfgs ref,all,...]

Per configuration it writes, in ntof.sub's format (<run> <subrun> <tag> <arm> <bundle dir> <out dir> [ntof_reco.py args]):

    jobs_<cfg>_AC.txt   chambers A and C   (submit with flavour workday: A about 2 h, C up to 5 h on 4 cores)
    jobs_<cfg>_D.txt    chamber D          (submit with flavour tomorrow: the large D acquisitions take 5.4-8.3 h)
    diag_<cfg>.txt      the WFT_DIAG string of the configuration (the submit macro)

Rows (revised at the re-audit of 2026-11-24):
  * the five beam sub-runs of the ringing study, per-row bundles as pass 1 with the ledger's canon
    (D of the L2 era and of run_104 on the L1 pin, D_E2_beam_L1);
  * chamber D on run_104 too.  Its lat_epoch is "unplaced"; the canon puts it under side/run104_D_L1, so its output
    directory is out/<cfg>/run104_D_L1/run_104/<subrun>/D (where ntof_stage3 looks for it with --side = the test's
    stage 2);
  * ten beam-off cosmic sub-runs of run_149 (about 140 A-C cosmic lines each): the only angle truth.  No configuration
    could be judged on angles without them.

Configurations.  The depth profiles (4.11) are stored in every configuration: they change no fit.
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NT = os.path.abspath(os.path.join(HERE, '..', '..'))
EOS = '/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1'
BEAM = [('145', 'stat090_0000'), ('145', 'stat090_0001'), ('145', 'stat090_0002'),
        ('104', 'stat090_0003'), ('139', 'stat090_0003')]
COSMIC = [('149', f'cosbounce_cos_{i:04d}') for i in range(10)]
SUBRUNS = BEAM + COSMIC
FLOOR, PRIOR, PROF = 'seed.SIG_FLOOR_LOCAL_MM=30', 'reco.TWO_TRACK_T0_PRIOR=1', 'reco.STORE_PROFILES=1'
#: 4.4 as corrected on 2026-11-24: the cap of five counts the clusters that survive the wide-band veto
RANK = 'seed.SEED_RANK=charge,seed.SEED_CAP_AFTER_VETO=1'
#: the n_TOF x-y column-time tolerance [ns] (quality_join.TC_TOL_NTOF); the bundles carry the bench 197 / 456 / 726
TC_TOL = {'A': 151, 'C': 296, 'D': 535}
#: cfg -> (WFT_DIAG, extra ntof_reco args for A/C, extra for D (None = the chamber is not run), D bundles 'pf' or 'osc').
#: "{tc}" in an argument string is replaced by the chamber's TC_TOL.
#: `all` keeps D on the pass-1 bundles: whether the oscillating-channel hot list (osc) helps D is judged on its own
#: configuration first (the sizing cluster's D comparison was confounded by a wrong pin, see README).
CFGS = {
    'ref':      (PROF, '', '', 'pf'),
    'all':      (','.join([FLOOR, PRIOR, RANK, PROF]), '--headon-rescue --ringing deflect_hits --tc-tol {tc}',
                 '--headon-rescue --tc-tol {tc}', 'pf'),
    'floor30':  (','.join([FLOOR, PROF]), '', '', 'pf'),
    't0prior':  (','.join([PRIOR, PROF]), '', '', 'pf'),
    'seedrank': (','.join([RANK, PROF]), '', '', 'pf'),
    'headon':   (PROF, '--headon-rescue', '--headon-rescue', 'pf'),
    'ringing':  (PROF, '--ringing deflect_hits', None, 'pf'),
    'tctol':    (PROF, '--tc-tol {tc}', '--tc-tol {tc}', 'pf'),
    'osc':      (PROF, None, '', 'osc'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--code', required=True, help='the deployed code commit (names the EOS test directory)')
    ap.add_argument('--outdir', default='jobs')
    ap.add_argument('--cfgs', default=','.join(CFGS))
    ap.add_argument('--no-cosmic', action='store_true', help='beam sub-runs only')
    a = ap.parse_args()
    base = f'{EOS}/tests/phase4_{a.code}'
    want = set(BEAM if a.no_cosmic else SUBRUNS)
    runs = ','.join(sorted({r for r, _ in want}))
    lines = subprocess.check_output(
        [sys.executable, os.path.join(NT, 'condor', 'make_ntof_jobs.py'), '--bundles', f'{EOS}/bundles', '--out', 'OUT',
         '--arms', 'ACD', '--runs', runs, '--per-row', '--unplaced', 'L1'], text=True).splitlines()
    rows = [l.split() for l in lines if l.strip() and (l.split()[0], l.split()[1]) in want]
    # the bundle and the directory the ledger's canon names for the row; make_ntof_jobs --per-row does not know the
    # canon, and ref must be pass 1 exactly
    sys.path.insert(0, NT)
    import ntof_ledger as L  # noqa: E402
    canon = L.load_canon(os.path.join(NT, 'ledger_canon.json'))
    reg = {(r['run'], r['subrun'], r['tag']): r for r in csv.DictReader(open(os.path.join(NT, 'ntof_registry.csv')))}
    sub_of = {}
    for r in rows:
        u = L.rule_for(canon, reg[(r[0], r[1], r[2])], r[3])
        if u and u.get('bundle'):
            r[4] = f'{EOS}/bundles/{u["bundle"]}'
        # a canon dir with a sub-directory (side/run104_D_L1) keeps it, so stage 3 finds the files (ntof_stage3.canonical)
        sub_of[tuple(r[:4])] = '/'.join(u['dir'].split('/')[1:]) if u else ''
    os.makedirs(a.outdir, exist_ok=True)
    for cfg in a.cfgs.split(','):
        diag, ex_ac, ex_d, bset = CFGS[cfg]
        out = {'AC': [], 'D': []}
        for r in rows:
            run, sub, tag, arm, bundle = r[:5]
            ex = ex_d if arm == 'D' else ex_ac
            if ex is None:
                continue
            if arm == 'D' and bset == 'osc':
                bundle = f'{base}/bundles_osc/{os.path.basename(bundle)}'
            od = '/'.join(filter(None, [f'{base}/out/{cfg}', sub_of[(run, sub, tag, arm)], f'run_{run}', sub, arm]))
            ex = ex.format(tc=TC_TOL[arm])
            out['D' if arm == 'D' else 'AC'].append(' '.join([run, sub, tag, arm, bundle, od] + ([ex] if ex else [])))
        for k, v in out.items():
            p = f'{a.outdir}/jobs_{cfg}_{k}.txt'
            if v:
                open(p, 'w').write('\n'.join(v) + '\n')
            elif os.path.exists(p):
                os.remove(p)
        open(f'{a.outdir}/diag_{cfg}.txt', 'w').write(diag + '\n')
        print(f'{cfg:9s} A/C {len(out["AC"]):4d}  D {len(out["D"]):4d}  diag: {diag or "-"}')


if __name__ == '__main__':
    main()
