#!/usr/bin/env python3
"""make_satfin_dag.py <code_dir> -- held-out gate of the error-model recipe across datasets (ERROR_MODEL.md, 10-29).

Arms (heldout_arm.py, NO rails: the production path has none -- under SAT_CAP=1200 the rail only matters for samples
clipped below 1200 ADC, see ERROR_MODEL.md "Rails"):
  base   the current algorithm (no WFT_DIAG)
  cand   model.SAT_CAP=1200, model.STRIP_TAU=0.05, model.QSOLVE_PLAIN=true
Datasets:
  golden caches (t15/caches/<det>.pkl, events [180, 1500), the fleet table's sample) and det3's 8000-event cache
  (events [1500, 6318), the "second sample"): base / cand without rails, to compare with the fleet's rail / capt arms;
  other datasets (det3: 6-28 night, 700 V, 500 V, 6-25 day; det2/4/6/7 one second dataset each): HITS (hits-chain
  event list, 03_alignment_and_tpc) -> CACHE (build_cache_wft.py, M3 reference placed by that row's own waveform
  alignment from FB / the dry run, plain CM) -> held-out chunks, all events of the cache, with the chamber's golden
  t15 base bundle (as the pass does: one bundle per chamber) and the row's depth-bin count K.
Writes satfin.dag + ho.sub (heldout via ~/x17rerun/satur/run_sat.sh) and stage.sub (stage_job.sh).
Outputs: /eos/user/d/dneff/x17rerun/satfin/{Analysis/<row>/..., ho/<name>_<arm>_c<i>.npz}.
"""
import argparse

E = '/eos/user/d/dneff/x17rerun'
OUT = f'{E}/satfin'
FB = f'{E}/followups/fb/Analysis'
DRY = f'{E}/Analysis_dryrun'
DIAG = {'base': 'none', 'cand': 'model.SAT_CAP=1200,model.STRIP_TAU=0.05,model.QSOLVE_PLAIN=true'}
# name -> (registry key, chamber, K, alignment.json of the row's own waveform reco, cache events)
ROWS = {
    'det3_p2': ('c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3', 'det3', 24,
                f'{FB}/mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3/wft/fb/alignment/alignment.json', 4000),
    'det3_d700': ('c26__mx17_det3_saturday_scan_6-27-26__drift_scan_resist_490V_drift_700V__mx17_3', 'det3', 24,
                  f'{DRY}/mx17_det3_saturday_scan_6-27-26/drift_scan_resist_490V_drift_700V/mx17_3/wft/vrefit/alignment0/alignment.json', 1500),
    'det3_d500': ('c26__mx17_det3_saturday_scan_6-27-26__drift_scan_resist_490V_drift_500V__mx17_3', 'det3', 30,
                  f'{DRY}/mx17_det3_saturday_scan_6-27-26/drift_scan_resist_490V_drift_500V/mx17_3/wft/vrefit/alignment0/alignment.json', 1500),
    'det3_day625': ('c26__mx17_det3_day_6-25-26__long_run__mx17_3', 'det3', 30,
                    f'{DRY}/mx17_det3_day_6-25-26/long_run/mx17_3/wft/vrefit/alignment0/alignment.json', 4000),
    'det2_o22long': ('c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 'det2', 24,
                     f'{FB}/mx17_det2_det3_overnight_6-22-26/long_run/mx17_2/wft/fb/alignment/alignment.json', 2000),
    'det4_o23': ('c26__mx17_det3_det4_overnight_6-23-26__long_run__mx17_4', 'det4', 24,
                 f'{FB}/mx17_det3_det4_overnight_6-23-26/long_run/mx17_4/wft/fb/alignment/alignment.json', 2000),
    'det6_longer': ('c26__mx17_det6_det7_overnight_6-26-26__longer_run__mx17_6', 'det6', 24,
                    f'{FB}/mx17_det6_det7_overnight_6-26-26/longer_run/mx17_6/wft/fb/alignment/alignment.json', 2000),
    'det7_longer': ('c26__mx17_det6_det7_overnight_6-26-26__longer_run__mx17_7', 'det7', 24,
                    f'{FB}/mx17_det6_det7_overnight_6-26-26/longer_run/mx17_7/wft/fb/alignment/alignment.json', 2000),
}
CHUNK = 1000
HO_SUB = '''executable     = $ENV(HOME)/x17rerun/satur/run_sat.sh
arguments      = $(code) $(diag) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/satfin.log
request_cpus   = 8
request_memory = 16000
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''
ST_SUB = '''executable     = $ENV(HOME)/x17rerun/stage_job.sh
arguments      = $(code) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/satfin.log
request_cpus   = $(cpus)
request_memory = $(mem)
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('code'); ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args(); L = []

    def job(name, sub, parents=(), **v):
        L.append(f'JOB {name} {sub}')
        L.append(f'VARS {name} nodename="{name}" code="{a.code}" ' + ' '.join(f'{k}="{x}"' for k, x in v.items()))
        L.append(f'RETRY {name} 1')
        L.extend(f'PARENT {p} CHILD {name}' for p in parents)

    def ho(name, det, cache, start, n, k, parents=()):
        i = 0
        for s in range(start, start + n, CHUNK):
            m = min(CHUNK, start + n - s)
            for arm, dg in DIAG.items():
                job(f'HO_{name}_{arm}_c{i}', 'ho.sub', parents, diag=dg,
                    args=f'--cache {cache} --bundle {E}/t15/bundles/{det}/base --out {OUT}/ho/{name}_{arm}_c{i}.npz '
                         f'--start {s} --n {m} --jobs 8 --k {k} --t0-fix')
            i += 1

    sel = lambda n: a.only is None or n in a.only
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        if sel(det):
            ho(det, det, f'{E}/t15/caches/{det}.pkl', 180, 1320, 24)
    if sel('det3big'):
        ho('det3big', 'det3', f'{E}/satur/big/det3_8000.pkl', 1500, 4818, 24)
    for name, (key, det, K, al, n) in ROWS.items():
        if not sel(name):
            continue
        pre = f'{key} --eos-analysis {OUT}/Analysis'
        job(f'HITS_{name}', 'stage.sub', cpus=4, mem=12000,
            args=f'{pre} --parts run_config,m3,hits --require cache/event_results_veto50.pkl '
                 f'--script mx_june_cosmic_qa/03_alignment_and_tpc.py')
        cache = f'{OUT}/caches/{name}.pkl'
        job(f'CACHE_{name}', 'stage.sub', [f'HITS_{name}'], cpus=2, mem=16000,
            args=f'{pre} --script rerun/w12/build_cache_wft.py -- --align {al} --out {cache} --n {n} --cm 0')
        # the cache's event count is known only after CACHE: chunks past its end are empty and harmless
        ho(name, det, cache, 0, n, K, [f'CACHE_{name}'])
    open('satfin.dag', 'w').write('\n'.join(L) + '\n')
    open('ho.sub', 'w').write(HO_SUB); open('stage.sub', 'w').write(ST_SUB)
    print('wrote satfin.dag:', sum(1 for l in L if l.startswith('JOB')), 'nodes')


if __name__ == '__main__':
    main()
