#!/usr/bin/env python3
"""make_pass_dag.py <phase> <code_dir> [--golden det=it_dir ...] -- DAGs for the re-run pass (RERUN_PLAN §4).

Phases (each writes <phase>.dag in the current dir; submit with condor_submit_dag from ~/x17rerun/pass):

calib_iter  the extra CAL4 iteration (Dylan 10-29) for the rows in --only: RECO4 (with the final calib_bundle) -> ALIGN4
          -> ANG4 -> W0KW4 (c4/bundle, a check); rerun/calib/conv_iter.py judges r3 vs r4 like for like (both bow)
calib     the from-scratch calibration of the five golden rows (PASS_PLAN_2026-10-28 E3), into Analysis_pass1, every
          product under <row>/wft/calib/:
          CAL0  HITS (03_alignment_and_tpc: the hits-chain corridor only) -> CACHE0 (c0/cache.pkl, 1500 events)
          CAL1  SEED0 (make_bundle: templates + latency + stated inputs) -> DRC0 (recal --free rc_D_y --t0-pin
                --unsat-only, training events 580-1079; det4: none, H4 value) -> MAKE0 (c0/bundle)
          CAL2  RECO1 (8000 matched) -> ALIGN1 (waveform) -> CACHE1 (build_cache_wft --align) -> SEED1 -> DRC1 -> MAKE1
                (c1/bundle)
          CAL3  RECO2 -> ALIGN2 -> ANG2 -> W0KW2 (--table --angles --sigma0 0.30,0.30: c2/bundle_nobow) -> BOW2
                (bow_node on bundle_nobow's w0/kw: c2/bow_block.json) -> W0KW2b (--bow-json --table: c2/bundle)
          CAL4  RECO3 (with the bow) -> ALIGN3 -> ANG3 -> W0KW3 (--table: <row>/wft/calib_bundle, the bow kept) -> BOW3
                (r3/bow_block.json, a check only) -> CONV (convergence.py: wft/calib/CALIB_CONVERGENCE.json);
                BUDGET (effbudget on the RECO3 table, for G1)
sps       every sps_registry.csv row (S3, PASS_PLAN E4): sps_stage.sh -> Analysis_pass1/sps/<key>/events.parquet with
          the pass's det4 calib_bundle (CAL4; the bench bow / v map / w0 / t0 are dropped in sps_reco).  Submit after
          the calib DAG's CONV_det4 (a separate DAG: no cross-DAG parent).
finalize  golden rows (S1 close).  Bundle = the converged W12 fit (w12/<it>/fit_pinned/bundle, --golden det=itN).
          RECO (8000 matched events with it) -> ALIGN (01) -> ANGLES (03) -> W0KW (W6: w0/kw measured under this
          bundle) -> <row>/wft/calib_bundle, the bundle every row of the chamber uses.
vrefit    one per condition group with refit=v (S1b, W7).  RECO with the chamber's golden calib_bundle (8000) ->
          ALIGN -> CACHE (ref from this row's waveform alignment, the group's K) -> FITV (v only, kernel pinned)
          -> RECO2 (8000, refitted bundle) -> ALIGN2 -> ANGLES2 -> W0KW -> <row>/wft/calib_bundle.
pass      every non-excluded row (S2 + S3).  RECO (all M3-matched events, the row's bundle: its own calib_bundle if
          it is a calibration row, else its group's) -> [calibration rows] ALIGN -> EFF (02), ANGLES (03), MAPS (04)
          with the group's waveform alignment.
"""
import argparse
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
#: the pass's own Analysis tree (PASS_PLAN E3: a new root, so no old product can be picked up)
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
#: DRC (TEMPLATE_STUDY T6/T8): pinned t0, unsaturated planes, p0 profiled +-2 mm, training events 580-1079 of the cache
DRC_ARGS = ('--free rc_D_y --objective cold --t0-pin --unsat-only --p0-profile --p0-window 2.0 --p0-step 0.2 '
            '--train-start 580 --n-train 500 --jobs 8 --maxiter 80')
GOLD_DET = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
#: chambers whose D_rc,y is a stated input (physics_inputs.json): no DRC node
DRC_FIXED = {'det4'}


def rel(r):
    return f"{r['run']}/{r['subrun']}/{r['det']}"


EXTRA = {'eos': '', 'limit': ''}


def node(L, name, sub, args, cpus, mem, code, parents=()):
    if sub == 'stage1.sub':
        k, rest = args.split(' ', 1)
        args = f"{k}{EXTRA['eos']} {rest}"
        if ' reco ' in rest and EXTRA['limit']:
            # a test limit replaces the stage's own event count (the calib phase's --limit 8000)
            args = re.sub(r' --limit \d+', '', args) + EXTRA['limit']
    L.append(f'JOB {name} {sub}')
    L.append(f'VARS {name} code="{code}" args="{args}" cpus="{cpus}" mem="{mem}" node="{name}"')
    L.append(f'RETRY {name} 3')            # transient EOS FUSE timeouts on workers (dry run 10-29: status 126)
    for p in parents:
        L.append(f'PARENT {p} CHILD {name}')


def calib_phase(L, rows, a, only):
    """E3: CAL0-CAL4 per golden row (module docstring)."""
    for r in rows:
        if r['refit'] != 'golden' or (only and r['key'] not in only):
            continue
        k, d, K = r['key'], GOLD_DET[r['det']], r['K']
        c = '{out}/wft/calib'
        # a dry run (--limit) keeps every node but shrinks the expensive ones: cache 700 events, DRC 60 events x 4 evals
        n_cache = 700 if a.limit else 1500
        drc = (DRC_ARGS.replace('--n-train 500', '--n-train 60').replace('--maxiter 80', '--maxiter 4') if a.limit
               else DRC_ARGS)
        S = lambda name, args, cpus, mem, parents=(): node(L, f'{name}_{d}', 'stage1.sub', f'{k} {args}', cpus, mem,
                                                           a.code, [f'{p}_{d}' for p in parents])
        bundle = lambda it, rc: (f'--parts run_config --script rerun/calib/make_bundle.py -- {d} --run-key {{key}} '
                                 f'--cache {c}/c{it}/cache.pkl --K {K} --force --out {c}/c{it}/'
                                 + ('seed' if rc is None else f'bundle{rc}'))
        reco = lambda b, t: f'--module wft.cli -- reco {{key}} --bundle {b} --out {t} --matched-only --limit 8000 --jobs 16'
        align = lambda t, o: f'--parts run_config,m3,hits --script mx_june_wft/01_alignment.py -- --table {t} --out {o}'
        ang = lambda t, al, o: (f'--parts run_config,m3,hits --script mx_june_wft/03_angles.py -- --table {t} '
                                f'--alignment {al} --out {o}')
        # CAL0
        S('HITS', '--parts run_config,m3,hits --require cache/event_results_veto50.pkl '
                  '--script mx_june_cosmic_qa/03_alignment_and_tpc.py', 4, 12000)
        S('CACHE0', f'--script rerun/w12/build_cache_wft.py -- --align {{out}}/alignment_tpc_veto50/alignment.json '
                    f'--out {c}/c0/cache.pkl --n {n_cache} --cm 0', 2, 16000, ['HITS'])
        prev = 'CACHE0'
        for it in (0, 1):
            if it == 1:
                t = f'{c}/r1/events.parquet'
                S('RECO1', reco(f'{c}/c0/bundle', t), 16, 24000, [f'MAKE0'])
                S('ALIGN1', align(t, f'{c}/r1/alignment'), 1, 8000, ['RECO1'])
                S('CACHE1', f'--script rerun/w12/build_cache_wft.py -- --align {c}/r1/alignment/alignment.json '
                            f'--out {c}/c1/cache.pkl --n {n_cache} --cm 0', 2, 16000, ['ALIGN1'])
                prev = 'CACHE1'
            if d in DRC_FIXED:
                S(f'MAKE{it}', bundle(it, ''), 1, 6000, [prev])
                continue
            S(f'SEED{it}', bundle(it, None), 1, 6000, [prev])
            S(f'DRC{it}', f'--parts run_config --script sps_beam_test_26/analysis/plane_ratio/recal.py -- '
                          f'--arm drc{it}_{{key}} --cache {c}/c{it}/cache.pkl --bundle {c}/c{it}/seed '
                          f'--depth-bins {K} {drc} --out {c}/c{it}/drc.json', 8, 12000, [f'SEED{it}'])
            S(f'MAKE{it}', bundle(it, '') + f' --rc-json {c}/c{it}/drc.json', 1, 6000, [f'DRC{it}'])
        # CAL3
        t2 = f'{c}/r2/events.parquet'
        S('RECO2', reco(f'{c}/c1/bundle', t2), 16, 24000, ['MAKE1'])
        S('ALIGN2', align(t2, f'{c}/r2/alignment'), 1, 8000, ['RECO2'])
        S('ANG2', ang(t2, f'{c}/r2/alignment/alignment.json', f'{c}/r2/angles'), 1, 8000, ['ALIGN2'])
        S('W0KW2', f'--parts run_config --script rerun/w12/set_w0kw.py -- --bundle {c}/c1/bundle --angles {c}/r2/angles '
                   f'--table {t2} --sigma0 0.30,0.30 --out {c}/c2/bundle_nobow', 1, 4000, ['ANG2'])
        # the bow fit maps w to tan through the bundle's w0/kw: a from-scratch c1 has none, so it takes W0KW2's
        # (measured on this same RECO2 table; dry run 10-29: KeyError 'x' with c1)
        S('BOW2', f'--parts run_config,m3 --script rerun/calib/bow_node.py -- {d} --events {t2} '
                  f'--alignment {c}/r2/alignment/alignment.json --bundle {c}/c2/bundle_nobow --out {c}/c2/bow_block.json',
          1, 8000, ['ALIGN2', 'W0KW2'])
        S('W0KW2b', f'--parts run_config --script rerun/w12/set_w0kw.py -- --bundle {c}/c2/bundle_nobow '
                    f'--angles {c}/r2/angles --table {t2} --bow-json {c}/c2/bow_block.json --out {c}/c2/bundle',
          1, 4000, ['W0KW2', 'BOW2'])
        # CAL4
        t3 = f'{c}/r3/events.parquet'
        S('RECO3', reco(f'{c}/c2/bundle', t3), 16, 24000, ['W0KW2b'])
        S('ALIGN3', align(t3, f'{c}/r3/alignment'), 1, 8000, ['RECO3'])
        S('ANG3', ang(t3, f'{c}/r3/alignment/alignment.json', f'{c}/r3/angles'), 1, 8000, ['ALIGN3'])
        S('W0KW3', f'--parts run_config --script rerun/w12/set_w0kw.py -- --bundle {c}/c2/bundle --angles {c}/r3/angles '
                   f'--table {t3} --out {{out}}/wft/calib_bundle', 1, 4000, ['ANG3'])
        S('BOW3', f'--parts run_config,m3 --script rerun/calib/bow_node.py -- {d} --events {t3} '
                  f'--alignment {c}/r3/alignment/alignment.json --bundle {c}/c2/bundle --out {c}/r3/bow_block.json',
          1, 8000, ['ALIGN3'])
        # G1 needs the loss budget of the final calibration reconstruction (gate_compare layout: <arm>/budget/)
        S('BUDGET', f'--parts run_config,m3,hits --script rerun/effbudget/budget.py -- {d} --table {t3} '
                    f'--alignment {c}/r3/alignment/alignment.json --out {c}/r3/budget', 1, 8000, ['ALIGN3'])
        S('CONV', f'--parts run_config --script rerun/calib/convergence.py -- --calib {c} --final {{out}}/wft/calib_bundle '
                  f'--out {c}/CALIB_CONVERGENCE.json', 1, 4000, ['W0KW3', 'BOW3'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['calib', 'calib_iter', 'finalize', 'vrefit', 'pass', 'sps'])
    ap.add_argument('code')
    ap.add_argument('--golden', nargs='*', default=[], help='det=itN (finalize)')
    ap.add_argument('--bundle-sub', default='fit_pinned/bundle',
                    help='finalize: bundle path under w12/<it>/ (arm D: timeD/bundle)')
    ap.add_argument('--out-tag', default='',
                    help='finalize: write wft/final_<tag>/ and wft/calib_bundle_<tag> (an arm next to the canonical one)')
    ap.add_argument('--keep-v', action='store_true',
                    help='vrefit: no waveform v fit (phys policy); the golden bundle + this group\'s own w0/kw')
    ap.add_argument('--registry', default=os.path.join(HERE, 'pass_registry.csv'))
    ap.add_argument('--only', default=None, help='comma list of keys (tests)')
    ap.add_argument('--analysis-root', default=None, help='tests: another Analysis tree (e.g. the dry-run one)')
    ap.add_argument('--limit', type=int, default=None, help='tests: events per reco')
    ap.add_argument('--skip-dets', default='', help='comma list (det2,det4): chambers held after G1 (Dylan 10-29: a chamber '
                                                    'that fails G1 is held, the others go on); sps is det4')
    a = ap.parse_args()
    global A
    if a.analysis_root:
        A = a.analysis_root
    if a.analysis_root:
        EXTRA['eos'] = f' --eos-analysis {a.analysis_root}'
    if a.limit:
        EXTRA['limit'] = f' --limit {a.limit}'
    rows = list(csv.DictReader(open(a.registry)))
    by_key = {r['key']: r for r in rows}
    skip = {f"mx17_{d.strip()[3:]}" for d in a.skip_dets.split(',') if d.strip()}
    if skip:
        rows = [r for r in rows if r['det'] not in skip]
        if a.phase == 'sps' and 'mx17_4' in skip:
            raise SystemExit('det4 is held: no sps DAG')
        print(f'held chambers (not in this DAG): {sorted(skip)}')
    only = set(a.only.split(',')) if a.only else None
    L = []
    if a.phase == 'calib':
        calib_phase(L, rows, a, only)
    elif a.phase == 'calib_iter':
        for r in rows:
            if r['refit'] != 'golden' or (only and r['key'] not in only):
                continue
            k, d, c = r['key'], GOLD_DET[r['det']], '{out}/wft/calib'
            t = f'{c}/r4/events.parquet'
            node(L, f'RECO4_{d}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {{out}}/wft/calib_bundle '
                 f'--out {t} --matched-only --limit 8000 --jobs 16', 16, 24000, a.code)
            node(L, f'ALIGN4_{d}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py -- '
                 f'--table {t} --out {c}/r4/alignment', 1, 8000, a.code, [f'RECO4_{d}'])
            node(L, f'ANG4_{d}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- '
                 f'--table {t} --alignment {c}/r4/alignment/alignment.json --out {c}/r4/angles', 1, 8000, a.code,
                 [f'ALIGN4_{d}'])
            node(L, f'W0KW4_{d}', 'stage1.sub', f'{k} --parts run_config --script rerun/w12/set_w0kw.py -- '
                 f'--bundle {{out}}/wft/calib_bundle --angles {c}/r4/angles --table {t} --out {c}/c4/bundle',
                 1, 4000, a.code, [f'ANG4_{d}'])
    elif a.phase == 'sps':
        det4 = next(r for r in rows if r['refit'] == 'golden' and r['det'] == 'mx17_4')
        b = f'{A}/{rel(det4)}/wft/calib_bundle'
        for r in csv.DictReader(open(os.path.join(HERE, 'sps_registry.csv'))):
            if only and r['key'] not in only:
                continue
            lim = f" --per-plateau {a.limit}" if a.limit else ''
            node(L, f"SPS_{r['key']}", 'sps.sub', f"{r['key']} {A}/sps/{r['key']} --bundle {b}{lim}", 8, 16000, a.code)
    elif a.phase == 'finalize':
        its = dict(g.split('=') for g in a.golden)            # det2=it1 ...
        for r in rows:
            d = r['det'].replace('mx17_', 'det')
            if r['refit'] != 'golden' or d not in its or (only and r['key'] not in only):
                continue
            k, it = r['key'], its[d]
            b = f'{{out}}/wft/w12/{it}/{a.bundle_sub}'
            sfx = f'_{a.out_tag}' if a.out_tag else ''
            fin = f'{{out}}/wft/final{sfx}'
            t = f'{fin}/events.parquet'
            node(L, f'RECO_{d}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {b} --out {t} '
                 f'--matched-only --limit 8000 --jobs 16', 16, 24000, a.code)
            node(L, f'ALIGN_{d}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py '
                 f'-- --table {t} --out {fin}/alignment', 1, 8000, a.code, [f'RECO_{d}'])
            node(L, f'ANG_{d}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- '
                 f'--table {t} --alignment {fin}/alignment/alignment.json --out {fin}/angles',
                 1, 8000, a.code, [f'ALIGN_{d}'])
            node(L, f'W0KW_{d}', 'stage1.sub', f'{k} --parts run_config --script rerun/w12/set_w0kw.py -- '
                 f'--bundle {b} --angles {fin}/angles --table {t} --out {{out}}/wft/calib_bundle{sfx}', 1, 4000, a.code,
                 [f'ANG_{d}'])
    elif a.phase == 'vrefit':
        gold = {r['det']: r for r in rows if r['refit'] == 'golden'}
        for r in rows:
            if r['refit'] != 'v' or (only and r['key'] not in only):
                continue
            k, n = r['key'], r['key'].replace('.', '_')
            g = f"{A}/{rel(gold[r['det']])}/wft/calib_bundle"
            t1, t2 = '{out}/wft/vrefit/events0.parquet', '{out}/wft/vrefit/events1.parquet'
            kk = f' --n-depth-bins {r["K"]}' if a.keep_v else ''
            node(L, f'RECO_{n}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {g} --out {t1} '
                 f'--matched-only --limit 8000 --jobs 16{kk}', 16, 24000, a.code)
            node(L, f'ALIGN_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py '
                 f'-- --table {t1} --out {{out}}/wft/vrefit/alignment0', 1, 8000, a.code, [f'RECO_{n}'])
            if a.keep_v:
                # phys policy (RERUN_PLAN §10/§11): the waveform fit does not determine v (C17) and the fit uses v only
                # for its seed and the final tan = (w - w0) / (kw v), so the group keeps the golden bundle (v, kernel,
                # mask) and gets its own w0/kw under it; the group's depth v is the timing v, measured downstream (S4)
                node(L, f'ANG_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- '
                     f'--table {t1} --alignment {{out}}/wft/vrefit/alignment0/alignment.json --out {{out}}/wft/vrefit/angles',
                     1, 8000, a.code, [f'ALIGN_{n}'])
                node(L, f'W0KW_{n}', 'stage1.sub', f'{k} --parts run_config --script rerun/w12/set_w0kw.py -- '
                     f'--bundle {g} --angles {{out}}/wft/vrefit/angles --table {t1} --out {{out}}/wft/calib_bundle --n-depth-bins {r["K"]} --fold-kw',
                     1, 4000, a.code,
                     [f'ANG_{n}'])
                continue
            node(L, f'HITS_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --require cache/event_results_veto50.pkl '
                 f'--script mx_june_cosmic_qa/03_alignment_and_tpc.py', 4, 12000, a.code)
            node(L, f'CACHE_{n}', 'stage1.sub', f'{k} --script rerun/w12/build_cache_wft.py -- '
                 f'--align {{out}}/wft/vrefit/alignment0/alignment.json --out {{out}}/wft/vrefit/cache.pkl --cm 0',
                 2, 16000, a.code, [f'ALIGN_{n}', f'HITS_{n}'])
            node(L, f'FITV_{n}', 'fitv1.sub', f'{rel(r)} {g} {r["K"]} vrefit/fit {A}', 8, 12000, a.code, [f'CACHE_{n}'])
            node(L, f'RECO2_{n}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {{out}}/wft/vrefit/fit/bundle '
                 f'--out {t2} --matched-only --limit 8000 --jobs 16', 16, 24000, a.code, [f'FITV_{n}'])
            node(L, f'ALIGN2_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py '
                 f'-- --table {t2} --out {{out}}/wft/vrefit/alignment1', 1, 8000, a.code, [f'RECO2_{n}'])
            node(L, f'ANG_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/03_angles.py -- '
                 f'--table {t2} --alignment {{out}}/wft/vrefit/alignment1/alignment.json --out {{out}}/wft/vrefit/angles',
                 1, 8000, a.code, [f'ALIGN2_{n}'])
            node(L, f'W0KW_{n}', 'stage1.sub', f'{k} --parts run_config --script rerun/w12/set_w0kw.py -- '
                 f'--bundle {{out}}/wft/vrefit/fit/bundle --angles {{out}}/wft/vrefit/angles --table {t2} --out {{out}}/wft/calib_bundle',
                 1, 4000, a.code, [f'ANG_{n}'])
    else:
        for r in rows:
            if r['cls'] == 'excluded' or (only and r['key'] not in only):
                continue
            k, n = r['key'], r['key'].replace('.', '_')
            ck = by_key[r['calib_key']]
            ak = r.get('align_key') or r['calib_key']          # the geometry: its own night (make_registry align_key)
            own = r['key'] == r['calib_key']
            own_al = r['key'] == ak
            b = '{out}/wft/calib_bundle' if own else f'{A}/{rel(ck)}/wft/calib_bundle'
            al = '{out}/wft/alignment/alignment.json' if own_al else f'{A}/{rel(by_key[ak])}/wft/alignment/alignment.json'
            node(L, f'RECO_{n}', 'stage1.sub', f'{k} --module wft.cli -- reco {{key}} --bundle {b} --matched-only '
                 f'--jobs 16', 16, 24000, a.code)
            after = [f'RECO_{n}']
            if own_al:
                node(L, f'ALIGN_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/01_alignment.py',
                     1, 8000, a.code, after)
                after = [f'ALIGN_{n}']
            for s, sc in (('EFF', '02_efficiency'), ('ANG', '03_angles'), ('MAPS', '04_maps')):
                node(L, f'{s}_{n}', 'stage1.sub', f'{k} --parts run_config,m3,hits --script mx_june_wft/{sc}.py -- '
                     f'--alignment {al}', 1, 8000, a.code, after)
        # a group's other rows read the group's alignment: they wait for it
        for r in rows:
            ak = r.get('align_key') or r['calib_key']
            if r['cls'] == 'excluded' or (only and r['key'] not in only) or r['key'] == ak:
                continue
            if only and ak not in only:
                continue
            n, cn = r['key'].replace('.', '_'), ak.replace('.', '_')
            for s in ('EFF', 'ANG', 'MAPS'):
                L.append(f'PARENT ALIGN_{cn} CHILD {s}_{n}')
    open(f'{a.phase}.dag', 'w').write('\n'.join(L) + '\n')
    print(f'wrote {a.phase}.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
