#!/usr/bin/env python3
"""
gate_eval.py — score the t0-prior gate arms for one golden key through the
STANDARD chain accounting (01_alignment → 02_efficiency → 03_angles), and
print the adopt/fallback comparison.

    ../../.venv/bin/python mx_june_wft/condor/gate_eval.py o22_long_det2 \
        --results /home/dylan/x17/cosmic_bench/condor_campaign/results

Expects the campaign staging dirs <results>/<key>/ (prod arm — frozen bundle,
no prior) and <results>/<key>__t0p/ (prior arm) from collect_results.py. Each
arm is evaluated by temporarily swapping its events.parquet into the live
<OUT_BASE>/wft/ (originals parked in wft/gate_backup/ and restored — even on
error), so both arms go through exactly the accounting FLEET_DIGEST uses.
Verdict rule (T0_PRIOR_2026-08-11 §8): ADOPT if nothing regresses beyond noise
(within5 −0.1, far +0.1, core +2 %) and something improves; else FALLBACK.
The verdict is advice — the campaign runbook says who decides.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa')]

FILES = ('events.parquet', 'events.candidates.parquet', 'events.meta.json')
PY = os.path.join(REPO, '.venv', 'bin', 'python')


def run(script, key):
    subprocess.run([PY, os.path.join(REPO, 'mx_june_wft', script), key],
                   check=True, cwd=REPO)


def eval_arm(cfg, key, arm_dir):
    wft = os.path.join(cfg.OUT_BASE, 'wft')
    bak = os.path.join(wft, 'gate_backup')
    os.makedirs(bak, exist_ok=True)
    moved = []
    for f in FILES:
        if os.path.exists(os.path.join(wft, f)):
            shutil.move(os.path.join(wft, f), os.path.join(bak, f))
            moved.append(f)
    try:
        for f in FILES:
            src = os.path.join(arm_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(wft, f))
        run('01_alignment.py', key)
        subprocess.run([PY, os.path.join(REPO, 'mx_june_wft',
                                         '02_efficiency.py'), key,
                        '--max-dropped', '-1'], check=True, cwd=REPO)
        run('03_angles.py', key)
        out = {}
        with open(os.path.join(wft, 'efficiency',
                               'efficiency_breakdown.json')) as f:
            e = json.load(f)
        out.update(within5=e['within_R'], far=e['reco_far'],
                   core=e['core_sigma_mm'], median=e['median_r_mm'])
        with open(os.path.join(wft, 'angles',
                               'angular_resolution.json')) as f:
            a = json.load(f)
        for p in ('x', 'y'):
            out[f'sig_{p}'] = a['planes'][p]['sigma_deg']
            out[f'vsp_{p}'] = a['planes'][p]['implied_v_spread']
        ev = os.path.join(arm_dir, 'eval')
        os.makedirs(ev, exist_ok=True)
        for sub in ('efficiency', 'angles', 'alignment'):
            if os.path.isdir(os.path.join(wft, sub)):
                shutil.copytree(os.path.join(wft, sub),
                                os.path.join(ev, sub), dirs_exist_ok=True)
        with open(os.path.join(ev, 'summary.json'), 'w') as f:
            json.dump(out, f, indent=1)
        return out
    finally:
        for f in FILES:
            p = os.path.join(wft, f)
            if os.path.exists(p):
                os.remove(p)
        for f in moved:
            shutil.move(os.path.join(bak, f), os.path.join(wft, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--results',
                    default='/home/dylan/x17/cosmic_bench/condor_campaign/'
                            'results')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(args.run_key)

    arms = {}
    for tag, name in (('prod', args.run_key),
                      ('t0p', args.run_key + '__t0p')):
        d = os.path.join(args.results, name)
        if not os.path.isdir(d):
            sys.exit(f'missing arm {d} — run collect_results.py first')
        print(f'=== evaluating {tag}: {d}')
        arms[tag] = eval_arm(cfg, args.run_key, d)

    a, b = arms['prod'], arms['t0p']
    print(f'\n{"metric":10s} {"prod":>9s} {"t0p":>9s}   delta')
    for m in ('within5', 'far', 'core', 'median', 'sig_x', 'sig_y',
              'vsp_x', 'vsp_y'):
        va, vb = a.get(m), b.get(m)
        if va is None or vb is None:
            continue
        print(f'{m:10s} {va:9.3f} {vb:9.3f}   {vb-va:+.3f}')
    ok = (b['within5'] >= a['within5'] - 0.1 and b['far'] <= a['far'] + 0.1
          and b['core'] <= a['core'] * 1.02)
    better = (b['within5'] > a['within5'] or b['far'] < a['far']
              or b['core'] < a['core'])
    print('\nVERDICT:', 'ADOPT t0 prior' if ok and better else
          'FALLBACK to no-prior bundle',
          '(rule: no regression beyond within5 −0.1 / far +0.1 / core +2%, '
          'and at least one improvement)')


if __name__ == '__main__':
    main()
