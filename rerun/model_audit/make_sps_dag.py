#!/usr/bin/env python3
"""make_sps_dag.py <code_dir> [--caches ...] -- SPS det4 known-angle arms (model audit 2026-10-27, RERUN_PLAN S4 R7).

One node per (plateau cache, arm): heldout_arm.py on the inclined run_63 operating_00 caches (sps_build_cache.py) with
the promoted bench det4 bundle, v = the span-route CF4 v(E) of the plateau, K covering the column at 64 samples, ZS
censoring, bench dead list dropped (other cabling).  Output /eos/user/d/dneff/x17rerun/model_audit/sps/<cache>_<arm>.npz.
"""
import argparse

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle'
S = '/eos/user/d/dneff/x17rerun/model_audit/sps'
# plateau -> (drift V, field V/cm over 28.8 mm, span-route v um/ns: EXTRACTION_2026-08-05 section 4)
PLATEAUS = {'d675': (675.5, 235, 14.0), 'd625': (625.3, 217, 13.2), 'd575': (575.3, 200, 12.4), 'd525': (525.2, 182, 11.8),
            'd475': (475.3, 165, 11.6), 'd425': (425.2, 148, 11.6), 'd325': (325.1, 113, 11.5)}
ARMS = {'base': ('WFT_ZS_CENSOR=1', ''), 't0i': ('WFT_ZS_CENSOR=1 WFT_RC_T0_INTERP=1', ''),
        'asymp': ('WFT_ZS_CENSOR=1', '--hyper {\\"rc_asym_x\\":0.13}'), 'asymn': ('WFT_ZS_CENSOR=1', '--hyper {\\"rc_asym_x\\":-0.13}'),
        'nozs': ('', '')}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('code')
    ap.add_argument('--subrun', default='operating_00'); ap.add_argument('--dataset', default='run63_rot25')
    ap.add_argument('--plateaus', nargs='*', default=['d675', 'd625', 'd575', 'd525', 'd475'])
    ap.add_argument('--arms', nargs='*', default=list(ARMS)); ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--out-dir', default=S); ap.add_argument('--tag', default='sps')
    a = ap.parse_args(); L = []
    for pl in a.plateaus:
        dv, E, v = PLATEAUS[pl]
        k = int(30000.0 / v / 60.0) + 5
        cache = f'{S}/{a.dataset}_{a.subrun}_{pl}.pkl'
        for arm in a.arms:
            env, extra = ARMS[arm]; name = f'SPS_{pl}_{arm}'
            env = env.replace(' ', ',') or '-'
            args = (f'--cache {cache} --bundle {A} --out {a.out_dir}/{a.dataset}_{a.subrun}_{pl}_{arm}.npz --start 0 --n {a.n} --jobs 8 '
                    f'--k {k} --nsamp 64 --v {v} --no-dead {extra}')
            L += [f'JOB {name} heldout.sub', f'VARS {name} nodename="{name}" code="{a.code}" env="{env}" args="{args}"', f'RETRY {name} 1']
    open(f'{a.tag}.dag', 'w').write('\n'.join(L) + '\n')
    print(f'wrote {a.tag}.dag: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
