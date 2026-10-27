#!/usr/bin/env python3
"""make_res_dag.py [--out ws_sps_res.dag] -- heldout_arm.py nodes for the SPS position-resolution caches (resolution_cache.py).

Submit from ~/x17rerun/audit (heldout.sub + run_heldout.sh live there): condor_submit_dag ws_sps_res.dag
"""
import argparse

B = '/eos/user/d/dneff/x17rerun/ws_sps'
BUN = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle'
CACHES = (('res_flat_flat700', 14.0), ('res_rot25_d425', 11.6), ('res_rot25_d325', 11.5))
ARMS = (('cens', 'WFT_ZS_CENSOR=1'), ('zsb4', 'WFT_ZS_BOUND=4'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='ws_sps_res.dag'); ap.add_argument('--code', default=f'{B}/code3')
    a = ap.parse_args(); L = []
    for c, v in CACHES:
        for arm, env in ARMS:
            n = f'RES_{c}_{arm}'
            args = (f'--cache {B}/data/{c}.pkl --bundle {BUN} --out {B}/out/{c}_{arm}.npz --start 0 --n 3000 --jobs 8 --k 60 '
                    f'--nsamp 64 --v {v} --no-dead')
            L += [f'JOB {n} heldout.sub', f'VARS {n} nodename="{n}" code="{a.code}" env="{env}" args="{args}"', f'RETRY {n} 1']
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(f'wrote {a.out}: {len(L) // 3} nodes')


if __name__ == '__main__':
    main()
