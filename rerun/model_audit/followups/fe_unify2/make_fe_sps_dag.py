#!/usr/bin/env python3
"""make_fe_sps_dag.py <code_dir> -- FE's SPS gate: det4 at H4, run_63 25.64 deg caches REBUILT without the 256-ADC
offset (ws_sps fix 7505338, /eos/user/d/dneff/x17rerun/ws_sps/data/fixed/), the ONE reconstruction (heldout_arm.py:
wft.reco.fit_plane per view, pivot start, band censoring, ZS samples as measured zeros, no per-event sigma0 choice),
bench det4 bundle with the FE content (fleet sigma0; t0 NOT pinned -- the beam trigger is not on the DREAM clock),
v = the span-route CF4 v(E) of the plateau.  Arms = the channel layer at H4 (sps_mask.py):
  none        no mask (every SPS arm before 10-27)
  h4          the mask measured at H4 with the uRWELL (rerun/channel_layer/det4_sps_h4.json) -- never run before
  h4_stripes  h4 + the strips under det4's dead amplification stripes
Output <out>/run63_rot25_operating_00_<plateau>_<arm>.npz (sps_summary.py reads them).  Writes fe_sps.dag + heldout.sub.
"""
import argparse, os

CACHES = '/eos/user/d/dneff/x17rerun/ws_sps/data/fixed'
BUNDLE = ('/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe/mx17_det4_day_6-24-26/long_run/mx17_4/wft/'
          'calib_bundle_fe1')
OUT = '/eos/user/d/dneff/x17rerun/followups/fe/sps'
PLATEAUS = {'d675': 14.0, 'd575': 12.4, 'd475': 11.6}
SUB = '''executable     = $ENV(HOME)/x17rerun/audit/run_heldout.sh
arguments      = $(code) $(env) $(args)
output         = log/$(nodename).out
error          = log/$(nodename).err
log            = log/fe_sps.log
request_cpus   = 8
request_memory = 16000
+JobFlavour    = "workday"
should_transfer_files = NO
queue
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('code'); ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--arms', nargs='*', default=['none', 'h4', 'h4_stripes'])
    a = ap.parse_args(); L = []
    for pl, v in PLATEAUS.items():
        k = int(30000.0 / v / 60.0) + 5
        for arm in a.arms:
            name = f'FESPS_{pl}_{arm}'
            args = (f'--cache {CACHES}/run63_rot25_operating_00_{pl}.pkl --bundle {BUNDLE} '
                    f'--out {OUT}/run63_rot25_operating_00_{pl}_{arm}.npz --start 0 --n {a.n} --jobs 8 --k {k} --nsamp 64 '
                    f'--v {v} --sps-mask {arm}')
            L += [f'JOB {name} fe_sps.sub', f'VARS {name} nodename="{name}" code="{a.code}" env="-" args="{args}"',
                  f'RETRY {name} 1']
    open('fe_sps.dag', 'w').write('\n'.join(L) + '\n'); open('fe_sps.sub', 'w').write(SUB)
    print(f'wrote fe_sps.dag: {len(L) // 3} nodes')


if __name__ == '__main__':
    main()
