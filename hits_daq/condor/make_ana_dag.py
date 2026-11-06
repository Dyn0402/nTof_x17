#!/usr/bin/env python3
"""make_ana_dag.py <dag dir> --code <archive> [--stage tiers,h2c,resim] -- the Arm D / Arm S analysis DAG on lxplus.

Inputs are the extracted windows on EOS (hits_daq/windows/, FUSE paths), outputs go to hits_daq/results/ (ana.sh).
  tiers   one job per window file: every V family x every ladder variant (tiers.py --table-only; scored locally on
          the merged tables with tiers.py --from-tables)
  h2c     H2c per window file, split in N_H2C parts (event parity), families H2C_FAMS
  resim   Arm S on the det3 Run 1 windows: the resistivity / pattern / footprint variants, split in N_RESIM parts
Every node RETRY 2 (EOS transients).  Flavour 'tomorrow' (24 h): bench H2c ~2.8 h, SPS H2c ~3 h, resim ~2 h per node
(smoke 2026-11-06) and worker speed varies; 'workday' (8 h) removed the SPS extraction twice.
"""
import argparse
import glob
import os

EOSW = '/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/hits_daq/windows'
FRAMES = '/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_floor/frames'
P2 = 'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3'
ROWS = {'sat_det3': ('sat_det3', 28.08), 'p2_det3': (P2, 28.08)}
N_H2C = 8
# SPS windows: 6000 planes of 64 samples, ~45 s/plane for the 7 families (smoke 2026-11-06) -> 24 parts of ~3 h
N_H2C_SPS = 24
H2C_FAMS = 'dream,peak,thr,tot,fd,binary,ideal'
N_RESIM = 4
VARIANTS = ('base:,rc0:rc_D_y=0,rc025:rc_D_y=x0.25,rc05:rc_D_y=x0.5,rc2:rc_D_y=x2,rc5:rc_D_y=x5,rc10:rc_D_y=x10,'
            'sheet:rc_D_x=@rc_D_y,s0_07:sigma_p0=0.07,s0_15:sigma_p0=0.15,s0_45:sigma_p0=0.45,'
            'metal:rc_D_y=0;sigma_p0=0.07,dp0:Dp=0,point:rc_D_y=0;sigma_p0=0.07;Dp=0')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dag')
    ap.add_argument('--code', required=True)
    ap.add_argument('--stage', default='tiers,h2c,resim')
    ap.add_argument('--windows', default=EOSW)
    a = ap.parse_args()
    st = a.stage.split(',')
    L = []

    def node(name, script, out, args, mem=4000):
        L.extend([f'JOB {name} ana.sub', f'VARS {name} code="{a.code}" script="{script}" name="{out}" '
                  f'args="{args}" mem="{mem}"', f'RETRY {name} 2'])

    for tag, (key, gap) in ROWS.items():
        files = sorted(glob.glob(os.path.join(a.windows, f'{tag}_p*of*.pkl')))
        ref = os.path.join(FRAMES, f'{key}.parquet')
        for i, f in enumerate(files):
            b = os.path.basename(f)[:-4]
            if 'tiers' in st:
                node(f'TT_{b}', 'tiers.py', f'tiers_{b}.parquet', f'{f} --ref {ref} --table-only')
            if 'h2c' in st:
                for j in range(N_H2C):
                    node(f'H_{b}_{j}', 'h2c_run.py', f'h2c_{b}_{j}of{N_H2C}.parquet',
                         f'{f} --ref {ref} --gap {gap} --families {H2C_FAMS} --part {j}/{N_H2C}')
            if 'resim' in st and tag == 'sat_det3':
                for j in range(N_RESIM):
                    node(f'R_{b}_{j}', 'resim.py', f'resim_{b}_{j}of{N_RESIM}.parquet',
                         f"{f} --variants '{VARIANTS}' --families {H2C_FAMS} --part {j}/{N_RESIM} --refit", mem=6000)
    for f in sorted(glob.glob(os.path.join(a.windows, 'sps_*.pkl'))):
        b = os.path.basename(f)[:-4]
        if 'tiers' in st:
            node(f'TT_{b}', 'tiers.py', f'tiers_{b}.parquet', f'{f} --table-only')
        if 'h2c' in st:
            for j in range(N_H2C_SPS):
                node(f'H_{b}_{j}', 'h2c_run.py', f'h2c_{b}_{j}of{N_H2C_SPS}.parquet',
                     f'{f} --gap 27.9 --families {H2C_FAMS} --part {j}/{N_H2C_SPS}')
    os.makedirs(os.path.join(a.dag, 'log'), exist_ok=True)
    open(os.path.join(a.dag, 'ana.dag'), 'w').write('\n'.join(L) + '\n')
    open(os.path.join(a.dag, 'ana.sub'), 'w').write(f"""# hits-DAQ analysis node (papers/paper2/hits_daq/condor/ana.sh)
executable     = /bin/bash
arguments      = $(code)/papers/paper2/hits_daq/condor/ana.sh $(code) $(script) $(name) $(args)
output         = log/$(name).out
error          = log/$(name).err
log            = log/ana.log
request_cpus   = 1
request_memory = $(mem)
+JobFlavour    = "tomorrow"
should_transfer_files = NO
requirements   = (Machine != "b9g42p2031.cern.ch")
queue
""")
    print(sum(1 for x in L if x.startswith('JOB ')), 'nodes')


if __name__ == '__main__':
    main()
