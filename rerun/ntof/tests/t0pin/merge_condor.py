#!/usr/bin/env python3
"""merge_condor.py -- concat the per-acquisition condor outputs of the t0 study into r149sNN_<arm>_<cfg> tables."""
import glob, os, re, sys
import pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else '/media/ucla/work/ntof_pass1/t0pin'
for cfgdir in sorted(glob.glob(f'{D}/condor/*')):
    lab = os.path.basename(cfgdir)
    for armdir in sorted(glob.glob(f'{cfgdir}/run_149/*/*')):
        sub, arm = armdir.split('/')[-2:]
        s = re.search(r'_(\d+)$', sub).group(1)[-2:]
        out = f'{D}/r149s{s}_{arm}_{lab}'
        if os.path.exists(out + '.candidates.parquet'):
            continue
        for kind in ('parquet', 'candidates.parquet'):
            fs = sorted(f for f in glob.glob(f'{armdir}/*.{kind}') if kind != 'parquet' or f.count('.') == 1)
            pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).to_parquet(f'{out}.{kind}')
        print(out, len(fs))
