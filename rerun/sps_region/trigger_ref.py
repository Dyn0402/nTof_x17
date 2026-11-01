#!/usr/bin/env python3
"""trigger_ref.py [run_63 operating_02 ...] -- how stable the H4 trigger is against the DREAM time base, from the uRWELL
telescope's own hits (banco's combined_hits, FEU 1: hit times are interpolated, not sample-quantised).  Per event the
time of the largest uRWELL hit (amp > 150 ADC, 200-1200 ns) is taken; its s68 over events is the uRWELL time resolution
(+) the trigger's phase against the 60 ns clock (uniform +-30 ns alone would give s68 ~ 20 ns).  It bounds the trigger
part of any det4 'time against the trigger'.  Run on lxplus (reads EOS); prints JSON."""
import glob, json, sys
import numpy as np
import uproot

RUNS = '/eos/experiment/ntof/data/x17/p2_sps_july/runs'
run, subs = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 2 else ('run_63', ['operating_02', 'operating_03'])
out = {}
for sub in subs:
    for f in sorted(glob.glob(f'{RUNS}/{run}/{sub}/combined_hits_root/*_feu-combined_hits.root')):
        a = uproot.open(f)['hits'].arrays(['eventId', 'feu', 'amplitude', 'time'], library='np')
        amp = np.abs(a['amplitude']); tm = a['time']; ev = a['eventId']
        m = (a['feu'] == 1) & (amp > 150) & (tm > 200) & (tm < 1200)
        o = np.lexsort((-amp[m], ev[m])); e2 = ev[m][o]; t2 = tm[m][o]
        lead = t2[np.r_[True, e2[1:] != e2[:-1]]]
        q = np.percentile(lead, [16, 50, 84])
        out[f.split('/')[-1]] = dict(n=int(len(lead)), median=float(q[1]), s68=float((q[2] - q[0]) / 2))
        print(f.split('/')[-1], out[f.split('/')[-1]], file=sys.stderr, flush=True)
s = np.array([v['s68'] for v in out.values()]); n = np.array([v['n'] for v in out.values()])
print(json.dumps(dict(run=run, files=out, s68_mean=float((s * n).sum() / n.sum()), s68_range=[float(s.min()), float(s.max())]), indent=1))
