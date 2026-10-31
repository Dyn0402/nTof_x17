#!/usr/bin/env python3
"""res_check.py <run> <subrun> <group> <feu3 dir> <nsamp> <out.json>: does the new oscillator handling touch the
resolution sample?  On one FEU 03 file of the telescope audit (rerun/sps_audit/tel_job.py), with tel_job's hot probe
(the PASS selection, seeded with SEED_BAD): the hot channels by the rule the telescope tables were made with (all +-4
neighbours, the first 2000 events) against the new rules (hot_in_blocks on live neighbours, near_oscillators,
coupled_channels).  Identical hot sets and no coupling = the resolution sample is unchanged by them."""
import glob, json, os, sys
import numpy as np

CODE = os.environ['WFT_REPO']
sys.path[:0] = [CODE, os.path.join(CODE, 'rerun', 'model_audit')]
import sps_build_cache as sbc   # noqa: E402
from det4_sps_map import POSITION_MM, VIEW   # noqa: E402


def hot_old(cnt, n):
    out = np.zeros(512, bool)
    for view in ('x', 'y'):
        chs = np.flatnonzero(VIEW == view); o = chs[np.argsort(POSITION_MM[chs])]
        for i, c in enumerate(o):
            nb = np.r_[o[max(0, i - 4):i], o[i + 1:i + 5]]
            out[c] = cnt[c] > 0.03 * n and cnt[c] > 10.0 * max(np.median(cnt[nb]), 1.0)
    return out


run, subrun, group, f3dir, nsamp, outp = sys.argv[1:7]
nsamp = int(nsamp); raw = False
sub_dir = os.path.join(sbc.RUNS, run, subrun)
f3dir = f3dir if os.path.isabs(f3dir) else os.path.join(sub_dir, f3dir)
f3 = sorted(glob.glob(os.path.join(f3dir, f'*_datrun_*_{group}_03.root')))[0]
import uproot   # noqa: E402
t = uproot.open(f3)['nt']; seeds, peaks = [], []
for lo in range(0, t.num_entries, 5000):
    arr = t.arrays(['sample', 'channel', 'amplitude'], entry_start=lo, entry_stop=min(lo + 5000, t.num_entries), library='np')
    for j in range(len(arr['channel'])):
        r = sbc.event_record(arr['channel'][j].astype(int), arr['sample'][j].astype(int), arr['amplitude'][j].astype(np.float32),
                             raw, nsamp, keep_peaks=True)
        if r is not None:
            seeds.append((r['seed_ch_x'], r['seed_ch_y'])); peaks.append(r['peak'])
    if len(seeds) >= 3000:
        break
seeds = seeds[:3000]; peaks = np.array(peaks[:3000])
cnt = np.zeros(512)
for sx, sy in seeds[:2000]:
    cnt[sx] += 1; cnt[sy] += 1
old = hot_old(cnt, min(len(seeds), 2000))
new = sbc.hot_in_blocks(seeds)
new |= sbc.near_oscillators(peaks, sbc.SEED_BAD | new)
cp = sbc.coupled_channels(peaks, sbc.SEED_BAD | new)
out = dict(file=os.path.basename(f3), n=len(seeds), hot_old=[int(c) for c in np.flatnonzero(old)],
           hot_new=[int(c) for c in np.flatnonzero(new)], coupled={str(k): v for k, v in cp.items()})
out['unchanged'] = out['hot_old'] == out['hot_new'] and not cp
json.dump(out, open(outp, 'w')); print(json.dumps(out))
