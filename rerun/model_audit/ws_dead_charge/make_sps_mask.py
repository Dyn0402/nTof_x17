#!/usr/bin/env python3
"""make_sps_mask.py -- the det4 H4 dead mask, in FEU 03 channel numbers, from the SPS uRWELL censuses (sps_census.py).

dead          the bench census rule (firing given a uRWELL crossing < 0.25 x its +-6 neighbours) in >= 2 of the 3 flat-mount
              datasets -- same rule as every bench bundle (one treatment); this is what sps_mask.py 'h4' applies
dead_stripes  also the strips under a dead amplification stripe (reference max(neighbours, half the view median)): they
              never avalanche but DO carry charge spread from live neighbours (40 % of the run63 X windows hold one
              with samples), so censoring them is a model choice that needs the gate; sps_mask.py 'h4_stripes'
Both from the 3 flat-mount datasets (run56 Ar/CO2, run63 CF4, run71 CF4 RAW); channels outside the
beam (not judged anywhere) are not masked -- no track reaches them.  For comparison the bench g_det4 bundle mask
remapped by position (CalibrationBundle.remap_channels) is written next to it.
-> rerun/channel_layer/det4_sps_h4.json
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
from det4_sps_map import POSITION_MM, VIEW            # noqa: E402
from wft.calib import CalibrationBundle               # noqa: E402

TAGS = ['run56_m10V', 'run63_flat', 'run71_raw']
S = {t: json.load(open(os.path.join(HERE, 'census', f'sps_{t}.json'))) for t in TAGS}
out = dict(source='rerun/model_audit/ws_dead_charge/sps_census.py on ' + ', '.join(TAGS),
           rule='dead: bench census rule (crossing firing < 0.25 x +-6 neighbours) in >= 2 of 3; dead_stripes: reference max(neighbours, half the view median), so the dead amplification stripes count', channel_numbering='SPS H4 FEU 03 (det4_sps_map)')
votes = {p: np.zeros(512, int) for p in 'xy'}; vst = {p: np.zeros(512, int) for p in 'xy'}
for t, s in S.items():
    for p in 'xy':
        votes[p][s[p]['dead_nb']] += 1; vst[p][s[p]['dead']] += 1
for p in 'xy':
    d = np.flatnonzero(votes[p] >= 2).tolist(); ds = np.flatnonzero((vst[p] >= 2) | (votes[p] >= 2)).tolist()
    out[p] = dict(dead=d, dead_pos_mm=[round(float(POSITION_MM[c]), 2) for c in d], dead_stripes=ds,
                  judged_any=sorted(set().union(*[s[p]['judged'] for s in S.values()])))
# bench mask by position, for the record
b = CalibrationBundle.load(os.path.expanduser('~/x17/cosmic_bench/model_audit/det4/calib_bundle'))
c = np.arange(512); bp = 49.92 * (c // 64) + (63 - c % 64) * 0.78
rec = b.remap_channels({'x': bp, 'y': bp}, {v: np.where(VIEW == v, POSITION_MM, np.nan) for v in 'xy'}, target='sps det4')
out['bench_remapped'] = dict(dead=b.dead, record=rec)
for p in 'xy':
    jd = set(out[p]['judged_any'])
    bset = set(b.dead[p]); sset = set(out[p]['dead'])
    out[p]['bench_remapped_in_beam'] = sorted(bset & jd)
    out[p]['agree'] = sorted(bset & sset)
    print(p, 'SPS mask', len(sset), 'bench remapped', len(bset), f'(in beam {len(bset & jd)})', 'agree', len(bset & sset),
          'sps-only', sorted(round(float(POSITION_MM[x]), 2) for x in sset - bset),
          'bench-only in beam', sorted(round(float(POSITION_MM[x]), 2) for x in (bset & jd) - sset))
json.dump(out, open(os.path.join(REPO, 'rerun', 'channel_layer', 'det4_sps_h4.json'), 'w'), indent=1)
