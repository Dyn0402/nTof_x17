"""Strip-level test of the shorts (hits used for QA only: does a channel fire, no geometry from
hit times). For each channel of det7 X, the firing probability (hit > 50 ADC) as a function of
where the M3 track crosses, in strips relative to the channel's own mapped strip. A pad-level
short at the pogo field joins strips 5 apart, so a shorted channel must also fire when the
track crosses the partner strip (offset +-5). Electronic crosstalk between M1/cable neighbours
would instead copy a neighbour channel's signal. Writes fire_offset.npz."""
import os, sys, glob, json, numpy as np
REPO = '/home/dylan/PycharmProjects/nTof_x17'
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis'),
                os.path.join(REPO, 'rerun', 'seedfit')]
from qa_config import get_config, setup_paths
setup_paths()
from wft.io import strip_position_map
import channel_census as CC
cfg = get_config('g_det7_long'); pm = np.asarray(strip_position_map(cfg)[6], float)
cl = json.load(open(f'{REPO}/rerun/channel_layer/det7_g_det7_long.json'))['x']
k = 'xy'.index(cl['m3_axis']); sgn = cl['m3_sign']; off = cl['m3_offset_mm']
OFF = np.arange(-12, 13)
fire = np.zeros((512, len(OFF))); cross = np.zeros((512, len(OFF)))
for s in ['long_run', 'longer_run']:
    T = CC.rays_at(cfg, s)
    for f in sorted(glob.glob(os.path.join(cfg.BASE_PATH, cfg.RUN, s, 'hits_root', '*_06_hits.root'))):
        a = CC.read_hits(f)
        e, c, q = a['eventId'], a['channel'].astype(int), a['amplitude']
        b = np.flatnonzero(np.diff(e)) + 1
        for s0, s1 in zip(np.r_[0, b], np.r_[b, len(e)]):
            t = T.get(int(e[s0]))
            if t is None: continue
            cc = c[s0:s1]; qq = q[s0:s1]
            if len(cc) > 40: continue
            loc = sgn * t[k] + off
            d = (loc - pm) / CC.PITCH                       # track position in strips relative to each channel
            j = np.rint(d).astype(int) - OFF[0]
            ok = (j >= 0) & (j < len(OFF)) & (np.abs(d - np.rint(d)) < 0.4)
            fired = np.zeros(512, bool); fired[cc[qq > CC.FIRE_ADC]] = True
            ch = np.flatnonzero(ok)
            cross[ch, j[ch]] += 1; fire[ch, j[ch]] += fired[ch]
np.savez_compressed('fire_offset.npz', fire=fire, cross=cross, OFF=OFF)
print('done', cross.sum())
