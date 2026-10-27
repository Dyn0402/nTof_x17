"""flank_census on the H4 det4 paired caches (FEU 03), each view separately -> census/flank_sps_<tag>.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
from det4_sps_map import POSITION_MM, VIEW   # noqa: E402
from flank_census import flank               # noqa: E402

for tag, path in (('run56_m10V', '/media/dylan/data/x17/sps_run53_det4_check/paired_npz/pair_m10V.npz'),
                  ('run63_flat', '/media/dylan/data/x17/sps_run53_det4_check/staging/run_63/pair_run63_flat.npz'),
                  ('run71_raw', '/media/dylan/data/x17/sps_run53_det4_check/staging/run_71/pair_run71_raw.npz')):
    P = np.load(path)
    n = min(len(P['h_ev']), 12_000_000)
    ev, ch, amp, sig = P['h_ev'][:n], P['h_ch'][:n].astype(int), P['h_amp'][:n], P['h_sig'][:n]
    out = dict(tag=tag, pair=path)
    for v in 'xy':
        pos = np.where(VIEW == v, POSITION_MM, np.nan)
        m = VIEW[ch] == v
        r = flank(ev[m], ch[m], amp[m], pos, sig[m])
        r['dead_pos_mm'] = sorted(round(float(POSITION_MM[c]), 2) for c in r['dead'])
        out[v] = r
        print(tag, v, 'judged', len(r['judged']), 'dead', r['dead_pos_mm'], 'silent', len(r['silent']), flush=True)
    json.dump(out, open(os.path.join(HERE, 'census', f'flank_sps_{tag}.json'), 'w'))
