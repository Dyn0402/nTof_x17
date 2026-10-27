"""det4: bench census (g_det4) against the SPS uRWELL census, in strip position, over the SPS-judged strips."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment'))
from det4_sps_map import POSITION_MM, VIEW   # noqa: E402

bench = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', 'det4_g_det4.json')))
bundle = json.load(open(os.path.expanduser('~/x17/cosmic_bench/model_audit/det4/calib_bundle/bundle.json')))
bpos = lambda c: 49.92 * (c // 64) + (63 - c % 64) * 0.78
out = {}
for tag in sys.argv[1:]:
    S = json.load(open(os.path.join(HERE, 'census', f'sps_{tag}.json')))
    R = {}
    for p in 'xy':
        judged_pos = {round(float(POSITION_MM[c]), 2) for c in S[p]['judged']}
        sps_dead = {round(float(POSITION_MM[c]), 2) for c in S[p]['dead_nb']}
        sps_dead_floor = {round(float(POSITION_MM[c]), 2) for c in S[p]['dead']}
        b_dead = {round(bpos(c), 2) for c in bench[p]['dead']}
        b_short = {round(bpos(c), 2) for c in bench[p]['shorts']}
        b_mask = {round(bpos(c), 2) for c in bundle['dead'][p]}
        inwin = lambda s: {x for x in s if x in judged_pos}
        R[p] = dict(n_judged=len(judged_pos), sps_dead=sorted(sps_dead), sps_dead_incl_stripes=len(sps_dead_floor),
                    bench_dead_in_window=sorted(inwin(b_dead)), bench_short_in_window=sorted(inwin(b_short)),
                    bench_mask_in_window=len(inwin(b_mask)),
                    both=sorted(sps_dead & inwin(b_dead)), sps_only=sorted(sps_dead - b_dead - b_short),
                    bench_only=sorted(inwin(b_dead) - sps_dead_floor),
                    bench_short_dead_at_sps=sorted(inwin(b_short) & sps_dead_floor))
        print(tag, p, {k: (v if not isinstance(v, list) or len(v) < 25 else f'{len(v)} items') for k, v in R[p].items()})
    out[tag] = R
json.dump(out, open(os.path.join(HERE, 'sps_vs_bench.json'), 'w'), indent=1)
