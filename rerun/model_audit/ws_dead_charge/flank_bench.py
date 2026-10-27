#!/usr/bin/env python3
"""flank_bench.py [keys...] -- flank_census on every June row of the five chambers -> census/flank_<det>_<key>.json"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]

ROWS = {'det2': ['ovn_det2', 'o22_det2', 'g_det2', 'o22_long_det2'],
        'det3': ['g_det3', 'o22_det3', 'o23_long_det3', 'g_det3_wknd', 'sat_det3'],
        'det4': ['o23_long_det4', 'g_det4'],
        'det6': ['g_det6', 'g_det6_longer', 'g_det6_long'],
        'det7': ['g_det7', 'g_det7_longer', 'g_det7_long']}
MAX_FILES = 8


def one(det, key):
    import uproot
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.io import strip_position_map
    from flank_census import flank
    cfg = get_config(key); pm = strip_position_map(cfg)
    sp = os.path.join(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN)
    out = dict(det=det, key=key, run=cfg.RUN, sub_run=cfg.SUB_RUN, feus=list(cfg.MX17_FEUS))
    for p, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        files = sorted(glob.glob(os.path.join(sp, 'hits_root', f'*_{feu:02d}_hits.root')))[:MAX_FILES]
        E, C, A, S = [], [], [], []
        for i, f in enumerate(files):
            a = uproot.open(f)['hits'].arrays(['eventId', 'channel', 'amplitude', 'significance'], library='np')
            E.append(a['eventId'].astype(np.int64) + i * 10_000_000); C.append(a['channel']); A.append(a['amplitude']); S.append(a['significance'])
        if not files:
            out[p] = None; continue
        pos = np.asarray(pm[feu], float)
        r = flank(np.concatenate(E), np.concatenate(C), np.concatenate(A), pos, np.concatenate(S))
        r['feu'] = feu; r['n_files'] = len(files)
        r['dead_pos_mm'] = sorted(round(float(pos[c]), 2) for c in r['dead'])
        out[p] = r
    json.dump(out, open(os.path.join(HERE, 'census', f'flank_{det}_{key}.json'), 'w'))
    s = ' '.join(f"{p}: {len(out[p]['judged'])} judged, dead {out[p]['dead']} weak {len(out[p]['weak'])}" for p in 'xy' if out[p])
    return f'{det} {key}: {s}'


if __name__ == '__main__':
    want = set(sys.argv[1:])
    jobs = [(d, k) for d, ks in ROWS.items() for k in ks if not want or k in want]
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(one, *zip(*jobs)):
            print(r, flush=True)
