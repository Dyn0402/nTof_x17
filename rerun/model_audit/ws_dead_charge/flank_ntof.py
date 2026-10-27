#!/usr/bin/env python3
"""flank_ntof.py RUN [RUN ...] -- flank_census on n_TOF combined hits (local runs), chambers A-D, both views.
Positions from the run's own run_config (every n_TOF plug 'inverted', 1:1 connectors).  Events with > MAX_HITS hits
on the FEU (gamma flash, showers) are dropped: a flash saturates every strip and the flank test needs track-like
clusters.  Output census/flank_ntof_<run>.json."""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, REPO]
from flank_census import flank                               # noqa: E402
from common.Mx17StripMap import RunConfig                     # noqa: E402

RUNS = '/media/dylan/data/x17/beam_july/runs'
MAX_HITS = 60
MAX_FILES = 20
BENCH = {'A': 'det3', 'B': 'det2', 'C': 'det6', 'D': 'det7'}


def main():
    for run in sys.argv[1:]:
        rcp = os.path.join(RUNS, run, 'run_config.json')
        rc = RunConfig(rcp, os.path.join(REPO, 'mx17_m1_map.csv'))
        files = sorted(glob.glob(os.path.join(RUNS, run, '*', 'combined_hits_root', '*_feu-combined_hits.root')))[:MAX_FILES]
        import uproot
        cols = ['eventId', 'channel', 'amplitude', 'significance', 'feu']
        D = {k: [] for k in cols}
        for i, f in enumerate(files):
            t = uproot.open(f)['hits']
            a = t.arrays([c for c in cols if c in t.keys()], library='np')
            if 'significance' not in a:                   # early July hits (run_55) carry no significance
                a['significance'] = np.full(len(a['eventId']), np.inf, np.float32)
            a['eventId'] = a['eventId'].astype(np.int64) + i * 100_000_000
            for k in cols:
                D[k].append(a[k])
        D = {k: np.concatenate(v) for k, v in D.items()}
        out = dict(run=run, files=[os.path.relpath(f, RUNS) for f in files], max_hits=MAX_HITS)
        for arm in 'ABCD':
            det = rc.get_detector(f'mx17_{arm}')
            feus = {}
            for key, (feu, conn) in det.dream_feus.items():
                feus.setdefault(key[0], feu)
            out[arm] = dict(bench=BENCH[arm])
            for p in 'xy':
                feu = feus[p]
                pos = np.full(512, np.nan)
                for c in range(512):
                    h = det.map_hit(feu, c)
                    if h is not None and h[0 if p == 'x' else 1] is not None:
                        pos[c] = h[0 if p == 'x' else 1]
                m = D['feu'] == feu
                ev, ch, amp, sig = D['eventId'][m], D['channel'][m].astype(int), D['amplitude'][m], D['significance'][m]
                ue, inv, cnt = np.unique(ev, return_inverse=True, return_counts=True)
                k = cnt[inv] <= MAX_HITS
                r = flank(ev[k], ch[k], amp[k], pos, sig[k])
                r['feu'] = int(feu)
                r['dead_pos_mm'] = sorted(round(float(pos[c]), 2) for c in r['dead'])
                r['pos'] = np.round(pos, 2).tolist()
                out[arm][p] = r
                print(f"{run} {arm} {p} (FEU {feu}): {r['n_events']} events, judged {len(r['judged'])}, dead {len(r['dead'])} "
                      f"at {r['dead_pos_mm'][:40]}; silent {len(r['silent'])}", flush=True)
        json.dump(out, open(os.path.join(HERE, 'census', f'flank_ntof_{run}.json'), 'w'))


if __name__ == '__main__':
    main()
