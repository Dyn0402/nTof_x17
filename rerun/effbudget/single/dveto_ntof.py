#!/usr/bin/env python3
"""dveto_ntof.py [--base /media/ucla/work/ntof_pass1/runs] [--run run_145] [--subrun stat090_0000] [--out json]
-- what the spark veto question looks like on n_TOF beam data (DISCHARGE_VETO_2026-11-02.md §5).

The n_TOF seeder (ntof_tracking.wft_beam.seeds_from_hits_beam) has no event-level veto: a PLANE whose floored hit count
exceeds BUSY_PLANE_HITS (150) is dropped and the other kept.  Per arm, on every file tag of the subrun (rr1_cm0 hits,
the seeder's own largest-pulse rule and 10 % floor): how many triggers the bench rule (> 50 floored hits x + y) would
drop, how many planes the beam rule drops, and, for the planes over 50, how many are one contiguous region (flash /
discharge: the largest 12 mm-gap cluster holds >= 80 % of the plane's strips) against several separate clusters
(particles).  Hits are candidate finding / QA only.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/media/ucla/work/ntof_pass1/runs')
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from ntof_tracking import wft_beam as wb
    from wft import seed as ws
    sub = os.path.join(a.base, a.run, a.subrun)
    hd = os.path.join(sub, 'combined_hits_root_rr1_cm0')
    if not os.path.isdir(hd):
        hd = os.path.join(sub, 'combined_hits_root')
    files = sorted(glob.glob(os.path.join(hd, '*.root')))
    out = dict(run=a.run, subrun=a.subrun, files=len(files), arms={})
    # arm -> (feu_x, feu_y) from the n_TOF registry columns (rerun/ntof/ntof_registry.csv)
    R = pd.read_csv(os.path.join(REPO, 'rerun', 'ntof', 'ntof_registry.csv'), dtype=str)
    row = R[(R.run.str.strip() == a.run.replace('run_', '')) & (R.subrun.str.strip() == a.subrun)].iloc[0]
    for arm in 'ABCD':
        fx, fy = int(float(row[f'feu_x_{arm}'])), int(float(row[f'feu_y_{arm}']))
        tot = dict(triggers=0, bench_veto=0, busy_x=0, busy_y=0, busy_any=0, over50_plane=0, over50_contig=0,
                   over50_multi=0, both_busy=0)
        for f in files:
            H = wb.read_hits_tag(f, (fx, fy))
            if not len(H):
                continue
            Hf = ws.apply_significance_floor(H, ws.SIG_REL_FLOOR)
            n = Hf.groupby(['eventId', 'feu']).size().unstack(fill_value=0).reindex(columns=[fx, fy], fill_value=0)
            ev = H.eventId.unique()
            n = n.reindex(ev, fill_value=0)
            tot['triggers'] += int(len(ev))
            tot['bench_veto'] += int((n.sum(axis=1) > ws.SPARK_VETO_HITS).sum())
            bx, by = n[fx] > wb.BUSY_PLANE_HITS, n[fy] > wb.BUSY_PLANE_HITS
            tot['busy_x'] += int(bx.sum()); tot['busy_y'] += int(by.sum())
            tot['busy_any'] += int((bx | by).sum()); tot['both_busy'] += int((bx & by).sum())
            # shape of planes over 50 strips: one region or several (channel order ~ position order on a plane)
            for feu in (fx, fy):
                big = n.index[n[feu] > ws.SPARK_VETO_HITS]
                if not len(big):
                    continue
                G = Hf[(Hf.feu == feu) & Hf.eventId.isin(big)]
                for _, g in G.groupby('eventId'):
                    ch = np.sort(g.channel.to_numpy())
                    gaps = np.flatnonzero(np.diff(ch) > int(12.0 / ws.PITCH_MM)) + 1
                    big_cl = max(len(x) for x in np.split(ch, gaps))
                    tot['over50_plane'] += 1
                    if big_cl >= 0.8 * len(ch):
                        tot['over50_contig'] += 1
                    else:
                        tot['over50_multi'] += 1
        for k in ('bench_veto', 'busy_any', 'both_busy'):
            tot[f'{k}_pct'] = round(100 * tot[k] / max(tot['triggers'], 1), 2)
        out['arms'][arm] = tot
        print(arm, tot, flush=True)
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
