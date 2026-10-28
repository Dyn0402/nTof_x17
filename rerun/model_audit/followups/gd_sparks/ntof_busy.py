#!/usr/bin/env python3
"""ntof_busy.py <run_dir> <file_tag> [--arms A C] [--n-wf 300] -- GD item 3: what the n_TOF busy-plane veto removes.

n_TOF has no event-level spark veto: ntof_tracking.wft_beam.seeds_from_hits_beam drops a PLANE whose floored hit count
exceeds BUSY_PLANE_HITS (150) and keeps the other.  Per arm, on one file tag of a local run (hits + decoded_root):
the floored count per plane (the beam seeder's own floor and largest-pulse rule), and for every busy plane plus a
sample of 100-150 planes and of ordinary ones the waveform features of gd_sparks/lx_sparks.wf_features (strips over SNR
6, on the raw rail, block common-mode step, fired span, clusters by the 12 mm gap).  Writes
~/x17/cosmic_bench/model_audit/gd_sparks/ntof_<run>_<tag>.parquet and prints the split.
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, HERE]
OUT = os.path.expanduser('~/x17/cosmic_bench/model_audit/gd_sparks')


def clusters(pos, on, gap=12.0):
    p = np.sort(pos[on & np.isfinite(pos)])
    if not len(p):
        return 0, 0
    g = np.split(p, np.flatnonzero(np.diff(p) > gap) + 1)
    return len(g), max(len(x) for x in g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir'); ap.add_argument('tag'); ap.add_argument('--arms', nargs='+', default=['A', 'B', 'C', 'D'])
    ap.add_argument('--n-wf', type=int, default=300)
    a = ap.parse_args()
    from ntof_tracking import wft_beam as wb
    from wft import seed as wseed, io as wio
    from lx_sparks import raw_events, wf_features
    sub = glob.glob(os.path.join(a.run_dir, 'stat*'))[0]
    hp = glob.glob(os.path.join(sub, 'combined_hits_root', f'*_{a.tag}_feu-combined_hits.root'))[0]
    run = os.path.basename(a.run_dir.rstrip('/'))
    rng = np.random.default_rng(5)
    rows = []
    for arm in a.arms:
        d = wb.BEAM_DETS[arm]
        fx, fy = d['feu_x'], d['feu_y']
        try:
            H = wb.read_hits_tag(hp, (fx, fy))
        except Exception as e:
            print(arm, 'no hits', e); continue
        Hf = wseed.apply_significance_floor(H, wseed.SIG_REL_FLOOR)
        n = Hf.groupby(['eventId', 'feu']).size().unstack(fill_value=0).reindex(columns=[fx, fy], fill_value=0)
        n.columns = ['nx', 'ny']
        busy = n[(n.nx > wb.BUSY_PLANE_HITS) | (n.ny > wb.BUSY_PLANE_HITS)].index
        mid = n[((n.nx > 100) | (n.ny > 100)) & ~n.index.isin(busy)].index
        low = n[(n.nx <= 100) & (n.ny <= 100) & ((n.nx > 0) & (n.ny > 0))].index
        pick = set(busy[:a.n_wf]) | set(rng.choice(mid, min(len(mid), a.n_wf // 2), replace=False)) | \
            set(rng.choice(low, min(len(low), a.n_wf // 2), replace=False))
        print(f'{run} {a.tag} arm {arm}: {len(n):,} events with hits; plane > {wb.BUSY_PLANE_HITS}: x {int((n.nx > 150).sum())} '
              f'y {int((n.ny > 150).sum())} both {int(((n.nx > 150) & (n.ny > 150)).sum())}; 100-150 {len(mid)}', flush=True)
        files = {}
        for v, feu in (('x', fx), ('y', fy)):
            f = glob.glob(os.path.join(sub, 'decoded_root', f'*_{a.tag}_{feu:02d}.root'))
            files[v] = f[0] if f else None
        if not all(files.values()):
            continue
        pm = wio.strip_position_map(wb.beam_config(arm, run=run))
        pos = {'x': pm[fx], 'y': pm[fy]}
        feats = {}
        for v in 'xy':
            r = wio.FeuReader(files[v])
            for eid, raw in raw_events(r, pick):
                f, W, band = wf_features(raw, r.ped, r.noise, pos[v])
                on = (W.max(1) / np.maximum(r.noise, 3.0)) > 6
                f['ncl'], f['cl_max'] = clusters(pos[v], on)
                feats.setdefault(eid, {})[v] = f
        for eid, pl in feats.items():
            if len(pl) < 2:
                continue
            row = dict(run=run, tag=a.tag, arm=arm, event_id=eid, nx=int(n.nx.get(eid, 0)), ny=int(n.ny.get(eid, 0)))
            for v in 'xy':
                row.update({f'{v}_{k}': x for k, x in pl[v].items()})
            rows.append(row)
    T = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    T.to_parquet(os.path.join(OUT, f'ntof_{run}_{a.tag}.parquet'))
    print(f'wrote {len(T)} rows')


if __name__ == '__main__':
    main()
