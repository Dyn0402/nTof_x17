#!/usr/bin/env python3
"""events.py <subrun> --out EVENTS.txt  -- run_149 events worth a D fit (studies only; the D fit is the slow step).

Union of
  (a) line candidates: D and a partner (A or B) both have a cluster in both views (centroids.py), and the line through
      the two CENTROIDS (no fit) is within |tan| < 0.8 of D's normal in both views.  The ratio window is
      |true tan| 0.15-0.45, so this pre-selection, made on positions only, does not cut into it.  C is left out: a line
      shallow in D meets D on its dead x half (local x > +40 mm, 137 dead channels);
  (b) wall candidates: D has a cluster on a trigger that clock_match put on the n_TOF clock, with exactly one of D's
      four wall groups lit (wall_kw.py uses only those).
cos_0000 tag _000 was fitted on the first, wider list (every D & (A|B|C) coincidence and every matched trigger).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, HERE]
WALL_HITS = '/media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale/arm_D/wall_hits_D.parquet'
TAN_PRE = 0.8


def candidates(subrun):
    """(line candidates, wall candidates) as event-id sets."""
    import line_kw as LK
    from ntof_cosmics import cosmic_wall_scale as CWS
    from ntof_scint_stack import ana as SA
    trs = LK.transforms()
    Dc = LK.centroids(subrun, 'D', strict=False)
    _, _, PD = LK.centroid_global(Dc, 'D', trs['mx17_D'])
    Dc = Dc.assign(px=PD[:, 0], py=PD[:, 1], pz=PD[:, 2])
    line = set()
    for parm in 'AB':
        C = LK.centroids(subrun, parm, strict=False)
        _, _, PP = LK.centroid_global(C, parm, trs[f'mx17_{parm}'])
        C = C.assign(qx=PP[:, 0], qy=PP[:, 1], qz=PP[:, 2])
        M = Dc[['event_id', 'px', 'py', 'pz']].merge(C[['event_id', 'qx', 'qy', 'qz']], on='event_id')
        Ll = (M[['px', 'py', 'pz']].to_numpy() - M[['qx', 'qy', 'qz']].to_numpy()) @ trs['mx17_D'].R
        ok = (np.abs(Ll[:, 0] / Ll[:, 2]) < TAN_PRE) & (np.abs(Ll[:, 1] / Ll[:, 2]) < TAN_PRE)
        line |= set(M.event_id[ok].tolist())
    CWS.ARM = 'D'
    H = pd.read_parquet(WALL_HITS)
    h = H[H.subrun == subrun]
    wall = set()
    if len(h):
        Wd = CWS.wide(h, CWS.peaks(H))
        one = SA.wall_groups_fired(Wd).sum(1) == 1
        z = np.load(f'{LK.CEN}/r149_{subrun}.npz')
        wall = set(Wd.event_id[one].tolist()) & set(z['D_eid'][z['D_x_has']].tolist())
    return line, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subrun'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    line, wall = candidates(a.subrun)
    ev = sorted(line | wall)
    with open(a.out, 'w') as f:
        f.write('\n'.join(map(str, ev)) + '\n')
    print(f'{a.subrun}: {len(line)} line candidates, {len(wall)} wall candidates, {len(ev)} events -> {a.out}')


if __name__ == '__main__':
    main()
