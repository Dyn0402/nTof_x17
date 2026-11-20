#!/usr/bin/env python3
"""pair_response.py -- how the reconstruction answers one and two leptons in a chamber, from digitised Geant4.

Input: a digitised library (run_digi.py g4 ... -> <label>.parquet, one row per (event, arm) with the reconstruction's
row, ``n_tracks`` and ``tracks_json``) and the Geant4 steps it was made from.  Truth per event: every Geant4 track
with >= MIN_GAP_MM of drift-gap extent in w (a "lepton" in the same sense as same_chamber_compare.DIGI), its
edep-weighted line taken to the mesh (u, v).  Reco: a track is a track_id present in both views with both planes
gated (the stage-3 `gated_reco` analogue; the |tan| acceptance is applied separately, true |tan| < TAN_MAX on the
truth and |tan| < TAN_MAX on the reco, as the data's `gated`).

Reported, per arm:
  * p1  = P(>= 1 track | 1 lepton), p2 = P(>= 2 tracks | 2 leptons), and the fake-pair rate
    f12 = P(>= 2 tracks | 1 lepton) -- a single lepton reconstructed as two tracks (a split, or a second track the
    overlay trigger itself carries); f12 < 40 mm (``p_ge2_given_1_lt40``) keeps only the close ones, the splits;
  * the separation of those fake pairs at the strip plane (x view |dp0|, y view |dp0|, and the larger of the two),
    against the true separation of the two-lepton events and whether both were found, in SEP_EDGES bins;
  * the same for the overlay alone: the muon smoke has the overlay's own second-track rate.

    python ntof_cosmics/g4_digi/pair_response.py --digi $WORK_DATA/ntof_pass1/g4_digi/g4_A_nose_pass1.parquet --arm A \
        --steps /media/dylan/data/x17/ntof_cosmics/g4_digi/steps_nose/*.parquet --out DIR
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_cosmics.g4_digi import digitise as DG  # noqa: E402

MIN_GAP_MM = 20.0
TAN_MAX = 0.5
CLOSE_MM = 40.0
SEP_EDGES = np.array([0, 6, 12, 24, 40, 80, 400], float)
G4_ARM = {'A': 2, 'C': 3}


def lepton_lines(S: pd.DataFrame) -> pd.DataFrame:
    """Per (file, eventID, trackID): gap extent, u/v at the mesh and tans (edep-weighted lines)."""
    rows = []
    for (f, e, t), g in S.groupby(['file', 'eventID', 'trackID'], sort=False):
        w, q = g.w.to_numpy(float), np.clip(g.edep.to_numpy(float), 0, None)
        ext = float(w.max() - w.min())
        if ext < MIN_GAP_MM or q.sum() <= 0:
            continue
        W = q.sum()
        mw = (q * w).sum() / W
        sww = (q * (w - mw) ** 2).sum()
        out = dict(file=f, eventID=int(e), trackID=int(t), extent=ext, edep=float(W))
        for c in ('u', 'v'):
            x = g[c].to_numpy(float)
            mx = (q * x).sum() / W
            b = (q * (w - mw) * (x - mx)).sum() / sww if sww > 0 else np.nan
            out[f'tan_{c}'] = float(b)
            out[f'{c}_mesh'] = float(mx + b * (DG.W_MESH - mw))
        rows.append(out)
    return pd.DataFrame(rows)


def reco_tracks(js: str):
    """[(p0_x, tan_x, p0_y, tan_y)] for track ids gated in both views."""
    tr = json.loads(js) if isinstance(js, str) else []
    by = {}
    for plane, tid, p0, tan, gated, _split in tr:
        by.setdefault(tid, {})[plane] = (p0, tan, gated)
    out = []
    for tid, d in sorted(by.items()):
        if 'x' in d and 'y' in d and d['x'][2] and d['y'][2]:
            if abs(d['x'][1]) < TAN_MAX and abs(d['y'][1]) < TAN_MAX:
                out.append((d['x'][0], d['x'][1], d['y'][0], d['y'][1]))
    return out


def analyse(R: pd.DataFrame, L: pd.DataFrame, arm: str) -> tuple:
    L = L[(L.tan_u.abs() < TAN_MAX) & (L.tan_v.abs() < TAN_MAX)]
    nl = L.groupby(['file', 'eventID']).size().rename('n_lep')
    R = R.merge(nl, left_on=['file', 'eventID'], right_index=True, how='left')
    R['n_lep'] = R.n_lep.fillna(0).astype(int)
    trk = R.tracks_json.map(reco_tracks)
    R['n_reco'] = trk.map(len)
    summ = dict(arm=arm, n_events=int(len(R)), n_lep_dist={int(k): int(v) for k, v in R.n_lep.value_counts().items()})
    for k in (1, 2):
        g = R[R.n_lep == k]
        summ[f'n_lep{k}'] = int(len(g))
        summ[f'p_ge1_given_{k}'] = float((g.n_reco >= 1).mean()) if len(g) else np.nan
        summ[f'p_ge2_given_{k}'] = float((g.n_reco >= 2).mean()) if len(g) else np.nan
    g0 = R[R.n_lep == 0]
    summ['p_ge2_given_0'] = float((g0.n_reco >= 2).mean()) if len(g0) else np.nan
    summ['p_ge1_given_0'] = float((g0.n_reco >= 1).mean()) if len(g0) else np.nan
    # fake pairs: one lepton, two reco tracks -> their separation at the plane
    rows = []
    for i in np.where((R.n_lep == 1).to_numpy() & (R.n_reco.to_numpy() >= 2))[0]:
        t = trk.iloc[i][:2]
        dx, dy = abs(t[0][0] - t[1][0]), abs(t[0][2] - t[1][2])
        rows.append(dict(kind='fake (1 lepton)', dx=dx, dy=dy, dmax=max(dx, dy), found_both=True))
    # true pairs: two leptons, their separation and whether two tracks came back
    Lk = L.set_index(['file', 'eventID'])
    for i in np.where((R.n_lep == 2).to_numpy())[0]:
        key = (R.file.iloc[i], int(R.eventID.iloc[i]))
        ll = Lk.loc[[key]]
        du, dv = abs(ll.u_mesh.iloc[0] - ll.u_mesh.iloc[1]), abs(ll.v_mesh.iloc[0] - ll.v_mesh.iloc[1])
        rows.append(dict(kind='true (2 leptons)', dx=du, dy=dv, dmax=max(du, dv), found_both=bool(R.n_reco.iloc[i] >= 2)))
    P = pd.DataFrame(rows)
    # the fake pairs that matter for a correlated-pair fraction: a single lepton split into two close tracks.  Far-apart
    # second tracks are the overlay trigger's own content -- accidentals, which the data's rate fit already removes.
    n1 = max(summ['n_lep1'], 1)
    for cut in (24.0, CLOSE_MM):
        k = int(((P.kind == 'fake (1 lepton)') & (P.dmax < cut)).sum()) if len(P) else 0
        summ[f'p_ge2_given_1_lt{int(cut)}'] = k / n1
        summ[f'n_fake_lt{int(cut)}'] = k
    sep = []
    if len(P):
        P['bin'] = pd.cut(P.dmax, SEP_EDGES, right=False)
        for (kind, b), g in P.groupby(['kind', 'bin'], observed=False):
            sep.append(dict(arm=arm, kind=kind, lo=b.left, hi=b.right, n=int(len(g)),
                            found_both=float(g.found_both.mean()) if len(g) else np.nan))
    return summ, pd.DataFrame(sep), P.assign(arm=arm) if len(P) else P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--digi', required=True)
    ap.add_argument('--arm', required=True, choices=('A', 'C'))
    ap.add_argument('--steps', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R = pd.read_parquet(a.digi)
    R = R[R.get('kind', 'g4') == 'g4'] if 'kind' in R else R
    S = pd.concat([pd.read_parquet(f) for f in a.steps], ignore_index=True)
    S = S[S.armID == G4_ARM[a.arm]]
    keys = set(zip(R.file, R.eventID))
    S = S[[k in keys for k in zip(S.file, S.eventID)]]
    L = lepton_lines(S)
    summ, sep, P = analyse(R, L, a.arm)
    os.makedirs(a.out, exist_ok=True)
    stem = Path(a.digi).stem
    json.dump(summ, open(os.path.join(a.out, f'{stem}_response.json'), 'w'), indent=1, default=float)
    sep.to_csv(os.path.join(a.out, f'{stem}_pairsep.csv'), index=False)
    if len(P):
        P.drop(columns='bin').to_parquet(os.path.join(a.out, f'{stem}_pairs.parquet'), index=False)
    print(json.dumps(summ, indent=1, default=float))
    print(sep.to_string(index=False))


if __name__ == '__main__':
    main()
