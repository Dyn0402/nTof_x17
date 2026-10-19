#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reduce_gap_wall.py -- MX17_Full_Geant HitTree -> one row per (event, arm) with
a drift-gap track: the truth-level "reconstructed" angle and where the track
actually reaches the SiPM wall.  Runs on lxplus (LCG_106 python, uproot) as a
condor job, one ROOT file per call.  HANDOFF_TRACKING_2026-10-06.md §10f.

THE QUESTION.  Data: beam tracks read ~0.92 x raw at A's wall, cosmic muons
~1.10-1.15.  Is that physics -- the ionisation in the gap of a few-MeV
electron does not point where the electron goes next (scattering in the gap,
mesh, PCB, air) -- or the reconstruction?  Here the reconstruction is replaced
by its ideal: an edep-weighted least-squares line u(w), v(w) through every
DriftGas step of the arm (all particles, as the strips see all charge).  If
this ideal already reads tan_wall / tan_gap ~ 0.85 for the beam population and
~1 for muons, the gap is physics.

Per (event, arm), prompt steps only (time < 1e8 ns: RadioactiveDecay is on and
28Al decays sit at 1e9-1e13 ns):
  gap   : n, edep, w span, edep-weighted fit tan_u, tan_v, u/v at the mesh
          (w = W_MESH), chi2-like rms; dominant track (largest gap edep), its
          particle, its share of the gap edep, its ke at its first gap step
  wall  : the dominant track's FIRST PlasticScint step (u, v, w, ke) if any;
          else the edep-weighted wall centroid; total wall edep
  truth : EventTree generator info is joined downstream by eventID

    python3 reduce_gap_wall.py <in.root> <out.parquet> [--arms 2]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import uproot

W_MESH = 30.1           # mm, local w of the micromesh (DriftGas spans 0.1-30.1)
T_MAX = 1e8             # ns
BR = ['eventID', 'trackID', 'parentID', 'armID', 'detType', 'particle', 'u', 'v', 'w',
      'edep', 'ke', 'time']


def _fit(w, x, q):
    W = q.sum()
    mw, mx = (q * w).sum() / W, (q * x).sum() / W
    sww = (q * (w - mw) ** 2).sum()
    b = (q * (w - mw) * (x - mx)).sum() / sww if sww > 0 else np.nan
    a = mx - b * mw
    rms = np.sqrt((q * (x - a - b * w) ** 2).sum() / W)
    return b, a + b * W_MESH, rms


def reduce_chunk(d: pd.DataFrame, arms) -> list[dict]:
    d = d[(d.time < T_MAX) & d.armID.isin(arms)]
    g = d[d.detType == 'DriftGas']
    g = g[g.edep > 0]
    wl = d[d.detType == 'PlasticScint']
    rows = []
    wl_groups = {k: v for k, v in wl.groupby(['eventID', 'armID'])}
    for (ev, arm), s in g.groupby(['eventID', 'armID']):
        if len(s) < 5:
            continue
        w, u, v, q = (s[c].to_numpy(float) for c in ('w', 'u', 'v', 'edep'))
        tu, u_mesh, rms_u = _fit(w, u, q)
        tv, v_mesh, rms_v = _fit(w, v, q)
        by = s.groupby('trackID').edep.sum()
        dom = int(by.idxmax())
        sd = s[s.trackID == dom].sort_values('time')
        r = dict(eventID=int(ev), arm=int(arm), n_gap=len(s), edep_gap=float(q.sum()),
                 w_lo=float(w.min()), w_hi=float(w.max()), tan_gap_u=tu, tan_gap_v=tv,
                 u_mesh=u_mesh, v_mesh=v_mesh, rms_u=rms_u, rms_v=rms_v,
                 dom_track=dom, dom_parent=int(sd.parentID.iat[0]), dom_particle=str(sd.particle.iat[0]),
                 dom_share=float(by.max() / q.sum()), dom_ke_gap=float(sd.ke.iat[0]),
                 n_tracks_gap=int(len(by)))
        # the dominant track alone (no delta rays): its own line
        if len(sd) >= 5:
            r['tan_dom_u'], r['u_mesh_dom'], _ = _fit(sd.w.to_numpy(float), sd.u.to_numpy(float),
                                                       sd.edep.to_numpy(float))
        x = wl_groups.get((ev, arm))
        if x is not None and len(x):
            r['edep_wall'] = float(x.edep.sum())
            xd = x[x.trackID == dom].sort_values('time')
            if len(xd):
                f = xd.iloc[0]
                r.update(u_wall=float(f.u), v_wall=float(f.v), w_wall=float(f.w), ke_wall=float(f.ke),
                         wall_same_track=True)
            else:
                qq = x.edep.to_numpy(float)
                r.update(u_wall=float((x.u * qq).sum() / qq.sum()), v_wall=float((x.v * qq).sum() / qq.sum()),
                         w_wall=float((x.w * qq).sum() / qq.sum()), ke_wall=np.nan, wall_same_track=False)
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--arms', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--step', type=int, default=3_000_000)
    a = ap.parse_args()
    f = uproot.open(a.inp)
    t = f['HitTree']
    rows, carry = [], None
    for arr in t.iterate(BR, step_size=a.step, library='np'):
        chunk = pd.DataFrame({k: (v.astype(str) if v.dtype == object else v) for k, v in arr.items()})
        # an event can straddle chunks: hold back the last event id
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk.eventID.iat[-1]
        carry = chunk[chunk.eventID == last]
        rows += reduce_chunk(chunk[chunk.eventID != last], a.arms)
    if carry is not None:
        rows += reduce_chunk(carry, a.arms)
    R = pd.DataFrame(rows)
    ev = pd.DataFrame({k: (v.astype(str) if v.dtype == object else v)
                       for k, v in f['EventTree'].arrays(library='np').items() if v.ndim == 1})
    keep = [c for c in ev.columns if c in ('eventID', 'event_type', 'neutron_E_eV', 'capture_vol', 'inv_mass',
                                            'vtx_x', 'vtx_y', 'vtx_z', 'em_ke', 'em_px', 'em_py', 'em_pz')]
    if len(R):
        R = R.merge(ev[keep], on='eventID', how='left')
    R.to_parquet(a.out, index=False)
    print(f'{a.inp}: {len(R)} (event, arm) gap tracks, {int(R.get("u_wall", pd.Series()).notna().sum())} reach the wall')
    return 0


if __name__ == '__main__':
    sys.exit(main())
