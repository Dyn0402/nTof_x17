#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pair_event_display.py -- 3D event displays (GIF + still) of selected pair
candidates: TPC charge, the reconstructed tracks, the scintillators that fired,
and the capsule at its inferred pose.

Built for the 2026-10-15 meeting on run_145 stat090_0000, the one sub-run whose
reco, waveforms and slim are all on the laptop.  Two steps:

  candidates   rebuild the stage-3 tracks of one sub-run from the is2_v2 reco
               (`build_tracks.build_arm`, at the capsule-focus k the pair
               analysis uses), confirm each leg on the SiPM wall
               (`det_a_scint.match_run`) and form the pairs
               (`det_a_intra.pairs`, the re-pass cuts: t0 <= 300 ns,
               |true tan| < 0.6, measured axis), plus the two-arm A-C pairs.
               Nothing here is a new selection: every flag is the analysis's
               own function, called unchanged.
  render       one GIF and one PNG per chosen trigger.

WHAT EACH ELEMENT IS, AND WHAT IT IS NOT
  * **TPC charge** is the raw DREAM waveform (`decoded_root`, pedestal + CNS as
    in `wft.io`), not `combined_hits`.  A strip plane measures (strip, time),
    so each view is drawn as a PROJECTION on the chamber face it belongs to:
    the x view on the top face (x, depth), the y view on the side face
    (y, depth).  No third coordinate is invented.  A sample at time t is put at
    depth (t - tau_peak - t0) * v / k: tau_peak is the bundle's own
    impulse-response peak (140 ns), t0 the fitted arrival time of the nearest
    track, v / k the in-situ drift speed stage 3 uses.  It is a display
    mapping, not a reconstruction: the line is the forward-model fit.
  * **Tracks** are the stage-3 lines (p0, d) -- thick inside the drift gap,
    thin out to the SiPM wall and back to the pair's closest approach.
  * **Scintillators**: every wall group, plastic bar and LS slab with an
    in-time hit (slim `prod` window, -100..+60 ns, not the control
    collection) is lit.  The wall is four groups of four bars; a group is lit
    as a group, because that is all the read-out resolves.
  * **Capsule** at the pose2 fit (HANDOFF_CAPSULE_POSE_FIT.md, Correction
    box): 15 mm toward -x, 10 mm low, top tilted 10 deg toward phi = 135 deg,
    beam centred at x = -10 mm.  The sideways position is degenerate with
    the beam's; the drawn one is the best-chi2 member of that family.

    python -m sept26_prelim_analysis.pair_event_display candidates
    python -m sept26_prelim_analysis.pair_event_display render --events 24576,38414
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                               # noqa: E402

RUN, SUBRUN = 'run_145', 'stat090_0000'
RECO = Path(os.environ.get(
    'PAIRDISP_RECO', '/media/dylan/data/x17/is2_v2_stage/fullpass'))
#: kcal_is2_v1 (run_145: A 0.976, C 0.962) x the two-sided rescale for is2_v2
#: (A 1.009, C 1.003) x the capsule-focus factor 1.12 the pair analysis adopted
#: (SAME_CHAMBER_PAIRS.md, "The k used"; HANDOFF_TRACKING §19, §22).
K_CF = {'A': 0.976 * 1.009 * 1.12, 'C': 0.962 * 1.003 * 1.12}
#: the measured source offset (HANDOFF_TRACKING §22: dx = -9.5 mm)
AXIS_OFFSET = (-9.5, 0.0, 0.0)
FEU = {'A': (3, 4), 'C': (7, 8)}
#: the bundle's impulse-response peak, ns after the charge arrives (both arms)
TAU_PEAK = 140.0
SAMPLE_NS = 60.0
#: capsule pose (pose2, 19.5 m, real EAR2 beam): see the module docstring
POSE = dict(x_c=-15.0, y_c=-10.0, z_c=0.0, theta=10.0, phi=135.0)
BEAM_X, BEAM_R = -10.0, 22.0
L_FLIGHT_M = 19.5
#: slim in-time window (det_a_scint.DT_WINDOW)
DT_WINDOW = (-100.0, 60.0)
WAL, PSS, LIQ = 0, 4, 8
ARMS = ('A', 'B', 'C', 'D')

OUT = paths.spell('out', 'pair_displays')
WF_DIR = OUT / 'wf'


def _slim_root() -> Path:
    return Path('/media/dylan/data/x17/slim/out_224670/'
                f'ntof_hits_{RUN}_{SUBRUN}_224670.root')


# =========================================================================== #
# step 1: candidates
# =========================================================================== #
def candidates(out: Path) -> pd.DataFrame:
    """Stage-3 tracks, wall-confirmed legs and the pair tables of one sub-run."""
    import uproot
    from sept26_prelim_analysis import build_tracks as BT
    from ntof_tracking.reco import geometry as G

    work = out / 'work'
    (work / 'stage3').mkdir(parents=True, exist_ok=True)
    (work / 'slim').mkdir(parents=True, exist_ok=True)
    cfg = json.loads((paths.root('runs') / RUN / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)
    T = []
    for arm in ('A', 'C'):
        df, prov = BT.build_arm(RECO / RUN / SUBRUN, arm, trs[f'mx17_{arm}'],
                                RUN, SUBRUN, k=K_CF[arm], tan_max_true=0.6)
        print(f'[{arm}] {len(df):,} segments, {prov["acceptance"]["n_gated"]:,} gated')
        T.append(df)
    T = pd.concat(T, ignore_index=True)

    f = uproot.open(_slim_root())
    h = pd.DataFrame(f['hits'].arrays(
        ['eventId', 'det', 'detn', 'tof', 'dt_ns', 'amp', 'is_control'], library='np'))
    h['subrun'] = SUBRUN
    h.to_parquet(work / 'slim' / f'ntof_hits_{RUN}_{SUBRUN}.parquet')
    ev = (pd.DataFrame(f['events'].arrays(['eventId', 't_dream_ns'], library='np'))
          .drop_duplicates('eventId').set_index('eventId'))
    T['t_since_flash_ns'] = T.event_id.map(ev.t_dream_ns)
    # det_a_intra reads these stage-1 flags; the per-leg wall match below is
    # what the pair selection actually uses, so they are written False here.
    for a in ARMS:
        for c in ('wall', 'plastic', 'coinc'):
            T[f'{c}_{a}'] = False
    T['n_coinc_arms'] = 0
    p = work / 'stage3' / f'tracks_{RUN}_{SUBRUN}.parquet'
    T.to_parquet(p)
    p.with_name(p.name.replace('.parquet', '.meta.json')).write_text(json.dumps(dict(
        y_convention='up', built=str(pd.Timestamp.now().date()),
        note='pair_event_display: build_tracks.build_arm on the is2_v2 reco, k = capsule focus')))
    T.to_parquet(out / 'tracks.parquet')

    # the pair analysis itself, run once per arm in a clean process so its
    # import-time configuration (environment) is the re-pass's
    import subprocess
    for arm in ('A', 'C'):
        env = dict(os.environ, DET_INTRA_WALL='1', DET_INTRA_T0_MAX='300',
                   DET_INTRA_TAN_MAX='0.6', DET_INTRA_AXIS='measured',
                   DET_INTRA_ARM=arm, X17_SEPT26_OUT=str(work),
                   PYTHONPATH=str(REPO))
        subprocess.run([sys.executable, __file__, '_pairs', arm, str(work)],
                       env=env, check=True)
    P = pd.concat([pd.read_parquet(work / f'pairs_{a}.parquet') for a in 'AC'],
                  ignore_index=True)
    P['kind'] = 'same-chamber'
    L = pd.concat([pd.read_parquet(work / f'legs_{a}.parquet') for a in 'AC'],
                  ignore_index=True)
    X = two_arm_pairs(L)
    P = pd.concat([P, X], ignore_index=True)
    tag = T.drop_duplicates('event_id').set_index('event_id').tag
    P['tag'] = P.event_id.map(tag)
    P['tof_ms'] = P.event_id.map(ev.t_dream_ns) / 1e6
    P.to_parquet(out / 'pairs.parquet')
    L.to_parquet(out / 'legs.parquet')
    show = P[P.wall2.fillna(False) | ((P.kind == 'two-arm') & P.wall)]
    cols = ['kind', 'arm', 'event_id', 'tag', 'open_deg', 'dca_pair_mm', 'v_r_mm',
            'vy_mm', 'dt0_ns', 'wall', 'wall2', 'tof_ms']
    print(show[cols].round(1).sort_values(['kind', 'dca_pair_mm']).to_string())
    return P


def _pairs_worker(arm: str, work: Path) -> None:
    """det_a_intra.tracks + pairs for one arm (run in its own process)."""
    from sept26_prelim_analysis import det_a_intra as DI
    DI._AXIS_OFFSET[RUN] = AXIS_OFFSET
    t = DI.tracks(RUN, [SUBRUN], work / 'stage3')
    P = DI.pairs(t, RUN, with_mixed=False)
    P['arm'] = arm
    P.to_parquet(work / f'pairs_{arm}.parquet')
    t['arm'] = arm
    t.to_parquet(work / f'legs_{arm}.parquet')
    print(f'[{arm}] {len(t):,} legs, {len(P):,} pairs, '
          f'wall {int(P.wall.sum())}, wall2 {int(P.wall2.sum())}')


def two_arm_pairs(L: pd.DataFrame) -> pd.DataFrame:
    """One leg in A and one in C in the same trigger: the X17 topology.

    The same leg selection as the same-chamber pairs (det_a_intra.tracks).
    ``wall`` = both legs confirmed on their own arm's wall; ``wall2`` is a
    same-chamber notion and is left NaN.  ``open_deg`` is the angle between the
    two OUTWARD directions, so a straight line through both chambers reads 180.
    """
    from sept26_prelim_analysis.source_imaging import _dca_two_lines
    a, c = L[L.arm == 'A'], L[L.arm == 'C']
    m = a.merge(c, on='event_id', suffixes=('_1', '_2'))
    if not len(m):
        return pd.DataFrame()
    p1 = m[['p0_x_1', 'p0_y_1', 'p0_z_1']].to_numpy(float)
    p2 = m[['p0_x_2', 'p0_y_2', 'p0_z_2']].to_numpy(float)
    d1 = _outward(p1, m[['d_x_1', 'd_y_1', 'd_z_1']].to_numpy(float))
    d2 = _outward(p2, m[['d_x_2', 'd_y_2', 'd_z_2']].to_numpy(float))
    V, dca = _dca_two_lines(p1, d1, p2, d2)
    X0, _, Z0 = AXIS_OFFSET
    return pd.DataFrame(dict(
        run=RUN, subrun=SUBRUN, event_id=m.event_id, kind='two-arm', arm='AC',
        open_deg=np.degrees(np.arccos(np.clip((d1 * d2).sum(1), -1, 1))),
        dca_pair_mm=dca, vx_mm=V[:, 0], vy_mm=V[:, 1], vz_mm=V[:, 2],
        v_r_mm=np.hypot(V[:, 0] - X0, V[:, 2] - Z0),
        u1=m.x_local_1, v1=m.y_local_1, u2=m.x_local_2, v2=m.y_local_2,
        dt0_ns=m.t0_1 - m.t0_2,
        wall=(m.match_wall_1 & m.match_wall_2).to_numpy(),
        dca_max_mm=np.maximum(m.dca_axis_mm_1, m.dca_axis_mm_2)))


def _outward(p0: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Unit directions oriented away from the beam axis (the y line)."""
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    s = np.sign(p0[:, 0] * d[:, 0] + p0[:, 2] * d[:, 2])
    return d * np.where(s == 0, 1.0, s)[:, None]


# =========================================================================== #
# waveforms
# =========================================================================== #
LX_SCRIPT = r'''
import sys, numpy as np, uproot, os
wanted = np.loadtxt(sys.argv[1], dtype=np.int64, ndmin=1)
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
base = '/eos/experiment/ntof/data/x17/july_beam/runs/{run}/{sub}/decoded_root/'
for t in {tags!r}:
    for f in {feus!r}:
        tr = uproot.open(base + 'Mx17_{subtag}_datrun_' + t + '_' + f + '.root')['nt']
        eid = tr['eventId'].array(library='np')
        a0 = tr.arrays(['amplitude'], entry_stop=300, library='np')['amplitude']
        idx = np.where(np.isin(eid, wanted))[0]
        sel = [tr.arrays(['eventId', 'amplitude', 'ftst'], entry_start=int(i),
                         entry_stop=int(i) + 1, library='np') for i in idx]
        np.savez_compressed(out + '/' + t[-3:] + '_' + f + '.npz',
            ped_amp=np.array([np.asarray(x, np.int16) for x in a0 if len(x) == len(a0[0])]),
            eid=np.array([int(s['eventId'][0]) for s in sel], np.int64),
            ftst=np.array([int(s['ftst'][0]) for s in sel], np.int64),
            amp=(np.array([np.asarray(s['amplitude'][0], np.int16) for s in sel])
                 if sel else np.zeros((0, 1), np.int16)))
        print(t, f, len(idx), flush=True)
'''


def lxplus_cut_script(tags) -> str:
    """The script that cuts decoded_root down to the wanted triggers on lxplus.

    The full files are 115 MB per FEU and tag; the cut keeps the first 300
    events (the pedestal, as `wft.io.FeuReader` builds it) and the candidates,
    ~4 MB.  Run with LCG_105:  python3 cut.py wanted.txt /tmp/<dir>
    """
    return LX_SCRIPT.format(run=RUN, sub=SUBRUN, subtag=SUBRUN.replace('stat', 'stat'),
                            tags=[f'260805_14H06_{t}' for t in tags],
                            feus=['03', '04', '07', '08'])


class Waveforms:
    """Pedestal + CNS corrected (512, n_sample) waveforms from the cut files."""

    def __init__(self, wf_dir: Path = WF_DIR):
        self.dir = Path(wf_dir)
        self._cache = {}

    def _load(self, tag3: str, feu: int):
        key = (tag3, feu)
        if key not in self._cache:
            z = np.load(self.dir / f'{tag3}_{feu:02d}.npz')
            ped_amp = z['ped_amp'].astype(np.float32)
            ns = ped_amp.shape[1] // 512
            stack = ped_amp.reshape(len(ped_amp), ns, 512)
            ped = np.median(stack, axis=(0, 1))
            sub = stack - ped[None, None, :]
            cms = np.median(sub.reshape(len(stack), ns, 8, 64), axis=3)
            sub -= np.repeat(cms, 64, axis=2)
            noise = 1.4826 * np.median(np.abs(sub), axis=(0, 1))
            self._cache[key] = dict(ped=ped, noise=np.maximum(noise, 3.0), ns=ns,
                                    eid=z['eid'], amp=z['amp'])
        return self._cache[key]

    def event(self, tag3: str, feu: int, event_id: int):
        """(W[512, ns], noise[512]) or None when the trigger was not cut."""
        d = self._load(tag3, feu)
        i = np.where(d['eid'] == event_id)[0]
        if not len(i):
            return None
        w = d['amp'][i[0]].astype(np.float32).reshape(-1, 512) - d['ped']
        cms = np.median(w.reshape(w.shape[0], 8, 64), axis=2)
        w -= np.repeat(cms, 64, axis=1)
        return w.T, d['noise']


def strip_maps():
    """{arm: (pos_x[512], pos_y[512])} strip positions on the 0..398.6 mm map."""
    from ntof_tracking import wft_beam
    from wft import io as wio
    out = {}
    for arm in ('A', 'C'):
        cfg = wft_beam.beam_config(arm, RUN, SUBRUN)
        pm = wio.strip_position_map(cfg)
        out[arm] = (pm[FEU[arm][0]], pm[FEU[arm][1]])
    return out


def charge_samples(arm: str, tracks: pd.DataFrame, wf: Waveforms, smap,
                   event_id: int, tag3: str, n_sigma: float = 5.0) -> dict:
    """Per view, the samples above ``n_sigma`` x noise as (coord, depth, adc).

    ``coord`` is the local strip coordinate (x_local / y_local, the frame the
    tracks' p0 lives in).  Each sample takes the t0 of the track it sits
    nearest to, so two tracks with different t0 are each drawn at their own
    depth scale.
    """
    from sept26_prelim_analysis.build_tracks import (
        IN_PLANE_SIGN, IN_PLANE_SIGN_Y, STRIP_MAP_HALF)
    v = tracks.v_drift_um_ns.iloc[0] / 1000.0          # in-situ, v / k
    out = {}
    for view, feu, pos, sign in (('x', FEU[arm][0], smap[arm][0], IN_PLANE_SIGN),
                                 ('y', FEU[arm][1], smap[arm][1], IN_PLANE_SIGN_Y)):
        r = wf.event(tag3, feu, event_id)
        if r is None:
            out[view] = None
            continue
        W, noise = r
        ns = W.shape[1]
        ok = np.isfinite(pos)
        ch = np.where(ok)[0]
        coord = sign * (pos[ch] - STRIP_MAP_HALF)
        A = W[ch]
        sig = A > n_sigma * noise[ch, None]
        t = np.arange(ns) * SAMPLE_NS
        # each track's line in this view: c(w) = c0 - tan * w
        c0 = tracks[f'{view}_local'].to_numpy(float)
        tn = tracks[f'tan{view}'].to_numpy(float)
        t0 = tracks[f'{view}_t0'].to_numpy(float)
        ii, jj = np.where(sig)
        cc, tt, aa = coord[ii], t[jj], A[ii, jj]
        best = np.full(len(ii), np.inf)
        dep = np.full(len(ii), np.nan)
        for k in range(len(c0)):
            w = (tt - TAU_PEAK - t0[k]) * v
            d = np.abs(cc - (c0[k] - tn[k] * w))
            m = d < best
            best[m], dep[m] = d[m], w[m]
        out[view] = dict(coord=cc, depth=dep, adc=aa, coord_all=coord,
                         W=A, noise=noise[ch], t=t, v=v, t0=t0, c0=c0, tan=tn)
    return out


# =========================================================================== #
# geometry of the scene
# =========================================================================== #
def scene_geometry():
    """Every drawn volume in the global frame, from run_config (det_a_scint)."""
    from sept26_prelim_analysis import det_a_scint as DS
    from ntof_tracking.reco import geometry as G
    cfg = json.loads((paths.root('runs') / RUN / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)
    pos = {d['name']: np.array([d['det_center_coords'][k] for k in 'xyz'], float)
           for d in cfg['detectors']}
    geo = {}
    for arm in ARMS:
        uh, wh = G.U_HAT[arm], G.W_HAT[arm]
        g = dict(tr=trs[f'mx17_{arm}'], u=uh, w=wh, v=G.V_HAT)
        g['bars'] = {int(n.split('_')[-1]): pos[n] for n in pos
                     if n.startswith(f'sipm_{arm}_')}
        lg = DS.layer_geometry(RUN, arm)
        g['plas'] = {int(n): pos[f'plastic_{arm}_{s}'] for s in ('L', 'R')
                     for n, u in lg['plas_u'].items()
                     if abs(pos[f'plastic_{arm}_{s}'] @ uh - u) < 1e-6}
        g['ls'] = pos[f'liquid_{arm}']
        geo[arm] = g
    return geo


def fired(slim: pd.DataFrame, event_id: int) -> dict:
    """{arm: dict(wall=set(groups), plas=set(detn), ls=bool)} in the prod window."""
    h = slim[(slim.eventId == event_id) & (slim.is_control == 0)
             & (slim.dt_ns >= DT_WINDOW[0]) & (slim.dt_ns <= DT_WINDOW[1])]
    out = {}
    for i, arm in enumerate(ARMS):
        out[arm] = dict(
            wall={int((n - 1) // 2) for n in h[h.det == WAL + i].detn},
            plas={int(n) for n in h[h.det == PSS + i].detn},
            ls=bool((h.det == LIQ + i).any()))
    return out


def _oriented_box(center, u, v, w, hu, hv, hw):
    """A pyvista box with half-sizes (hu, hv, hw) along the unit axes (u, v, w)."""
    import pyvista as pv
    b = pv.Box(bounds=(-hu, hu, -hv, hv, -hw, hw))
    R = np.column_stack([u, v, w])
    b.points = np.asarray(b.points) @ R.T + np.asarray(center, float)
    return b


def capsule_meshes():
    """(Al vessel, He-3 gas) polycones at the inferred pose."""
    import pyvista as pv
    from sept26_prelim_analysis import capsule_pose as CP
    ax = CP.axis_of(POSE['theta'], POSE['phi'])
    ctr = np.array([POSE['x_c'], POSE['y_c'], POSE['z_c']])
    out = []
    for S, R in ((CP.S_AL, CP.R_AL), (CP.S_GAS, CP.R_GAS)):
        prof = np.column_stack([R, np.zeros_like(R), S])
        m = pv.lines_from_points(prof).extrude_rotate(resolution=72, capping=True)
        # profile axis is +z; map z -> the capsule axis
        z = np.array([0, 0, 1.0])
        k = np.cross(z, ax)
        s, c = np.linalg.norm(k), float(z @ ax)
        if s < 1e-12:
            Rm = np.eye(3)
        else:
            k = k / s
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rm = np.eye(3) + s * K + (1 - c) * K @ K
        m.points = np.asarray(m.points) @ Rm.T + ctr
        out.append(m)
    return out


# =========================================================================== #
# the drawing
# =========================================================================== #
COL = dict(bg='#f5f3ee', ink='#1c2230', mut='#5a6170', rule='#d8d3c8',
           mm='#7d8796', leg1='#1f5fa8', leg2='#c8601a', lit='#c99318',
           ls_lit='#2f8a5b', plas_lit='#c63d4f', vessel='#9aa3ad', gas='#7a52b8',
           beam='#c63d4f', vertex='#1c2230')
LEG = (COL['leg1'], COL['leg2'])
HALF_U, HALF_V, GAP = 190.0, 170.0, 30.0


def _tube(pl, a, b, color, radius, opacity=1.0):
    import pyvista as pv
    pl.add_mesh(pv.Tube(pointa=tuple(a), pointb=tuple(b), radius=radius,
                        n_sides=14), color=color, opacity=opacity,
                smooth_shading=True)


def _local(tr, xl, yl, w):
    return tr.local_to_global(np.asarray(xl, float), np.asarray(yl, float),
                              np.asarray(w, float))


def build_scene(pl, ev: dict, geo: dict, fire: dict, charge: dict):
    """Add every mesh for one trigger to plotter ``pl``."""
    import pyvista as pv
    import matplotlib
    from matplotlib import colors as mcolors

    # --- chambers: all four, the pair's chambers stronger
    for arm in ARMS:
        g = geo[arm]
        ctr = _local(g['tr'], 0.0, 0.0, GAP / 2)
        box = _oriented_box(ctr, g['u'], g['v'], -g['w'], HALF_U, HALF_V, GAP / 2)
        on = arm in ev['arms']
        pl.add_mesh(box, color=COL['mm'], opacity=0.14 if on else 0.04,
                    show_edges=True, edge_color=COL['mm'] if on else COL['rule'],
                    line_width=1.2 if on else 0.6)

    # --- scintillators: unfired = one faint outline per read-out unit,
    #     fired = lit (flat-shaded, so the colour reads as the colour)
    lit_kw = dict(lighting=False)
    for arm in ARMS:
        g, f = geo[arm], fire[arm]
        for grp in range(4):
            ps = [g['bars'][4 * grp + 1 + i] for i in range(4)]
            if grp in f['wall']:
                for p in ps:
                    pl.add_mesh(_oriented_box(p, g['u'], g['v'], g['w'], 11.8, 250.0, 2.0),
                                color=COL['lit'], opacity=0.9, **lit_kw)
            else:
                c = np.mean(ps, axis=0)
                pl.add_mesh(_oriented_box(c, g['u'], g['v'], g['w'], 49.0, 250.0, 1.5),
                            color=COL['rule'], opacity=0.10, **lit_kw)
        for n, p in g['plas'].items():
            if n in f['plas']:
                pl.add_mesh(_oriented_box(p, g['u'], g['v'], g['w'], 99.0, 150.0, 10.0),
                            color=COL['plas_lit'], opacity=0.55, **lit_kw)
            elif arm in ev['arms']:
                pl.add_mesh(_oriented_box(p, g['u'], g['v'], g['w'], 99.0, 150.0, 10.0),
                            style='wireframe', color=COL['mut'], opacity=0.18,
                            line_width=0.6)
        box = _oriented_box(g['ls'], g['u'], g['v'], g['w'], 225.6, 225.3, 10.6)
        if f['ls']:
            pl.add_mesh(box, color=COL['ls_lit'], opacity=0.30, **lit_kw)
        elif arm in ev['arms']:
            pl.add_mesh(box, style='wireframe', color=COL['mut'], opacity=0.12,
                        line_width=0.5)

    # --- beam and capsule
    beam = pv.Cylinder(center=(BEAM_X, 0, 0), direction=(0, 1, 0),
                       radius=BEAM_R, height=900, resolution=48, capping=False)
    pl.add_mesh(beam, color=COL['beam'], opacity=0.07)
    pl.add_mesh(pv.Arrow(start=(BEAM_X, -470, 0), direction=(0, 1, 0), scale=140,
                         tip_length=0.25, tip_radius=0.08, shaft_radius=0.025),
                color=COL['beam'], opacity=0.6)
    al, gas = capsule_meshes()
    pl.add_mesh(al, color=COL['vessel'], opacity=0.35, smooth_shading=True)
    pl.add_mesh(gas, color=COL['gas'], opacity=0.85, smooth_shading=True)

    # --- tracks
    for i, t in enumerate(ev['legs'].itertuples()):
        g = geo[t.arm]
        col = LEG[i % 2]
        p0 = np.array([t.p0_x, t.p0_y, t.p0_z])
        d = np.array([t.d_x, t.d_y, t.d_z])
        d = d / np.linalg.norm(d)
        # in the gap: from the strips (w = 0) to the cathode (w = 30)
        a = _local(g['tr'], t.x_local, t.y_local, 0.0)
        b = _local(g['tr'], t.x_local - t.tanx * GAP, t.y_local - t.tany * GAP, GAP)
        _tube(pl, a, b, col, 3.2)
        # outward to the SiPM wall plane, inward to the vertex
        wh = g['w']
        w_wall = float(next(iter(g['bars'].values())) @ wh)
        s_out = (w_wall - p0 @ wh) / (d @ wh)
        _tube(pl, a, p0 + s_out * d, col, 1.2, 0.9)
        vin = ev['vertex'] if ev['vertex'] is not None else _axis_foot(p0, d)
        s_in = (vin - p0) @ d
        _tube(pl, b, p0 + s_in * d, col, 1.0, 0.55)
    if ev['vertex'] is not None:
        pl.add_mesh(pv.Sphere(radius=5.0, center=tuple(ev['vertex'])),
                    color=COL['vertex'], opacity=0.9)

    # --- TPC charge, as the two views' projections on the chamber faces
    cmap = matplotlib.colormaps['magma_r']
    for arm, ch in charge.items():
        g = geo[arm]
        for view, face in (('x', +HALF_V), ('y', +HALF_U)):
            c = ch.get(view)
            if c is None or not len(c['coord']):
                continue
            keep = (c['depth'] > -2) & (c['depth'] < GAP + 2)
            cc, dd, aa = c['coord'][keep], np.clip(c['depth'][keep], 0, GAP), c['adc'][keep]
            if not len(cc):
                continue
            if view == 'x':
                P = _local(g['tr'], cc, np.full_like(cc, face), dd)
            else:
                P = _local(g['tr'], np.full_like(cc, face), cc, dd)
            norm = mcolors.LogNorm(vmin=max(aa.min(), 20), vmax=max(aa.max(), 40))
            rgb = (cmap(0.15 + 0.85 * norm(aa))[:, :3] * 255).astype(np.uint8)
            cloud = pv.PolyData(P)
            cloud['rgb'] = rgb
            pl.add_mesh(cloud, scalars='rgb', rgb=True, point_size=6.0,
                        render_points_as_spheres=True, lighting=False)
            # the tracks' projections on the same face, so the eye can link them
            for i, t in enumerate(ev['legs'].itertuples()):
                if t.arm != arm:
                    continue
                ws = np.array([0.0, GAP])
                if view == 'x':
                    q = _local(g['tr'], t.x_local - t.tanx * ws, np.full(2, face), ws)
                else:
                    q = _local(g['tr'], np.full(2, face), t.y_local - t.tany * ws, ws)
                pl.add_mesh(pv.Line(q[0], q[1]), color=LEG[i % 2], line_width=2.5,
                            opacity=0.9)


def _axis_foot(p0, d):
    """Point on the line closest to the beam axis (x = BEAM_X, z = 0)."""
    q = p0 - np.array([BEAM_X, 0, 0])
    den = d[0] ** 2 + d[2] ** 2
    s = -(q[0] * d[0] + q[2] * d[2]) / den if den > 1e-12 else 0.0
    return p0 + s * d


# =========================================================================== #
# the 2D insets: strip vs depth per view, the fitted lines over the raw charge
# =========================================================================== #
def inset_figure(ev: dict, charge: dict, size_px=(620, 820)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    arms = [a for a in ev['arms'] if a in charge]
    n = len(arms)
    fig, axs = plt.subplots(2 * n, 1, figsize=(size_px[0] / 100, size_px[1] / 100),
                            dpi=100, squeeze=False)
    fig.patch.set_facecolor(COL['bg'])
    k = 0
    for arm in arms:
        for view in ('x', 'y'):
            ax = axs[k, 0]
            k += 1
            ax.set_facecolor('#fffdf9')
            c = charge[arm].get(view)
            legs = ev['legs'][ev['legs'].arm == arm]
            if c is None:
                ax.text(0.5, 0.5, 'no waveform', ha='center', transform=ax.transAxes)
                continue
            lo = np.nanmin(legs[f'{view}_local']) - 45
            hi = np.nanmax(legs[f'{view}_local']) + 45
            sel = (c['coord_all'] >= lo) & (c['coord_all'] <= hi)
            o = np.argsort(c['coord_all'][sel])
            xs = c['coord_all'][sel][o]
            Wm = c['W'][sel][o]
            Wm = np.ma.masked_less(Wm, 5.0 * c['noise'][sel][o][:, None])
            t0 = float(np.nanmean(c['t0']))
            dep = (c['t'] - TAU_PEAK - t0) * c['v']
            if len(xs) > 1:
                ax.pcolormesh(xs, dep, Wm.T, shading='nearest', cmap='magma_r',
                              norm=mcolors.LogNorm(vmin=max(float(Wm.min()), 15), vmax=max(float(Wm.max()), 60)),
                              rasterized=True)
            for i, t in enumerate(ev['legs'].itertuples()):
                if t.arm != arm:
                    continue
                ws = np.array([0.0, GAP])
                cl = getattr(t, f'{view}_local') - getattr(t, f'tan{view}') * ws
                ax.plot(cl, ws, color=LEG[i % 2], lw=1.1, alpha=0.85, ls=(0, (4, 2)))
            ax.axhspan(0, GAP, color=COL['mm'], alpha=0.06, lw=0)
            ax.axhline(0, color=COL['mut'], lw=0.6, ls=':')
            ax.axhline(GAP, color=COL['mut'], lw=0.6, ls=':')
            ax.set_ylim(GAP + 12, -8)
            ax.set_xlim(lo, hi)
            ax.set_ylabel('depth (mm)', fontsize=9, color=COL['mut'])
            ax.set_title(f'chamber {arm} · {view} strips', fontsize=10.5,
                         color=COL['ink'], loc='left', fontweight='bold')
            ax.tick_params(labelsize=8, colors=COL['mut'])
            for s in ax.spines.values():
                s.set_color(COL['rule'])
    axs[-1, 0].set_xlabel('strip position, local (mm)', fontsize=9, color=COL['mut'])
    fig.tight_layout(h_pad=0.9)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


# =========================================================================== #
# frames
# =========================================================================== #
def _font(size, bold=False):
    from matplotlib import font_manager
    from PIL import ImageFont
    p = font_manager.findfont(font_manager.FontProperties(
        family='DejaVu Sans', weight='bold' if bold else 'normal'))
    return ImageFont.truetype(p, size)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def camera_path(n: int, ev: dict, geo: dict):
    """(pos, focal) per frame.  The loop opens and closes on a side-on close
    view of the pair (capsule -> chamber -> wall in profile), swinging slowly
    there, and makes one fast wide orbit of the whole set-up in the middle.
    Side-on = looking along the arm's u axis, so no scintillator sits between
    the camera and the tracks."""
    focal_w = np.array([0.0, -20.0, 0.0])
    legs = ev['legs']
    if len(ev['arms']) == 1:
        g = geo[ev['arms'][0]]
        ch = _local(g['tr'], legs.x_local.mean(), legs.y_local.mean(), GAP / 2)
        vtx = ev['vertex'] if ev['vertex'] is not None else np.array([BEAM_X, ch[1], 0.0])
        focal_c = 0.5 * (ch + vtx) + 0.12 * (ch - vtx)
        r_c = 760.0
        az_n = float(np.degrees(np.arctan2(g['w'][0], g['w'][2])))
    else:
        focal_c = np.array([0.0, ev['vertex'][1] if ev['vertex'] is not None else 0.0, 0.0])
        r_c = 1250.0
        az_n = 0.0                         # the A-C axis is global z
    az0 = az_n + 90.0
    r_w, el_w = 1700.0, 26.0
    el_c = 48.0 if len(ev['arms']) == 1 else 16.0   # a same-chamber pair is split in u
    out = []
    for i in range(n):
        f = i / n
        g_ = f - 0.75 / (2 * np.pi) * np.sin(2 * np.pi * f)   # slow near f = 0, 1
        z = 1.0 - _smooth((f - 0.10) / 0.22) + _smooth((f - 0.68) / 0.22)
        az = az0 + 360.0 * g_
        r = r_w + (r_c - r_w) * z
        el = np.radians(el_w + (el_c - el_w) * z)
        foc = focal_w + (np.asarray(focal_c, float) - focal_w) * z
        a = np.radians(az)
        pos = foc + r * np.array([np.cos(el) * np.sin(a), np.sin(el),
                                  np.cos(el) * np.cos(a)])
        out.append((pos, foc))
    return out


def header_lines(ev: dict) -> tuple:
    p = ev['pair']
    tof = float(p['tof_ms'])
    v = L_FLIGHT_M / (tof * 1e-3)
    e_mev = 0.5 * 939.565e6 * (v / 2.998e8) ** 2 * 1e3      # meV
    if p['kind'] == 'two-arm':
        what = 'two-arm pair, chambers A + C'
    else:
        what = f'same-chamber pair, chamber {p["arm"]}'
    flags = []
    if bool(p.get('wall2', False)) is True:
        flags.append('both legs on different, fired wall groups (wall2)')
    elif bool(p.get('wall', False)):
        flags.append('both legs confirmed on the SiPM wall')
    title = f'{what}  ·  run_145 trigger {int(p["event_id"])}'
    sub = (f'opening angle {p["open_deg"]:.0f}°  ·  lines meet within '
           f'{p["dca_pair_mm"]:.0f} mm at y = {p["vy_mm"]:+.0f} mm, '
           f'{p["v_r_mm"]:.0f} mm from the beam axis  ·  Δt₀ {p["dt0_ns"]:+.0f} ns  ·  '
           f'{tof:.1f} ms after the γ-flash (≈{e_mev:.1f} meV)')
    return title, sub, '  ·  '.join(flags)


def legend_items():
    return [(COL['leg1'], 'track 1'), (COL['leg2'], 'track 2'),
            ('#8a3a6a', 'TPC charge (x / y view projections)'),
            (COL['lit'], 'SiPM wall group fired'), (COL['plas_lit'], 'plastic fired'),
            (COL['ls_lit'], 'liquid scint. fired'),
            (COL['gas'], '³He capsule, inferred pose')]


def render_event(ev: dict, geo: dict, fire: dict, charge: dict, out_dir: Path,
                 n_frames: int = 96, size=(1600, 900), fps: int = 14) -> dict:
    import pyvista as pv
    from PIL import Image, ImageDraw
    pv.OFF_SCREEN = True
    W, H = size
    head, foot = 112, 46
    inset_w = 560
    view_w, view_h = W - inset_w, H - head - foot
    inset = inset_figure(ev, charge, size_px=(inset_w, view_h))
    title, sub, flags = header_lines(ev)

    pl = pv.Plotter(off_screen=True, window_size=(view_w, view_h), border=False)
    pl.set_background(COL['bg'])
    pl.enable_anti_aliasing('ssaa')
    build_scene(pl, ev, geo, fire, charge)
    frames = []
    for pos, foc in camera_path(n_frames, ev, geo):
        pl.camera_position = [tuple(pos), tuple(foc), (0, 1, 0)]
        pl.camera.view_angle = 30.0
        pl.reset_camera_clipping_range()
        pl.render()
        img = pl.screenshot(return_img=True)
        canvas = Image.new('RGB', (W, H), COL['bg'])
        canvas.paste(Image.fromarray(img), (0, head))
        canvas.paste(Image.fromarray(inset), (view_w, head))
        dr = ImageDraw.Draw(canvas)
        dr.text((28, 18), title, font=_font(30, True), fill=COL['ink'])
        dr.text((28, 60), sub, font=_font(17), fill=COL['mut'])
        if flags:
            dr.text((28, 84), flags, font=_font(16, True), fill=COL['lit'])
        x = 28
        for c, lab in legend_items():
            dr.rounded_rectangle([x, H - foot + 14, x + 18, H - foot + 30], radius=3, fill=c)
            dr.text((x + 25, H - foot + 13), lab, font=_font(15), fill=COL['ink'])
            x += 25 + int(dr.textlength(lab, font=_font(15))) + 26
        dr.text((W - 150, H - foot + 13), 'PRELIMINARY', font=_font(15, True),
                fill=COL['mut'])
        frames.append(canvas)
    pl.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'pair_{ev["pair"]["kind"].replace("-", "")}_{int(ev["pair"]["event_id"])}'
    still = out_dir / f'{stem}.png'
    frames[0].save(still)
    gif = out_dir / f'{stem}.gif'
    # ONE palette for the whole loop: a per-frame adaptive palette re-picks the
    # colours every frame, and the legend swatches visibly flicker
    sample = Image.new('RGB', (W, H * 4))
    for k, i in enumerate(np.linspace(0, n_frames - 1, 4).astype(int)):
        sample.paste(frames[i], (0, k * H))
    ref = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=int(1000 / fps),
                loop=0, optimize=False, disposal=1)
    try:
        import imageio.v2 as iio
        mp4 = out_dir / f'{stem}.mp4'
        with iio.get_writer(mp4, fps=fps, codec='libx264', quality=8,
                            macro_block_size=8, pixelformat='yuv420p') as wr:
            for f in frames:
                wr.append_data(np.asarray(f))
    except Exception as exc:                    # no ffmpeg: the GIF is the product
        print(f'  [mp4 skipped: {exc}]')
    return dict(gif=str(gif), still=str(still), mb=round(gif.stat().st_size / 1e6, 1))


def event_record(P: pd.DataFrame, L: pd.DataFrame, T: pd.DataFrame, event_id: int,
                 kind: str | None = None) -> dict:
    """The pair row, its two legs (with the stage-3 columns) and the vertex."""
    rows = P[P.event_id == event_id]
    if kind:
        rows = rows[rows.kind == kind]
    if not len(rows):
        raise KeyError(f'trigger {event_id}: no pair in the candidate table')
    p = rows.sort_values('dca_pair_mm').iloc[0]
    legs = L[L.event_id == event_id]
    if p['kind'] == 'two-arm':
        a = legs[(legs.arm == 'A') & np.isclose(legs.x_local, p.u1)]
        c = legs[(legs.arm == 'C') & np.isclose(legs.x_local, p.u2)]
        legs = pd.concat([a.head(1), c.head(1)])
    else:
        legs = legs[(legs.arm == p['arm'])
                    & (np.isclose(legs.x_local, p.u1) | np.isclose(legs.x_local, p.u2))]
    # the stage-3 columns det_a_intra drops (t0 per view, v) come from the table
    full = T[T.event_id == event_id]
    legs = legs.merge(full[['arm', 'x_local', 'y_local', 'x_t0', 'y_t0',
                            'v_drift_um_ns', 'tag', 'track_id']]
                      .drop_duplicates(['arm', 'x_local', 'y_local']),
                      on=['arm', 'x_local', 'y_local'], how='left',
                      suffixes=('', '_s3'))
    for c in ('x_t0', 'y_t0'):
        if f'{c}_s3' in legs.columns:
            legs[c] = legs[c].fillna(legs[f'{c}_s3'])
    if len(legs) != 2:
        raise ValueError(f'trigger {event_id}: found {len(legs)} legs, expected 2')
    vtx = np.array([p.vx_mm, p.vy_mm, p.vz_mm]) if p.dca_pair_mm < 60 else None
    return dict(pair=p, legs=legs.reset_index(drop=True), vertex=vtx,
                arms=sorted(set(legs.arm)), tag3=str(legs.tag.iloc[0])[-3:])


def render(events, out: Path, n_frames: int, size, fps: int) -> list:
    P = pd.read_parquet(out / 'pairs.parquet')
    L = pd.read_parquet(out / 'legs.parquet')
    T = pd.read_parquet(out / 'tracks.parquet')
    slim = pd.read_parquet(out / 'work' / 'slim' / f'ntof_hits_{RUN}_{SUBRUN}.parquet')
    geo = scene_geometry()
    smap = strip_maps()
    wf = Waveforms()
    res = []
    for e in events:
        kind = None
        if ':' in str(e):
            kind, e = str(e).split(':')
            kind = {'s': 'same-chamber', 't': 'two-arm'}.get(kind, kind)
        ev = event_record(P, L, T, int(e), kind)
        charge = {}
        for arm in ev['arms']:
            legs = ev['legs'][ev['legs'].arm == arm]
            charge[arm] = charge_samples(arm, legs, wf, smap, int(e), ev['tag3'])
        r = render_event(ev, geo, fired(slim, int(e)), charge, out / 'gif',
                         n_frames=n_frames, size=size, fps=fps)
        r.update(event_id=int(e), kind=ev['pair']['kind'])
        print(f'  {e}: {r["gif"]} ({r["mb"]} MB)')
        res.append(r)
    return res


# =========================================================================== #
# report.html: the gallery, generated from the tables
# =========================================================================== #
PROMPT_NS = 75.0          # det_a_intra.PROMPT_NS
STRAIGHT_DEG = 170.0      # a two-arm "pair" this open is one line through both


def caveats(p: pd.Series) -> list:
    """What each display does NOT show, computed from the pair's own numbers."""
    out = []
    if p['kind'] == 'two-arm' and p['open_deg'] >= STRAIGHT_DEG:
        out.append(f'{p["open_deg"]:.0f}° is nearly a straight line through both '
                   'chambers: most likely ONE through-going particle, not a pair.')
    if p['kind'] == 'same-chamber' and abs(p['dt0_ns']) > PROMPT_NS:
        out.append(f'Δt₀ {p["dt0_ns"]:+.0f} ns is outside the ±{PROMPT_NS:.0f} ns '
                   'prompt window: the two legs may not be simultaneous.')
    if p['v_r_mm'] > 20 or p['dca_pair_mm'] > 30:
        out.append(f'Fails the analysis vertex cut (lines within 30 mm, within '
                   f'20 mm of the axis): {p["dca_pair_mm"]:.0f} mm apart, '
                   f'{p["v_r_mm"]:.0f} mm from the axis.')
    if not out:
        out.append('Passes the vertex cut.')
    return out


def report(out: Path) -> Path:
    import html
    P = pd.read_parquet(out / 'pairs.parquet')
    idx = json.loads((out / 'gif' / 'index.json').read_text())
    sc = P[P.kind == 'same-chamber']
    ta = P[P.kind == 'two-arm']
    n = dict(sc=len(sc), sc_wall2=int(sc.wall2.fillna(False).astype(bool).sum()),
             sc_vtx=int((sc.v_r_mm < 20).mul(sc.dca_pair_mm < 30).sum()),
             ta=len(ta), ta_wall=int(ta.wall.astype(bool).sum()),
             ta_straight=int((ta.wall.astype(bool) & (ta.open_deg >= STRAIGHT_DEG)).sum()))
    cards = []
    for r in idx:
        p = P[(P.event_id == r['event_id']) & (P.kind == r['kind'])].sort_values('dca_pair_mm').iloc[0]
        rel = lambda x: 'gif/' + Path(x).name                       # noqa: E731
        mp4 = rel(r['gif']).replace('.gif', '.mp4')
        title = (f'Chamber {p["arm"]}, same-chamber pair' if p['kind'] == 'same-chamber'
                 else 'Chambers A + C, two-arm pair')
        cav = ''.join(f'<li>{html.escape(c)}</li>' for c in caveats(p))
        cards.append(f'''
<figure class="card">
  <video src="{mp4}" poster="{rel(r['still'])}" autoplay loop muted playsinline></video>
  <figcaption>
    <h3>{title} · trigger {int(p['event_id'])}</h3>
    <p class="num">opening {p['open_deg']:.0f}° · lines {p['dca_pair_mm']:.0f} mm apart at
       y = {p['vy_mm']:+.0f} mm, {p['v_r_mm']:.0f} mm from the axis · Δt₀ {p['dt0_ns']:+.0f} ns ·
       {p['tof_ms']:.1f} ms after the flash</p>
    <ul>{cav}</ul>
    <p class="dl"><a href="{rel(r['gif'])}" download>GIF</a> · <a href="{mp4}" download>MP4</a> ·
       <a href="{rel(r['still'])}" download>PNG</a></p>
  </figcaption>
</figure>''')
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pair event displays</title>
<style>
:root {{ --bg:#f5f3ee; --card:#fffdf9; --ink:#1c2230; --mut:#5a6170; --rule:#d8d3c8; --acc:#c99318; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#151a24; --card:#1d2330; --ink:#eef0f3; --mut:#a3abb9; --rule:#333b4a; }} }}
:root[data-theme="dark"] {{ --bg:#151a24; --card:#1d2330; --ink:#eef0f3; --mut:#a3abb9; --rule:#333b4a; }}
body {{ background:var(--bg); color:var(--ink); font:16px/1.5 'IBM Plex Sans',Helvetica,Arial,sans-serif;
  margin:0; padding:24px 16px 48px; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:28px; margin:0 0 4px; }} h2 {{ font-size:19px; margin:32px 0 8px; }}
.lede {{ color:var(--mut); max-width:900px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:18px 0; }}
.stat {{ background:var(--card); border:1px solid var(--rule); border-radius:8px; padding:10px 14px; }}
.stat b {{ font-size:24px; display:block; font-variant-numeric:tabular-nums; }} .stat span {{ color:var(--mut); font-size:13px; }}
.card {{ background:var(--card); border:1px solid var(--rule); border-radius:10px; margin:0 0 22px; overflow:hidden; }}
.card video {{ width:100%; display:block; background:#f5f3ee; }}
figcaption {{ padding:12px 18px 14px; }} figcaption h3 {{ margin:0 0 4px; font-size:17px; }}
.num {{ color:var(--mut); margin:0 0 6px; font-variant-numeric:tabular-nums; }}
ul {{ margin:4px 0 6px 18px; padding:0; }} .dl a {{ color:var(--acc); font-weight:600; }}
.keys li {{ margin-bottom:4px; }}
</style></head><body><main>
<h1>Pair event displays</h1>
<p class="lede">Selected pair candidates from run_145, sub-run stat090_0000, on the is2_v2 re-pass
reconstruction at the capsule-focus angle scale (k<sub>A</sub> {K_CF['A']:.3f}, k<sub>C</sub> {K_CF['C']:.3f}).
Each loop opens on a close view of the pair, orbits the whole set-up, and comes back.
The panels on the right show raw strip charge against drift depth, with the fitted track dashed over it.
<b>PRELIMINARY.</b></p>
<div class="stats">
 <div class="stat"><b>{n['sc']}</b><span>same-chamber pairs (A + C, re-pass cuts)</span></div>
 <div class="stat"><b>{n['sc_wall2']}</b><span>of them wall2: two different fired wall groups</span></div>
 <div class="stat"><b>{n['sc_vtx']}</b><span>pass the vertex cut in this sub-run</span></div>
 <div class="stat"><b>{n['ta_wall']}</b><span>two-arm A + C pairs, both legs on the wall</span></div>
 <div class="stat"><b>{n['ta_straight']}</b><span>of them ≥ {STRAIGHT_DEG:.0f}° (one straight line)</span></div>
</div>
<h2>The events</h2>
{''.join(cards)}
<h2>What is drawn</h2>
<ul class="keys">
<li><b>TPC charge</b>: raw DREAM waveforms (decoded_root, pedestal and common-mode subtracted, samples above 5σ),
not the hit tables. A strip plane measures (strip, time), so in 3D each view is drawn as a projection on its own
chamber face: x strips on the top face, y strips on the side face. A sample at time t sits at depth
(t − 140 ns − t₀) · v/k, where 140 ns is the bundle's pulse peaking time.</li>
<li><b>Tracks</b>: the stage-3 lines. They are thick inside the 30 mm gap, run out to the SiPM wall, and run back
to the pair's closest approach (black dot).</li>
<li><b>Scintillators</b>: wall groups (4 bars each, all the read-out resolves), plastics and LS slabs with an
in-time hit (−100…+60 ns, not the control collection) are lit.</li>
<li><b>Capsule</b>: the pose2 fit, 15 mm toward −x, 10 mm low, top tilted 10° toward φ 135°, beam centred at
x = −10 mm (radius 22 mm). The sideways position is degenerate with the beam's; the one drawn is the best-χ²
member of that family.</li>
</ul>
<h2>What these displays do not show</h2>
<ul class="keys">
<li>They are hand-picked from one sub-run (1/29 of run_145), chosen to be clean. They illustrate the topology;
they are not a sample of it. The rates are the pair analysis's numbers (SAME_CHAMBER_PAIRS.md), not these.</li>
<li>A same-chamber "wall2" pair needs two particles, but not two from one vertex. Two-arm pairs near 180° are
mostly single through-going particles.</li>
<li>The angle scale is the capsule-focus k, 12 % above the in-beam muon scale (open). At the muon k every
opening angle shrinks by about that much.</li>
<li>Stage 3 here was rebuilt on the laptop with build_tracks.build_arm from the is2_v2 reco. The wall match and
pair flags are det_a_scint.match_run and det_a_intra.pairs, unchanged, but the table is not the desktop's
stage3_is2_v2_cf file.</li>
</ul>
<p class="lede">Generated by sept26_prelim_analysis/pair_event_display.py ({pd.Timestamp.now():%Y-%m-%d}).</p>
</main></body></html>'''
    dst = out / 'report.html'
    dst.write_text(page)
    return dst


def inline_note(out: Path, width: int = 1280, crf: int = 30) -> Path:
    """report.html with every asset embedded, for the notes site (which takes
    only self-contained pages).  Videos are re-encoded smaller; the GIF and
    PNG download links are dropped (they live beside report.html)."""
    import base64
    import re
    import subprocess
    import imageio_ffmpeg
    from PIL import Image
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    page = (out / 'report.html').read_text()
    small = out / 'gif' / 'note'
    small.mkdir(exist_ok=True)

    def mp4(m):
        src = out / m.group(1)
        dst = small / src.name
        subprocess.run([ff, '-y', '-loglevel', 'error', '-i', str(src), '-vf',
                        f'scale={width}:-2', '-c:v', 'libx264', '-crf', str(crf),
                        '-preset', 'slow', '-pix_fmt', 'yuv420p', '-an', str(dst)],
                       check=True)
        return 'src="data:video/mp4;base64,' + base64.b64encode(dst.read_bytes()).decode() + '"'

    def poster(m):
        src = out / m.group(1)
        dst = small / (src.stem + '.jpg')
        im = Image.open(src).convert('RGB')
        im.resize((width, round(im.height * width / im.width))).save(dst, quality=82)
        return 'poster="data:image/jpeg;base64,' + base64.b64encode(dst.read_bytes()).decode() + '"'

    page = re.sub(r'src="(gif/[^"]+\.mp4)"', mp4, page)
    page = re.sub(r'poster="(gif/[^"]+\.png)"', poster, page)
    page = re.sub(r'<p class="dl">.*?</p>', '', page, flags=re.S)
    dst = out / 'report_note.html'
    dst.write_text(page)
    print(f'{dst} ({dst.stat().st_size / 1e6:.1f} MB)')
    return dst


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == '_pairs':
        _pairs_worker(sys.argv[2], Path(sys.argv[3]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('candidates')
    sp.add_parser('report')
    sp.add_parser('note')
    r = sp.add_parser('render')
    r.add_argument('--events', required=True,
                   help='comma list of trigger ids; prefix s: / t: to pick the '
                        'same-chamber or two-arm pair when a trigger has both')
    r.add_argument('--frames', type=int, default=96)
    r.add_argument('--fps', type=int, default=14)
    r.add_argument('--size', default='1600x900')
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'candidates':
        candidates(a.out)
    elif a.cmd == 'report':
        print(report(a.out))
    elif a.cmd == 'note':
        inline_note(a.out)
    else:
        w, h = (int(x) for x in a.size.split('x'))
        res = render(a.events.split(','), a.out, a.frames, (w, h), a.fps)
        (a.out / 'gif' / 'index.json').write_text(json.dumps(res, indent=1))
        print(report(a.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
