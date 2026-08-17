#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_coincidence.py -- one real muon through all five SPS chambers.

    ../.venv/bin/python make_coincidence.py [--theme light|dark|both]
                                            [--views hero,beam,side]
                                            [--event N] [--draft]

The setup is the same scene ``make_sps.py`` renders -- two EIC uRWELL
references at z = 0 / 1370 mm around P2_IN / P2_MID / P2_OUT at 320 / 630 /
940 mm -- but instead of a sheaf of sampled beam tracks it draws **a single
measured muon**, with the pad it fired highlighted on each of the three P2
fans and a marker at its cluster on each uRWELL.

MX17 is deliberately absent: it was installed on 1 August, and the run this
event comes from (``highstat_eff_1``) was taken on 25 July.

Two levels of provenance, and the figure says which one it used:

  * ``data/coincident_events.json`` (from ``tools/extract_coincident_event.py``,
    run on lxplus) -- a **five-fold measured coincidence**: the uRWELL track and
    a real P2 cluster in all three stations, same ``geventId``.  Preferred.
  * ``data/urwell_tracks.csv`` alone -- a **measured uRWELL track** with the
    crossed pads computed from the published alignment.  The track is real; the
    three pads are where the track goes, not confirmed hits.

Frames.  The pads are placed by ``geometry.p2_pad_to_lab``, which carries the
renderer's beam's-eye ``P2_MIRROR``; the published alignment instead maps a
uRWELL point into the *pad* frame.  Composing the two naively is a reflection
(x agrees, y flips) -- see ``tools/check_track_pad_frames.py``.  So the track is
drawn through the same pad-frame -> lab composition as the pads, using one
station's transform for the whole line so it stays straight.  The residual
disagreement is then the station-to-station spread of the fit itself,
0.4-1.3 mm, which the check script prints.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geometry as G          # noqa: E402
import meshes as M            # noqa: E402
import style as S             # noqa: E402
import scenes_sps as X        # noqa: E402
import annotate as A          # noqa: E402

FIG = os.path.join(HERE, 'figures')
ALIGN = os.path.join(os.path.dirname(HERE), 'sps_beam_test_26', 'analysis',
                     'urw_mapping', 'mapping_alignment.json')
COINC_JSON = os.path.join(HERE, 'data', 'coincident_events.json')
TRACKS_CSV = os.path.join(HERE, 'data', 'urwell_tracks.csv')

Z_FRONT, Z_BACK = 0.0, 1370.0
P2_NAMES = ('P2_IN', 'P2_MID', 'P2_OUT')
TRACK_FRAME = 'P2_MID'        # whose transform draws the line

VIEWS = {
    'hero': dict(pos=(-2300, 1450, -1700), focal=(-20, 330, 620), up=(0, 1, 0),
                 angle=27.0),
    'beam': dict(pos=(-300, 470, -2700), focal=(0, 330, 700), up=(0, 1, 0),
                 angle=24.0),
    'side': dict(pos=(-4900, 1000, -1150), focal=(0, 330, 660), up=(0, 1, 0),
                 angle=17.0),
    # tight on the three fans, so the three lit pads are readable
    'pads': dict(pos=(-1250, 700, -1450), focal=(10, 268, 630), up=(0, 1, 0),
                 angle=19.0),
}


# --------------------------------------------------------------- geometry --- #
def rigid(align, name):
    """(2x2, t, z) taking a uRWELL track point at z into the P2 pad frame."""
    st = align['urwell_to_p2_frame']['stations'][name]
    r = st['rigid_mean']
    th = np.deg2rad(r['theta_deg'])
    A2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return A2, np.array([r['dx'], r['dy']]), float(st['z_mm'])


def urw_point(front, back, z):
    """Linear interpolation of the two-point uRWELL track, in uRWELL local mm."""
    f, b = np.asarray(front, float), np.asarray(back, float)
    return f + (b - f) * (z - Z_FRONT) / (Z_BACK - Z_FRONT)


def lab_at(align, apex, front, back, z, frame=TRACK_FRAME):
    """Lab (x, height) of the track at ``z``, in the drawn pad frame."""
    A2, t, _ = rigid(align, frame)
    return G.p2_pad_to_lab(A2 @ urw_point(front, back, z) + t, apex=apex)


def pad_at_station(align, pad_map, apex, front, back, name):
    """Index and channel_id of the pad the track crosses at ``name``."""
    A2, t, z = rigid(align, name)
    pred = A2 @ urw_point(front, back, z) + t
    d = np.hypot(pad_map.pad_cx.values - pred[0],
                 pad_map.pad_cy.values - pred[1])
    i = int(np.argmin(d))
    return i, int(pad_map.channel_id.values[i]), float(d[i]), z


# ------------------------------------------------------------------ event --- #
def load_event(index=0):
    """Return (front, back, per-station dict, provenance string)."""
    if os.path.exists(COINC_JSON):
        d = json.load(open(COINC_JSON))
        evs = d['events']
        if index >= len(evs):
            raise SystemExit(f'only {len(evs)} coincident events in '
                             f'{COINC_JSON}')
        e = evs[index]
        u = e['urwell']
        prov = (f'five-fold measured coincidence, '
                f'{d["source"]["run"]}/{d["source"]["sub_run"]}, '
                f'geventId {e["geventId"]}')
        st = {k: dict(channel_id=v['channel_id'],
                      residual_mm=v['residual_mm'],
                      measured=True)
              for k, v in e['stations'].items()}
        return ((u['front_x_mm'], u['front_y_mm']),
                (u['back_x_mm'], u['back_y_mm']), st, prov)

    rows = list(csv.DictReader(open(TRACKS_CSV)))
    if not rows:
        raise SystemExit(f'{TRACKS_CSV} is empty')
    return None, None, None, rows          # caller picks; see choose_track


def choose_track(align, pad_map, apex, rows, sectors):
    """Pick the measured uRWELL track that displays best.

    Wanted: all three crossed pads inside the instrumented sectors (so the
    lit pad sits on a bright, read-out part of the fan) and each crossing
    close to its pad centre (so the highlight is unambiguous).
    """
    lo, hi = G.P2_INSTRUMENTED_SECTORS
    best = None
    for r in rows:
        front = (float(r['front_x_mm']), float(r['front_y_mm']))
        back = (float(r['back_x_mm']), float(r['back_y_mm']))
        st, worst, ok = {}, 0.0, True
        for name in P2_NAMES:
            i, cid, d, z = pad_at_station(align, pad_map, apex, front, back,
                                          name)
            if not (lo <= sectors[i] <= hi):
                ok = False
                break
            st[name] = dict(channel_id=cid, residual_mm=d, measured=False)
            worst = max(worst, d)
        if not ok:
            continue
        if best is None or worst < best[0]:
            best = (worst, front, back, st, int(r['eventId']))
    if best is None:
        raise SystemExit('no track crosses instrumented pads on all three '
                         'stations')
    worst, front, back, st, eid = best
    prov = (f'measured uRWELL track (eventId {eid}), crossed pads from the '
            f'published alignment — pads not confirmed as hits')
    return front, back, st, prov


# ------------------------------------------------------------------ scene --- #
def build(theme='light', size=(2600, 1750), ssaa=True, shadows=True,
          event=0, spot=True):
    align = json.load(open(ALIGN))
    pads_lab, sectors, pad_map = X.load_pads_lab()
    _m, _polys, apex = G.load_p2_pads()
    lo, hi = G.P2_INSTRUMENTED_SECTORS
    live = (sectors >= lo) & (sectors <= hi)

    front, back, stations, prov = load_event(event)
    if front is None:                       # fallback path
        front, back, stations, prov = choose_track(align, pad_map, apex, prov,
                                                   sectors)
    print(f'  event: {prov}')

    illum = None
    if spot:
        s = X.load_illumination('P2_MID')
        if s is not None:
            illum = s.reindex(pad_map.channel_id.values).fillna(0.0).values[live]
            illum = np.clip(illum / max(illum.max(), 1e-9), 0, 1)

    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=70.0)
    X.add_table(p, theme, legs=False)

    anchors, outlines, hit_markers = {}, [], []
    cid_to_i = {int(c): i for i, c in enumerate(pad_map.channel_id.values)}

    for st in G.SPS_STATIONS:
        if st.kind == 'mx17':
            continue                        # not installed on 25 July
        if st.kind == 'p2':
            parts = X.add_p2(
                p, st.z, pads_lab, sectors, x=st.x, y=st.y,
                spot=illum if (st.name == 'P2_MID' and illum is not None)
                else None)
            # the fired pad, drawn just in front of the pad plane so it reads
            # as lit rather than as a different pad
            i = cid_to_i[stations[st.name]['channel_id']]
            zf = st.z - (G.P2_THICK_MM / 2 + 1.6)
            # the pad itself, lit
            p.add_mesh(M.quads_mesh(pads_lab[i:i + 1], zf),
                       **S.mat('glow', S.COL['track_mu']))
            # a halo behind it, so a 12 mm pad still reads on a 600 mm fan
            ctr = pads_lab[i].mean(axis=0)
            p.add_mesh(M.quads_mesh(
                (ctr + (pads_lab[i] - ctr) * 2.1)[None, :, :], zf + 0.6),
                **S.mat('glow', S.COL['track_mu'], opacity=0.32))
            hit_markers.append((ctr[0] + st.x, ctr[1] + st.y, zf - 2.0))
            anchors[st.name] = (st.x,
                                G.P2_APEX_HEIGHT - G.P2_R_ACTIVE[1] - 40 + st.y,
                                st.z)
        else:
            parts = X.add_urwell(p, st.z, x=st.x, y=st.y)
            xy = lab_at(align, apex, front, back, st.z)
            hit_markers.append((xy[0] + st.x, xy[1] + st.y,
                                st.z - G.URW_THICK_MM / 2 - 4.0))
            anchors[st.name] = (st.x,
                                G.SPS_BEAM_HEIGHT + G.URW_PCB_MM / 2 + 40 + st.y,
                                st.z)
        outlines.append(parts['outline'])

    # One marker per plane, on the upstream face.  The boards occlude the tube
    # between stations -- correctly, they are solid -- so the five markers are
    # what actually carries "the same muon, in all five chambers".
    import pyvista as pv
    for c in hit_markers:
        p.add_mesh(pv.Sphere(radius=11.0, center=c, theta_resolution=36,
                             phi_resolution=36),
                   **S.mat('glow', S.COL['track_mu']))

    if shadows:
        S.add_ground_shadows(p, outlines, 0.0, plane_axis='y', up='y',
                             theme=theme)

    # the muon itself, drawn through the same composition as the pads
    pad_ = 320.0
    a_xy = lab_at(align, apex, front, back, Z_FRONT)
    b_xy = lab_at(align, apex, front, back, Z_BACK)
    a = np.array([a_xy[0], a_xy[1], Z_FRONT])
    b = np.array([b_xy[0], b_xy[1], Z_BACK])
    u = (b - a) / np.linalg.norm(b - a)
    X.add_tracks(p, [(a - u * pad_, b + u * pad_)], radius=2.6,
                 color=S.COL['track_mu'])

    S.add_light_rig(p, np.array([0.0, 300.0, 685.0]), 900.0, theme=theme,
                    shadows=False)
    return p, anchors, stations, prov


def render(view, theme, out, **kw):
    p, anchors, stations, prov = build(theme=theme, **kw)
    v = VIEWS[view]
    p.camera.position = v['pos']
    p.camera.focal_point = v['focal']
    p.camera.up = v['up']
    p.camera.view_angle = v['angle']
    p.renderer.reset_camera_clipping_range()
    px = A.project(p, anchors)
    S.finish(p, out)
    print(f'  [{out}]')
    return px, stations, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--views', default='hero,beam,pads')
    ap.add_argument('--event', type=int, default=0)
    ap.add_argument('--no-spot', action='store_true')
    ap.add_argument('--no-shadows', action='store_true')
    ap.add_argument('--size', nargs=2, type=int, default=[2600, 1750])
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    size = (1100, 750) if args.draft else tuple(args.size)
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]

    for theme in themes:
        for view in args.views.split(','):
            out = os.path.join(FIG, f'coincidence_{view}_{theme}.png')
            _px, stations, prov = render(
                view, theme, out, event=args.event,
                spot=not args.no_spot, shadows=not args.no_shadows,
                size=size, ssaa=not args.draft)
    print('\n  fired pads')
    for name in P2_NAMES:
        s = stations[name]
        tag = 'measured hit' if s.get('measured') else 'crossed (predicted)'
        print(f'    {name:7s} channel_id {s["channel_id"]:4d}   '
              f'{s["residual_mm"]:5.2f} mm from the pad centre   {tag}')
    print(f'  provenance: {prov}')


if __name__ == '__main__':
    main()
