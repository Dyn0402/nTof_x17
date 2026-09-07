#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ntof.py -- the n_TOF setup, rendered as a build-up sequence.

    ../.venv/bin/python make_ntof.py                  # all nine stages
    ../.venv/bin/python make_ntof.py --only full      # one frame
    ../.venv/bin/python make_ntof.py --view top       # a different camera
    ../.venv/bin/python make_ntof.py --draft          # small and fast

Writes ``figures/ntof_build_<n>_<tag>_<theme>.png`` -- one frame per stage, in
four acts (see ``STAGES``): the frames inside an act share a camera and a size
exactly, so dropping them on successive slides grows the detector onto a still
picture as you talk.  ``--only full`` re-renders just the last one, which is
also the standalone setup figure.

The camera sits in the (-X, +Z) quadrant: the two arms nearest it (B in front,
A on the right) are drawn as outlines, and the two the simulated pair actually
crossed (D, C) stay solid behind them.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import annotate as A             # noqa: E402
import style as S                # noqa: E402
import scenes_ntof as N          # noqa: E402

FIG = os.path.join(HERE, 'figures')
CENTER = (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Labels on the capsule frame
# --------------------------------------------------------------------------- #
# The vessel is a three-layer polycone and the two outer layers are 0.6 and
# 0.9 mm thick on a 20 mm bore, so at any size that fits a slide they are two
# thin bands and nothing on the picture says which is which.  The first frame
# therefore carries its own leader lines.  Anchors are world points; the wall
# ones sit on the cut face, whose outward direction in the frame is the camera's
# own right, so they follow the camera instead of being typed in per view.
def capsule_labels(view):
    look = np.asarray(view['pos'], float) - np.asarray(view['focal'], float)
    t = np.array([look[2], 0.0, -look[0]])
    t /= np.linalg.norm(t)                       # frame-right, on the floor
    r_al = (N.PG.RO_GAS.max() + N.PG.RO_AL.max()) / 2 * N.CM
    r_cf = (N.PG.RO_AL.max() + N.PG.RO_CFRP.max()) / 2 * N.CM
    kw = beam_arrow_kw(view)
    out = {}
    if kw is not None:
        # the dart is on the axis now, so the leader comes off its own tip
        out['beam'] = (np.array([0.0, kw['y_mid'] + kw['length'] / 2, 0.0]),
                       dict(text='EAR2 neutron beam,\nfrom below',
                            dx=-0.19, dy=0.09, ha='right'))
    out.update({
        'gas': (np.array([0.0, 2.0, 0.0]) - 4.0 * t,
                dict(text='500 bar $^{3}$He\nØ20 mm bore, 40 mm long',
                     dx=-0.15, dy=-0.19, ha='right')),
        'al': (np.array([0.0, 14.0, 0.0]) + r_al * t,
               dict(text='0.6 mm Al vessel', dx=0.12, dy=-0.12, ha='left')),
        'cfrp': (np.array([0.0, -16.0, 0.0]) + r_cf * t,
                 dict(text='0.9 mm CFRP overwrap', dx=0.12, dy=0.10,
                      ha='left')),
    })
    return out


def anchor_points(labels):
    """Flatten ``labels`` into the name -> world point map ``A.project`` wants.

    A label may name SEVERAL places at once -- a layer exists in four arms and
    the frame draws two of them solid -- so its anchor is allowed to be a list,
    and each entry gets its own leader off the same block of type.
    """
    out = {}
    for key, (anchor, _) in labels.items():
        pts = anchor if np.asarray(anchor).ndim == 2 else [anchor]
        for i, pt in enumerate(pts):
            out[f'{key}#{i}'] = pt
    return out


def _leader_start(bb, target, pad):
    """Where a leader to ``target`` leaves the text's box (data coords)."""
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    dx, dy = target[0] - cx, target[1] - cy
    x0, y0, x1, y1 = bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad
    ts = []
    if abs(dx) > 1e-9:
        ts.append(((x1 if dx > 0 else x0) - cx) / dx)
    if abs(dy) > 1e-9:
        ts.append(((y1 if dy > 0 else y0) - cy) / dy)
    if not ts:
        return cx, cy
    t = min(min(ts), 1.0)
    return cx + t * dx, cy + t * dy


def overlay(png, px, labels, theme='light'):
    """Draw leader lines and text onto an existing frame, in place.

    ``annotate.compose`` is the package's usual label path, but it grows the
    canvas by a gutter and a title band and writes an opaque page.  These
    frames have to stay exactly the size of their neighbours in the sequence
    and keep their alpha channel, so the type goes on top of the render at 1:1.

    The leaders are drawn by hand rather than by ``annotate(arrowprops=...)``
    because one block of type has to serve several anchors: each line is struck
    from the text's own bounding box towards its target, so two leaders off one
    label leave it from the two different edges that face their arms.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from PIL import Image

    img = np.asarray(Image.open(png).convert('RGBA'))
    h, w = img.shape[:2]
    ink = '#f2f5f9' if theme == 'dark' else '#141b24'
    lead = '#9aa7b6' if theme == 'dark' else '#5d6874'
    halo = '#0a0d13' if theme == 'dark' else '#ffffff'

    dpi = 200.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    # 1:1, so NEAREST -- a resampling filter with negative lobes overshoots the
    # alpha channel at the render's silhouette and leaves a dashed opaque line
    # down the edge of anything translucent
    ax.imshow(img, interpolation='nearest')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    fs = A.TEXT_FRAC * w * 72.0 / dpi * 1.15

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    for key, (_, spec) in labels.items():
        targets = []
        for i in range(16):
            if f'{key}#{i}' not in px:
                break
            targets.append(px[f'{key}#{i}'])
        if not targets:
            continue
        # ``pos`` pins the text to a place in the FRAME rather than to an offset
        # from its anchor: a layer name belongs in whatever corner the apparatus
        # leaves free, and the leader does the pointing.
        if 'pos' in spec:
            tx, ty = spec['pos'][0] * w, spec['pos'][1] * h
        else:
            tx = targets[0][0] + spec.get('dx', 0) * w
            ty = targets[0][1] + spec.get('dy', 0) * h
        f = fs * spec.get('scale', 1.0)
        txt = ax.text(
            tx, ty, spec['text'],
            ha=spec.get('ha', 'left'), va=spec.get('va', 'center'),
            fontsize=f, color=ink, fontweight=spec.get('weight', 'medium'),
            linespacing=1.35,
            path_effects=[pe.withStroke(linewidth=f * 0.34, foreground=halo,
                                        alpha=0.95)],
            zorder=6, **A.FONT)
        e = txt.get_window_extent(rend)
        (bx0, by0), (bx1, by1) = inv.transform([(e.x0, e.y0), (e.x1, e.y1)])
        bb = (min(bx0, bx1), min(by0, by1), max(bx0, bx1), max(by0, by1))
        pad = f * 0.55
        for tgt in targets:
            sx, sy = _leader_start(bb, tgt, pad)
            d = np.hypot(tgt[0] - sx, tgt[1] - sy)
            if d < pad:                       # the anchor is under the type
                continue
            k = max(0.0, (d - f * 0.5) / d)   # stop just short of the anchor
            ax.plot([sx, sx + k * (tgt[0] - sx)], [sy, sy + k * (tgt[1] - sy)],
                    color=lead, lw=f * 0.055, alpha=0.95, solid_capstyle='round',
                    zorder=5)

    fig.savefig(png, dpi=dpi, transparent=True)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Naming the layer as it arrives
# --------------------------------------------------------------------------- #
# One label per build frame, on the layer that just appeared and gone by the
# next frame -- so the picture says what it is showing without the audience
# having to find it in the bullets, and without four names piling up on the
# finished apparatus.  The anchor is the layer's own centre in one arm
# (scenes_ntof.layer_anchor, straight from the geometry); the text is pinned to
# a corner of the frame, since the apparatus is a pinwheel and its corners are
# the only reliably empty places.
#
# ONE LABEL, TWO LEADERS.  A layer is four objects, and the frame draws two of
# them solid; a single leader picks one arm out and quietly implies the label is
# about that one.  Both solid arms therefore get a line off the same block of
# type, which is also what makes the picture say "these are a pair" without a
# word.  Which arms those are is read off the view's own ``near``, so it cannot
# fall out of step with the ghosting.
#
# Sizes in CENTIMETRES: this figure is looked at from across a room, and the
# numbers on it are meant to be read at a glance, not to be quoted.
LAYER_LABEL = {
    'mm': dict(text='Micromegas TPC\n40 × 36 cm, 3 cm drift', v_frac=0.45),
    'sipm': dict(text='SiPM trigger wall\n50 × 50 cm, 20 bars', v_frac=0.70),
    # no "2 x" -- there are four leaders on the frame doing that job
    'plastic': dict(text='Plastic scintillators\n20 × 30 × 2 cm', v_frac=0.72),
    'ls': dict(text='Liquid scintillator\n45 × 45 cm, 6.5 L per arm',
               v_frac=0.0),
}
# Where the type sits, per camera: (x, y) as a fraction of the frame, plus the
# alignment that keeps it inside.  Tuned once per act -- the frames within an
# act share a camera exactly, so one entry serves all of them.
#
# The close-up act puts it at the BOTTOM, over the near arms' outlines: at that
# scale the chambers fill the frame corner to corner and the only empty space
# left is the see-through one, which is exactly the place a label costs nothing.
# ``v_frac`` here overrides the layer's own: from the bottom of the close-up
# frame a leader to the middle of a chamber has to cross the whole picture and
# the target with it, so that act reaches for the near edge instead.
LAYER_POS = {
    'close': dict(pos=(0.045, 0.945), ha='left', va='bottom', v_frac=-0.62),
    'hero': dict(pos=(0.030, 0.075), ha='left', va='top'),
    'over': dict(pos=(0.030, 0.075), ha='left', va='top'),
}


def layer_labels(view, layer, view_name):
    """The label naming the layer just added, with a leader to each solid arm."""
    spec = LAYER_LABEL.get(layer)
    if spec is None:
        return None
    where = dict(LAYER_POS.get(view_name,
                               dict(pos=(0.03, 0.08), ha='left', va='top')))
    v_frac = where.pop('v_frac', spec.get('v_frac', 0.0))
    near = set(view.get('near', N.NEAR_ARMS))
    arms = [a for a in range(len(N.ARMS)) if a not in near]
    # one leader per drawn OBJECT, not per arm: the plastics are two separate
    # bars, and a single line to the pair's centre lands in the gap between them
    anchors = [N.layer_anchor(layer, a, v_frac=v_frac, part=k)
               for a in arms for k in range(N.LAYER_PARTS.get(layer, 1))]
    return {'layer': (anchors, dict(text=spec['text'], scale=1.16,
                                    weight='semibold', **where))}


def spherical(elev_deg, azim_deg, r, angle, focal=(0.0, 0.0, 0.0),
              track_scale=1.0, near=(1, 2), size=None, arrow=0.0,
              arrow_y=-0.55, bare=False, cut=True):
    """A camera at (elevation, azimuth) about the beam axis.

    Azimuth is measured from +X towards +Z, so the quadrant the camera sits in
    names the two arms it looks *through* -- which is what ``NEAR_ARMS`` has to
    agree with.

    ``arrow`` and ``arrow_y`` place the beam-direction dart as fractions of the
    frame's own half-height.  Only the close-up act asks for one: on the build
    frames the column is already cut off at the target, which says "from below"
    on its own, and a dart small enough to stay inside a 17 mm column is a
    smudge on a 1.4 m frame.
    """
    e, a = np.radians(elev_deg), np.radians(azim_deg)
    pos = (focal[0] + r * np.cos(e) * np.cos(a),
           focal[1] + r * np.sin(e),
           focal[2] + r * np.cos(e) * np.sin(a))
    # World +Y projected into the image plane.  At a normal elevation this is
    # what VTK derives from up = (0, 1, 0) anyway, so nothing moves; looking
    # almost straight DOWN, (0, 1, 0) is parallel to the view direction and
    # the camera has no defined roll at all, which is what this avoids.
    f = -np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])
    up = np.array([0.0, 1.0, 0.0]) + np.sin(e) * f
    up = up / np.linalg.norm(up)
    return dict(pos=pos, focal=focal, up=tuple(up), angle=angle,
                track_scale=track_scale, near=near, size=size, bare=bare,
                half_h=r * np.tan(np.radians(angle) / 2), cut=cut,
                arrow=arrow, arrow_y=arrow_y)


def beam_arrow_kw(view):
    """The direction dart's geometry for one view, in world units.

    On the axis, inside the column.  Only its LENGTH comes from the frame; its
    girth is set from the beam radius by ``scenes_ntof.beam_arrow``, so it can
    never grow out through the column it is inside.
    """
    h = view.get('half_h')
    f = view.get('arrow', 0.0)
    if not h or not f:
        return None
    return dict(length=f * h,
                y_mid=view['focal'][1] + view.get('arrow_y', -0.55) * h)


VIEWS = {
    # three-quarter from above, over the corner between the two near arms
    'hero': spherical(47, 160, 3050, 26.0, focal=(25, -20, -25)),
    # Straight down (89 deg -- 90 would leave the camera with no defined roll).  The layers are stacked RADIALLY, so
    # from a three-quarter view the plastics hide behind the trigger wall in
    # front of them; from up here the stack opens out into four nested rings
    # and each layer's own deposit sits in its own.  Frames 8-9 use it.
    #
    # BARE: looking down the beam the four arms are seen through each other,
    # and the aluminium that reads perfectly well from three-quarters becomes
    # an opaque lid.  Frames and boards drop to a whisper here and only the
    # active volumes keep their colour.  No direction dart either -- straight
    # down it is foreshortened to a dot, and the beam is long established.
    #
    # cut=False: the capsule's cutaway plane contains the view direction, so
    # from up here it does not open the vessel, it deletes the half of it
    # nearest the bottom of the frame.  Whole, and whispered by BARE like every
    # other passive shell, is both honest and legible.
    'over': spherical(89, 160, 2850, 26.0, focal=(0, -20, 0), bare=True,
                      cut=False),
    # The capsule act: the same elevation and azimuth as 'hero', ten times
    # closer, so the cut between acts reads as a zoom rather than as a move.
    # The subject is a narrow vertical object, so it gets a portrait frame.
    # view_angle is the VERTICAL angle, so this crops width without changing
    # the scale of anything.  --size overrides it.
    'micro': spherical(47, 160, 300, 27.0, focal=(0, 6, 0),
                       track_scale=0.115, arrow=0.36, arrow_y=-0.81,
                       size=(1060, 1400)),
    # Halfway out: the four chambers closing in around the capsule, filling the
    # frame.  This is where the detector first appears, and where the frame
    # turns from portrait to the landscape shape the apparatus has.
    'close': spherical(47, 160, 1550, 26.0, focal=(0, -5, 0),
                       track_scale=0.52, size=(1500, 1400)),
    # steeper: the pinwheel and the layer depths together
    'high': spherical(48, 152, 2850, 26.5, focal=(20, -10, -20)),
    # low three-quarter, the "photograph" angle
    'low': spherical(24, 150, 3000, 26.5, focal=(20, -10, -20)),
    # along the beam from above: the pinwheel at its most legible
    'top': dict(pos=(0, 2900, 0), focal=(0, 0, 0), up=(0, 0, 1), angle=27.0,
                track_scale=1.0, near=(1, 2), cut=False),
}

# The sequence is in four acts, because the subject changes scale by a factor
# of fifty and then changes what it is trying to show:
#
#   micro   the target and the event in it -- 23 mm of vessel filling the frame
#   close   one frame, where the chambers arrive around it
#   hero    the apparatus, built up at one fixed camera
#   over    the same apparatus from above, so the OUTER layers are visible --
#           the plastics sit behind the trigger wall and a three-quarter view
#           cannot show a track reaching them
#
# Within each act the frames share a camera and a size exactly, so the layers
# appear to grow onto a still picture; the cuts between acts are deliberate and
# each lands on something the previous camera could not show.
#
# The fourth field is the layer the frame NAMES on the picture: each build
# frame carries the name of the layer it just added and nothing else, so the
# label moves outward with the build instead of accumulating.
_CORE = ('beam', 'capsule', 'neutron', 'pair')
_MM = _CORE + ('mm',)
STAGES = [
    ('capsule', ('beam', 'capsule'), 'micro', None),
    ('neutron', ('beam', 'capsule', 'neutron'), 'micro', None),
    ('pair', _CORE, 'micro', None),
    ('mm_near', _MM, 'close', 'mm'),
    ('mm', _MM, 'hero', 'mm'),
    ('sipm', _MM + ('sipm',), 'hero', 'sipm'),
    ('plastic', _MM + ('sipm', 'plastic'), 'hero', 'plastic'),
    ('plastic_top', _MM + ('sipm', 'plastic'), 'over', 'plastic'),
    ('full', N.PARTS, 'over', 'ls'),
]


def build(theme='light', size=(1500, 1400), ssaa=True, show=N.PARTS,
          event=None, track_scale=1.0, near=(1, 2), cut_normal=None,
          transparent=True, arrow_kw=None, bare=False):
    N.set_near_arms(near)
    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=25.0,
                       transparent=transparent)
    # This scene is layered translucent shells nested inside each other, which
    # is exactly the case VTK's back-to-front ordering gets wrong: without
    # depth peeling the liquid shows through the plastics that are in front of
    # it, in whichever order the meshes happened to be added.
    p.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
    N.build(p, show=show, event=event, theme=theme, beam_arrow_kw=arrow_kw,
            track_scale=track_scale, cut_normal=cut_normal, bare=bare)
    S.add_light_rig(p, CENTER, N.scene_scale(), theme=theme, shadows=False,
                    up='y')
    return p


def set_cam(p, view):
    p.camera.position = view['pos']
    p.camera.focal_point = view['focal']
    p.camera.up = view['up']
    p.camera.view_angle = view['angle']
    # VTK keeps the auto-framed clipping range, which cuts a manually placed
    # camera's scene away entirely
    p.renderer.reset_camera_clipping_range()


def render(out, view, theme='light', transparent=True, labels=None, **kw):
    # the capsule is cut open towards the camera, so the cut plane follows
    # whichever view is being rendered -- unless that view is looking down the
    # axis the plane contains, where the cut takes half the vessel away instead
    # of opening it (see VIEWS['over'])
    look = np.asarray(view['pos'], float) - np.asarray(view['focal'], float)
    cut = (look[0], 0.0, look[2]) if view.get('cut', True) else None
    p = build(theme=theme, track_scale=view.get('track_scale', 1.0),
              near=view.get('near', (1, 3)), transparent=transparent,
              arrow_kw=beam_arrow_kw(view), bare=view.get('bare', False),
              cut_normal=cut, **kw)
    set_cam(p, view)
    px = A.project(p, anchor_points(labels)) if labels else None
    S.finish(p, out, transparent=transparent)
    if labels:
        overlay(out, px, labels, theme=theme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light',
                    choices=['light', 'dark', 'both'])
    ap.add_argument('--view', default=None, choices=sorted(VIEWS),
                    help='override the camera each stage asks for')
    ap.add_argument('--only', default=None,
                    help='comma-separated stage tags, e.g. "full" or '
                         '"capsule,mm"')
    ap.add_argument('--size', nargs=2, type=int, default=[1500, 1400],
                    help='the frame is close to the projected shape of the '
                         'full setup, so the object fills it')
    ap.add_argument('--opaque', action='store_true',
                    help='keep the theme background instead of writing the '
                         'frames with an alpha channel (they are transparent '
                         'by default, so a slide of any colour shows through)')
    ap.add_argument('--no-labels', action='store_true',
                    help='drop the in-image type: the capsule frame\'s '
                         'wall-thickness leaders and the layer name each build '
                         'frame carries (both are drawn by default)')
    ap.add_argument('--draft', action='store_true')
    args = ap.parse_args()
    args.size_given = any(a == '--size' for a in sys.argv)

    size = (620, 580) if args.draft else tuple(args.size)
    themes = ['light', 'dark'] if args.theme == 'both' else [args.theme]
    want = None if args.only is None else set(args.only.split(','))

    event = N.load_event()
    prov = event['provenance']
    print(f'event: pair #{prov["pair_event"]} + neutron '
          f'#{prov["neutron_event"]}, spliced with a '
          f'{prov["vertex_residual_mm"]} mm residual')

    tag = '' if args.view is None else f'_{args.view}'
    for theme in themes:
        for i, (name, show, view_name, layer) in enumerate(STAGES, start=1):
            if want and name not in want:
                continue
            vname = args.view or view_name
            view = VIEWS[vname]
            sz = size if args.size_given or args.draft \
                else (view.get('size') or size)
            out = os.path.join(FIG, f'ntof_build_{i}_{name}{tag}_{theme}.png')
            if args.no_labels:
                lab = None
            elif name == 'capsule':
                lab = capsule_labels(view)
            else:
                lab = layer_labels(view, layer, vname)
            print(f'{name}  [{sz[0]}x{sz[1]}]  {args.view or view_name}'
                  f'{"  +labels" if lab else ""}')
            render(out, view, theme=theme, size=sz, ssaa=not args.draft,
                   show=show, event=event, transparent=not args.opaque,
                   labels=lab)


if __name__ == '__main__':
    main()
