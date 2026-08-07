#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_anim.py -- turntables and build-up sequences.

    ../.venv/bin/python make_anim.py                    # everything
    ../.venv/bin/python make_anim.py --only turn_sps
    ../.venv/bin/python make_anim.py --draft            # small, few frames

Two products, both from the same scenes as the still figures:

  **Turntables** -- the camera orbits the scene's vertical axis at the
  elevation of the still's hero view.  Written as MP4 (for Keynote / Beamer)
  and GIF (for anywhere that won't take video), plus the frames themselves.

  **Build-up sequences** -- a fixed camera, with the setup assembled one
  element at a time.  Written as numbered stills (drop them on successive
  slides and it animates itself) and as a slow MP4.

Turntables loop seamlessly: the last frame is one step short of a full turn.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import style as S             # noqa: E402
import make_sps as MS         # noqa: E402
import make_bench as MB       # noqa: E402
import make_chamber as MC     # noqa: E402
import scenes_chamber as C    # noqa: E402

ANIM = os.path.join(HERE, 'animations')


# --------------------------------------------------------------------------- #
def orbit(view, frac, up='y'):
    """The hero camera, rotated by ``frac`` of a turn about the vertical axis.

    Elevation, distance and focal point are taken straight from the still's
    preset, so a turntable frame at frac = 0 is the still.
    """
    pos = np.array(view['pos'], float)
    foc = np.array(view['focal'], float)
    d = pos - foc
    a = 2 * math.pi * frac
    c, s = math.cos(a), math.sin(a)
    if up == 'y':                      # rotate in the x-z plane
        d = np.array([c * d[0] + s * d[2], d[1], -s * d[0] + c * d[2]])
    else:                              # z up: rotate in the x-y plane
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
    return foc + d


def write_video(frames, base, fps=25, gif_fps=12, gif_width=900):
    """MP4 + GIF from a list of RGB arrays."""
    import imageio.v2 as iio

    # MP4: even dimensions required by the H.264 encoder
    h, w = frames[0].shape[:2]
    crop = [f[:h - h % 2, :w - w % 2] for f in frames]
    iio.mimwrite(base + '.mp4', crop, fps=fps, quality=8,
                 macro_block_size=1)
    print(f'  wrote {base}.mp4  ({len(frames)} frames)')

    step = max(1, round(fps / gif_fps))
    small = []
    for f in frames[::step]:
        im = Image.fromarray(f)
        im = im.resize((gif_width, round(gif_width * im.height / im.width)),
                       Image.LANCZOS)
        small.append(im.convert('P', palette=Image.ADAPTIVE, colors=192))
    small[0].save(base + '.gif', save_all=True, append_images=small[1:],
                  duration=int(1000 / gif_fps), loop=0, optimize=True)
    print(f'  wrote {base}.gif  ({len(small)} frames)')


def grab(p, close=True):
    """Screenshot + downsample, mirroring style.finish without writing a file.

    The explicit ``render()`` is load-bearing: ``screenshot()`` hands back the
    already-rendered buffer, so on a plotter that is being reused across frames
    a camera move has no effect until something re-renders.  The still figures
    never hit this because ``annotate.project`` renders before projecting; a
    turntable that only moves the camera produces 90 identical frames without
    it.
    """
    p.render()
    img = np.asarray(p.screenshot())
    out = getattr(p, '_mpgd_out_size', None)
    if close:
        p.close()
    if img.std() < 1e-6:
        raise RuntimeError('VTK returned a uniform frame')
    if out and (img.shape[1], img.shape[0]) != tuple(out):
        img = np.asarray(Image.fromarray(img).resize(tuple(out), Image.LANCZOS))
    return img


def set_cam(p, pos, view, clip=True):
    p.camera.position = tuple(pos)
    p.camera.focal_point = view['focal']
    p.camera.up = view['up']
    p.camera.view_angle = view['angle']
    if clip:
        p.renderer.reset_camera_clipping_range()


# --------------------------------------------------------------------------- #
def turntable(name, builder, view, up, n_frames, size, ssaa, theme,
              save_frames=True):
    frame_dir = os.path.join(ANIM, f'{name}_frames')
    if save_frames:
        os.makedirs(frame_dir, exist_ok=True)
    # Build the scene ONCE and only move the camera.  Rebuilding it per frame
    # costs seconds each (1280 P2 pads plus every strip quad) for no change in
    # what is drawn, and turns a 90-frame turntable into a coffee break.
    p = builder(theme=theme, size=size, ssaa=ssaa)
    frames = []
    for i in range(n_frames):
        set_cam(p, orbit(view, i / n_frames, up=up), view)
        img = grab(p, close=False)
        frames.append(img)
        if save_frames:
            Image.fromarray(img).save(
                os.path.join(frame_dir, f'{name}_{i:03d}.png'))
        print(f'    frame {i + 1}/{n_frames}', end='\r', flush=True)
    p.close()
    print(' ' * 30, end='\r')
    write_video(frames, os.path.join(ANIM, name))


def buildup(name, builder, stages, view, size, ssaa, theme, hold=18):
    """Numbered stills plus a slow MP4 that holds on each stage."""
    os.makedirs(ANIM, exist_ok=True)
    frames = []
    for i, (tag, show) in enumerate(stages, start=1):
        p = builder(theme=theme, size=size, ssaa=ssaa, show=show)
        set_cam(p, view['pos'], view)
        img = grab(p)
        out = os.path.join(ANIM, f'{name}_{i}_{tag}.png')
        Image.fromarray(img).save(out)
        print(f'  wrote {out}')
        frames.extend([img] * hold)
    write_video(frames, os.path.join(ANIM, name), fps=18, gif_fps=6)


# --------------------------------------------------------------------------- #
def _sps(theme, size, ssaa, show=MS.PARTS):
    return MS.build(theme=theme, size=size, ssaa=ssaa, mx17=False, show=show)[0]


def _bench(slots):
    def f(theme, size, ssaa, show=MB.PARTS):
        return MB.build(theme=theme, size=size, ssaa=ssaa, slots=slots,
                        show=show)[0]
    return f


def _chamber(theme, size, ssaa, show=None):
    p = S.make_plotter(theme=theme, size=size, ssaa=ssaa, ssao_radius=6.0)
    C.build(p)
    S.add_light_rig(p, np.array([0, 0, 56]), 70.0, theme=theme, shadows=False,
                    up='z')
    return p


SPS_STAGES = [
    ('table', ('table',)),
    ('urwell', ('table', 'urwell')),
    ('p2', ('table', 'urwell', 'p2')),
    ('beam', ('table', 'urwell', 'p2', 'tracks')),
]

BENCH_STAGES = [
    ('rack', ('structure',)),
    ('trigger', ('structure', 'scint')),
    ('reference', ('structure', 'scint', 'm3')),
    ('chambers', ('structure', 'scint', 'm3', 'dut')),
    ('muons', ('structure', 'scint', 'm3', 'dut', 'tracks')),
]

JOBS = {
    'turn_sps':      dict(kind='turn', builder=_sps, view=MS.VIEWS['hero'],
                          up='y', size=(1600, 1000)),
    'turn_bench':    dict(kind='turn', builder=_bench(('mx17', 'mx17')),
                          view=MB.VIEWS['hero'], up='z', size=(1100, 1400)),
    'turn_bench_p2': dict(kind='turn', builder=_bench(('p2', 'p2')),
                          view=MB.VIEWS['hero'], up='z', size=(1100, 1400)),
    'turn_chamber':  dict(kind='turn', builder=_chamber, view=MC.VIEW,
                          up='z', size=(1000, 1300)),
    'build_sps':     dict(kind='build', builder=_sps, view=MS.VIEWS['hero'],
                          stages=SPS_STAGES, size=(1900, 1200)),
    'build_bench':   dict(kind='build', builder=_bench(('mx17', 'mx17')),
                          view=MB.VIEWS['hero'], stages=BENCH_STAGES,
                          size=(1300, 1650)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theme', default='light', choices=['light', 'dark'])
    ap.add_argument('--only', default=None,
                    help='comma-separated subset of ' + ','.join(JOBS))
    ap.add_argument('--frames', type=int, default=90,
                    help='turntable frames per full turn')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--no-frames', action='store_true',
                    help='skip writing the individual turntable frames')
    args = ap.parse_args()

    os.makedirs(ANIM, exist_ok=True)
    names = list(JOBS) if args.only is None else args.only.split(',')
    n = 24 if args.draft else args.frames

    for name in names:
        j = JOBS[name]
        size = tuple(int(x * 0.5) for x in j['size']) if args.draft \
            else j['size']
        print(f'{name}  [{size[0]}x{size[1]}]')
        if j['kind'] == 'turn':
            turntable(name, j['builder'], j['view'], j['up'], n, size,
                      not args.draft, args.theme,
                      save_frames=not args.no_frames)
        else:
            buildup(name, j['builder'], j['stages'], j['view'], size,
                    not args.draft, args.theme)


if __name__ == '__main__':
    main()
