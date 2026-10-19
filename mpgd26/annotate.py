#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate.py -- put labels on a rendered scene, in 2-D, after the fact.

VTK's own 3-D text is hard to place well and reads badly at print size.  The
approach here is instead:

  1. render the scene to a raster,
  2. project the 3-D anchor points through the *same* camera to pixels
     (``project``, called while the plotter is still alive),
  3. lay the type out over the raster with matplotlib (``compose``), where
     leader lines, fonts and a caption block are under full control.

The result is a figure that can be saved as both PNG and vector-text PDF.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.patheffects as pe      # noqa: E402
import numpy as np                       # noqa: E402
from PIL import Image                    # noqa: E402

FONT = {'family': 'DejaVu Sans'}
TEXT_FRAC = 0.0155      # label cap-height as a fraction of the image width


def project(plotter, anchors):
    """3-D world points -> (x, y) pixels in the screenshot, y measured from the
    TOP (matplotlib image convention).

    Must be called after the camera is set and before the plotter is closed.
    """
    from pyvista import _vtk

    plotter.render()
    ren = plotter.renderer
    w, h = plotter.window_size
    # the plotter renders supersampled; the delivered raster is 1/k of that
    k = getattr(plotter, '_mpgd_scale', 1.0)
    out = {}
    for name, pt in anchors.items():
        c = _vtk.vtkCoordinate()
        c.SetCoordinateSystemToWorld()
        c.SetValue(float(pt[0]), float(pt[1]), float(pt[2]))
        x, y = c.GetComputedDoubleDisplayValue(ren)
        out[name] = (x / k, (h - y) / k)
    return out


def column_labels(px, items, size, side='right', x_frac=1.015, top=0.04,
                  bottom=0.98, min_gap=None, color=None, gutter=0.30):
    """Lay labels out in a non-overlapping column in ``compose``'s gutter.

    ``items`` is a list of (anchor_key, text) in the order they should appear
    top to bottom.  Each label keeps its own leader line back to the projected
    anchor, but the text itself is snapped to a tidy column and pushed apart
    until nothing overlaps -- which is what stops a six-station telescope's
    labels from piling up on each other wherever the camera happens to be.

    ``x_frac`` is in units of the RENDER width, so > 1 puts the column in the
    gutter ``compose`` adds.  ``min_gap`` defaults to the height the tallest
    label actually needs, derived from its line count and the canvas aspect --
    hand-tuning it per figure is exactly the thing that breaks when a camera
    moves.
    """
    w, h = size
    if min_gap is None:
        lines = max(t.count('\n') + 1 for _, t in items) if items else 1
        canvas_w = w * (1 + gutter)
        min_gap = (lines * 1.34 + 0.75) * TEXT_FRAC * canvas_w / h

    ys = []
    for key, _ in items:
        ys.append(px[key][1] / h if key in px else 0.5)

    # sort-stable push-apart in normalised units
    ys = list(np.clip(ys, top, bottom))
    for _ in range(400):
        moved = False
        for i in range(len(ys) - 1):
            d = ys[i + 1] - ys[i]
            if d < min_gap:
                shift = (min_gap - d) / 2
                ys[i] -= shift
                ys[i + 1] += shift
                moved = True
        ys = list(np.clip(ys, top, bottom))
        if not moved:
            break

    out = []
    for (key, text), y in zip(items, ys):
        if key not in px:
            continue
        ax, ay = px[key]
        tx = x_frac * w if side == 'right' else (1 - x_frac) * w
        out.append(dict(xy=(ax, ay), dx=tx - ax, dy=y * h - ay, text=text,
                        ha='left' if side == 'right' else 'right',
                        color=color))
    return out


def side_labels(px, items, size, x_pad=0.018, top=0.032, bottom=0.972,
                min_gap=None):
    """Labels laid out ON the render's own background, in two columns.

    ``column_labels`` puts every label in a gutter that ``compose`` adds beside
    the render.  That is the right answer for a wide scene and the wrong one for
    a tall narrow one: the gutter is dead space down the whole height, and the
    picture has to be made smaller to pay for it.  This lays the labels out
    *inside* the image instead, left and right of the subject -- which only works
    because a 20 m beam line drawn 1:1 leaves the sides of the frame empty, so
    render wide enough that it does.

    ``items`` is ``(anchor_key, text, side)`` with ``side`` in {'left', 'right'},
    in the order they should appear top to bottom.  Each side is pushed apart
    independently, the same way ``column_labels`` does it.  Returns an ordered
    ``{key: label_dict}`` rather than a list, so a caller building a BUILD-UP can
    solve the whole set once and then select by key -- which is what stops a
    label moving between frames.
    """
    w, h = size
    out = {}
    for want in ('left', 'right'):
        grp = [(k, t) for k, t, s in items if s == want and k in px]
        if not grp:
            continue
        gap = min_gap
        if gap is None:
            lines = max(t.count('\n') + 1 for _, t in grp)
            gap = (lines * 1.34 + 0.60) * TEXT_FRAC * w / h
        ys = list(np.clip([px[k][1] / h for k, _ in grp], top, bottom))
        for _ in range(400):
            moved = False
            for i in range(len(ys) - 1):
                d = ys[i + 1] - ys[i]
                if d < gap:
                    shift = (gap - d) / 2
                    ys[i] -= shift
                    ys[i + 1] += shift
                    moved = True
            ys = list(np.clip(ys, top, bottom))
            if not moved:
                break
        for (key, text), y in zip(grp, ys):
            ax, ay = px[key]
            tx = x_pad * w if want == 'left' else (1 - x_pad) * w
            out[key] = dict(xy=(ax, ay), dx=tx - ax, dy=y * h - ay, text=text,
                            ha='left' if want == 'left' else 'right')
    return out


def compose(png, labels, out_base, title=None, subtitle=None, caption=None,
            theme='light', dpi=300, gutter=0.30, header=0.13, footer=0.16):
    """Lay the render out on a titled canvas and write ``out_base``.png/.pdf.

    The render is NOT covered by the type: the canvas is grown by a right-hand
    ``gutter`` for the label column and by a ``header``/``footer`` band for the
    title and caption (all as fractions of the render's own size).  Leader lines
    then run from the anchor, over the render, into clean space -- which is what
    keeps a six-label figure readable at slide size.

    ``labels`` is a list of dicts in RENDER pixel coordinates:
        {'xy': (x, y), 'text': ..., 'dx': ..., 'dy': ..., 'ha': ...}
    """
    img = np.asarray(Image.open(png).convert('RGB'))
    h, w = img.shape[:2]

    gut = int(round(gutter * w))
    head = int(round(header * h))
    foot = int(round(footer * h))
    W, H = w + gut, h + head + foot

    ink = '#f2f5f9' if theme == 'dark' else '#141b24'
    muted = '#9aa7b6' if theme == 'dark' else '#5d6874'
    halo = '#0a0d13' if theme == 'dark' else '#ffffff'
    page = '#0a0d13' if theme == 'dark' else '#ffffff'

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=page)

    imax = fig.add_axes([0, foot / H, w / W, h / H])
    imax.imshow(img, interpolation='lanczos')
    imax.axis('off')

    # one overlay axes in canvas-pixel coordinates for everything else
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    ax.patch.set_alpha(0)

    # Type is sized as a fraction of the canvas WIDTH, converted to points: a
    # fixed point size would be tiny on a 2800 px hero and enormous on a draft,
    # since the figure's physical size is width/dpi inches.
    fs = TEXT_FRAC * W * 72.0 / dpi

    for L in labels:
        x, y = L['xy'][0], L['xy'][1] + head
        tx, ty = x + L.get('dx', 0), y + L.get('dy', 0)
        ax.annotate(
            L['text'], xy=(x, y), xytext=(tx, ty),
            ha=L.get('ha', 'left'), va=L.get('va', 'center'),
            fontsize=L.get('size', fs), color=L.get('color') or ink,
            fontweight=L.get('weight', 'medium'),
            arrowprops=dict(arrowstyle='-', color=L.get('lead', muted),
                            lw=fs * 0.06, alpha=0.9,
                            shrinkA=fs * 0.4, shrinkB=fs * 0.3,
                            connectionstyle=L.get('conn', 'arc3,rad=0')),
            path_effects=[pe.withStroke(linewidth=fs * 0.34, foreground=halo,
                                        alpha=0.9)],
            zorder=5, **FONT)

    pad = 0.030 * W
    if title:
        ax.text(pad, head * 0.40, title, ha='left', va='center',
                fontsize=fs * 1.95, color=ink, fontweight='bold', **FONT)
    if subtitle:
        ax.text(pad, head * 0.78, subtitle, ha='left', va='center',
                fontsize=fs * 0.95, color=muted, **FONT)
    if caption:
        # wrap to the canvas rather than trusting hand-inserted newlines: the
        # same caption is used at draft and full resolution
        cfs = fs * 0.66
        char_px = cfs * dpi / 72.0 * 0.52          # DejaVu Sans mean advance
        ncols = max(40, int((W - 2 * pad) / char_px))
        wrapped = '\n'.join(textwrap.fill(par, ncols)
                            for par in caption.replace('\n', ' ').split('\n\n'))
        ax.text(pad, H - foot * 0.5, wrapped, ha='left', va='center',
                fontsize=cfs, color=muted, linespacing=1.6, **FONT)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + '.png', dpi=dpi, facecolor=page)
    fig.savefig(out_base + '.pdf', facecolor=page)
    plt.close(fig)
    print(f'  wrote {out_base}.png/.pdf')
