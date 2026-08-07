#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
style.py -- palette, PBR material presets, lighting rigs and the render harness.

Everything visual that is shared between the two scenes lives here so the SPS
and cosmic-bench figures read as one set: same materials for the same physical
object, same light rig, same background, same output pipeline.

Rendering is VTK/PyVista with physically-based shading:
  * a procedurally generated studio cubemap drives the PBR reflections (metals
    are otherwise flat and lifeless in VTK),
  * a four-light rig (key / fill / rim / bounce),
  * screen-space ambient occlusion to seat objects against each other,
  * super-sampled anti-aliasing at 2x the delivered resolution.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pyvista as pv
from PIL import Image

pv.OFF_SCREEN = True

SHADOW_MAP_RES = 4096

# direction from the scene centre towards the key light, canonical frame
KEY_DIR = (0.95, 0.95, 0.50)

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
THEMES = {
    'light': dict(
        bg_bottom='#e9ecf1', bg_top='#fbfcfe',
        ink='#1b2430', muted='#6a7583',
        floor='#d4d9e0', floor_opacity=1.0,
        cubemap='studio',
    ),
    'dark': dict(
        bg_bottom='#0a0d13', bg_top='#1c2431',
        ink='#f2f5f9', muted='#9aa7b6',
        floor='#141a24', floor_opacity=1.0,
        cubemap='studio_dark',
    ),
}

# Physical objects -> a fixed colour, used identically in both scenes.
COL = {
    'pcb':        '#1a5b50',   # FR4 readout board, dark solder-mask green
    'pcb_urwell': '#25617f',   # uRWELL board, distinguishable blue-green
    'copper':     '#d18a44',   # metallised strips / pads (instrumented)
    'copper_dead': '#7d6247',  # metallised but not read out
    'copper_hot': '#f0b660',   # highlighted readout region
    'alu':        '#ccd4dd',   # aluminium frames and rails
    'alu_dark':   '#98a1ac',   # bench structure
    'gas':        '#6fd0e8',   # drift volume
    'mesh':       '#9aa3ad',   # micromesh / amplification plane
    'scint':      '#63d2ff',   # plastic scintillator paddle
    'guide':      '#23272c',   # light guide: perspex, wrapped in black tape,
                               # which is what makes it read as a guide and not
                               # as more scintillator
    'pmt':        '#3a4049',   # photomultiplier housing
    'track_mu':   '#ff4f36',   # charged-particle track (both scenes)
    'track_beam': '#ff4f36',   # H4 beam particle -- same red, one visual language
    'm3':         '#2f6fa8',   # M3 reference chamber accent
    'm3_pcb':     '#2a4f7a',   # M3 readout board, distinct from the MX17 green
    'mx17':       '#8a3f8f',   # MX17 chamber accent
    'p2':         '#1f8a70',   # P2 BASKET accent
    'beam_env':   '#ffd166',   # beam envelope / trigger slab
}

# Material presets.
#
# Only the genuinely metallic parts use VTK's PBR path.  VTK drives PBR almost
# entirely from the environment cubemap (image-based lighting), which is what
# makes aluminium and copper read as metal -- but it also swamps the directional
# lights, so a scene rendered entirely in PBR comes out flat and casts no
# visible shadows.  Everything non-metallic therefore uses classic Phong, where
# ambient/diffuse/specular are under direct control and the light rig (and its
# shadow map) actually does the modelling.
MAT = {
    # --- PBR: metals, lit by the studio cubemap ---
    'copper':   dict(pbr=True, metallic=0.82, roughness=0.36),
    'alu':      dict(pbr=True, metallic=0.52, roughness=0.32),
    'mesh':     dict(pbr=True, metallic=0.90, roughness=0.24),
    # --- Phong: everything else ---
    'pcb':      dict(pbr=False, ambient=0.34, diffuse=0.90, specular=0.24,
                     specular_power=28),
    'alu_matte': dict(pbr=False, ambient=0.34, diffuse=0.86, specular=0.32,
                      specular_power=22),
    'plastic':  dict(pbr=False, ambient=0.32, diffuse=0.88, specular=0.35,
                     specular_power=40),
    'gas':      dict(pbr=False, opacity=0.10, specular=0.25, specular_power=20,
                     ambient=0.40, diffuse=0.55),
    'scint':    dict(pbr=False, opacity=0.30, specular=0.9, specular_power=60,
                     ambient=0.30, diffuse=0.70),
    'glow':     dict(pbr=False, ambient=0.85, diffuse=0.45, specular=0.6,
                     specular_power=40),
    'envelope': dict(pbr=False, opacity=0.10, ambient=0.55, diffuse=0.4,
                     specular=0.0),
}


def illumination_cmap():
    """Copper -> amber -> white: shows the measured beam spot *on* the pads
    without the zero end going black the way `inferno` does."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'padspot', ['#d18a44', '#d78b3e', '#f0993a', '#ffb545', '#ffd77a',
                    '#fff3d2', '#fffdf6'])


def microtpc_cmap(theme='light'):
    """Drift-time colour scale that survives its own background.

    ``plasma`` runs to a very pale yellow, which is fine on black and almost
    invisible on white -- and the pale end is the SHALLOW end of the gap, so on
    a light theme the figure loses exactly the clusters nearest the cathode.
    The light theme therefore uses a truncated plasma that stops before the
    pale end; the dark theme drops the near-black start for the same reason.
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as _plt
    lo, hi = (0.0, 0.78) if theme == 'light' else (0.14, 1.0)
    base = _plt.get_cmap('plasma')
    return LinearSegmentedColormap.from_list(
        f'microtpc_{theme}', base(np.linspace(lo, hi, 256)))


def mat(name, color=None, **over):
    """Material preset merged with a colour and per-call overrides."""
    kw = dict(MAT[name])
    if color is not None:
        kw['color'] = color
    kw.update(over)
    return kw


# --------------------------------------------------------------------------- #
# Procedural studio cubemap (drives the PBR reflections)
# --------------------------------------------------------------------------- #
_CUBEMAP_CACHE = {}


def _face(size, kind, variant):
    """One 'studio' cubemap face: gradient sky, horizon, floor, soft key blob.

    The walls carry a bright key and a dimmer fill on OPPOSITE sides, and the
    two remaining walls get a weak wash.  Concentrating all the light on one
    wall makes a metal's brightness depend strongly on its azimuth, so two
    identical chambers would render quite differently purely because a measured
    alignment turns one of them by 90 deg.
    """
    y, x = np.mgrid[0:size, 0:size] / (size - 1.0)

    if variant == 'studio_dark':
        sky_hi, sky_lo = np.array([0.62, 0.68, 0.78]), np.array([0.20, 0.23, 0.29])
        flr_hi, flr_lo = np.array([0.17, 0.18, 0.22]), np.array([0.07, 0.08, 0.10])
        key_gain = 3.0
    else:
        sky_hi, sky_lo = np.array([1.00, 1.00, 1.00]), np.array([0.55, 0.59, 0.66])
        flr_hi, flr_lo = np.array([0.38, 0.40, 0.44]), np.array([0.20, 0.21, 0.24])
        key_gain = 2.4

    if kind == 'py':                      # +Y, the ceiling: broad soft light
        r = np.hypot(x - 0.42, y - 0.40)
        img = sky_hi[None, None, :] * np.ones((size, size, 1))
        img = img * (0.72 + 0.62 * np.exp(-(r / 0.34) ** 2))[..., None]
        img *= 1.0 if variant == 'studio' else 0.9
    elif kind == 'ny':                    # -Y, the floor
        img = flr_lo[None, None, :] * np.ones((size, size, 1))
    else:                                 # the four walls: sky -> horizon -> floor
        t = y[..., None]
        top = sky_hi + (sky_lo - sky_hi) * np.clip(t / 0.5, 0, 1)
        bot = flr_hi + (flr_lo - flr_hi) * np.clip((t - 0.5) / 0.5, 0, 1)
        img = np.where(t < 0.5, top, bot)
        if kind == 'px':                  # one wall carries the key softbox
            r = np.hypot(x - 0.55, y - 0.30)
            img = img * (1.0 + key_gain * np.exp(-(r / 0.20) ** 2))[..., None]
        if kind == 'nx':                  # a dimmer fill box opposite the key
            r = np.hypot(x - 0.40, y - 0.34)
            img = img * (1.0 + 0.45 * key_gain * np.exp(-(r / 0.26) ** 2))[..., None]
        if kind in ('pz', 'nz'):          # weak wash so azimuth matters less
            r = np.hypot(x - 0.50, y - 0.33)
            img = img * (1.0 + 0.28 * key_gain * np.exp(-(r / 0.34) ** 2))[..., None]

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def studio_cubemap(variant='studio', size=512):
    """Build (once) and return a pv.Texture cubemap for PBR image-based lighting."""
    if variant in _CUBEMAP_CACHE:
        return _CUBEMAP_CACHE[variant]
    tmp = tempfile.mkdtemp(prefix='mpgd_cubemap_')
    order = ['px', 'nx', 'py', 'ny', 'pz', 'nz']
    paths = []
    for kind in order:
        p = os.path.join(tmp, f'{kind}.png')
        Image.fromarray(_face(size, kind, variant)).save(p)
        paths.append(p)
    tex = pv.cubemap_from_filenames(image_paths=paths)
    _CUBEMAP_CACHE[variant] = tex
    return tex


# --------------------------------------------------------------------------- #
# Plotter factory
# --------------------------------------------------------------------------- #
SS_MAX_DIM = 3800   # see supersample_factor()
SSAO_KERNEL = 128   # see make_plotter()


def supersample_factor(size, ssaa=True, max_dim=SS_MAX_DIM):
    """How much bigger than the delivered size to render.

    VTK's own SSAA pass and the SSAO pass each allocate full-resolution
    off-screen buffers, and *together* they fail -- silently, returning an
    all-black frame -- once the internal buffer passes roughly 3800 px on a
    side.  With SSAA on, that ceiling is hit at a delivered size of only about
    1900 px.  So the supersampling is done here instead: render at k x the
    delivered size with SSAO on and VTK's SSAA off, then Lanczos-downsample.
    Same anti-aliasing, no silent black frames, and k is chosen to keep the
    buffer under the ceiling at any output size.
    """
    if not ssaa:
        return 1.0
    return max(1.0, min(2.0, max_dim / max(size)))


def make_plotter(theme='light', size=(2400, 1600), ssaa=True, ssao_radius=None):
    """A plotter with the shared background, cubemap and (optionally) SSAO.

    SSAO has to be installed *before* the shadow pass -- both rewrite the
    renderer's pass chain, and whichever is enabled last wins -- so it is set
    up here and ``add_light_rig`` turns shadows on afterwards.

    The plotter carries ``_mpgd_scale`` (the supersample factor) and
    ``_mpgd_out_size``; ``finish`` and ``annotate.project`` both need them.
    """
    th = THEMES[theme]
    k = supersample_factor(size, ssaa)
    win = [int(round(size[0] * k)), int(round(size[1] * k))]

    p = pv.Plotter(off_screen=True, window_size=win, lighting='none')
    # PyVista blocks new public attributes on Plotter; private ones are fine
    p._mpgd_scale = k
    p._mpgd_out_size = tuple(size)
    p.set_background(th['bg_bottom'], top=th['bg_top'])
    p.set_environment_texture(studio_cubemap(th['cubemap']), is_srgb=True)
    if ssao_radius:
        try:
            # kernel_size is the real size limit here, not the window: VTK's
            # SSAO pass with 256 samples returns an all-black frame once the
            # buffer passes ~3000 px on a side.  128 is visually identical and
            # survives everything we render.
            p.enable_ssao(radius=ssao_radius, bias=ssao_radius * 0.01,
                          kernel_size=SSAO_KERNEL, blur=True)
        except Exception:                                    # pragma: no cover
            pass
    return p


def add_light_rig(p, center, scale, theme='light', shadows=True, up='y'):
    """Four-light rig sized to the scene.

    ``center`` is the scene centroid and ``scale`` its characteristic radius;
    the lights sit on a sphere a few times that radius, so the same rig
    transfers unchanged between the two very differently sized scenes.
    ``up`` names the scene's vertical axis ('y' on the SPS rail, 'z' on the
    cosmic bench) so "key from above" means the same thing in both.
    """
    c = np.asarray(center, float)
    # the dark theme gets a BRIGHTER cubemap but WEAKER direct light: the
    # metals then read from their reflections instead of blowing out white
    boost = 1.0 if theme == 'light' else 0.80
    # rig in a canonical frame (x right, y up, z toward the viewer)
    rig = [
        (KEY_DIR, 1.05 * boost, '#fff4e8', True),                # key, warm
        ((-0.85, +0.25, +0.80), 0.46 * boost, '#dcebff', False),  # fill, cool
        ((-0.20, +0.40, -1.00), 0.34 * boost, '#ffffff', False),  # rim / back
        ((0.00, -1.00, +0.15), 0.12 * boost, '#c9d6e6', False),  # bounce
    ]
    # map canonical (x, y, z) onto the scene's axes
    if up == 'y':
        def rot(d):
            return np.array([d[0], d[1], d[2]])
    else:                       # z up: canonical +y -> scene +z, +z -> scene -y
        def rot(d):
            return np.array([d[0], -d[2], d[1]])

    for d, inten, col, is_key in rig:
        d = rot(np.asarray(d, float))
        d /= np.linalg.norm(d)
        lt = pv.Light(position=c + d * scale * 4.5, focal_point=c,
                      color=col, intensity=inten)
        lt.positional = False          # directional: VTK's shadow map needs it
        if is_key:
            lt.shadow_attenuation = 0.10
        p.add_light(lt)
    if shadows:                       # VTK shadow maps: see meshes.ground_shadow
        try:
            p.enable_shadows()
            sm = p.renderer._render_passes._shadow_map_pass
            if sm is not None:
                sm.GetShadowMapBakerPass().SetResolution(SHADOW_MAP_RES)
        except Exception:                                    # pragma: no cover
            pass
    return p


def key_direction(up='y'):
    """The unit vector from the scene centre towards the key light -- what
    ``meshes.ground_shadow`` needs to place a cast shadow consistent with the
    rig above."""
    d = np.array(KEY_DIR, float)
    if up != 'y':
        d = np.array([d[0], -d[2], d[1]])
    return d / np.linalg.norm(d)


def add_ground_shadows(p, outlines, plane_value, plane_axis='y', up='y',
                       theme='light', opacity=0.055):
    """Cast every outline in ``outlines`` onto the ground plane."""
    import meshes as _M
    col = '#000000' if theme == 'light' else '#000000'
    d = key_direction(up)
    for o in outlines:
        for poly, w in _M.ground_shadow(o, d, plane_value,
                                        plane_axis=plane_axis):
            p.add_mesh(poly, color=col, opacity=opacity * w, pbr=False,
                       ambient=1.0, diffuse=0.0, specular=0.0,
                       lighting=False)
    return p


def finish(p, path, transparent=False):
    """Render, downsample the supersampled frame and write ``path``.

    Also guards against the silent all-black frame VTK produces when a render
    pass runs out of buffer: better to fail loudly here than to ship a black
    figure.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.asarray(p.screenshot(transparent_background=transparent))
    p.close()

    if img.std() < 1e-6:
        raise RuntimeError(
            f'{path}: VTK returned a uniform frame -- a render pass almost '
            f'certainly overflowed at window size {img.shape[1]}x{img.shape[0]}')

    out_size = getattr(p, '_mpgd_out_size', None)
    if out_size and (img.shape[1], img.shape[0]) != tuple(out_size):
        img = np.asarray(Image.fromarray(img).resize(
            tuple(out_size), Image.LANCZOS))
    Image.fromarray(img).save(path)
    print(f'  wrote {path}')
