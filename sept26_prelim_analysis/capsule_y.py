#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capsule_y.py -- where the He-3 capsule sits along the beam, by ray tracing the
trigger's acceptance in v.

READ FIRST (2026-09-14, after the campaign pass).  This module was built to turn
the band crossing into a fitted height, and it is NOT that.  The acceptance fit
reads the SHAPE of the v distribution, which a chamber's own eff(v), dead strips
and noisy columns dominate: per 100 % efficiency tilt it moves 32-65 mm against
0.3-3.7 mm for the band crossing (:func:`sensitivity`).  So:

  * quote the HEIGHT from the band crossing (``band_mm``; ~+33 +- 3 mm, A and C);
  * use the acceptance fit for what only it can do -- exclude a COMMON v-origin
    error (its gain to a frame offset is ~2.1, the band's is 1);
  * never read the per-chamber output of :func:`separate` (``delta``,
    ``y_source``) as an alignment or a height.  It absorbs eff(v); the first
    reading of this pass called it an 11 mm A-C misalignment, and it was not.

The long-form account is the Outcome box of ``HANDOFF_CAPSULE_Y.md`` and the
"mistake" section of the report.

THE QUESTION (HANDOFF_CAPSULE_Y.md).  Three chambers' scale-free y band
crossings put the source at ~+31 mm along the beam where the CAD polycone puts
+0.8, they agree to ~3 mm, and the capsule's height was never surveyed.  The
band crossing is an ENSEMBLE estimator on a selected sample with the acceptance
left out, so it is not yet the number to put in the geometry.  This module turns
it into a fitted number: predict what each chamber should see, with the
capsule's y as the one free parameter, and fit it.

WHAT IS FITTED, AND WHY IT IS NOT THE SAME MEASUREMENT AGAIN.  The observable is
the distribution of ``v`` -- the track's position along the beam ON THE STRIP
PLANE -- of the trigger-matched sample.  That is a POSITION, read off the y
strip map, and no angle enters it.  Contrast the campaign forward comparison
(`source_imaging.y_forward_model`, `campaign_imaging`'s ``y_per_run``), which
compares ``target_y_mm`` = v + tan_y * 234.6: that carries the uncalibrated y
angle scale AND the tan_y resolution, which is why its width ratios came out
2.4-5.6 and why STATUS.md could not separate "the polycone acceptance model" from
"the y reconstruction".  Here the y angle scale cannot enter the data side at
all, because the data side has no angle in it.

WHAT SHAPES THE v PROFILE, in order of how much it matters (measured, Sec. 4 of
the report):

  1. **the plastic bars**, 300 mm tall at ~189 mm past the strips.  A track from
     a source at ``sv`` crossing the strips at ``v`` reaches the plastic at
     ``sv + rho (v - sv)`` with ``rho = (D + L)/D = 1.80``, so |v_p| <= 150
     clips v to a window of half-width ~83 mm whose CENTRE AND ASYMMETRY move
     with the source height.  This is the estimator.
  2. **the 1/r^2 flux**, which peaks at v = the source height.
  3. **the chamber's own live area in v**: the measured passivation band
     (`common/mx17_active_area.py`, +-180 mm, confirmed on run_79 beam data),
     dead v ranges found in the occupancy, and this run's noisy v columns.
  4. **the SiPM wall** -- which turns out to do NOTHING inside the fiducial:
     16 contiguous 25 mm bars span 400 mm in u and 500 mm in v, and at 97.4 mm
     past the strips that clips v only beyond +-177 mm.  It is imposed anyway,
     because "the wall adds nothing" is a result and not an assumption.

THE TRAP THE HANDOFF WARNS ABOUT (Sec. 6.4) IS REAL AND IS HANDLED.  The
triggering particle need not be the reconstructed track, so a few per cent of
the matched sample sits outside any window the plastic can impose.  The model
therefore carries a second component -- the same ray trace with the
scintillators NOT required -- at a free fraction ``f``.  An edge-finder cannot
do this; a profile fit can.

WHAT THE STRAIGHT-LINE TRACE CANNOT DO WITHOUT.  A fitted Gaussian smearing of
the acceptance edge, ``sigma``.  The edge is imposed 190 mm PAST the strips and a
few-MeV electron scatters on the way there, so the v at which the trigger turns
off is smeared even though the v at which the track is MEASURED is not.  Without
it chi2/ndf roughly doubles.  See :class:`Profile`.

AND WHAT IT STILL CANNOT DESCRIBE, which is why the fit uses the acceptance
EDGES and not the whole profile: over all bins the model is rejected outright
(chi2/ndf 4.6 on A, 6.8 on C), with a coherent "too peaked in the middle, too
thin in the wings" residual over the plateau that neither a flat background nor
the bystander term absorbs.  The plateau also measures nothing on its own --
fitted alone it rails at the end of the scan grid.  See :data:`V_EDGE`, which
states the three answer-independent grounds for the range.

WHAT IS NOT IN IT AT ALL.  Energy loss, the reconstruction's own dependence on
incidence angle, and any v dependence of the chamber efficiency beyond the hard
live mask -- there is no measured eff(v) map (`efficiency.py` bins u only).  The
last is the DOMINANT systematic: pinning a linear tilt at +-10 % moves y0 by a
few mm, and letting it float freely is close to a reparametrisation of the
source height on the edges alone.

    python -m sept26_prelim_analysis.capsule_y --validate     # the ray trace
    python -m sept26_prelim_analysis.capsule_y --run run_145
    python -m sept26_prelim_analysis.capsule_y --campaign --jobs 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: `pair_vertex_imaging.z_image` owns the noisy-column definition this analysis
#: masks with, and it lives under the Athens package.
ATHENS = os.path.join(REPO, 'ntof_athens_26')
for _p in (REPO, ATHENS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sept26_prelim_analysis import paths  # noqa: E402

SCHEMA = 'sept26_prelim/capsule_y/1'

#: The three chambers with a drift field and a certified angle scale.  B has
#: neither, but it needs neither here -- the fit uses positions only -- so B is
#: measured too and reported separately.  A and C decide (handoff Sec. 6.6).
ARMS = ('A', 'C', 'D')
ALL_ARMS = ('A', 'B', 'C', 'D')
DECIDE = ('A', 'C')

#: Fit binning in v, and the fiducial.  |v| <= 170 is the rest of this
#: analysis's active-area cut and it is what removes arm A's rail at v ~ -195
#: (`det_a_scint`, handoff Sec. 6.3).
V_FID = 170.0
V_BIN = 10.0

#: The fit uses the bins with |v| >= V_EDGE and not the plateau inside it.
#:
#: This is where the acceptance information is, and it is chosen on three
#: grounds that are all independent of the answer:
#:
#:   1. the estimator this test was built on is the plastic's v CLIP, whose
#:      turn-off sits at |v| ~ 85 mm -- the handoff says so before any fit;
#:   2. the plateau is not described by a straight-line model (chi2/ndf 4.6 on A
#:      and 6.8 on C over all bins, against ~2.0 on the edges alone), and a
#:      fitted parameter from a model that does not fit is not a measurement;
#:   3. the plateau on its own does not measure y0 at all -- fitted alone it
#:      rails at the end of the scan grid, because a flat top has no feature for
#:      the source height to move.
#:
#: It is not tuning: the edge-only answer is FURTHER from the band crossing than
#: the all-bins answer is, not closer.  ``all bins`` is kept as a variant.
V_EDGE = 60.0
#: Sub-bin the model on the noisy-column grid so a 2 mm hot column can be
#: removed from a 10 mm fit bin exactly, in the model as well as in the data.
V_SUB = 2.0

#: Nominal: the gas polycone's own volume centroid in the detector frame.
NOMINAL_Y0 = 0.8

#: Strips to the beam axis along the normal -- the band crossing's lever.
D_PERP_MM = 234.6

#: Source-height grid the profile likelihood is scanned on, then refined.  The
#: nominal is put ON the grid so the CAD answer's own likelihood is evaluated
#: with the same model as the fitted one.
Y0_GRID = np.sort(np.append(np.arange(-30.0, 90.1, 4.0), NOMINAL_Y0))

#: Ray-trace sampling.  The source quadrature is built ONCE per shape and reused
#: at every y0, so the profile likelihood is smooth in y0 rather than carrying an
#: independent sampling error at each point.  The beam direction gets most of the
#: points because that is the direction being measured; the capsule is only 20 mm
#: across at a 234 mm lever, so the transverse integral is nearly trivial.
N_Y, N_R, N_PHI = 48, 2, 6
N_SRC = N_Y * N_R * N_PHI
#: u samples across the plane.  The feature that sets this is the plastic bars'
#: OUTER edges, which map back to |u| ~ 112 mm on the plane and carry a ~10 mm
#: penumbra from the source's own width; 4 mm samples resolve that comfortably.
N_U = 100

#: Where the capsule is transversely, from 33 runs of scale-free crossings
#: (`imaging_campaign/per_arm.csv`): X from the mean of A and C, Z from D.
#: Read, not typed -- see :func:`capsule_xz`.


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
_XZ = {}


def capsule_xz() -> tuple:
    """(X, Z) of the capsule in the detector frame, from the campaign."""
    if not _XZ:
        pa = pd.read_csv(paths.root('out') / 'imaging_campaign' / 'per_arm.csv'
                         ).set_index('arm')
        _XZ['v'] = (0.5 * (float(pa.loc['A', 'median_mm'])
                           + float(pa.loc['C', 'median_mm'])),
                    float(pa.loc['D', 'median_mm']))
    return _XZ['v']


def polycone() -> tuple:
    """The He-3 active gas polycone (y, R) [mm], from the geometry module."""
    from ntof_tracking.reco import geometry as G
    return np.asarray(G.HE3_GAS_Y, float), np.asarray(G.HE3_GAS_R, float)


def active_v_band(arm: str) -> tuple:
    """(v_lo, v_hi) of the LIVE strip plane along the beam, measured.

    The y plane is passivated ~19 mm at each edge -- measured on five chambers
    on the June cosmic bench and confirmed independently on run_79 beam data in
    the n_TOF configuration (`common/mx17_active_area.py`).  Converted into this
    analysis's v with `build_tracks.IN_PLANE_SIGN_Y`, so the band is where the
    tracks are and not where the strip map ends.
    """
    from common import mx17_active_area as AA
    from sept26_prelim_analysis.build_tracks import IN_PLANE_SIGN_Y
    from ntof_tracking import run145_target_imaging as TI
    #: run_config aliases: A = mx17_3, B = mx17_2, C = mx17_6, D = mx17_7.
    det = {'A': 'mx17_3', 'B': 'mx17_2', 'C': 'mx17_6', 'D': 'mx17_7'}[arm]
    lo, hi = AA.TRUE_ACTIVE_BY_DET[det]['y']
    v = sorted(IN_PLANE_SIGN_Y * (np.array([lo, hi]) - TI.STRIP_MAP_HALF))
    return float(v[0]), float(v[1])


def scintillators(run: str, arm: str, trs) -> dict:
    """The plastic bars and the SiPM wall of one arm, in that arm's own frame.

    Read from the DAQ's ``run_config.json``, which places every scintillator in
    the GLOBAL frame, so this needs no depth constant and no offset convention:
    each element's surveyed centre is projected onto the arm's own (u, outward)
    axes.  That is deliberately NOT `geometry.py`'s constants, whose active-PVT
    mid-plane sits 2.5 mm short of the surveyed bar centre
    (`make_overhead_figure.plastics`, 2026-09-13).

    Self-checked against the two conventions the coincidence already uses: the
    plastic pair is centred on the MM and sits +-PLASTIC_U_OFFSET from it.
    """
    from ntof_tracking.reco import geometry as G
    cfg = json.loads((paths.root('runs') / run / 'run_config.json').read_text())
    tr = trs[f'mx17_{arm}']
    uhat = tr.R @ np.array([1.0, 0.0, 0.0])
    what = tr.R @ np.array([0.0, 0.0, 1.0])
    # The hard-coded axes are what every other module in this analysis assumes;
    # check them against the run's own orientation rather than trusting either.
    if not (np.allclose(uhat, G.U_HAT[arm], atol=1e-6)
            and np.allclose(what, G.W_HAT[arm], atol=1e-6)):
        raise AssertionError(f'{arm}: run_config axes {uhat}, {what} disagree '
                             f'with geometry.U_HAT/W_HAT')
    acc = {'plastic': [], 'sipm': []}
    for det in cfg.get('detectors', []):
        name = det.get('name', '')
        kind = name.split('_')[0]
        if kind not in acc or name.split('_')[1] != arm:
            continue
        c = det['det_center_coords']
        p = np.array([c['x'], c.get('y', 0.0), c['z']], float) - tr.center
        acc[kind].append((float(p @ uhat), float(p @ what)))

    pl = sorted(acc['plastic'])
    if len(pl) != 2:
        raise AssertionError(f'{arm}: {len(pl)} plastic bars in the config, not 2')
    mid = 0.5 * (pl[0][0] + pl[1][0])
    if abs(mid) > 1.0:
        raise AssertionError(f'{arm}: plastic midpoint {mid:.2f} mm is not on '
                             f'the MM centre')
    if abs(abs(pl[1][0] - pl[0][0]) / 2 - G.PLASTIC_U_OFFSET) > 1.0:
        raise AssertionError(f'{arm}: plastic bar centres {[p[0] for p in pl]} '
                             f'are not +-{G.PLASTIC_U_OFFSET} mm apart')
    wl = sorted(acc['sipm'])
    return dict(
        plastic_u=[p[0] for p in pl],
        plastic_depth=float(np.mean([p[1] for p in pl])),
        plastic_half_u=G.PLASTIC_HALF_U, plastic_half_v=G.PLASTIC_HALF_V,
        wall_u=[w[0] for w in wl], wall_n=len(wl),
        wall_depth=float(np.mean([w[1] for w in wl])),
        wall_half_u=G.SIPM_BAR_W / 2.0, wall_half_v=G.SIPM_HALF_V,
        # the source, in this arm's own frame: transverse offset in u, and the
        # perpendicular distance (negative = on the target side of the plane)
        src_u=float(np.array([capsule_xz()[0], 0.0, capsule_xz()[1]]) @ uhat
                    - tr.center @ uhat),
        src_w=float(np.array([capsule_xz()[0], 0.0, capsule_xz()[1]]) @ what
                    - tr.center @ what),
    )


# --------------------------------------------------------------------------- #
# the source shapes -- the systematic of handoff Sec. 6.1
# --------------------------------------------------------------------------- #
def source_points(shape: str, n_y: int = N_Y) -> tuple:
    """((N, 3) source points about the shape's EMISSION centroid, that centroid).

    DETERMINISTIC, by equal-probability quantiles rather than by random draws.
    That matters more than it looks: a random 200-point draw from an 80 mm
    source carries ~1.5 mm of sampling error in the very moments this fit reads
    the source position from, and it would reappear as a seed-dependent shift in
    ``y0`` -- checked against an independent Monte Carlo (the module's own
    validation, 2026-09-14).  Quantiles have none, and the same points are
    reused at every ``y0`` so the profile likelihood stays smooth.

    Recentring on each shape's own centroid is what makes ``y0`` mean the same
    thing for all three shapes: "where the emission centroid sits in the
    detector frame", directly comparable with the band crossing.

    ``gas``    uniform in the polycone volume -- the CAD gas, the nominal.
    ``shell``  uniform over the polycone's lateral SURFACE, the stand-in for a
               thin-wall aluminium source (handoff Sec. 6.1: the pairs are
               expected from the capsule's aluminium, not the helium).
    ``caps``   the same surface but only outside the 20 mm barrel -- the two
               tapered ends alone, the most extreme reading of Sec. 6.1 and so
               the systematic's far end.
    """
    ys, rs = polycone()
    y = np.linspace(ys.min(), ys.max(), 20001)
    R = np.interp(y, ys, rs)
    if shape == 'gas':
        w = np.pi * R ** 2                      # dV = pi R^2 dy
    else:
        dR = np.gradient(R, y)
        w = 2 * np.pi * R * np.sqrt(1 + dR ** 2)   # dA of the lateral surface
        if shape == 'caps':
            w = np.where(R >= rs.max() - 1e-6, 0.0, w)
        elif shape != 'shell':
            raise ValueError(f'unknown source shape {shape!r}')
    c = np.cumsum(w)
    c = c / c[-1]
    # equal-probability quantiles: the exact centroid and spread, no noise
    q = (np.arange(n_y) + 0.5) / n_y
    yy = np.interp(q, c, y)
    Ry = np.interp(yy, ys, rs)
    if shape == 'gas':
        # r ~ r dr on [0, R]: equal-area shells
        frac = np.sqrt((np.arange(N_R) + 0.5) / N_R)
    else:
        frac = np.ones(1)                        # on the surface, r = R(y)
    ph = 2 * np.pi * (np.arange(N_PHI) + 0.5) / N_PHI
    YY = np.repeat(yy, len(frac) * N_PHI)
    RR = (Ry[:, None] * frac[None, :]).repeat(N_PHI, axis=1).ravel()
    PH = np.tile(ph, len(yy) * len(frac))
    # the true centroid of the CONTINUOUS shape, not of the sample
    cen = float((y * w).sum() / w.sum())
    return np.c_[RR * np.cos(PH), YY - cen, RR * np.sin(PH)], cen


# --------------------------------------------------------------------------- #
# the ray trace, projected in v
# --------------------------------------------------------------------------- #
def _intervals(centres, half) -> list:
    """Bar centres -> merged (lo, hi) intervals in u.

    The SiPM wall is 16 ABUTTING 25 mm bars, so testing them one at a time would
    build sixteen boolean arrays where one interval will do; the plastics stay
    two intervals because the 3.4 mm gap between the wrapped bars is real and is
    the reason the measured hit maps have a dip down the middle.
    """
    out = []
    for c in sorted(centres):
        lo, hi = c - half, c + half
        if out and lo <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(x) for x in out]


def _in_any(up, spans):
    ok = np.zeros(up.shape, bool)
    for lo, hi in spans:
        ok |= (up >= lo) & (up <= hi)
    return ok


def v_model(geo: dict, y0: float, src: np.ndarray, v_sub: np.ndarray,
            u_live: np.ndarray, dead_edge: float = 0.0,
            wall: bool = True) -> tuple:
    """(trigger-matched, scintillator-free) predicted v densities.

    For every (source point, surface point) pair: the straight line from the
    source to the strip plane, weighted by the solid angle that surface element
    subtends at the source (``cos theta / r^2`` = ``|w_src| / r^3``, which is
    what makes an isotropic source come out right), extended to the wall plane
    and the plastic plane.  ``u_live`` carries the chamber's live u -- dead
    readout ranges are OFF the plane, not merely unlit, so they must be removed
    from the u integral and not from the answer afterwards.

    Returns two densities over ``v_sub``: with both scintillators required, and
    with neither.  The second is the bystander shape of handoff Sec. 6.4.
    """
    U = u_live[:, None]                            # (nu, 1)
    V = v_sub[None, :]                             # (1, nv)
    hit = np.zeros((len(u_live), len(v_sub)))
    free = np.zeros_like(hit)
    #: ``dead_edge`` is inactive scintillator at each bar edge and is applied to
    #: the PLASTICS only: on a 25 mm wall bar 5 mm per side would be a 40 % dead
    #: wall, which is a different hypothesis, not a systematic on this one.
    layers = [(geo['plastic_depth'],
               _intervals(geo['plastic_u'], geo['plastic_half_u'] - dead_edge),
               geo['plastic_half_v'] - dead_edge)]
    if wall:
        layers.append((geo['wall_depth'],
                       _intervals(geo['wall_u'], geo['wall_half_u']),
                       geo['wall_half_v']))

    for k in range(0, len(src), 25):
        S = src[k:k + 25]
        su = (geo['src_u'] + S[:, 0])[:, None, None]
        sv = (y0 + S[:, 1])[:, None, None]
        sw = (geo['src_w'] + S[:, 2])[:, None, None]
        du, dv = U[None] - su, V[None] - sv
        r2 = du * du + dv * dv + sw * sw
        w = np.abs(sw) / (r2 * np.sqrt(r2))
        free += w.sum(0)
        # the line reaches depth L at t = 1 - L/sw (t = 1 is the strip plane)
        ok = np.ones(w.shape, bool)
        for depth, spans, hv in layers:
            t = 1.0 - depth / sw
            ok &= _in_any(su + t * du, spans) & (np.abs(sv + t * dv) <= hv)
        hit += (w * ok).sum(0)
    return hit.sum(0), free.sum(0)


# --------------------------------------------------------------------------- #
# the fit
# --------------------------------------------------------------------------- #
#: The nuisances, in the order they are packed into the parameter vector.
#: ``sigma`` and ``flat`` are the two the sharp point-source ray trace cannot do
#: without, and both are FREE by default; ``tilt`` is not, and is the systematic.
NUISANCE = ('f', 'flat', 'sigma', 'tilt')
BOUNDS = dict(f=(0.0, 1.0), flat=(0.0, 1.0), sigma=(0.0, 60.0),
              tilt=(-0.8, 0.8))
START = dict(f=0.05, flat=0.10, sigma=12.0, tilt=0.0)
#: Nuisances free in the baseline fit.
FREE = ('f', 'flat', 'sigma')


class Profile:
    """Binds one chamber's binning and live mask to the model it is fitted with.

    Three components, and only the first is geometry:

    ``A``     the ray trace with both scintillators required -- the estimator.
    ``B``     the same ray trace with NEITHER required, at fraction ``f``: the
              handoff's Sec. 6.4 population, whose trigger was some other
              particle, so no plastic edge applies to it.  Still from the
              capsule, so still peaked at the source.
    flat      a v-uniform component at fraction ``flat``: everything that is not
              from the capsule at all.  Needed, and measured to be needed -- with
              only A and B the residuals are a coherent "too peaked in the middle,
              too thin in the wings" pattern at 3-6 sigma a bin.

    ``sigma`` is the fourth thing the sharp-edged trace cannot do without.  The
    acceptance edge is imposed 190 mm PAST the strips, and a few-MeV electron
    scatters in the chamber gas, the field cage, the air and the wall's container
    on the way there, so the v at which the trigger turns off is smeared even
    though the v at which the track is MEASURED is not.  The smearing is applied
    on the fine grid (i.e. to the acceptance, in true v) and the live mask after
    it (i.e. to the measurement, in read-out v), because that is the order the
    apparatus applies them in.

    ``use`` restricts the likelihood to a subset of the fit bins, which is how
    the one question the poor chi2 raises gets answered: is the fitted ``y0``
    coming from the ACCEPTANCE EDGES -- the feature this test is built on -- or
    from the core, where the model is demonstrably wrong?  Fit each alone and
    compare.  The normalisation is then conditional on the used bins, so a
    sub-range fit is a real fit and not the full one with bins hidden.
    """

    def __init__(self, counts, models, v_sub, vc, live_sub, nsub, use=None):
        from scipy.ndimage import gaussian_filter1d
        self.gf = gaussian_filter1d
        self.n, self.models = counts, models
        self.v_sub, self.vc = v_sub, vc
        self.live = live_sub.astype(float)
        self.nsub = nsub
        self.use = np.ones(len(vc), bool) if use is None else np.asarray(use, bool)
        self.N = counts[self.use].sum()
        # the flat component is flat in v but still blind where the chamber is:
        # it is a source hypothesis, not an exemption from the readout
        self.uni = self._bin(self.live)
        self.uni = self.uni / max(self.uni.sum(), 1e-12)

    def _bin(self, fine):
        return fine.reshape(-1, self.nsub).sum(1)

    def _unit(self, fine, p):
        if p['sigma'] > V_SUB / 2:
            fine = self.gf(fine, p['sigma'] / V_SUB, mode='nearest')
        b = self._bin(fine * self.live)
        return b / max(b.sum(), 1e-12)

    def shape(self, A, B, p: dict):
        """Expected fraction per fit bin, normalised over the USED bins."""
        s = (self._unit(A, p) + p['f'] * self._unit(B, p)
             + p['flat'] * self.uni)
        s = s * (1.0 + p['tilt'] * self.vc / V_FID)
        s = np.clip(s, 1e-12, None)
        return s / s[self.use].sum()

    def mu(self, A, B, p: dict):
        return self.N * self.shape(A, B, p)

    def nll(self, A, B, p: dict) -> float:
        mu, m = self.mu(A, B, p), self.use
        return float((mu[m] - self.n[m] * np.log(mu[m])).sum())

    def deviance(self, A, B, p: dict, n_par: int = 5) -> tuple:
        """Poisson deviance and dof -- does the model describe the profile?

        A fitted ``y0`` from a model that does not fit is not a measurement, and
        the likelihood gap against the CAD position cannot be read as a
        significance unless the winning model is itself acceptable.  Bins the
        model predicts empty (a fully dead 10 mm bin) carry no information and
        are not counted.
        """
        mu = self.mu(A, B, p)
        use = self.use & (mu > 1e-9)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(self.n[use] > 0,
                            self.n[use] * np.log(self.n[use] / mu[use]), 0.0)
        d = 2.0 * float((mu[use] - self.n[use] + term).sum())
        return d, int(use.sum() - n_par)

    def with_counts(self, counts):
        return Profile(counts, self.models, self.v_sub, self.vc,
                       self.live, self.nsub, self.use)

    def with_use(self, use):
        return Profile(self.n, self.models, self.v_sub, self.vc,
                       self.live, self.nsub, use)


def _pack(free, fixed=None) -> tuple:
    """Start point and bounds: free nuisances float, the rest are pinned.

    ``fixed`` pins one to a stated VALUE rather than to zero, which is how a
    nuisance that is degenerate with ``y0`` gets turned into a systematic: a
    free linear eff(v) tilt fitted on the acceptance edges alone is very nearly
    a re-parametrisation of the source height, so it runs away and measures
    nothing.  Pinning it at a defensible +-10 % and reading the shift in ``y0``
    is the useful statement.
    """
    fixed = fixed or {}
    x0, bnd = [], []
    for k in NUISANCE:
        if k in fixed:
            x0.append(float(fixed[k]))
            bnd.append((float(fixed[k]), float(fixed[k])))
        elif k in free:
            x0.append(START[k])
            bnd.append(BOUNDS[k])
        else:
            x0.append(0.0)
            bnd.append((0.0, 0.0))
    return x0, bnd


def profile_nll(P: Profile, free=FREE, fixed=None):
    """NLL(y0) with the nuisances profiled out at every grid point."""
    from scipy.optimize import minimize
    x0, bnd = _pack(free, fixed)
    out = []
    for y0, (A, B) in P.models.items():
        if not free:
            p = dict(zip(NUISANCE, x0))
            out.append((y0, P.nll(A, B, p)) + tuple(x0))
            continue
        r = minimize(lambda x: P.nll(A, B, dict(zip(NUISANCE, x))), x0,
                     bounds=bnd, method='L-BFGS-B')
        out.append((y0, float(r.fun)) + tuple(float(v) for v in r.x))
    return np.array(out)


def _parabola_min(y, q):
    """Vertex and curvature error of a parabola through the three lowest points."""
    i = int(np.argmin(q))
    if i == 0 or i == len(q) - 1:
        return float(y[i]), np.nan, i
    x1, x2, x3 = y[i - 1], y[i], y[i + 1]
    q1, q2, q3 = q[i - 1], q[i], q[i + 1]
    d = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(d) < 1e-12:
        return float(y[i]), np.nan, i
    a = (x3 * (q2 - q1) + x2 * (q1 - q3) + x1 * (q3 - q2)) / d
    b = (x3 ** 2 * (q1 - q2) + x2 ** 2 * (q3 - q1) + x1 ** 2 * (q2 - q3)) / d
    if a <= 0:
        return float(y[i]), np.nan, i
    # NLL is -log L, so Delta NLL = 1/2 is one sigma: a dy^2 = 1/2
    return float(-b / (2 * a)), float(np.sqrt(0.5 / a)), i


def fit_y0(P: Profile, free=FREE, fixed=None) -> dict:
    if not P.use.any():
        return dict(y0=np.nan, err_curv=np.nan, nll=np.nan, best={},
                    on_edge=True, grid_y0=float(list(P.models)[0]),
                    profile=np.empty((0, 2 + len(NUISANCE))))
    Q = profile_nll(P, free, fixed)
    y0, err, i = _parabola_min(Q[:, 0], Q[:, 1])
    best = dict(zip(NUISANCE, (float(x) for x in Q[i, 2:])))
    return dict(y0=y0, err_curv=err, nll=float(Q[i, 1]), best=best,
                on_edge=bool(i in (0, len(Q) - 1)), grid_y0=float(Q[i, 0]),
                profile=Q)


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def subruns_of(reco: Path, run: str) -> list:
    d = reco / run
    return [p.name for p in sorted(d.iterdir())
            if p.is_dir() and p.name.startswith('stat090_')
            and all((p / f'mx17_{a}' / 'events_prelim.parquet').exists()
                    for a in ALL_ARMS)]


def matched_sample(run: str, subruns, arm: str, reco: str,
                   charge_window: bool = True) -> dict:
    """v of the trigger-matched tracks, and the funnel that produced them.

    ``charge_window=False`` drops `k_arm`'s 25-75 % charge window.  That window
    is a systematic on THIS measurement in a way it is not on the angle scale: a
    track at large |v| arrives more inclined, so its path in the 30 mm gap is
    longer and its charge higher, and a cut on charge is then a soft cut on |v|.
    Both are run and reported.
    """
    from sept26_prelim_analysis import k_arm as K
    v, ty, n_coin, n_hot = [], [], 0, 0
    saved = K.CHARGE_WINDOW
    try:
        if not charge_window:
            K.CHARGE_WINDOW = (0.0, 100.0)
        for s in subruns:
            S = K.coincident_tracks(run, s, arm, reco)
            v.append(S['yl'])
            ty.append(S['ty'])
            n_coin += S['n_coincident'] + S['n_hotstrip']
            n_hot += S['n_hotstrip']
    finally:
        K.CHARGE_WINDOW = saved
    return dict(v=np.concatenate(v), ty=np.concatenate(ty),
                n_coincident=n_coin, n_hotstrip=n_hot)


def live_mask(run: str, subruns, arm: str, reco: str, v_sub: np.ndarray) -> tuple:
    """Per-sub-bin live flag in v, and the live u values, for one chamber.

    Three independent things make a v sub-bin dead, and all three are found in
    THIS run's data rather than transcribed: the measured passivation band, a
    zero-occupancy v range (chamber D), and a noisy v column.
    """
    from sept26_prelim_analysis import source_imaging as SI
    from pair_vertex_imaging import z_image as ZI
    from sept26_prelim_analysis.build_tracks import IN_PLANE_SIGN_Y
    from ntof_tracking import run145_target_imaging as TI

    raw = []
    for s in subruns:
        p = paths.require(Path(reco) / s / f'mx17_{arm}' / 'events_prelim.parquet')
        raw.append(pd.read_parquet(p, columns=['x_p0', 'y_p0']))
    raw = pd.concat(raw, ignore_index=True)

    lo, hi = active_v_band(arm)
    ok = (v_sub >= lo) & (v_sub <= hi)
    # dead v ranges, found the same way the dead u ranges are
    yv = raw.y_p0.to_numpy(float)
    dead_v = SI.dead_ranges(yv[np.isfinite(yv)])
    for a, b in dead_v:
        vv = sorted(IN_PLANE_SIGN_Y * (np.array([a, b]) - TI.STRIP_MAP_HALF))
        ok &= ~((v_sub >= vv[0]) & (v_sub <= vv[1]))
    # noisy v columns: `z_image`'s finder, on this run's whole gated v occupancy
    vloc = IN_PLANE_SIGN_Y * (yv - TI.STRIP_MAP_HALF)
    vloc = vloc[np.isfinite(vloc)]
    _, tab = ZI.hot_columns(vloc, np.full(len(vloc), arm), run)
    hot = tab.dropna(subset=['lo'])
    for a, b in hot[['lo', 'hi']].itertuples(False):
        ok &= ~((v_sub >= a) & (v_sub < b))

    # live u: the plane's own active width less the dead readout ranges
    xv = raw.x_p0.to_numpy(float)
    dead_u = [tuple(sorted(TI.IN_PLANE_SIGN * (np.array([a, b]) - TI.STRIP_MAP_HALF)))
              for a, b in SI.dead_ranges(xv[np.isfinite(xv)])]
    u = np.linspace(-TI.STRIP_MAP_HALF, TI.STRIP_MAP_HALF, N_U)
    keep = np.ones(len(u), bool)
    for a, b in dead_u:
        keep &= ~((u >= a) & (u <= b))
    return ok, u[keep], dict(dead_v=dead_v, n_hot_v=len(hot), dead_u=dead_u,
                             active_band=[lo, hi], u_live_frac=float(keep.mean()))


def bin_data(v: np.ndarray, ok_sub: np.ndarray, v_sub: np.ndarray,
             v_edges: np.ndarray) -> tuple:
    """Counts per fit bin, keeping only tracks in LIVE sub-bins.

    Data and model have to see one detector: a 10 mm fit bin holding a 2 mm
    noisy column loses those tracks here and the matching 20 % of its model
    weight in :func:`one_arm`, so the bin stays usable instead of being thrown
    away whole.
    """
    idx = np.floor((v + V_FID) / V_SUB).astype(int)
    inside = (idx >= 0) & (idx < len(v_sub))
    keep = inside.copy()
    keep[inside] &= ok_sub[idx[inside]]
    return np.histogram(v[keep], bins=v_edges)[0].astype(float), keep


def one_arm(run: str, subruns, arm: str, reco: str, trs, shape: str = 'gas',
            dead_edge: float = 0.0, wall: bool = True, v_fid: float = V_FID,
            charge_window: bool = True, edge_cut: float = V_EDGE,
            y0_grid=Y0_GRID) -> tuple:
    """The whole measurement for one chamber of one run."""
    v_edges = np.arange(-v_fid, v_fid + 0.5 * V_BIN, V_BIN)
    vc = 0.5 * (v_edges[:-1] + v_edges[1:])
    v_sub = np.arange(-v_fid + V_SUB / 2, v_fid, V_SUB)
    nsub = int(round(V_BIN / V_SUB))

    S = matched_sample(run, subruns, arm, reco, charge_window)
    ok_sub, u_live, info = live_mask(run, subruns, arm, reco, v_sub)
    geo = scintillators(run, arm, trs)
    src, shape_centroid = source_points(shape)

    n, keep = bin_data(S['v'], ok_sub, v_sub, v_edges)
    #: models are stored on the FINE grid, unsmeared and un-masked, so the
    #: scattering nuisance can be varied at fit time without rebuilding them
    models = {float(y0): v_model(geo, float(y0), src, v_sub, u_live,
                                 dead_edge=dead_edge, wall=wall)
              for y0 in y0_grid}
    P = Profile(n, models, v_sub, vc, ok_sub, nsub,
                use=np.abs(vc) >= edge_cut)

    R = fit_y0(P)
    out = dict(run=run, arm=arm, shape=shape, dead_edge=dead_edge, wall=wall,
               v_fid=v_fid, charge_window=charge_window, edge_cut=edge_cut,
               n_bins_used=int(P.use.sum()), n_in_fit=int(n[P.use].sum()),
               n_tracks=int(keep.sum()), n_dropped_dead=int((~keep).sum()),
               n_coincident=S['n_coincident'], n_hotstrip=S['n_hotstrip'],
               live_v_frac=float(ok_sub.mean()),
               shape_centroid=shape_centroid,
               plastic_depth=geo['plastic_depth'], wall_depth=geo['wall_depth'],
               src_u=geo['src_u'], src_w=geo['src_w'],
               y0=R['y0'], err_curv=R['err_curv'], nll=R['nll'],
               on_edge=R['on_edge'], f_bystander=R['best']['f'],
               f_flat=R['best']['flat'], sigma=R['best']['sigma'],
               **{k: (v if not isinstance(v, list) else json.dumps(v))
                  for k, v in info.items()})

    # the fit's own goodness, at the nearest grid point to the fitted y0 -- the
    # models are only built on the grid, and 4 mm from the optimum changes the
    # deviance far less than it would have to to change the verdict
    near = models[R['grid_y0']]
    out['chi2'], out['ndf'] = P.deviance(*near, R['best'])
    out['chi2_ndf'] = out['chi2'] / max(out['ndf'], 1)

    # the same fit with each nuisance held instead of fitted.  y0 moving here is
    # the systematic that matters: it is the model's shape freedom, not counting
    out['y0_sharp'] = fit_y0(P, ('f', 'flat'))['y0']       # no scattering
    out['y0_noflat'] = fit_y0(P, ('f', 'sigma'))['y0']     # no flat background
    out['y0_f0'] = fit_y0(P, ('flat', 'sigma'))['y0']      # no bystanders
    # An unmeasured eff(v).  A FREE linear tilt is nearly a reparametrisation of
    # y0 on the edges alone, so it is pinned at a defensible +-10 % across the
    # plane instead and the shift in y0 is the systematic.
    out['y0_tilt_p10'] = fit_y0(P, FREE, dict(tilt=+0.10))['y0']
    out['y0_tilt_m10'] = fit_y0(P, FREE, dict(tilt=-0.10))['y0']
    T = fit_y0(P, FREE + ('tilt',))
    out['y0_tilt_free'], out['tilt_free'] = T['y0'], T['best'].get('tilt')

    # Where the answer comes from.  The baseline is the edges; these two say what
    # the discarded plateau would have done, and are the evidence for discarding
    # it -- the core fit rails at the end of the grid, so it measures nothing.
    A_ = fit_y0(P.with_use(np.ones(len(vc), bool)))
    Cc = fit_y0(P.with_use(np.abs(vc) < edge_cut))
    out['y0_allbins'], out['y0_core'] = A_['y0'], Cc['y0']
    out['core_on_edge'] = Cc['on_edge']
    out['chi2_ndf_allbins'] = (lambda d: d[0] / max(d[1], 1))(
        P.with_use(np.ones(len(vc), bool)).deviance(
            *models[A_['grid_y0']], A_['best']))

    # the CAD answer's own likelihood, under the same model and the same nuisances
    out['nll_nominal'] = P.nll(*models[NOMINAL_Y0], R['best'])
    out['dnll_nominal'] = out['nll_nominal'] - out['nll']
    out['chi2_nominal'] = P.deviance(*models[NOMINAL_Y0], R['best'])[0]

    # the OTHER estimator, on exactly these tracks, with its own v gain -- the
    # pair of gains is what separates the capsule's height from the v origin
    vk, tk = S['v'][keep], S['ty'][keep]
    b = band_crossing(vk, tk)
    out['band_mm'], out['band_scale'], out['band_n'] = b['mm'], b['scale'], b['n']
    out['band_v_gain'] = float(
        (band_crossing(vk, tk, +10.0)['mm']
         - band_crossing(vk, tk, -10.0)['mm']) / 20.0)
    # the two curves a reader needs to see, evaluated here rather than rebuilt
    # by the figure code from a stored model and a stored parameter vector
    return out, dict(v_centres=vc, counts=n, models=models, profile=R['profile'],
                     best=R['best'], P=P, v_sub=v_sub, live_sub=ok_sub,
                     v_raw=vk, ty_raw=tk, v_edges=v_edges, use=P.use,
                     mu_best=P.mu(*models[R['grid_y0']], R['best']),
                     mu_nominal=P.mu(*models[NOMINAL_Y0], R['best']))


def bootstrap(P: Profile, n_boot=40, seed=7) -> float:
    """Statistical error on y0: Poisson-resample the bins and refit."""
    rng = np.random.default_rng(seed)
    out = [fit_y0(P.with_counts(rng.poisson(P.n).astype(float)))['y0']
           for _ in range(n_boot)]
    return float(np.nanstd(out))


#: The variant grid.  ``baseline`` first, and its ``curves`` are what the
#: figures are drawn from; every other row is one knob moved off it.
def _variants(shapes, dead_edges, full: bool) -> list:
    out = [dict(shape=shapes[0], dead_edge=0.0, wall=True, v_fid=V_FID,
                charge_window=True, edge_cut=V_EDGE, name='baseline')]
    if not full:
        return out
    for s in shapes[1:]:
        out.append(dict(shape=s, dead_edge=0.0, wall=True, v_fid=V_FID,
                        charge_window=True, edge_cut=V_EDGE,
                        name=f'source={s}'))
    for de in dead_edges:
        if de:
            out.append(dict(shape=shapes[0], dead_edge=de, wall=True,
                            v_fid=V_FID, charge_window=True, edge_cut=V_EDGE,
                            name=f'plastic dead edge {de:g} mm'))
    base = dict(shape=shapes[0], dead_edge=0.0, wall=True, v_fid=V_FID,
                charge_window=True, edge_cut=V_EDGE)
    out += [dict(base, wall=False, name='no SiPM wall'),
            dict(base, v_fid=150.0, name='fiducial |v| <= 150'),
            dict(base, v_fid=180.0, name='fiducial |v| <= 180'),
            dict(base, charge_window=False, name='no charge window'),
            dict(base, edge_cut=50.0, name='edge cut |v| >= 50'),
            dict(base, edge_cut=70.0, name='edge cut |v| >= 70'),
            dict(base, edge_cut=0.0, name='all bins (plateau included)')]
    return out


def one_run(run: str, reco: str, shapes, dead_edges, n_boot: int,
            full: bool = True) -> tuple:
    """(rows, curves, error) -- runs in a worker process."""
    try:
        from sept26_prelim_analysis import source_imaging as SI
        subs = subruns_of(Path(reco), run)
        if not subs:
            return run, None, None, 'no usable sub-runs'
        trs = SI.transforms(run)
        rd = str(Path(reco) / run)
        rows, curves = [], {}
        for arm in ARMS:
            for k, vv in enumerate(_variants(shapes, dead_edges, full)):
                name = vv.pop('name')
                r, c = one_arm(run, subs, arm, rd, trs, **vv)
                r['variant'] = name
                r['n_subruns'] = len(subs)
                if k == 0:
                    r['err_boot'] = bootstrap(c['P'], n_boot)
                    r['v_origin_gain'] = v_origin_gain(c)
                    curves[arm] = c
                rows.append(r)
        return run, pd.DataFrame(rows), curves, ''
    except Exception:
        return run, None, None, traceback.format_exc(limit=4)


# --------------------------------------------------------------------------- #
def v_origin_gain(c: dict, delta: float = 10.0) -> float:
    """d(fitted y0) / d(rigid v offset) -- the number that makes this decisive.

    A rigid offset ``delta`` in a chamber's v coordinate moves the BAND CROSSING
    by exactly ``delta``, and all four chambers share one strip-map convention
    (``y_local = +-(y_p0 - 199.29)``), so their mutual agreement cannot test it
    (handoff Sec. 5).  It moves THIS fit by a different amount, because the
    acceptance edges come from the scintillators, which are surveyed
    independently in ``run_config.json``.

    Read it like this.  Let the true source height be ``y_s`` and the v origin
    be wrong by ``delta``.  The band reports ``y_s + delta``.  The plastic clip
    responds with gain ``rho/(rho - 1) = 2.25`` (``rho = (D+L)/D = 1.80``) and
    the 1/r^2 flux with gain 1, so this fit reports ``y_s + g delta`` with
    ``1 < g < 2.25``.  Two estimators, two different gains, two unknowns: the
    pair separates the capsule's height from the v origin, which neither does
    alone.  Measured here rather than assumed, because which of the two features
    dominates is a property of this acceptance and not of the algebra.

    ``delta`` is defined as "every measured v is larger by delta", the same
    convention as for the band crossing below, so the two gains are comparable
    without a sign argument.
    """
    out = []
    for d in (-delta, delta):
        # shift the DATA, keep the detector: noisy columns and the passivation
        # band are properties of the chamber and do not move with a frame offset
        nb = np.histogram(c['v_raw'] + d, bins=c['v_edges'])[0].astype(float)
        out.append(fit_y0(c['P'].with_counts(nb))['y0'])
    return float((out[1] - out[0]) / (2 * delta))


def band_crossing(v, ty, delta: float = 0.0) -> dict:
    """The scale-free y band crossing of THESE tracks, and its own v gain.

    Computed on exactly the sample the acceptance fit uses, rather than quoted
    from `make_overhead_figure`'s y-cleaned selection, because the two estimators
    are only combinable on one sample.  Construction is `y_image`'s: a track from
    a source at ``y_s`` has ``tan_y * L = s (y_s - v)``, so a robust line of
    ``tan_y * L`` against ``v`` crosses zero at ``y_s`` whatever the unknown y
    angle scale ``s`` is.  The sign of ``q`` is the one `make_overhead_figure`
    measured: one step inward moves y by ``-tan_y``.
    """
    from ntof_tracking import run145_target_imaging as TI
    q = -np.asarray(ty, float) * D_PERP_MM
    vv = np.asarray(v, float) + delta
    ok = np.isfinite(q) & np.isfinite(vv)
    if ok.sum() < 200:
        return dict(mm=np.nan, scale=np.nan, n=int(ok.sum()))
    sl, ic = TI._robust_line(vv[ok], q[ok])
    return dict(mm=float(-ic / sl) if sl else np.nan, scale=float(-sl),
                n=int(ok.sum()))


def validate(run: str = 'run_145', arm: str = 'A', n: int = 4_000_000,
             seed: int = 5) -> dict:
    """Check :func:`v_model` against an independent isotropic Monte Carlo.

    This module's ray trace is a deterministic quadrature: it samples the SOURCE
    and integrates over the strip plane with a cos(theta)/r^2 weight, which is
    what makes an isotropic source come out right without being thrown
    isotropically.  That weight is the one step in the chain that is asserted
    rather than obvious, so it is checked against a plain isotropic throw
    through the same geometry -- the construction `source_imaging.y_forward_model`
    already uses, which was written independently and for another purpose.

    Both are given the SAME geometry for the comparison: no capsule transverse
    offset, `geometry.py`'s plastic depth rather than the survey's, plastics
    only, and the plane at +-190 x +-170 mm.

    Measured 2026-09-14: mean v agrees to 0.26 mm and the r.m.s. to 0.03 mm on
    a 96 mm-wide acceptance -- an order of magnitude below the systematic.
    """
    from sept26_prelim_analysis import source_imaging as SI
    from ntof_tracking.reco import geometry as G
    trs = SI.transforms(run)
    centre = trs[f'mx17_{arm}'].center
    dp = G.PLASTIC_W0[arm] - G.W_STRIP + G.PLASTIC_THICK / 2
    geo = dict(scintillators(run, arm, trs), src_u=0.0, plastic_depth=dp,
               src_w=-float(np.abs(centre @ G.W_HAT[arm])))

    v_sub = np.arange(-V_FID + V_SUB / 2, V_FID, V_SUB)
    u_live = np.linspace(-190.0, 190.0, 152)
    src, cen = source_points('gas')
    A, _ = v_model(geo, cen, src, v_sub, u_live, wall=False)

    rng = np.random.default_rng(seed)
    ys, rs = polycone()
    yy = rng.uniform(ys.min(), ys.max(), n)
    rr = G.HE3_R_MAX * np.sqrt(rng.uniform(0, 1, n))
    keep0 = rr <= np.interp(yy, ys, rs)
    yy, rr = yy[keep0], rr[keep0]
    ph = rng.uniform(0, 2 * np.pi, len(yy))
    P = np.c_[rr * np.cos(ph), yy, rr * np.sin(ph)]
    ct = rng.uniform(-1, 1, len(yy))
    st = np.sqrt(1 - ct ** 2)
    az = rng.uniform(0, 2 * np.pi, len(yy))
    D = np.c_[st * np.cos(az), ct, st * np.sin(az)]
    wh, uh, vh = G.W_HAT[arm], G.U_HAT[arm], G.V_HAT
    dn = D @ wh

    def at(depth):
        s = (centre @ wh + depth - P @ wh) / np.where(np.abs(dn) < 1e-9,
                                                      np.nan, dn)
        X = P + s[:, None] * D
        return X @ uh - centre @ uh, X @ vh, s

    u0, v0, s0 = at(0.0)
    up, vp, _ = at(dp)
    keep = (np.isfinite(u0) & (s0 > 0) & (np.abs(u0) < 190.0)
            & (np.abs(v0) < V_FID) & (np.abs(vp) < G.PLASTIC_HALF_V)
            & (np.abs(np.abs(up) - G.PLASTIC_U_OFFSET) < G.PLASTIC_HALF_U))

    e = np.arange(-V_FID, V_FID + V_BIN, V_BIN)
    c = 0.5 * (e[:-1] + e[1:])
    hm = np.histogram(v0[keep], bins=e)[0].astype(float)
    hm /= hm.sum()
    hq = A.reshape(-1, int(round(V_BIN / V_SUB))).sum(1)
    hq /= hq.sum()
    mom = lambda h: (float((c * h).sum()),  # noqa: E731
                     float(np.sqrt((c ** 2 * h).sum() - (c * h).sum() ** 2)))
    mq, rq = mom(hq)
    mm, rm = mom(hm)
    return dict(n_thrown=int(len(yy)), n_accepted=int(keep.sum()),
                n_src=len(src), mean_quad=mq, mean_mc=mm, d_mean=mq - mm,
                rms_quad=rq, rms_mc=rm, d_rms=rq - rm,
                mc_mean_err=rm / np.sqrt(max(keep.sum(), 1)))


#: Efficiency tilts imposed on the data by thinning, as fractions across
#: |v| = 170 mm.  +-0.3 is deliberately generous: the point is to show which
#: estimator does not care.
TILTS = (-0.3, -0.1, 0.0, 0.1, 0.3)


def _thin(v, a, rep):
    """Keep each track with probability (1 + a v/170)/(1 + |a|): an eff(v) tilt."""
    w = (1 + a * v / V_FID) / (1 + abs(a))
    return np.random.default_rng(1000 + rep).uniform(size=len(v)) < w


def sensitivity(run: str, reco: str, n_rep: int = 40, with_ray: bool = False,
                trs=None) -> pd.DataFrame:
    """Which estimator of the source height a chamber's own eff(v) can move.

    THE MISTAKE THIS EXISTS TO PREVENT (2026-09-14).  The first reading of this
    module's campaign pass treated the acceptance fit as unbiased apart from a
    rigid v offset, solved the two estimators chamber by chamber (:func:`separate`)
    and got an "11 mm relative v offset between A and C" and a +-8 mm error on the
    capsule height.  Both were artefacts.  The two estimators are not equally
    exposed to the thing that differs most between chambers:

      band crossing   uses the CORRELATION of tan_y with v -- whether tracks
                      point back to one height.  Thinning tracks along v
                      re-weights points along a line and does not move the line.
      acceptance fit  uses the SHAPE of the v distribution, which is exactly what
                      a chamber's eff(v), dead strips and noisy columns distort.

    This measures that directly, on the fit's own sample: thin the tracks by an
    imposed tilt (1 + a v/170) and refit.  Measured on run_145: a +-30 % tilt moves
    the band crossing by <= 1 mm; a +-10 % tilt moves the acceptance fit by 3-4 mm.

    Also the side-view histogram check: the MEDIAN of y at the target,
    ``v - tan_y L``, is set by the marginals mean(v) and mean(tan_y) and is not a
    pointing statement -- shuffling tan_y across a chamber's tracks, which
    destroys pointing, leaves it nearly where it was.  Only the band crossing's
    slope carries per-track pointing.

    ``with_ray`` also thins the data for the acceptance fit (a like-for-like
    comparison, ~1 min per chamber).  Without it the fit's response comes from
    the pinned model tilt ``y0_tilt_p10/m10`` in the fits table, which is the same
    effect seen from the model side.
    """
    from sept26_prelim_analysis import source_imaging as SI
    trs = trs or SI.transforms(run)
    subs = subruns_of(Path(reco), run)
    rd = str(Path(reco) / run)
    v_edges = np.arange(-V_FID, V_FID + 0.5 * V_BIN, V_BIN)
    v_sub = np.arange(-V_FID + V_SUB / 2, V_FID, V_SUB)
    rng = np.random.default_rng(29)
    rows = []
    for arm in ARMS:
        S = matched_sample(run, subs, arm, rd)
        ok_sub, _, _ = live_mask(run, subs, arm, rd, v_sub)
        _, keep = bin_data(S['v'], ok_sub, v_sub, v_edges)
        v, ty = S['v'][keep], S['ty'][keep]
        yt = v - ty * D_PERP_MM
        row = dict(run=run, arm=arm, n=int(len(v)), median_v=float(np.median(v)),
                   median_yt=float(np.median(yt)),
                   median_yt_shuffled=float(np.median(
                       v - rng.permutation(ty) * D_PERP_MM)),
                   band_mm=band_crossing(v, ty)['mm'])
        for a in TILTS:
            if a == 0.0:
                continue
            bs = [band_crossing(v[m], ty[m])['mm']
                  for m in (_thin(v, a, r) for r in range(n_rep))]
            row[f'band_tilt_{a:+.1f}'] = float(np.nanmedian(bs))
        if with_ray:
            r0, c = one_arm(run, subs, arm, rd, trs)
            row['ray_mm'] = r0['y0']
            for a in TILTS:
                if a == 0.0:
                    continue
                ys = []
                for r in range(min(n_rep, 10)):
                    nb = np.histogram(c['v_raw'][_thin(c['v_raw'], a, r)],
                                      bins=c['v_edges'])[0].astype(float)
                    ys.append(fit_y0(c['P'].with_counts(nb))['y0'])
                row[f'ray_tilt_{a:+.1f}'] = float(np.nanmedian(ys))
        # responses in mm per unit tilt (i.e. per 100 % across +-170 mm)
        row['band_per_tilt'] = (row['band_tilt_+0.3'] - row['band_tilt_-0.3']) / 0.6
        if with_ray:
            row['ray_per_tilt'] = (row['ray_tilt_+0.3'] - row['ray_tilt_-0.3']) / 0.6
        rows.append(row)
    return pd.DataFrame(rows)


def _sensitivity_job(run, reco, n_rep, with_ray):
    try:
        return run, sensitivity(run, reco, n_rep, with_ray), ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def separate(base: pd.DataFrame) -> pd.DataFrame:
    """Solve the two-estimator system for the capsule height and the v origin.

    Two measurements of the same chamber, with two different responses to a
    rigid offset ``delta`` in its v coordinate:

        band crossing      B = y_s + g_b delta      (g_b = 1, measured)
        acceptance fit     R = y_s + g_r delta      (g_r ~ 2.1, measured)

    so ``delta = (R - B)/(g_r - g_b)`` and ``y_s = B - g_b delta``.

    **READ THE COMMON PART ONLY.**  The model has room for one chamber-specific
    effect, a rigid shift, and :func:`sensitivity` shows the acceptance fit is
    10-30x more exposed than the band crossing to the other one -- the chamber's
    own eff(v).  So the per-chamber ``delta`` and ``y_source`` absorb eff(v) and
    are NOT an alignment or a capsule height (the 2026-09-14 campaign pass read
    them as an 11 mm A-C v offset; it was not).  What survives is the average
    over chambers: a COMMON v offset large enough to explain the band crossing
    would put every acceptance fit near B (1 + g), and no plausible eff(v) closes
    that gap.  The capsule height itself is best taken from the band crossing.
    """
    rows = []
    for r in base.itertuples():
        gb, gr = r.band_v_gain, r.v_origin_gain
        if not np.isfinite(gb * gr) or abs(gr - gb) < 0.2:
            rows.append(dict(run=r.run, arm=r.arm, delta=np.nan, y_source=np.nan))
            continue
        d = (r.y0 - r.band_mm) / (gr - gb)
        rows.append(dict(run=r.run, arm=r.arm, band_mm=r.band_mm, ray_mm=r.y0,
                         g_band=gb, g_ray=gr, delta=d,
                         y_source=r.band_mm - gb * d))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--campaign', action='store_true',
                    help='every run under the reco tree, not just --run')
    ap.add_argument('--reco', default=None)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--n-boot', type=int, default=40)
    ap.add_argument('--shapes', default='gas,shell,caps')
    ap.add_argument('--dead-edges', default='0,5')
    ap.add_argument('--validate', action='store_true',
                    help='check the ray trace against an independent isotropic '
                         'Monte Carlo through the same geometry, and exit')
    ap.add_argument('--sensitivity', action='store_true',
                    help='measure how an imposed eff(v) tilt moves each '
                         'estimator (band crossing on every run; the '
                         'acceptance fit too on --run), write '
                         'capsule_y_sensitivity.csv, and exit. Does not touch '
                         'the fits table.')
    a = ap.parse_args()

    if a.sensitivity:
        reco = Path(a.reco) if a.reco else paths.root('out') / 'reco_fullpass'
        runs = ([p.name for p in sorted(reco.iterdir())
                 if p.name.startswith('run_')] if a.campaign else [a.run])
        rows, bad = [], {}
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(_sensitivity_job, r, str(reco), 40, r == a.run)
                    for r in runs]
            for f in as_completed(futs):
                run, df, err = f.result()
                if err:
                    bad[run] = err
                    print(f'  {run:<10} --  {err}', flush=True)
                    continue
                rows.append(df)
                print(f'  {run:<10} ok  ' + '  '.join(
                    f'{x.arm} band {x.band_mm:+5.1f} ({x.band_per_tilt:+5.1f}/tilt)'
                    for x in df.itertuples()), flush=True)
        T = pd.concat(rows, ignore_index=True).sort_values(['run', 'arm'])
        T.to_csv(paths.out('capsule_y') / 'capsule_y_sensitivity.csv', index=False)
        cols = [c for c in T.columns if c.startswith(('band_tilt', 'ray_tilt'))]
        print('\n' + T[T.run == a.run][['arm', 'n', 'median_v', 'median_yt',
                                        'median_yt_shuffled', 'band_mm', 'ray_mm']
                                       + cols + ['band_per_tilt', 'ray_per_tilt']]
              .to_string(index=False, float_format=lambda x: f'{x:7.1f}'))
        print(f'\nwrote -> {paths.out("capsule_y") / "capsule_y_sensitivity.csv"}'
              + (f'   failed: {bad}' if bad else ''))
        return 0

    if a.validate:
        v = validate(a.run)
        print(f'ray trace vs independent MC ({a.run}, arm A, plastics only)')
        print(f'  quadrature  {v["n_src"]} source points')
        print(f'  MC          {v["n_accepted"]:,} accepted of {v["n_thrown"]:,}')
        print(f'  mean v      {v["mean_quad"]:+7.3f} vs {v["mean_mc"]:+7.3f} mm'
              f'   (diff {v["d_mean"]:+.3f}, MC stat err {v["mc_mean_err"]:.3f})')
        print(f'  r.m.s. v    {v["rms_quad"]:7.3f} vs {v["rms_mc"]:7.3f} mm'
              f'   (diff {v["d_rms"]:+.3f})')
        json.dump(v, open(paths.out('capsule_y') / 'validation.json', 'w'),
                  indent=1, default=float)
        return 0

    reco = Path(a.reco) if a.reco else paths.root('out') / 'reco_fullpass'
    od = paths.out('capsule_y')
    shapes = tuple(a.shapes.split(','))
    dead_edges = tuple(float(x) for x in a.dead_edges.split(','))

    runs = ([p.name for p in sorted(reco.iterdir()) if p.name.startswith('run_')]
            if a.campaign else [a.run])
    print(f'{len(runs)} run(s) from {reco}\n')

    R, bad, curves = [], {}, {}
    if len(runs) == 1:
        run, rows, cv, err = one_run(runs[0], str(reco), shapes, dead_edges,
                                     a.n_boot, full=True)
        if err:
            print(err, file=sys.stderr)
            return 1
        R.append(rows)
        curves = cv
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(one_run, r, str(reco), shapes, dead_edges,
                              a.n_boot, r == a.run): r for r in runs}
            for f in as_completed(futs):
                run, rows, cv, err = f.result()
                if err:
                    bad[run] = err.strip().splitlines()[-1]
                    print(f'  {run:<10} --  {bad[run]}', flush=True)
                    continue
                R.append(rows)
                if run == a.run:
                    curves = cv
                b = rows[rows.variant == 'baseline']
                print(f'  {run:<10} ok  ' + '  '.join(
                    f'{x.arm} {x.y0:+6.1f}' for x in b.itertuples()), flush=True)
    R = pd.concat(R, ignore_index=True).sort_values(
        ['run', 'arm', 'variant'], ignore_index=True)
    R.to_csv(od / 'capsule_y_fits.csv', index=False)

    base = R[R.variant == 'baseline']
    print('\n-- baseline fit: gas polycone, both scintillators, |v| <= 170')
    cols = ['run', 'arm', 'n_in_fit', 'y0', 'err_curv', 'err_boot', 'chi2_ndf',
            'ndf', 'sigma', 'f_bystander', 'f_flat', 'y0_sharp',
            'y0_tilt_m10', 'y0_tilt_p10', 'y0_allbins', 'y0_core',
            'dnll_nominal']
    print(base[[c for c in cols if c in base]].to_string(
        index=False, float_format=lambda x: f'{x:8.2f}'))

    SEP = separate(base)
    SEP.to_csv(od / 'capsule_y_separated.csv', index=False)
    print('\n-- the two estimators, and the split they make (per run, per arm)')
    print(SEP.to_string(index=False, float_format=lambda x: f'{x:8.2f}'))
    dec = SEP[SEP.arm.isin(DECIDE)].dropna(subset=['y_source'])
    if len(dec):
        sd = (lambda s: s.std(ddof=1) if len(s) > 1 else float('nan'))
        print(f'\n   A and C together:  v origin offset {dec.delta.mean():+.1f} mm '
              f'(spread {sd(dec.delta):.1f}), '
              f'capsule height {dec.y_source.mean():+.1f} mm '
              f'(spread {sd(dec.y_source):.1f})')
        # One capsule, two chambers: y_s is shared and delta is not, so the
        # DIFFERENCE of the two deltas is the relative v alignment of A against
        # C -- the v counterpart of the A-C number `campaign_imaging` reports in
        # X, and the thing no single chamber can produce.
        p = dec.pivot_table(index='run', columns='arm', values='delta')
        if set(DECIDE) <= set(p.columns):
            rel = (p[DECIDE[0]] - p[DECIDE[1]]).dropna()
            print(f'   relative v alignment {DECIDE[0]} - {DECIDE[1]}: '
                  f'{rel.mean():+.1f} mm over {len(rel)} run(s) '
                  f'(run-to-run s.d. {sd(rel):.1f})')
            ys = dec.pivot_table(index='run', columns='arm', values='y_source')
            d = (ys[DECIDE[0]] - ys[DECIDE[1]]).dropna()
            print(f'   the two chambers\' capsule heights differ by '
                  f'{d.mean():+.1f} mm (s.d. {sd(d):.1f}) -- the consistency '
                  f'check, since there is only one capsule')

    # variants are only run on --run; comparing them against a campaign-mean
    # baseline would report run-to-run scatter as a systematic
    print(f'\n-- every variant, y0 per arm ({a.run} only)')
    piv = R[R.run == a.run].pivot_table(index='variant', columns='arm',
                                        values='y0', aggfunc='mean')
    off = piv.subtract(piv.loc['baseline'], axis=1)
    print(pd.concat([piv, off.add_suffix('  d')], axis=1).to_string(
        float_format=lambda x: f'{x:8.2f}'))

    meta = dict(schema=SCHEMA, reco=str(reco), runs=runs, runs_failed=bad,
                shapes=list(shapes), dead_edges=list(dead_edges),
                capsule_xz=list(capsule_xz()), nominal_y0=NOMINAL_Y0,
                v_fid=V_FID, v_bin=V_BIN, v_sub=V_SUB, n_src=N_SRC, n_u=N_U,
                y0_grid=[float(x) for x in Y0_GRID], n_boot=a.n_boot,
                curve_run=a.run)
    if curves:
        np.savez_compressed(
            od / 'capsule_y_curves.npz',
            **{f'{arm}|{k}': np.asarray(v) for arm, c in curves.items()
               for k, v in (('counts', c['counts']), ('v', c['v_centres']),
                            ('profile', c['profile']), ('use', c['use']),
                            ('mu_best', c['mu_best']),
                            ('mu_nominal', c['mu_nominal']),
                            ('live_sub', c['live_sub']), ('v_sub', c['v_sub']))},
            **{f'{arm}|model|{y0:g}|{j}': m[j]
               for arm, c in curves.items() for y0, m in c['models'].items()
               for j in (0, 1)})
        meta['curves'] = 'capsule_y_curves.npz'
    json.dump(meta, open(od / 'capsule_y.meta.json', 'w'), indent=1, default=float)
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
