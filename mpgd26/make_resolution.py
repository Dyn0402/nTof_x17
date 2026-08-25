#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_resolution.py -- the angular-resolution figures for the MPGD2026 talk.

    ../.venv/bin/python make_resolution.py
    ../.venv/bin/python make_resolution.py --only correlation
    ../.venv/bin/python make_resolution.py --print      # numbers only

Two figures, both for the "Sub-degree angle, sub-mm position" slide, and one
JSON of the numbers that go on it as type:

  angle_correlation.png       reconstructed vs reference track angle, X and Y,
                              as a 2-D density with the line of equality drawn.
                              The plot that says the fit MEASURES the angle
                              rather than regressing towards the mean of it.
  angle_resolution.png        sigma68 of (reconstructed - reference) in bins of
                              |reference angle|, both planes, against the ~1 deg
                              physics floor.  Flat, INCLUDING the head-on bin.
  angle_resolution.json       every number the two figures and the slide quote.

Why this file exists
--------------------
The slide used to carry `angular_resolution.png` and `spatial_residuals.png`,
both lifted from `mx_june_cosmic_qa/engineer_package/figures/` and both
**hit-time chain, 2026-07-14 vintage** -- the estimator RECONSTRUCTION_BASIS.md
forbids for geometry, showing 1.66 deg while the tile beside them advertised
1.7 deg "hybrid" and the deck's own text claimed 1.0-1.1 deg from the forward
fit.  Three numbers, one slide, one of them from a superseded basis.

Everything here is computed from the frozen waveform-first table,
`<OUT_BASE>/wft/events.parquet`, against the same M3 reference recipe
(chi2 < 1.0, NClus = 4) and through the same rotation into the strip frame that
`mx_june_wft/03_angles.py` uses -- this is that script's accounting, drawn for
an audience instead of for a log file.  Cross-check: the full-sample sigma68
printed here must match `wft/angles/angular_resolution.json`'s `s68_deg`.

Coverage note (inherited from 03_angles.py, 2026-08-13): NO `slope_reliable`
gate.  That gate was a hits-chain inheritance -- a time ladder has no lever arm
on a head-on track -- and the forward fit measures the head-on band natively.
Gating it away would have hidden the most interesting bin on the figure.

THE SPATIAL AND TIMING NUMBERS ON THE SAME SLIDE DO NOT COME FROM HERE:

  * position   det4 in the SPS H4 beam, `sps_beam_test_26/analysis/
                spatial_resolution/` -- 176 um at normal incidence with the
                uRWELL reference fitted out rather than modelled.  The bench
                cannot give a per-axis position resolution honestly (its
                residual is reference- and scattering-limited); H4 can, because
                the DUT sits BETWEEN two reference planes.  The conservative
                band quoted on the slide is assembled in `spatial_band()`.
  * timing     the June bench, `mx_june_cosmic_qa/42_time_resolution.py` --
                33 ns plane-to-plane.  That measurement is already telescope-
                free (the X and Y layers time the same drifting electrons), so
                unlike the position number it has nothing to gain from a beam.
                There is no SPS timing extraction: det4 at H4 was mounted flat,
                so there is no drift-time ladder in that data at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import plotstyle as P  # noqa: E402

OUT = os.path.join(HERE, 'slides', 'assets', 'img')
DATA = os.path.join(HERE, 'data')

RUN_KEY = 'sat_det3'          # detector A, the chamber the slide is about

# |theta_ref| bin edges, degrees.  Chosen so every bin holds >= 200 planes on
# both views at this run's statistics -- see --print.  The first bin is the one
# that matters: it is the head-on band the hit-time ladder cannot do at all.
ANGLE_EDGES = [0.0, 2.0, 4.0, 7.0, 10.0, 14.0, 18.0]

# The physics floor: diffusion and the granularity of primary ionisation, from
# the toy closure of WAVEFORM_FIRST_THREADING.md §12.  Not a fit result -- a
# statement about what a 30 mm gap of this gas can do.
FLOOR_DEG = 1.0

PLANES = {'x': ('X view', P.DET_COLOR['A'], P.DET_MARKER['A']),
          'y': ('Y view', P.DET_COLOR['B'], P.DET_MARKER['B'])}


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #

def load_angles(run_key: str = RUN_KEY) -> dict:
    """{plane: (theta_ref_deg, theta_fit_deg)} from the frozen wft table.

    Deliberately the same seven steps as mx_june_wft/03_angles.py, in the same
    order, including the duplicate-event-id drop -- if this ever disagrees with
    that script's JSON, this file is the one that is wrong.
    """
    sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                    os.path.join(REPO, 'cosmic_bench_analysis')]
    from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
    setup_paths()
    import cosmic_micro_tpc_analysis as cm
    from M3RefTracking import M3RefTracking, get_xy_angles
    from wft import compat

    cfg = get_config(run_key)
    table = os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    align = os.path.join(cfg.OUT_BASE, 'wft', 'alignment', 'alignment.json')
    params = cm.load_alignment(align)
    df = compat.load_table(table)
    results = compat.as_event_results(df)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, _ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)

    ref = {}
    for r in results:
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, params)
        ref[int(r.event_id)] = (tx, ty)

    dup = df['event_id'].duplicated(keep=False)
    if dup.any():
        df = df[~dup]
    idx = df.set_index('event_id')

    out = {'run_key': run_key, 'table': table, 'planes': {}}
    for plane in ('x', 'y'):
        eids = [e for e in idx.index if e in ref and idx.loc[e, f'{plane}_ok']]
        tan_ref = np.array([ref[e][0 if plane == 'x' else 1] for e in eids])
        tan_fit = idx.loc[eids, f'{plane}_tan_theta'].to_numpy()
        ok = np.isfinite(tan_ref) & np.isfinite(tan_fit)
        out['planes'][plane] = (np.degrees(np.arctan(tan_ref[ok])),
                                np.degrees(np.arctan(tan_fit[ok])))
    return out


def s68(d: np.ndarray) -> tuple[float, float]:
    """(bias, sigma68) -- the median and the 68th percentile about it.

    sigma68 rather than a Gaussian fit on purpose: the residual has a tail from
    the events where the fit locked onto a second track or onto noise, and a
    Gaussian sigma either ignores the tail (fit the core) or is inflated by it
    (fit everything).  sigma68 is what it says it is at either extreme.
    """
    if not len(d):
        return float('nan'), float('nan')
    med = float(np.median(d))
    return med, float(np.percentile(np.abs(d - med), 68))


def s68_err(d: np.ndarray, n_boot: int = 400, seed: int = 20260817) -> float:
    """Bootstrap error on sigma68 -- there is no closed form worth trusting."""
    if len(d) < 20:
        return float('nan')
    rng = np.random.default_rng(seed)
    vals = [s68(d[rng.integers(0, len(d), len(d))])[1] for _ in range(n_boot)]
    return float(np.std(vals))


def resolve(angles: dict) -> dict:
    """Full-sample and per-|theta|-bin resolution, both planes."""
    res = {'run_key': angles['run_key'], 'floor_deg': FLOOR_DEG,
           'angle_edges_deg': ANGLE_EDGES, 'planes': {}}
    for plane, (thr, thf) in angles['planes'].items():
        d = thf - thr
        bias, sig = s68(d)
        rows = []
        for lo, hi in zip(ANGLE_EDGES[:-1], ANGLE_EDGES[1:]):
            m = (np.abs(thr) >= lo) & (np.abs(thr) < hi)
            b, s = s68(d[m])
            rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), bias_deg=b,
                             s68_deg=s, s68_err_deg=s68_err(d[m])))
        res['planes'][plane] = dict(n=int(len(d)), bias_deg=bias, s68_deg=sig,
                                    bins=rows)
    return res


# --------------------------------------------------------------------------- #
# The position number, from a different experiment
# --------------------------------------------------------------------------- #

SPS_JSON = os.path.join(REPO, 'sps_beam_test_26', 'analysis',
                        'spatial_resolution', 'results.json')


def spatial_band() -> dict:
    """det4's H4 position resolution with a deliberately pessimistic error.

    The fit's own uncertainty is +-10 um, and quoting that on a slide would be
    dishonest by omission: it is the error on an INTERCEPT, extrapolated to
    zero reference pitch, from three points, and three separate things about it
    are assumed rather than measured.  The band below adds, in quadrature:

      stat x sqrt(chi2/dof)  the fit's error, inflated because chi2/1 dof = 2.8
                             -- with one degree of freedom that is not evidence
                             of a bad model, but it is not evidence of a good
                             one either, and inflating is the cautious reading.
      zone spread            half the range of the five per-zone "det4 alone"
                             values (176-212 um).  Each zone illuminates a
                             different patch of the chamber, so this is a real
                             systematic and not statistical scatter.
      front plane            +-25 um, the report's own number for a factor-two
                             error on the front reference plane's assumed
                             resolution (its weight in the variance is 3.3 %).

    What the band does NOT cover, because no arithmetic can: this is det4, the
    fleet's WORST chamber by efficiency, and transferring it to det3 assumes
    the bench chambers are at least as good.  That assumption is stated on the
    slide, not hidden in an error bar.
    """
    with open(SPS_JSON) as f:
        d = json.load(f)
    fit = d['fit']
    zones = [z['sigma_det4'] for v in d['zones'].values() for z in v]
    stat = fit['sigma_det4_err'] * np.sqrt(max(fit['chi2'], 1.0))
    zone = 0.5 * (max(zones) - min(zones))
    front = 0.025
    tot = float(np.hypot(np.hypot(stat, zone), front))
    return dict(sigma_mm=fit['sigma_det4'], stat_mm=fit['sigma_det4_err'],
                stat_inflated_mm=float(stat), zone_mm=float(zone),
                front_mm=front, total_mm=tot,
                zone_values_mm=sorted(zones), chi2=fit['chi2'],
                f_back=fit['f_back'], f_binary=fit['f_binary'],
                pitch_mm=0.78)


# --------------------------------------------------------------------------- #
# Figure 1 -- reconstructed against reference
# --------------------------------------------------------------------------- #

def density_cmap():
    """White -> deck blue.  A density, so it wants a one-hue ramp."""
    return LinearSegmentedColormap.from_list(
        'density', ['#f4f7fb', '#c5d9ec', '#7fadd6', '#3d82bd', '#0072B2',
                    '#004c78'])


def fig_correlation(angles: dict, res: dict, lim: float = 20.0) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.1), sharey=True)
    bins = np.linspace(-lim, lim, 81)
    cmap = density_cmap()
    for ax, (plane, (label, colour, _mk)) in zip(axs, PLANES.items()):
        thr, thf = angles['planes'][plane]
        h, _, _, im = ax.hist2d(thr, thf, bins=[bins, bins], cmap=cmap,
                                norm=LogNorm(vmin=1, vmax=None), zorder=2)
        ax.plot([-lim, lim], [-lim, lim], color=P.INK, lw=1.2, ls='--',
                zorder=3, alpha=0.65)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('reference track angle  [deg]')
        ax.grid(True, color=P.LINE, lw=0.6, alpha=0.6, zorder=0)
        r = res['planes'][plane]
        ax.text(0.035, 0.965, label, transform=ax.transAxes, va='top',
                ha='left', fontsize=13, fontweight='bold', color=colour,
                zorder=4)
        ax.text(0.035, 0.895,
                f'{r["n"]:,} planes\nσ₆₈ {r["s68_deg"]:.2f}°\n'
                f'bias {r["bias_deg"]:+.2f}°',
                transform=ax.transAxes, va='top', ha='left', fontsize=11,
                color=P.INK, linespacing=1.5, zorder=4)
        P.strip(ax)
    axs[0].set_ylabel('reconstructed angle  [deg]')
    fig.colorbar(im, ax=axs, fraction=0.028, pad=0.02,
                 label='planes per bin')
    P.save(fig, os.path.join(OUT, 'angle_correlation.png'))


# --------------------------------------------------------------------------- #
# Figure 2 -- and how it behaves with angle
# --------------------------------------------------------------------------- #

def fig_vs_angle(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    ax.axhspan(0, FLOOR_DEG, color=P.ACCENT, alpha=0.09, lw=0, zorder=0)
    ax.axhline(FLOOR_DEG, color=P.ACCENT, lw=1.3, ls='--', zorder=1)
    # inside the shaded band and BELOW the lowest point (0.94 deg), so the one
    # bin that dips under the floor does not have to share pixels with it
    ax.text(0.35, 0.855,
            'the ~1° physics floor:  diffusion + ionisation granularity',
            ha='left', va='center', fontsize=10.5, color=P.ACCENT,
            fontweight='bold', zorder=4)

    for plane, (label, colour, marker) in PLANES.items():
        rows = res['planes'][plane]['bins']
        ctr = [0.5 * (r['lo'] + r['hi']) for r in rows]
        val = [r['s68_deg'] for r in rows]
        err = [r['s68_err_deg'] for r in rows]
        wid = [0.5 * (r['hi'] - r['lo']) for r in rows]
        ax.errorbar(ctr, val, yerr=err, xerr=wid, fmt=marker, color=colour,
                    ms=7, lw=0, elinewidth=1.6, capsize=0, zorder=3)
        ax.plot(ctr, val, color=colour, lw=1.6, alpha=0.55, zorder=2)
        # the direct label sits in the clear band at the top right, tied to its
        # own series by colour AND marker (the palette check requires both)
        ax.plot([18.9], [1.60 if plane == 'x' else 1.50], marker=marker,
                color=colour, ms=7, zorder=3)
        P.end_label(ax, 19.4, 1.60 if plane == 'x' else 1.50, label, colour)

    ax.set_xlim(0, 22.4)
    ax.set_ylim(0.78, 1.85)
    ax.set_xlabel('|reference track angle|  [deg]')
    ax.set_ylabel('σ₆₈ (reconstructed − reference)  [deg]')
    ax.set_xticks([0, 5, 10, 15, 20])
    P.strip(ax)

    # The head-on bin is the whole point of the figure -- say so on it.
    x = res['planes']['x']['bins'][0]
    y = res['planes']['y']['bins'][0]
    ax.annotate('head-on: no drift-time lever arm,\n'
                'and the fit still measures the angle',
                xy=(1.9, max(x['s68_deg'], y['s68_deg'])),
                xytext=(4.2, 1.63), fontsize=10.5, color=P.INK,
                ha='left', va='top', linespacing=1.4,
                arrowprops=dict(arrowstyle='->', color=P.MUTED, lw=1.2,
                                connectionstyle='arc3,rad=0.22'))
    P.save(fig, os.path.join(OUT, 'angle_resolution.png'))


# --------------------------------------------------------------------------- #

def report(res: dict, band: dict) -> None:
    print(f'\nangular resolution -- {res["run_key"]}, waveform-first')
    for plane, r in res['planes'].items():
        print(f'  {plane}: n={r["n"]:,}  bias {r["bias_deg"]:+.2f} deg  '
              f'sigma68 {r["s68_deg"]:.2f} deg')
        for b in r['bins']:
            print(f'      |theta| {b["lo"]:4.0f}-{b["hi"]:<4.0f} '
                  f'n={b["n"]:5d}  bias {b["bias_deg"]:+.2f}  '
                  f'sigma68 {b["s68_deg"]:.2f} +- {b["s68_err_deg"]:.2f}')
    print(f'\nposition (det4, SPS H4, normal incidence)')
    print(f'  fit intercept      {band["sigma_mm"]*1e3:.0f} um '
          f'+- {band["stat_mm"]*1e3:.0f} um (stat)')
    print(f'  inflated by chi2   +- {band["stat_inflated_mm"]*1e3:.0f} um '
          f'(chi2/1 dof = {band["chi2"]:.1f})')
    print(f'  zone spread        +- {band["zone_mm"]*1e3:.0f} um  '
          f'({", ".join(f"{z*1e3:.0f}" for z in band["zone_values_mm"])} um)')
    print(f'  front plane        +- {band["front_mm"]*1e3:.0f} um')
    print(f'  CONSERVATIVE       {band["sigma_mm"]*1e3:.0f} '
          f'+- {band["total_mm"]*1e3:.0f} um  '
          f'({(band["sigma_mm"]-band["total_mm"])*1e3:.0f}-'
          f'{(band["sigma_mm"]+band["total_mm"])*1e3:.0f} um), on a '
          f'{band["pitch_mm"]:.2f} mm pitch')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=('correlation', 'vsangle'), default=None)
    ap.add_argument('--print', dest='show', action='store_true')
    args = ap.parse_args()

    angles = load_angles()
    res = resolve(angles)
    band = spatial_band()
    res['spatial_sps'] = band
    report(res, band)
    if args.show:
        return

    P.use()
    matplotlib.rcParams['savefig.dpi'] = 220      # projector legibility
    if not args.only or args.only == 'correlation':
        fig_correlation(angles, res)
    if not args.only or args.only == 'vsangle':
        fig_vs_angle(res)

    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, 'angle_resolution.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'  -> {os.path.join(DATA, "angle_resolution.json")}')


if __name__ == '__main__':
    main()
