#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scint_acceptance.py -- what the arm-A scintillators' acceptance looks like from
the chamber, using the n_TOF <-> DREAM merge.

The chamber gives a track (position *and* angle, from the waveform-first fit in
`ntof_tracking/wft_beam.py`); the merge says which n_TOF scintillator fired the
trigger for that same event.  Extrapolating the track to each scintillator's
plane and asking "was it tagged?" as a function of where it landed maps out the
detector's acceptance.

Two honest limits, both measured here rather than assumed away:

* **Pointing resolution.**  Not every trigger particle is the one the chamber
  reconstructed, the angle scale is ~0.8 of truth (RUN79_PRELIM §3), and the
  lever arm to the plastics is 190 mm.  Every model below is therefore a sharp
  geometric shape *convolved with a Gaussian* whose width is a free parameter.
* **Accidental tags.**  The DREAM trigger is an OR over all four arms, so an
  arm-A track can carry a tag that some other particle in the same bunch
  produced.  That is a flat pedestal under every acceptance curve, also free.

With both floated, the *edges* are what the data constrains; the plateau height
is not an efficiency.

Run:
    .venv/bin/python -m ntof_active_area.scint_acceptance
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
MERGED = Path('/media/dylan/data/x17/beam_july/analysis/wft/run_79/'
              'stat090_0000/mx17_A/merged_prelim.parquet')

# arm-A geometry, run_79/run_config.json `det_center_coords` (structure frame,
# origin at the He-3 target; for arm A the tangential coordinate is global x).
Z_MM, Z_WALL, Z_PSS, Z_LIQ = 234.6, 332.0, 425.22, 483.1
MM_CENTRE_X = -16.35          # chamber A's pinwheel shift
# The chamber's own u axis runs ANTI-parallel to structure x -- established by
# the plastic L/R split below and consistent with the 'descending' wall-segment
# mapping in RUN79_PRELIM_2026-07-30 §4.  structure_x = -u - MM_CENTRE_X... i.e.
#   u(structure_x) = -(structure_x - MM_CENTRE_X)
U_SIGN = -1.0


def u_of_structure_x(x):
    return U_SIGN * (np.asarray(x, float) - MM_CENTRE_X)


# nominal scintillator footprints, structure frame, from run_config.json
PLASTIC = {1: (-118.07, 200.0), 2: (85.37, 200.0)}      # detn -> (centre x, width)
PLASTIC_V_FULL = 300.0
WALL_BARS = np.arange(-212.5, 162.5 + 1, 25.0)          # 16 bar centres
WALL_V_FULL = 500.0
WALL_SEG_WIDTH = 100.0                                  # 4 bars per n_TOF channel


def _erf(z):
    from scipy.special import erf
    return erf(z)


def smeared_step(u, u0, sigma, lo, hi):
    """P(inside [lo, hi]) for a point measured with Gaussian error `sigma`,
    where the true position is `u - u0`."""
    s = max(sigma, 1e-6)
    z = (np.asarray(u, float) - u0)
    return 0.5 * (_erf((hi - z) / (np.sqrt(2) * s))
                  - _erf((lo - z) / (np.sqrt(2) * s)))


def load_tracks() -> pd.DataFrame:
    d = pd.read_parquet(MERGED)
    q = d[d.x_ok & d.x_quality_ok & d.y_ok & d.y_quality_ok].copy()
    for name, z in (('wall', Z_WALL), ('pss', Z_PSS), ('liq', Z_LIQ)):
        # the outward extrapolation carries a minus sign (RUN79_PRELIM §4)
        q[f'u_{name}'] = q.u_mm - (z - Z_MM) * q.x_tan_theta
        q[f'v_{name}'] = q.v_mm - (z - Z_MM) * q.y_tan_theta
    good = np.ones(len(q), bool)
    for c in ('u_wall', 'v_wall', 'u_pss', 'v_pss'):
        good &= np.isfinite(q[c]) & (q[c].abs() < 1500)
    return q[good]


def fit_binary(x, y, model, p0, bounds):
    """Least-squares fit of a binary outcome binned in `x`."""
    from scipy.optimize import curve_fit
    edges = np.histogram_bin_edges(x, bins=28)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    n = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    k = np.bincount(idx, weights=y.astype(float), minlength=len(edges) - 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    ok = n >= 25
    p_hat = k[ok] / n[ok]
    err = np.sqrt(np.maximum(p_hat * (1 - p_hat), 0.01) / n[ok])
    p, cov = curve_fit(model, c[ok], p_hat, p0=p0, sigma=err,
                       absolute_sigma=True, bounds=bounds, maxfev=40000)
    return p, np.sqrt(np.diag(cov)), (c[ok], p_hat, err, n[ok])


def plastic_lr_boundary(q):
    """Where the L bar stops and the R bar starts, in chamber-u at the plastic
    plane.  The two bars abut on the chamber's own centre line, so the geometry
    predicts u0 = 0; this is the sharpest geometric statement the merge can make
    because it is a *boundary*, not an edge, so no acceptance falls off at it.
    """
    m = q.pss_detn.notna()
    x, y = q.u_pss[m].to_numpy(), (q.pss_detn[m] == 1).to_numpy()

    def model(u, u0, sigma, amp, base):
        return base + amp * 0.5 * (1 + _erf((u - u0) / (np.sqrt(2) * max(sigma, 1e-6))))

    p, e, binned = fit_binary(x, y, model, [0.0, 60.0, 0.7, 0.15],
                              ([-150, 5, 0.1, 0.0], [150, 400, 1.0, 0.5]))
    return dict(u0_mm=float(p[0]), u0_err_mm=float(e[0]),
                sigma_mm=float(p[1]), sigma_err_mm=float(e[1]),
                amplitude=float(p[2]), pedestal=float(p[3]),
                predicted_u0_mm=float(u_of_structure_x(
                    0.5 * (PLASTIC[1][0] + PLASTIC[1][1] / 2
                           + PLASTIC[2][0] - PLASTIC[2][1] / 2))),
                n=int(m.sum()), binned=[a.tolist() for a in binned])


def half_length(q, coord, tag, nominal_half, sigma_guess):
    """Half-extent of a detector along one axis, from where its tag turns off.

    Model: pedestal + amp * P(|position| < half), the box smeared by `sigma`.
    `half` and `sigma` both float, so the fit separates "the detector ends here"
    from "our pointing is this blurry".
    """
    x, y = q[coord].to_numpy(), tag.to_numpy()

    def model(v, half, sigma, amp, base):
        return base + amp * smeared_step(v, 0.0, sigma, -half, half)

    p, e, binned = fit_binary(x, y, model, [nominal_half, sigma_guess, 0.4, 0.15],
                              ([40, 5, 0.05, 0.0], [700, 400, 1.0, 0.6]))
    # Is the answer the data's or the model's?  The edge is only constrained
    # when the blur is small against the extent being measured AND the tagged
    # plateau stands well clear of the accidental pedestal.  Both fail here for
    # every outer dimension, so the flag is part of the result, not a footnote.
    contrast = float(p[2] / (p[2] + p[3])) if (p[2] + p[3]) > 0 else 0.0
    constrained = bool(p[1] < 0.35 * p[0] and contrast > 0.75
                       and e[0] < 0.15 * p[0])
    return dict(half_mm=float(p[0]), half_err_mm=float(e[0]),
                sigma_mm=float(p[1]), sigma_err_mm=float(e[1]),
                amplitude=float(p[2]), pedestal=float(p[3]),
                contrast=contrast, constrained=constrained,
                nominal_half_mm=nominal_half, n=int(len(x)),
                binned=[a.tolist() for a in binned])


def wall_segments(q):
    """Per n_TOF wall channel-pair, the mean chamber-u of the tracks it tagged.

    The four segments are 100 mm wide and adjacent, so their tagged-track means
    must climb (or fall) by 100 mm a step.  Pointing blur compresses that slope
    towards the sample mean without moving its sign, so this measures the
    segment *ordering* and the *offset* of the readout window well and its pitch
    only after the compression is divided out -- which is what `slope_ratio`
    below reports rather than hides.
    """
    m = q.wal_detn.notna()
    seg = ((q.wal_detn[m].to_numpy().astype(int) - 1) // 2)
    u = q.u_wall[m].to_numpy()
    rows = []
    for s in range(4):
        k = seg == s
        if k.sum() < 30:
            continue
        lo = -225.0 + 100.0 * s              # structure-frame span of segment s
        centre_u = float(u_of_structure_x(lo + 50.0))
        rows.append(dict(seg=int(s), n=int(k.sum()),
                         u_median=float(np.median(u[k])),
                         u_mean=float(np.mean(u[k])),
                         predicted_centre_u_mm=centre_u))
    if len(rows) >= 2:
        obs = np.array([r['u_mean'] for r in rows])
        pred = np.array([r['predicted_centre_u_mm'] for r in rows])
        slope = float(np.polyfit(pred, obs, 1)[0])
        corr = float(np.corrcoef(pred, obs)[0, 1])
    else:
        slope = corr = float('nan')
    return dict(segments=rows, slope_ratio=slope, ordering_corr=corr,
                n=int(m.sum()))


def measure(save: bool = True) -> dict:
    q = load_tracks()
    res = {'source': str(MERGED), 'n_tracks': int(len(q)),
           'note': ('arm A only; the merge exists for run_79/stat090_0000 tags '
                    '000-002 (RUN79_PRELIM_2026-07-30)'),
           'plastic_lr_boundary': plastic_lr_boundary(q),
           'plastic_v': half_length(q, 'v_pss', q.pss_detn.notna(),
                                    PLASTIC_V_FULL / 2, 80.0),
           'plastic_u': half_length(q, 'u_pss', q.pss_detn.notna(),
                                    PLASTIC[1][1], 80.0),
           'wall_v': half_length(q, 'v_wall', q.wal_detn.notna(),
                                 WALL_V_FULL / 2, 60.0),
           'wall_segments': wall_segments(q)}
    if save:
        (OUT / 'results_scint.json').write_text(json.dumps(res, indent=1))
    return res


def _print(r):
    b = r['plastic_lr_boundary']
    print(f"tracks {r['n_tracks']}")
    print(f"\nplastic L/R boundary  u0 = {b['u0_mm']:+.1f} +- {b['u0_err_mm']:.1f} mm "
          f"(geometry says {b['predicted_u0_mm']:+.1f})"
          f"   pointing sigma = {b['sigma_mm']:.0f} +- {b['sigma_err_mm']:.0f} mm")
    for key, label, nom in (('plastic_v', 'plastic half-length along beam', 150.0),
                            ('plastic_u', 'plastic pair half-width', 200.0),
                            ('wall_v', 'SiPM wall half-length along beam', 250.0)):
        f = r[key]
        print(f"{label:36s} {f['half_mm']:6.0f} +- {f['half_err_mm']:.0f} mm "
              f"(survey {nom:.0f})   sigma {f['sigma_mm']:.0f} mm, "
              f"contrast {f['contrast']:.2f}  -> "
              f"{'CONSTRAINED' if f['constrained'] else 'NOT constrained'}")
    w = r['wall_segments']
    print(f"\nwall segments (n={w['n']}), ordering corr {w['ordering_corr']:+.3f}, "
          f"slope {w['slope_ratio']:.2f} of geometric")
    for s in w['segments']:
        print(f"  seg {s['seg']}  n={s['n']:5d}  mean u {s['u_mean']:+7.1f} "
              f"(predicted centre {s['predicted_centre_u_mm']:+7.1f})")


if __name__ == '__main__':
    _print(measure())
