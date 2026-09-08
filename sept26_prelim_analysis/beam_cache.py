#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
beam_cache.py -- a ref-free training cache, so a bundle can be fitted on BEAM data.

WHY THIS EXISTS.  Every one of the four beam bundles is a bench transfer: its
kernel was fitted on cosmics against the M3 reference telescope and only ``v``,
``sat_adc`` and the sample grid were replaced.  Chamber B is the one chamber
that will not calibrate, five other explanations have been eliminated
(STATUS.md "Chamber B"), and a kernel that does not describe B in the beam is
what is left.  But ``wft.calibrate.build_cache`` cannot run on beam data at
all: it selects its training events along the M3 corridor and fits the model
against per-event reference track parameters.  There is no telescope in EAR2.

THE BEAM'S REFERENCE IS THE TARGET.  The source is the He-3 capsule on the beam
axis, ``d_perp`` = 234.6 mm from every strip plane, and a track from it must
satisfy

    tan = (u - foot_x) / d_perp

so the *position* fixes the *angle* with no telescope and no drift velocity.
That is the same relation :mod:`k_arm` uses to measure the angle scale; here it
supplies the truth a hyper fit needs.

**AND IT IS GOOD IN X ONLY.**  This is the design constraint, so it is measured
rather than assumed (``geometry.HE3_R_MAX``, ``HE3_GAS_Y``):

    plane   what the capsule is        truth precision per track
    x       a point, radius 10 mm      d(tan) = 0.043 -- 21 % of a typical
                                       |tan| ~ 0.20, so 1.6 % on the mean of 180
    y       80.2 mm long along y       d(tan) = 0.171 -- 85 % per track,
                                       comparable to the angle itself

So the x plane can pin a fit and the y plane cannot.  The consequence is
built in rather than papered over: ``truth_ok_y`` is False on every event, and
:func:`hypers_to_fit` returns the shared kernel parameters only, leaving ``kY``
-- the one parameter that is specifically about the Y plane -- at its
transferred value.  A caller that wants kY from the beam has to supply another
constraint; there is nothing here that can honestly provide it.

CIRCULARITY, AND WHY IT IS TOLERABLE.  ``u`` comes from the existing
reconstruction's ``p0``, which was produced with the very bundle being
replaced.  Three things make that acceptable, and the third is the one that
must be checked rather than argued:

  1. ``p0`` does not depend on the drift velocity at all (the fit estimates
     position, transverse speed and t0; only tan = w/v needs v).
  2. The sharing kernel redistributes charge across strips roughly
     symmetrically, so it moves a centroid far less than it moves a slope.
  3. It has to be a fixed point.  :mod:`k_arm` demonstrated exactly this for
     the angle scale -- re-deriving the sample with k applied moved 8-21 % of
     the rows and changed k by less than a grid step -- and the same iteration
     is the acceptance test here: fit hypers, re-reconstruct, re-fit, and
     require the hypers to stop moving.  Until that is run, a bundle from this
     cache is a candidate and nothing more.

WHAT HAPPENED WHEN IT WAS RUN (2026-09-08).  It works mechanically and has no
power.  chi2 improved by 0.028 % on 60 training events and 0.080 % on 180,
against the 23-27 % the bench ref-pinned fits achieve on the same seven
parameters; at 180 the optimiser wandered to c1 = 0.743, tau_s = 1.9 ns and
sigma_s = 5 ns for that 0.08 %.  The reason is the circularity above surfacing
as powerlessness rather than as a wrong answer: the chi2 can be satisfied by
moving the track instead of by getting the kernel right, because the truth is a
function of the fit's own p0.  **The target constrains one number -- the
position-angle relation, which is what k_arm measures -- not a seven-parameter
kernel.**

AND NO EXTERNAL REFERENCE IN THE BEAM IS GOOD ENOUGH EITHER.  A calibration
needs truth independent of the waveform, so the candidates are the
scintillators, and their granularity settles it (uniform segments, so sigma =
half-width/sqrt(3)):

    pair                      lever     d(tan)
    target + wall group       331 mm     0.089
    target + plastic bar      421 mm     0.138
    wall + plastic             90 mm     0.720

The best of them is 0.089, twice as coarse as the circular target truth that
already failed, and 70 % of a typical |tan| in chamber B.  The wall's u
granularity is 100 mm because detn resolves 4 groups of 4 bars; its 8 values
are those 4 groups x top/bottom, and the parity is a y distinction, not a finer
u one.

SO: this module stands as a working, validated selector and as the measurement
that the approach cannot calibrate a kernel.  The remaining avenue is a
DIFFERENT KIND of calibration -- ensemble rather than per-event.  The target
already pins one number from a distribution; a kernel might be pinned the same
way, by matching distributions the kernel controls (cluster width, chi2/dof,
the residual structure across strips) rather than by per-event truth.  That is
new method development.  A second, smaller idea: the wall's top/bottom
amplitude ratio should give position along the bar, which is the y handle the
capsule's 80 mm length denies -- useful for kY, useless for the shared kernel.

    python -m sept26_prelim_analysis.beam_cache --arm B --report
    python -m sept26_prelim_analysis.beam_cache --arm B --build --events 400
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.build_tracks import (  # noqa: E402
    IN_PLANE_SIGN, IN_PLANE_SIGN_Y, STRIP_MAP_HALF)

D_PERP_MM = 234.6
#: Window along the drift direction the training corridor spans, in mm, and the
#: padding either side.  Same numbers build_cache uses, so the channel windows
#: are comparable between a bench and a beam cache.
Z_LO, Z_HI, PAD_MM = -3.0, 33.0, 5.0

#: Hypers the x-plane truth can honestly constrain.  ``kY`` is excluded by
#: construction -- it scales the Y kernel, and the capsule is 80 mm long along
#: y, so there is no y truth to fit it against.
SHARED_HYPERS = ('c1', 'c2', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')
Y_ONLY_HYPERS = ('kY',)


def hypers_to_fit(with_y: bool = False) -> tuple:
    """Which hypers this cache can pin.  ``with_y`` is a caller's override and
    is not recommended: see the module docstring for what the y truth is worth."""
    return SHARED_HYPERS + (Y_ONLY_HYPERS if with_y else ())


def truth_from_target(x_local, y_local, foot_x):
    """(tan_x, tan_y, ref_mesh_x, ref_mesh_y) from the target constraint.

    ``tan_y`` is returned for completeness and is NOT truth -- see
    ``truth_ok_y``.  It assumes the capsule's mid-point, which is wrong by up
    to its half-length on any given track.
    """
    from ntof_tracking.reco import geometry as G
    y_mid = 0.5 * (float(G.HE3_GAS_Y[0]) + float(G.HE3_GAS_Y[-1]))
    return (np.asarray(x_local - foot_x, float) / D_PERP_MM,
            np.asarray(y_local - y_mid, float) / D_PERP_MM,
            np.asarray(x_local, float), np.asarray(y_local, float))


def truth_precision() -> dict:
    """What the capsule geometry is worth as a reference, per plane."""
    from ntof_tracking.reco import geometry as G
    y = np.asarray(G.HE3_GAS_Y, float)
    return dict(d_tan_x=float(G.HE3_R_MAX) / D_PERP_MM,
                d_tan_y=float(y.max() - y.min()) / 2.0 / D_PERP_MM,
                capsule_r_mm=float(G.HE3_R_MAX),
                capsule_len_mm=float(y.max() - y.min()))


def training_events(run: str, subrun: str, arm: str, merged_dir: str) -> pd.DataFrame:
    """The purest sample available, with target-derived truth attached.

    Purity is the pointing coincidence -- the track extrapolates to the wall
    segment AND the plastic bar that actually fired -- intersected with the
    charge window that k_arm uses.  Both are already measured selections; this
    module invents no new cut.
    """
    from sept26_prelim_analysis import k_arm as KA
    from ntof_tracking import run145_target_imaging as TI

    p = paths.require(os.path.join(merged_dir, subrun, f'mx17_{arm}',
                                   'events_prelim.parquet'),
                      f'reconstruction for {arm}/{subrun}')
    df = pd.read_parquet(p)
    d = os.path.join(str(paths.root('runs')), run, subrun, 'ntof_hits')
    slim = sorted(f for f in os.listdir(d) if f.endswith('.root'))
    if not slim:
        raise FileNotFoundError(f'no slim n_TOF file under {d}')
    sel = (df['x_ok'].to_numpy() & df['y_ok'].to_numpy()
           & (df['n_tracks'].to_numpy() > 0))
    coin, _ = TI.pointing_coincidence(os.path.join(d, slim[0]), arm, df, sel,
                                      foot_x=TI.PINWHEEL[arm])
    q = df['x_q_sum'].to_numpy()
    m = coin & sel & np.isfinite(q) & (q > 0)
    lo, hi = np.percentile(q[m], KA.CHARGE_WINDOW)
    m &= (q >= lo) & (q <= hi)

    g = df[m]
    xl = IN_PLANE_SIGN * (g['x_p0'].to_numpy() - STRIP_MAP_HALF)
    yl = IN_PLANE_SIGN_Y * (g['y_p0'].to_numpy() - STRIP_MAP_HALF)
    tx, ty, mx, my = truth_from_target(xl, yl, TI.PINWHEEL[arm])
    return pd.DataFrame(dict(
        event_id=g['event_id'].to_numpy(), tan_x=tx, tan_y=ty,
        ref_mesh_x=mx, ref_mesh_y=my,
        tan_x_reco=g['x_tan_theta'].to_numpy(),
        tan_y_reco=g['y_tan_theta'].to_numpy(),
        q_sum=g['x_q_sum'].to_numpy(),
        truth_ok_x=True, truth_ok_y=False))


def build(run: str, subrun: str, arm: str, merged_dir: str, n_events: int,
          out_path: str | None = None) -> dict:
    """A cache in ``wft.calibrate.build_cache``'s format, from beam data.

    Same keys, so ``calibrate()`` consumes it unchanged: per event the truth
    (tan_x, tan_y, ref_mesh_x, ref_mesh_y) and, per plane, the channel window
    with its waveforms and noise.
    """
    from ntof_tracking.wft_beam import beam_config
    from wft import io as wio

    T = training_events(run, subrun, arm, merged_dir)
    print(f'[beam-cache] {len(T):,} pointing-coincident tracks in the charge window')
    events = {int(r.event_id): dict(eid=int(r.event_id), tan_x=float(r.tan_x),
                                    tan_y=float(r.tan_y),
                                    ref_mesh_x=float(r.ref_mesh_x),
                                    ref_mesh_y=float(r.ref_mesh_y))
              for r in T.itertuples(index=False)}

    cfg = beam_config(arm, run, subrun)
    pos_maps = wio.strip_position_map(cfg)
    for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        pm = pos_maps[feu]
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            rdr = wio.FeuReader(f)
            want = set(events) & set(rdr.event_ids.tolist())
            if not want:
                continue
            for eid, ftst, wfm in rdr.iter_events(want):
                ev = events[eid]
                # The corridor is built in the STRIP-MAP frame, so undo the
                # in-plane sign that put ref_mesh into the local frame.
                sign = IN_PLANE_SIGN if plane == 'x' else IN_PLANE_SIGN_Y
                p0 = sign * ev[f'ref_mesh_{plane}'] + STRIP_MAP_HALF
                tn = sign * ev[f'tan_{plane}']
                a, b = p0 + Z_LO * tn, p0 + Z_HI * tn
                lo, hi = min(a, b) - PAD_MM, max(a, b) + PAD_MM
                ch = np.where((pm >= lo) & (pm <= hi))[0]
                ch = ch[np.argsort(pm[ch])]
                if len(ch) < 4:
                    continue
                ev[plane] = dict(ch=ch.astype(np.int16),
                                 pos=pm[ch].astype(np.float32),
                                 W=wfm[ch].astype(np.float32),
                                 noise=np.maximum(rdr.noise[ch], 3.0).astype(np.float32))
                ev[f'ftst_{plane}'] = ftst
            if sum(1 for e in events.values() if plane in e) >= n_events:
                break
    events = {k: v for k, v in events.items() if 'x' in v and 'y' in v}
    events = {k: events[k] for k in sorted(events)[:n_events]}
    print(f'[beam-cache] {len(events):,} events with both waveform windows')
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as fh:
            pickle.dump(events, fh, protocol=4)
        print(f'[beam-cache] wrote {out_path}')
    return events


def report(run: str, subrun: str, arm: str, merged_dir: str) -> None:
    """What the target reference is worth for this arm, before building anything."""
    pr = truth_precision()
    T = training_events(run, subrun, arm, merged_dir)
    tan = np.abs(T.tan_x_reco.to_numpy())
    med = float(np.median(tan))
    print(f'arm {arm}  {run}/{subrun}')
    print(f'  training candidates      {len(T):,}')
    print(f'  capsule: r = {pr["capsule_r_mm"]:.0f} mm, length {pr["capsule_len_mm"]:.1f} mm')
    print(f'  truth d(tan_x)           {pr["d_tan_x"]:.4f}')
    print(f'  truth d(tan_y)           {pr["d_tan_y"]:.4f}   (NOT usable)')
    print(f'  median |tan_x| in data   {med:.3f}')
    print(f'  -> x truth is {100 * pr["d_tan_x"] / max(med, 1e-9):.0f} % per track, '
          f'{100 * pr["d_tan_x"] / max(med, 1e-9) / np.sqrt(min(len(T), 180)):.1f} % '
          f'on a {min(len(T), 180)}-event mean')
    print(f'  hypers this cache can pin: {", ".join(hypers_to_fit())}')
    print(f'  left at the transferred value: {", ".join(Y_ONLY_HYPERS)}')
    # How far the current reconstruction is from the target constraint -- the
    # thing a refit would be trying to reduce.
    r = T.tan_x_reco.to_numpy() / np.where(T.tan_x.to_numpy() == 0, np.nan,
                                           T.tan_x.to_numpy())
    print(f'  median tan_reco / tan_target = {np.nanmedian(r):.3f} '
          f'(1/k, and k_arm measures the same thing)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--arm', default='B')
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--merged', default=str(paths.out('fullpass') / 'run_145'))
    ap.add_argument('--events', type=int, default=400)
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    if a.build:
        out = a.out or str(paths.out('kcal', 'beam_cache')
                           / f'{a.run}_{a.subrun}_mx17_{a.arm}.pkl')
        build(a.run, a.subrun, a.arm, a.merged, a.events, out_path=out)
        return 0
    report(a.run, a.subrun, a.arm, a.merged)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
