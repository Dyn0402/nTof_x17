#!/usr/bin/env python3
"""
measure.py -- run the angled-mount kernel measurement and write results.json.

Three measurements, all on run_63's own flat / 25.64 deg A/B (see kernel_lib
for why that pairing is clean and which view is tilted):

  M1  KERNEL A/B.  The X view is at normal incidence in BOTH mounts, so its
      kernel observables must not care that the chamber was rotated.  This is
      the "do the flat calibration numbers hold" test, and it comes with a
      2.2x drift lever for free (233 / 142 / 108 V/cm).

  M2  GEOMETRY LEVER.  The Y view is normal in the flat mount and carries the
      25.64 deg ladder in the rotated one.  The difference between M1 and M2 is
      pure track geometry, at a known angle, with an external telescope.

  M3  LATERAL WIDTH vs DRIFT FIELD.  Charge-weighted rms of the X view about
      the telescope-predicted impact point.  Transverse diffusion scales as
      1/sqrt(E); the film does not.  So a field-independent width bounds the
      diffusion term directly.

    ../../../.venv/bin/python measure.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_lib as K                                    # noqa: E402

ARM_ORDER = ['flat700', 'rot_d425', 'rot_d325', 'rot_d225']
Q0 = (200.0, 3000.0)


def main():
    out = {'meta': dict(q0_adc=list(Q0), gap_mm=K.GAP_MM, tilt_deg=K.TILT_DEG,
                        pitch_mm=float(K.PITCH_MM), sample_ns=K.SNS,
                        zs_sigma=4.0, resist_V=769.8,
                        gas='Ar/CF4/iso 88/10/2',
                        note='run_63 both mounts, one TAX access apart')}
    for arm in ARM_ORDER:
        a = K.ARMS[arm]
        d = K.load_arm(arm)
        if len(d['x']) < 300:
            print(f'{arm}: only {len(d["x"])} events -- skipped')
            continue
        rec = dict(drift_V=a['drift_V'], mount=a['mount'],
                   E_Vcm=a['drift_V'] / K.GAP_MM * 10.0,
                   tilted_view=a['tilted'], n_events=len(d['x']))
        for v in ('x', 'y'):
            sl_signed, sl_abs, n_sl = K.ladder_slope(d[v])
            o = K.neighbour_stack(d[v], q0lo=Q0[0], q0hi=Q0[1])
            o['ladder_signed_ns_per_mm'] = sl_signed
            o['ladder_abs_ns_per_mm'] = sl_abs
            o['incidence'] = ('25.64 deg' if a['tilted'] == v else 'normal')
            # implied drift velocity from the ladder, where there IS one
            if a['tilted'] == v and np.isfinite(sl_signed) and abs(sl_signed) > 50:
                o['v_from_ladder_um_ns'] = float(
                    1e3 / abs(sl_signed) / np.tan(np.radians(K.TILT_DEG)))
            rec[v] = o
        m = K.telescope_map(d, 'x')
        if m is not None:
            coef, mad, nfit = m
            rec['telescope_x'] = dict(coef=[float(q) for q in coef],
                                      mad_mm=mad, n_fit=nfit)
            rec['width_x'] = K.width_vs_time(d, 'x', coef, q0lo=Q0[0],
                                             q0hi=Q0[1])
        out[arm] = rec
        print(f'{arm}: done ({rec["n_events"]} ev)', flush=True)

    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote results.json')

    # ---- console summary ------------------------------------------------
    print('\nM1/M2 -- neighbour kernel, trim20 stacks, ZS 4 sigma throughout')
    print(f"{'arm':10}{'view':5}{'incidence':>11}{'E':>6}{'n':>6}{'wid':>5}"
          f"{'pm1 pk':>8}{'pm2 pk':>8}{'pm2/pm1':>9}{'pm1 ar':>8}"
          f"{'sh+1':>6}{'sh-1':>6}{'asym':>6}{'det1':>6}")
    for arm in ARM_ORDER:
        if arm not in out:
            continue
        for v in ('x', 'y'):
            o = out[arm][v]
            if 'pk_+1' not in o:
                continue
            print(f"{arm:10}{v:5}{o['incidence']:>11}{out[arm]['E_Vcm']:6.0f}"
                  f"{o['n_events']:6d}{o['width_20pct']:5.0f}"
                  f"{K.sym(o,'pk'):8.4f}{K.sym(o,'pk',2):8.4f}"
                  f"{K.sym(o,'pk',2)/K.sym(o,'pk'):9.4f}{K.sym(o,'area'):8.4f}"
                  f"{o['shift_p1_ns']:+6.0f}{o['shift_m1_ns']:+6.0f}"
                  f"{o['shift_asym_ns']:+6.0f}{K.sym(o,'detfrac'):6.3f}")

    print('\nM3 -- lateral width about the telescope impact point (X view)')
    print(f"{'arm':10}{'E':>6}{'n':>7}{'MAD':>7}{'sigma@pk':>10}"
          f"{'sigma+900ns':>13}{'nstrip@pk':>11}")
    for arm in ARM_ORDER:
        if arm not in out or 'width_x' not in out[arm]:
            continue
        w = out[arm]['width_x']
        s = np.array(w['sigma_mm']); q = np.array(w['charge'])
        ns = np.array(w['nstrip'])
        pk = int(np.nanargmax(q))
        j = min(pk + 15, len(s) - 1)
        print(f"{arm:10}{out[arm]['E_Vcm']:6.0f}{w['n_events']:7d}"
              f"{out[arm]['telescope_x']['mad_mm']:7.3f}{s[pk]:10.3f}"
              f"{s[j]:13.3f}{ns[pk]:11.2f}")


if __name__ == '__main__':
    main()
