#!/usr/bin/env python3
"""ntof_bundle.py -- P3 of NTOF_PASS_PLAN_2026-11-02: an n_TOF bundle from the chamber's pass-1 golden bundle.

    ntof_bundle.py <arm> --out DIR [--src DIR] [--v V] [--k K] [--sat 3550] [--no-bow] [--latency lat.json]

What carries, what is replaced, and why (each choice is written into the bundle's provenance):

  carried (chamber properties, pass 1 code_6a0bb1f)
      physical RC kernel hypers (sigma_p0 0.30/0.30, rc_D_y, rc_D_x 0, rc_tmpl 'x'), the X template, the dead list
      (census dead + shorts; the bench and n_TOF channel -> strip maps are identical on all four chambers, both views,
      all 512 channels -- checked 2026-11-02, so the list carries by channel number), kw (provisional, to be re-measured
      in situ), and the BOW block (Dylan 11-02: carried, then tested bow-on vs bow-off on the run_149 A-C line and the
      capsule focus).  Of the bow block the tilt (a_*), scale (s_*) and mesh-height (h_*) surfaces carry; the bench
      rigid tilt (w0_const_*) belongs to the bench mounting and is dropped with w0.
  dropped
      v_map (bench v(x,y) of one night), w0 (rigid tilt of the bench mounting), the bench t0_abs (the bench latency).
  replaced
      v_drift (argument; the column-duration measurement replaces it, P3), K = n_depth_bins (default 20 = the whole
      1.2 us window; is2's 18 railed q_uend at 1080 ns on ~40 % of tracks), sample_ns / n_samples / conditions from
      the registry, sat_adc (default 3550, the locked bench clip rule; Q5 checks 3700), and t0: free unless
      --latency gives an n_TOF latency (then t0_abs = C - 10 ns * ftst, sigma from --t0-sigma).
  still bench values, flagged
      Dp (gas-dependent: bench 95/5, n_TOF 90/10) -- re-derived in P3 from Magboltz + the n_TOF head-on stack.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

PASS1 = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
#: arm -> (bench chamber, pass-1 golden row) (docs/PARAMETERS.md "Golden calibration rows")
GOLDEN = {
    'A': ('det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3'),
    'B': ('det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2'),
    'C': ('det6', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6'),
    'D': ('det7', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'),
}
#: starting v [um/ns] until the column-duration measurement (P3) exists: the is2 in-situ values for A and C
#: (run_149 A-C cosmic line), D from its wall (1.525 x production raw at 42.6 -> ~28), B the Magboltz 90/10 prior
V_START = {'A': 38.0, 'B': 42.6, 'C': 28.7, 'D': 28.0}
BOW_DROP = ('w0_const_x', 'w0_const_y')


def build(arm, out, src=None, v=None, k=20, sat=3550.0, bow=True, latency=None, t0_sigma=5.0, tan_max=None,
          n_samples=20, sample_ns=60.0, conditions=None):
    from wft.calib import CalibrationBundle
    chamber, row = GOLDEN[arm]
    src = src or os.path.join(PASS1, row, 'wft', 'calib_bundle')
    cal = CalibrationBundle.load(src)
    prov = dict(cal.provenance or {})
    rec = dict(source_bundle=src, source_chamber=chamber, source_code=prov.get('code_commit'),
               plan='rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md P3')
    rec['dropped'] = dict(v_map=bool(cal.v_map), w0=dict(cal.w0 or {}),
                          t0_abs_bench=bool(cal.t0_abs), t0_prior_sigma_bench=cal.t0_prior_sigma)
    cal.v_map, cal.w0 = {}, {}
    cal.t0_abs, cal.t0_prior_sigma = {}, 0.0
    if bow and cal.bow:
        b = {kk: vv for kk, vv in cal.bow.items() if kk not in BOW_DROP}
        b['valid'] = ('carried from the bench golden row to n_TOF (Dylan 2026-11-02); under test: bow-on vs bow-off on '
                      'the run_149 A-C line and the capsule focus')
        b['dropped_at_ntof'] = {kk: cal.bow.get(kk) for kk in BOW_DROP}
        cal.bow = b
        rec['bow'] = 'carried (tilt a_*, scale s_*, height h_*); w0_const_* dropped'
    else:
        rec['bow'] = 'dropped' if cal.bow else 'none in source'
        cal.bow = {}
    cal.v_drift = float(v if v is not None else V_START[arm])
    rec['v_drift'] = f'{cal.v_drift} um/ns ({"given" if v is not None else "V_START: is2 in-situ / wall / Magboltz"}); ' \
                     'NOT the column-duration measurement yet'
    cal.n_depth_bins = int(k)
    cal.sat_adc = float(sat)
    cal.tan_max = None if tan_max is None else float(tan_max)
    rec['tan_max'] = cal.tan_max if tan_max is not None else 'reco default (0.6)'
    cal.sample_ns = float(sample_ns)
    if latency:
        from wft.calibrate import t0_abs_from_latency
        lat = json.load(open(latency))
        cal.t0_abs = t0_abs_from_latency(lat)
        cal.t0_prior_sigma = float(t0_sigma)
        rec['t0'] = dict(source=latency, C={p: d['C'] for p, d in lat.items()}, sigma=t0_sigma)
        prov['kept_t0_prior'] = 'n_TOF latency measured in situ (rerun/ntof), not the bench one'
    else:
        rec['t0'] = 'free'
    rec['not_revalidated'] = 'Dp (bench 95/5 value; n_TOF gas 90/10), template (bench X template, 20-sample window), kw'
    cal.run_key = f'ntof/{arm}'
    cal.conditions = dict(conditions or {}, n_samples=int(n_samples), sample_ns=float(sample_ns))
    prov['ntof_bundle'] = rec
    cal.provenance = prov
    cal.save(out, note=f'n_TOF pass-1 bundle for arm {arm} from {chamber} ({row})')
    print(cal.summary())
    print(json.dumps(rec, indent=1, default=str))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('arm', choices=list(GOLDEN))
    ap.add_argument('--out', required=True)
    ap.add_argument('--src', default=None, help='pass-1 bundle dir (default: the golden row on EOS)')
    ap.add_argument('--v', type=float, default=None)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--sat', type=float, default=3550.0)
    ap.add_argument('--no-bow', action='store_true')
    ap.add_argument('--latency', default=None, help='n_TOF latency json (measure_latency output) -> pinned t0')
    ap.add_argument('--t0-sigma', type=float, default=5.0)
    ap.add_argument('--tan-max', type=float, default=None, help='candidate |tan| limit for this dataset (registry tan_max)')
    a = ap.parse_args()
    build(a.arm, a.out, a.src, a.v, a.k, a.sat, not a.no_bow, a.latency, a.t0_sigma, a.tan_max,
          conditions=dict(gas='Ar/Iso 90/10', latency=27))


if __name__ == '__main__':
    main()
