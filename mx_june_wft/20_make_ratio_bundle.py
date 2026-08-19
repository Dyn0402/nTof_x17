#!/usr/bin/env python3
"""
20_make_ratio_bundle.py -- write the c2-slaved calibration bundle.

Takes the hypers 19_ratio_recal.py fitted, re-measures the absolute-t0 table
under them (a table measured with a different kernel puts the pulse somewhere
else, and the sigma = 5 ns prior then pulls the fit to the wrong place), and
saves a sibling bundle beside the production one.

TWO THINGS IT DELIBERATELY DOES NOT DO.

  w0 / kw.  The per-plane angle-mapping constants are measured from an existing
  reconstruction (bench/set_w0.py reads events.parquet), so they cannot be
  re-derived without re-running the reco.  They are copied across UNCHANGED and
  the bundle is stamped `w0_kw_stale`, because they were measured under the old
  kernel and a kernel change moves the raw w -> tan slope by ~1.4 % on Y --
  the same size as kw - 1.  Re-measure them after the first reco pass with the
  new bundle, before anything downstream quotes an absolute angle.

  Freeze.  It writes `calib_bundle_<tag>` and touches nothing the MPGD26
  manifest points at.  Re-freezing is a separate, deliberate act.

    ../.venv/bin/python mx_june_wft/20_make_ratio_bundle.py sat_det3 \\
        --ratio 0.6 --src calib_bundle_lp2_t0p
Output: <OUT_BASE>/wft/calib_bundle_r<ratio>
"""
import argparse
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--ratio', type=float, default=0.6)
    ap.add_argument('--src', default='calib_bundle_lp2_t0p')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--n-train', type=int, default=180)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle

    cfg = get_config(args.run_key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    src = os.path.join(W, args.src)
    cache = os.path.join(W, 'calib_work', 'calib_cache.pkl')
    rec = json.load(open(os.path.join(W, 'kernel_arms', 'ratio_recal.json')))
    key = f'ratio{args.ratio:g}'
    if key not in rec:
        raise SystemExit(f'[bundle] {key} not in ratio_recal.json '
                         f'(have {sorted(rec)}) -- run 19_ratio_recal.py first')
    h = {k: float(q) for k, q in rec[key]['hyper'].items()
         if k != 'c2_implied'}

    cal = CalibrationBundle.load(src)
    old = dict(cal.hyper)
    cal.hyper = dict(h)
    with open(cache, 'rb') as f:
        events = pickle.load(f)
    train = {e: events[e] for e in sorted(events)[:args.n_train]}

    tag = args.tag or f'r{args.ratio:g}'.replace('.', '')
    out = os.path.join(W, f'calib_bundle_{tag}')
    tmp = os.path.join(W, 'calib_work', f'provisional_{tag}')
    cal.save(tmp, note='hypers set, t0_abs not yet re-measured')
    t0abs, t0sig = wc.measure_t0_abs(train, tmp, h, float(cal.v_drift))
    cal.t0_abs = t0abs
    cal.provenance = dict(cal.provenance)
    # The source's fit record does NOT describe this fit. Carry the ratio
    # refit's own chi2 and drop the inherited fields, or the bundle claims a
    # chi2 it never achieved (and a code_commit that is not in this repo).
    cal.provenance.pop('code_commit', None)
    cal.provenance.update(
        chi2=float(rec[key]['chi2']),
        chi2_init=float(rec[key]['chi2_seed']),
        chi2_source=float(cal.provenance.get('chi2', float('nan'))),
        chi2_note='chi2 is the ratio-pinned refit on n_train events; '
                  'chi2_source is the free-ratio fit it is derived from, on '
                  'the same events -- pinning the ratio costs fit quality and '
                  'the two numbers say how much',
        fitted='mx_june_wft/19_ratio_recal.py',
        derived_from=args.src,
        fitted_by='mx_june_wft/19_ratio_recal.py',
        c2_over_c1=args.ratio,
        c2_slaved=True,
        w0_kw_stale=True,
        w0_kw_note='w0/kw copied from the source bundle; they were measured '
                   'from a reco under the OLD kernel and must be re-measured '
                   '(bench/set_w0.py) after the first pass with this bundle',
        superseded_hyper={k: float(v) for k, v in old.items()},
        evidence='sps_beam_test_26/analysis/sharing_kernel -- the head-on beam '
                 'measures c2/c1 = 0.45 +- 0.02 and near-vertical bench '
                 'cosmics 0.63 +- 0.09; the shipped bundles carried > 1')
    cal.save(out, note=f'c2 slaved to {args.ratio:g} x c1, t0_abs re-measured')
    print(f'[bundle] {src}\n     ->  {out}')
    print(f'[bundle] c2/c1  {old["c2"] / old["c1"]:.2f}  ->  {args.ratio:.2f}')
    for k in ('c1', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp'):
        print(f'[bundle]   {k:9} {old.get(k, float("nan")):10.4g} -> '
              f'{h[k]:10.4g}')
    print(f'[bundle]   c2 (slaved) {old["c2"]:8.4g} -> '
          f'{h["c1"] * args.ratio:10.4g}')
    print(f'[bundle] t0_abs re-measured on {len(train)} events, '
          f'spread x {t0sig["x"] if isinstance(t0sig, dict) else t0sig}')


if __name__ == '__main__':
    main()
