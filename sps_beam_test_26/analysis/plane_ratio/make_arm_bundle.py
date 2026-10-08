#!/usr/bin/env python3
"""make_arm_bundle.py -- turn a recal.py arm into a calibration bundle.

Mirrors mx_june_wft/20_make_ratio_bundle.py (the tool that made r06): copy the
source bundle (template, gain, dead/hot maps, dt_xy), replace the hypers with
the arm's, re-measure the absolute-t0 table under the NEW kernel on the arm's
own training events (a table from another kernel puts the pulse elsewhere and
the 5 ns prior then drags every fit), and record the fit in the provenance.
w0/kw are copied and stamped stale -- they must be re-measured from the first
reco pass under this bundle (bench/set_w0.py), exactly as for r06.

Refuses a ``diag_*`` arm (those may break c2 < c1 and are diagnostics only),
and the bundle's own load-time gate (check_kernel_ordering, per view) is the
second line.

    make_arm_bundle.py --src <bundle dir> --arm-json arm.json \
        --cache <training cache> --out <new bundle dir>
"""
import argparse
import json
import os
import pickle
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--arm-json', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-train', type=int, default=180)
    a = ap.parse_args()

    from wft import calibrate as wc
    from wft.calib import CalibrationBundle, check_kernel_ordering

    rec = json.load(open(a.arm_json))
    if rec['arm'].startswith('diag'):
        raise SystemExit(f'{rec["arm"]} is a diagnostic arm; no bundle is made from it')
    h = {k: float(v) for k, v in rec['hyper'].items()}
    check_kernel_ordering(h, where=a.arm_json)

    cal = CalibrationBundle.load(a.src)
    old = dict(cal.hyper)
    cal.hyper = dict(h)
    with open(a.cache, 'rb') as f:
        events = pickle.load(f)
    train = {e: events[e] for e in sorted(events)[:a.n_train]}
    tmp = a.out + '_provisional'
    cal.save(tmp, note='hypers set, t0_abs not yet re-measured')
    t0abs, t0sig = wc.measure_t0_abs(train, tmp, h, float(cal.v_drift))
    cal.t0_abs = t0abs
    prov = dict(cal.provenance)
    prov.pop('code_commit', None)
    prov.update(
        fitted='sps_beam_test_26/analysis/plane_ratio/recal.py',
        arm=rec['arm'], objective=rec.get('objective'),
        p0_profile=rec.get('p0_profile', False),
        chi2=float(rec['chi2']), chi2_init=float(rec['chi2_seed']),
        chi2_note='deterministic cold-t0 objective on n_train events; NOT '
                  'comparable with the warm-objective chi2 of earlier bundles',
        n_train=int(rec['n_train']), derived_from=os.path.abspath(a.src),
        w0_kw_stale=True,
        w0_kw_note='w0/kw copied from the source bundle; re-measure from the '
                   'first reco pass under this bundle (bench/set_w0.py)',
        superseded_hyper={k: float(v) for k, v in old.items()},
        evidence='sps_beam_test_26/analysis/plane_ratio (per-view head-on '
                 'closure; bench + H4 run_71)')
    cal.provenance = prov
    cal.save(a.out, note=f'per-view kernel arm {rec["arm"]}, t0_abs re-measured')
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'[bundle] {a.src} -> {a.out}')
    print(f'[bundle] {cal.summary()}')
    print(f'[bundle] t0_abs re-measured on {len(train)} events, spread {t0sig}')


if __name__ == '__main__':
    main()
