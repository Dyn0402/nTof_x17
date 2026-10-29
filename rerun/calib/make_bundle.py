#!/usr/bin/env python3
"""make_bundle.py <det> --cache <calibration cache.pkl> --out <bundle dir> [--rc-json <recal out>] -- E1: a bundle from
nothing (PASS_PLAN_2026-10-28 §2).

Every piece is either MEASURED here on the row's own calibration cache, or a STATED physics input with provenance:
  measured   templates  wft.calibrate.measure_templates (bright strips of inclined M3 tracks; X template drives both
                        views, rc_tmpl 'x')
             t0_abs     wft.calibrate.measure_latency -> t0_abs_from_latency (t0 = C - 10 ns * ftst), pinned
                        (t0_prior_sigma from physics_inputs)
             rc_D_y     DRC's recal.py output (--rc-json: --free rc_D_y --t0-pin --unsat-only); without it the
                        physics_inputs seed (det4: the H4 value, never refit)
  stated     sigma_p0_x/y, Dp, rc_D_x, v (timing), sat_adc, unit gains   (rerun/calib/physics_inputs.json)
  masks      dead / hot from the one-rule bench masks after F7 (model_audit/followups/fe_unify2/masks/bench_<det>.json)
  absent     bow, w0/kw, angle_map, v_map, tan_floor, dt_xy, xy_pairing: measured later in the calib DAG (W0KW, BOW2)
             on a reconstruction made with this bundle.  No copy-kernel keys (c1, c2, kY, tau_s, sigma_s): the RC kernel
             does not use them.

Provenance names every input (paths, sizes, the latency numbers, the template counts).  The save-time checks
(check_kernel_ordering, check_xy_pairing) still run.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
PHYS = os.path.join(HERE, 'physics_inputs.json')
MASKS = os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fe_unify2', 'masks')


def _file_id(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return dict(path=os.path.abspath(p), bytes=os.path.getsize(p), sha1=h.hexdigest()[:12])


def make(det, events, cache_id, phys, masks, rc=None, K=24, run_key='', conditions=None, rc_id=None):
    """The CalibrationBundle (not saved).  events: the calibration cache dict; phys: physics_inputs.json; masks: the
    bench mask json; rc: recal.py output dict or None."""
    from wft.calib import CalibrationBundle
    from wft.calibrate import measure_templates, measure_latency, t0_abs_from_latency
    com, me = phys['common'], phys[det]
    grid, tmpl = measure_templates(events, sat_adc=float(com['sat_adc']['value']))
    n_tmpl = {p: int(sum(1 for e in events.values() if p in e)) for p in ('x', 'y')}
    lat = measure_latency(events)
    if set(lat) != {'x', 'y'}:
        raise SystemExit(f'[make_bundle] latency measured on {sorted(lat)} only (need both views, >= 30 tracks each)')
    rc_cfg = me['rc_D_y']
    if rc is not None:
        if not rc_cfg.get('free', True):
            raise SystemExit(f'[make_bundle] {det}: rc_D_y is a stated input ({rc_cfg}); refusing a DRC fit for it')
        if 'rc_D_y' not in rc.get('free', []):
            raise SystemExit(f'[make_bundle] --rc-json did not free rc_D_y (free={rc.get("free")})')
        if not (rc.get('t0_pin') and rc.get('unsat_only')):
            raise SystemExit('[make_bundle] DRC must run --t0-pin --unsat-only (TEMPLATE_STUDY T6)')
        rc_D_y, rc_src = float(rc['hyper']['rc_D_y']), dict(rc_id or {}, chi2=rc.get('chi2'), chi2_seed=rc.get('chi2_seed'),
                                                             converged=rc.get('converged'), n_train=rc.get('n_train'))
    else:
        rc_D_y = float(rc_cfg['value'] if 'value' in rc_cfg else rc_cfg['seed'])
        rc_src = 'physics_inputs ' + ('value (stated)' if 'value' in rc_cfg else 'seed (DRC not yet run)')
    s0 = com['sigma_p0_mm']
    hyper = dict(sigma_p0=float(s0['x']), sigma_p0_x=float(s0['x']), sigma_p0_y=float(s0['y']), Dp=float(me['Dp']),
                 rc_D_y=rc_D_y, rc_D_x=float(com['rc_D_x']['value']),
                 rc_tmpl=1.0 if com['rc_tmpl']['value'] == 'x' else 0.0)
    planes = masks['planes']
    dead = {p: sorted(int(c) for c in planes[p].get('dead', [])) for p in ('x', 'y')}
    hot = {p: sorted(int(c) for c in planes[p].get('hot', [])) for p in ('x', 'y') if planes[p].get('hot')}
    cal = CalibrationBundle(
        hyper=hyper, v_drift=float(me['v_timing_um_ns']), grid=np.asarray(grid, float),
        tmpl={p: np.asarray(tmpl[p], float) for p in ('x', 'y')},
        gain={p: np.full(512, float(com['gain']['value'])) for p in ('x', 'y')},
        t0_abs=t0_abs_from_latency(lat), t0_prior_sigma=float(com['t0_prior_sigma']['value']),
        dead=dead, hot=hot, n_depth_bins=int(K), sat_adc=float(com['sat_adc']['value']),
        detector=f"mx17_{det[3:]}", run_key=run_key, conditions=dict(conditions or {}),
        provenance=dict(
            made_by='rerun/calib/make_bundle.py (E1, from nothing)',
            cache=cache_id, n_cache_events=len(events), n_template_windows=n_tmpl,
            latency={p: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for p, d in lat.items()},
            rc_D_y=rc_src, physics_inputs=dict(_file_id(PHYS), det=me, common={k: v.get('value', v) for k, v in com.items()}),
            masks=dict(path=os.path.relpath(masks.get('_path', ''), REPO) if masks.get('_path') else None,
                       rule=masks.get('rule', '')[:200], n_dead={p: len(dead[p]) for p in dead},
                       hot={p: hot.get(p, []) for p in ('x', 'y')}),
            measured_later='w0/kw, tan_floor, dt_xy, xy_pairing (W0KW on a reconstruction with this bundle); bow (BOW2)'))
    return cal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('det', choices=['det2', 'det3', 'det4', 'det6', 'det7'])
    ap.add_argument('--cache', required=True, help='calibration cache (wft.calibrate.build_cache / build_cache_wft.py)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rc-json', default=None, help="DRC: recal.py output with rc_D_y free (det2/3/6/7)")
    ap.add_argument('--K', type=int, default=24, help='depth bins (registry K of the golden row)')
    ap.add_argument('--run-key', default='')
    ap.add_argument('--physics', default=PHYS)
    ap.add_argument('--masks', default=None, help='default: fe_unify2/masks/bench_<det>.json')
    ap.add_argument('--note', default='')
    ap.add_argument('--force', action='store_true', help='overwrite an existing bundle (a DAG node retried)')
    a = ap.parse_args(argv)
    if os.path.exists(os.path.join(a.out, 'bundle.json')) and not a.force:
        raise SystemExit(f'{a.out} exists: not overwriting')
    phys = json.load(open(a.physics))
    mp = a.masks or os.path.join(MASKS, f'bench_{a.det}.json')
    masks = json.load(open(mp)); masks['_path'] = mp
    events = pickle.load(open(a.cache, 'rb'))
    rc = json.load(open(a.rc_json)) if a.rc_json else None
    cal = make(a.det, events, _file_id(a.cache), phys, masks, rc=rc, K=a.K, run_key=a.run_key,
               rc_id=_file_id(a.rc_json) if a.rc_json else None)
    cal.save(a.out, note=a.note)
    print(f'[make_bundle] {cal.summary()}\n  -> {a.out}')


if __name__ == '__main__':
    main()
