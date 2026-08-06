#!/usr/bin/env python3
"""
make_bundle_variant.py — derive a calibration bundle from another one.

For the transfer ablation (and, later, for run_79): take a bundle's kernel and
template and give it a different drift velocity and/or a different set of
angle-mapping constants, keeping the provenance of both parents. This is the
operation TRACK_PLAN_08 proposes for n_TOF — freeze the bench kernel, replace
the gas-dependent constants — so it is worth having as a tool rather than as an
ad-hoc edit.

    ../../.venv/bin/python mx_june_wft/bench/make_bundle_variant.py \
        --src <bundle> --out <bundle> [--v 40.26] [--w0kw-from <bundle>] \
        [--sample-ns 60] [--k 15] [--sat-adc 4000] [--set c1=0.05,kY=2.4]
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--v', type=float, default=None, help='v_drift [um/ns]')
    ap.add_argument('--w0kw-from', default=None,
                    help='take w0/kw from this bundle instead')
    ap.add_argument('--sample-ns', type=float, default=None)
    ap.add_argument('--k', type=int, default=None, help='n_depth_bins')
    ap.add_argument('--sat-adc', type=float, default=None)
    ap.add_argument('--set', default=None,
                    help='comma-separated hyper overrides, e.g. c1=0.05,kY=2.4')
    ap.add_argument('--note', default='')
    args = ap.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.out):
        raise SystemExit('refusing to overwrite the source bundle')
    os.makedirs(args.out, exist_ok=True)
    shutil.copy(os.path.join(args.src, 'arrays.npz'),
                os.path.join(args.out, 'arrays.npz'))
    with open(os.path.join(args.src, 'bundle.json')) as f:
        m = json.load(f)

    derived = {'kernel_from': args.src}
    if args.v is not None:
        derived['v_drift'] = [m['v_drift'], args.v]
        m['v_drift'] = args.v
    if args.w0kw_from:
        with open(os.path.join(args.w0kw_from, 'bundle.json')) as f:
            o = json.load(f)
        derived['w0kw_from'] = args.w0kw_from
        derived['w0_replaced'] = [m.get('w0'), o.get('w0')]
        m['w0'], m['kw'] = o.get('w0', {}), o.get('kw', {})
    for key, val in (('sample_ns', args.sample_ns), ('n_depth_bins', args.k),
                     ('sat_adc', args.sat_adc)):
        if val is not None:
            derived[key] = [m.get(key), val]
            m[key] = val
    if args.set:
        for kv in args.set.split(','):
            k, v = kv.split('=')
            derived.setdefault('hyper', {})[k.strip()] = [
                m['hyper'].get(k.strip()), float(v)]
            m['hyper'][k.strip()] = float(v)

    prov = dict(m.get('provenance', {}))
    prov['derived'] = derived
    if args.note:
        prov['note'] = args.note
    m['provenance'] = prov
    with open(os.path.join(args.out, 'bundle.json'), 'w') as f:
        json.dump(m, f, indent=1)
    print(f'wrote {args.out}\n  v={m["v_drift"]}  w0={m.get("w0")}  '
          f'kw={m.get("kw")}  sample_ns={m.get("sample_ns")}  '
          f'K={m.get("n_depth_bins")}  sat={m.get("sat_adc")}')


if __name__ == '__main__':
    main()
