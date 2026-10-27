#!/usr/bin/env python3
"""promote_phys.py [--mask det3,det7,...] [--apply] [--force] -- S1 close for the pass (RERUN_PLAN §10/§11).

On lxplus (EOS). For each golden calibration row, the finalize arm's bundle -> <row>/wft/calib_bundle, the bundle every
row of the chamber reads in the pass (directly, or through vrefit --keep-v):
  chambers in --mask   <row>/wft/calib_bundle_physmask: the phys bundle with the channel mask (census dead + shorts,
                       W3, adopted on the mask gate) and w0/kw measured UNDER the mask (finalize --out-tag physmask;
                       the mask moves det7's X ladder by 0.7 %, so the unmasked w0/kw would be stale)
  the others           <row>/wft/calib_bundle_phys
Dry run unless --apply; refuses an existing calib_bundle unless --force. Writes <row>/wft/calib_bundle/PROMOTED.json.
An empty ``conditions`` (run / sub_run) is filled from the golden row, so CalibrationBundle.check_conditions can check
the bundle (model audit F10: det7's promoted bundle had run = sub_run = '').  --fill-conditions does only that, in
place, on the bundles already promoted.
"""
import argparse, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
GOLD = {'det2': ('o22_long_det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2'),
        'det3': ('sat_det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3'),
        'det4': ('g_det4', 'mx17_det4_day_6-24-26/long_run/mx17_4'),
        'det6': ('g_det6_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6'),
        'det7': ('g_det7_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7')}


def fill_conditions(b: dict, rel: str) -> bool:
    """Fill an empty run / sub_run in bundle dict ``b`` from the golden row's path; True if anything changed."""
    run, sub = rel.split('/')[:2]
    c = b.setdefault('conditions', {})
    changed = False
    for k, v in (('run', run), ('sub_run', sub)):
        if not c.get(k):
            c[k] = v
            changed = True
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', default='', help='comma list of chambers that take the channel mask')
    ap.add_argument('--apply', action='store_true'); ap.add_argument('--force', action='store_true')
    ap.add_argument('--root', default=A)
    ap.add_argument('--fill-conditions', action='store_true', help='only fill empty conditions of the promoted bundles')
    a = ap.parse_args()
    if a.fill_conditions:
        for det, (key, rel) in GOLD.items():
            path = f'{a.root}/{rel}/wft/calib_bundle/bundle.json'
            b = json.load(open(path))
            if fill_conditions(b, rel):
                print(f'{det}: conditions -> {b["conditions"]}' + ('' if a.apply else '  (dry run)'))
                if a.apply:
                    json.dump(b, open(path, 'w'), indent=1)
            else:
                print(f'{det}: conditions {b.get("conditions")} (kept)')
        return
    masked = {m for m in a.mask.split(',') if m}
    bad = masked - set(GOLD)
    if bad:
        sys.exit(f'unknown chamber(s) {bad}')
    for det, (key, rel) in GOLD.items():
        tag = 'physmask' if det in masked else 'phys'
        src, dst = f'{a.root}/{rel}/wft/calib_bundle_{tag}', f'{a.root}/{rel}/wft/calib_bundle'
        if not os.path.isfile(f'{src}/bundle.json'):
            sys.exit(f'{det}: no source {src}')
        b = json.load(open(f'{src}/bundle.json'))
        if b.get('provenance', {}).get('w0_kw_stale', True):
            sys.exit(f'{det}: source w0/kw not measured under this bundle')
        dead = b.get('dead', {})
        if (det in masked) != bool(dead):
            sys.exit(f'{det}: source mask {({v: len(x) for v, x in dead.items()})} does not match --mask')
        print(f'{det}: {src} -> {dst}  v {b["v_drift"]:.2f}  K {b.get("n_depth_bins")}  '
              f'mask {({v: len(x) for v, x in dead.items()} or "none")}')
        if not a.apply:
            continue
        if os.path.exists(dst):
            if not a.force:
                sys.exit(f'{det}: {dst} exists (use --force)')
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        if fill_conditions(b, rel):
            json.dump(b, open(f'{dst}/bundle.json', 'w'), indent=1)
        json.dump(dict(source=src, mask=dead, time=time.strftime('%Y-%m-%d %H:%M:%S'), tool='rerun/promote_phys.py'),
                  open(f'{dst}/PROMOTED.json', 'w'), indent=1)
    if not a.apply:
        print('dry run (use --apply)')


if __name__ == '__main__':
    main()
