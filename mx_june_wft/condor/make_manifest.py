#!/usr/bin/env python3
"""
make_manifest.py — enumerate every wft-processable cosmic-bench subrun for the
MPGD26 full-June condor campaign, one row per (subrun, mx17 detector).

Walks the local bench mirror (the ground truth for what the 2026-07-24
reprocessing covered), reads each run's run_config.json for the detector→FEU
mapping, geometry and per-subrun HV, and classifies every row against the
detector's frozen calibration-bundle conditions:

    tier A  "direct"   same resist HV, same drift HV, same gas as the bundle
                       → reco directly with the frozen bundle
    tier B  "vrefit"   same resist HV + gas, different drift HV
                       → v_drift (+w0) refit first, kernel hypers pinned
    tier C  "held"     different resist HV / gas / no config / pre-June /
                       stale-duplicate tree → not in the automatic pass;
                       listed with the reason so nothing silently drops

The bundle-conditions table is read from the golden runs' own configs, not
hardcoded, so the manifest cannot disagree with the bundles it points at.

    ../../.venv/bin/python mx_june_wft/condor/make_manifest.py \
        [--bench /home/dylan/x17/cosmic_bench] [--eos-index eos_index.json] \
        [--out campaign_manifest.csv]

`--eos-index` is {run_name: eos_run_dir} (see README — one ssh find builds it);
without it the eos_path column is left blank, nothing else changes.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# ---------------------------------------------------------------- constants

# The frozen production bundle per detector (FREEZE_MPGD26_2026-08-12.md).
BUNDLE = {
    'mx17_2': 'calib_bundle_lp',
    'mx17_3': 'calib_bundle_lp2_t0p',
    'mx17_4': 'calib_bundle_lp',
    'mx17_6': 'calib_bundle_lp',
    'mx17_7': 'calib_bundle_lp',
}

# Golden (bundle-fit) dataset per detector: tree, run, subrun.
GOLDEN = {
    'mx17_2': ('det2_det3', 'mx17_det2_det3_overnight_6-22-26', 'longer_run'),
    'mx17_3': ('det3', 'mx17_det3_saturday_scan_6-27-26',
               'long_run_resist_490V_drift_1000V'),
    'mx17_4': ('det4_day', 'mx17_det4_day_6-24-26', 'long_run'),
    'mx17_6': ('det6_det7', 'mx17_det6_det7_overnight_6-26-26', 'long_run'),
    'mx17_7': ('det6_det7', 'mx17_det6_det7_overnight_6-26-26', 'long_run'),
}

# Registry keys for the golden rows (everything else gets a synthesized key).
REGISTRY_KEY = {
    ('mx17_det2_det3_overnight_6-22-26', 'longer_run', 'mx17_2'): 'o22_long_det2',
    ('mx17_det3_saturday_scan_6-27-26',
     'long_run_resist_490V_drift_1000V', 'mx17_3'): 'sat_det3',
    ('mx17_det4_day_6-24-26', 'long_run', 'mx17_4'): 'g_det4',
    ('mx17_det6_det7_overnight_6-26-26', 'long_run', 'mx17_6'): 'g_det6_long',
    ('mx17_det6_det7_overnight_6-26-26', 'long_run', 'mx17_7'): 'g_det7_long',
}

# Known stale duplicates of det4_day (june-cosmics-reprocessing memory / audit).
STALE_TREES = {'det4', 'det_4day'}

SKIP_TOP = {'Analysis', 'pedestals', '_m3check'}
RUN_DATE_RE = re.compile(r'(\d{1,2})-(\d{1,2})-26$')


# ---------------------------------------------------------------- helpers

def run_date(run_name):
    m = RUN_DATE_RE.search(run_name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_config(run_dir):
    p = os.path.join(run_dir, 'run_config.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def mx17_detectors(cfg):
    """{det_name: dict(feu_x, feu_y, det_z, hv_channels)} for mx17 dets."""
    out = {}
    for det in cfg.get('detectors', []):
        if det.get('det_type') != 'mx17':
            continue
        feus = det.get('dream_feus', {})
        if 'x_1' not in feus or 'y_1' not in feus:
            continue
        out[det['name']] = dict(
            feu_x=feus['x_1'][0], feu_y=feus['y_1'][0],
            det_z=det.get('det_center_coords', {}).get('z'),
            hv=det.get('hv_channels', {}))
    return out


def subrun_hv(cfg, subrun, hv_channels):
    """(resist_V, drift_V) for one detector in one subrun, or (None, None)."""
    for sr in cfg.get('sub_runs', []):
        if sr.get('sub_run_name') != subrun:
            continue
        hvs = sr.get('hvs', {})

        def get(kind):
            ch = hv_channels.get(kind)
            if not ch:
                return None
            return hvs.get(str(ch[0]), {}).get(str(ch[1]))
        return get('resist'), get('drift')
    return None, None


def dir_stats(d, feus=None):
    """(n_files, total_bytes) of *.root under d, optionally only FEUs given."""
    if not os.path.isdir(d):
        return 0, 0
    files = glob.glob(os.path.join(d, '*.root'))
    if feus is not None:
        pats = tuple(f'_{f:02d}.root' for f in feus)
        files = [f for f in files if f.endswith(pats)]
    return len(files), sum(os.path.getsize(f) for f in files)


def reference_conditions(bench):
    """Per detector: (gas, resist_V, drift_V) at the bundle's golden subrun."""
    ref = {}
    for det, (tree, run, subrun) in GOLDEN.items():
        cfg = load_config(os.path.join(bench, tree, run))
        if cfg is None:
            sys.exit(f'FATAL: golden run config missing for {det}: '
                     f'{tree}/{run} — cannot define tier A')
        dets = mx17_detectors(cfg)
        if det not in dets:
            sys.exit(f'FATAL: {det} not in {run} run_config detectors')
        r, dr = subrun_hv(cfg, subrun, dets[det]['hv'])
        ref[det] = dict(gas=cfg.get('gas'), resist=r, drift=dr)
    return ref


# ---------------------------------------------------------------- main walk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default='/home/dylan/x17/cosmic_bench')
    ap.add_argument('--eos-index', default=None,
                    help='JSON {run_name: eos_run_dir}')
    ap.add_argument('--out', default=os.path.join(HERE, 'campaign_manifest.csv'))
    args = ap.parse_args()

    eos = {}
    if args.eos_index:
        with open(args.eos_index) as f:
            eos = json.load(f)

    ref = reference_conditions(args.bench)
    rows = []

    for tree in sorted(os.listdir(args.bench)):
        top = os.path.join(args.bench, tree)
        if not os.path.isdir(top) or tree in SKIP_TOP \
                or tree.startswith('condor'):
            continue
        for run in sorted(os.listdir(top)):
            run_dir = os.path.join(top, run)
            if not os.path.isdir(run_dir):
                continue
            cfg = load_config(run_dir)
            dets = mx17_detectors(cfg) if cfg else {}
            date = run_date(run)
            for subrun in sorted(os.listdir(run_dir)):
                sub = os.path.join(run_dir, subrun)
                if not os.path.isdir(sub) or subrun.startswith('_'):
                    continue
                dec = os.path.join(sub, 'decoded_root')
                n_dec_all, _ = dir_stats(dec)
                if n_dec_all == 0:
                    continue                     # nothing to reconstruct
                has_hits = os.path.isdir(os.path.join(sub,
                                                      'combined_hits_root'))
                has_m3v2 = os.path.isdir(os.path.join(sub,
                                                      'm3_tracking_root_v2'))
                has_m3v1 = os.path.isdir(os.path.join(sub,
                                                      'm3_tracking_root'))

                if not dets:
                    rows.append(dict(
                        tree=tree, run=run, subrun=subrun, det='?',
                        key='', tier='C', reason='no run_config.json',
                        gas='', resist_V='', drift_V='',
                        feu_x='', feu_y='', det_z='',
                        n_dec_files=n_dec_all, dec_mb=0,
                        has_hits=int(has_hits), has_m3v2=int(has_m3v2),
                        has_m3v1=int(has_m3v1),
                        bundle='', eos_run_dir=eos.get(run, '')))
                    continue

                for det, info in sorted(dets.items()):
                    r, dr = subrun_hv(cfg, subrun, info['hv'])
                    gas = cfg.get('gas')
                    n_dec, dec_b = dir_stats(
                        dec, feus=[info['feu_x'], info['feu_y']])
                    tier, reason = 'A', ''
                    if tree in STALE_TREES:
                        tier, reason = 'C', f'stale duplicate tree {tree}/'
                    elif date is None or date[0] < 6:
                        tier, reason = 'C', 'pre-June run'
                    elif not has_hits:
                        tier, reason = 'C', 'no combined_hits_root (no seeds)'
                    elif not (has_m3v2 or has_m3v1):
                        tier, reason = 'C', 'no M3 tracking'
                    elif n_dec == 0:
                        tier, reason = 'C', 'no decoded files for det FEUs'
                    elif det not in ref:
                        tier, reason = 'C', 'no frozen bundle for this detector'
                    elif gas != ref[det]['gas']:
                        tier, reason = 'C', (f'gas {gas!r} != bundle '
                                             f'{ref[det]["gas"]!r}')
                    elif r is None or dr is None:
                        tier, reason = 'C', 'subrun HV not in run_config'
                    elif r != ref[det]['resist']:
                        tier, reason = 'C', (f'resist {r} V != bundle '
                                             f'{ref[det]["resist"]} V')
                    elif dr != ref[det]['drift']:
                        tier, reason = 'B', (f'drift {dr} V vs bundle '
                                             f'{ref[det]["drift"]} V')

                    key = REGISTRY_KEY.get((run, subrun, det)) or \
                        f'c26__{run}__{subrun}__{det}'
                    rows.append(dict(
                        tree=tree, run=run, subrun=subrun, det=det, key=key,
                        tier=tier, reason=reason, gas=gas,
                        resist_V=r, drift_V=dr,
                        feu_x=info['feu_x'], feu_y=info['feu_y'],
                        det_z=info['det_z'],
                        n_dec_files=n_dec, dec_mb=round(dec_b / 1e6),
                        has_hits=int(has_hits), has_m3v2=int(has_m3v2),
                        has_m3v1=int(has_m3v1),
                        bundle=BUNDLE.get(det, ''),
                        eos_run_dir=eos.get(run, '')))

    fields = ['tree', 'run', 'subrun', 'det', 'key', 'tier', 'reason', 'gas',
              'resist_V', 'drift_V', 'feu_x', 'feu_y', 'det_z',
              'n_dec_files', 'dec_mb', 'has_hits', 'has_m3v2', 'has_m3v1',
              'bundle', 'eos_run_dir']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n = {'A': 0, 'B': 0, 'C': 0}
    mb = {'A': 0, 'B': 0, 'C': 0}
    for row in rows:
        n[row['tier']] += 1
        mb[row['tier']] += row['dec_mb'] or 0
    print(f'wrote {args.out}: {len(rows)} rows')
    for t in 'ABC':
        print(f'  tier {t}: {n[t]:4d} rows, {mb[t]/1e3:7.1f} GB det-FEU '
              f'decoded input')
    missing_eos = sorted({row['run'] for row in rows
                          if not row['eos_run_dir']})
    if args.eos_index and missing_eos:
        print(f'  no EOS mapping for {len(missing_eos)} runs: '
              + ', '.join(missing_eos[:8]) + ('…' if len(missing_eos) > 8 else ''))


if __name__ == '__main__':
    main()
