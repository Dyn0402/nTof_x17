#!/usr/bin/env python3
"""make_jobs.py <code_dir> [--bundles-only DIR]  -- FA study bundles + the condor job list (one line per job).

Bundles (built from the local copies of the promoted golden bundles, ~/x17/cosmic_bench/model_audit/<det>/calib_bundle,
and the ws_t0 latency bundles), written to <DIR> and copied to EOS by hand (scp):
  <det>_base    the promoted bundle, unchanged (the reference arm, reconstructed with today's code)
  <det>_s0c     one common sigma0 for X and Y (S0C mm) and no per-event wide/narrow choice (sigma_p0_prof_* = [])
  <det>_pin     the ws_t0 latency bundle: t0 pinned at C - 10 ns * ftst (t0_prior_sigma 0.05)
  <det>_pins0c  both
  det3_v32      det3's bundle with v_drift = 32.0 um/ns, for the 600 V-drift rows of 6-23 (the reconstruction's column
                window and seed probes scale with v; the angle route uses the raw fitted w, never the bundle v)
Jobs (fa_row.sh <code> <mode> <key> <bundle> [limit]):
  colsum on the five golden rows (item 1), arms on det3/det4/det6 golden rows (item 2), reco + alignment + table on
  every other decodable bench row of every chamber (item 3, and item 2's det3 HV rows).
"""
import json
import os
import shutil
import sys

LOCAL = os.path.expanduser('~/x17/cosmic_bench/model_audit')
EOSB = '/eos/user/d/dneff/x17rerun/followups/fa/bundles'
S0C = 0.29
GOLD = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
C = 'c26__'
ROWS = {   # item 3 (+ item 2 HV rows): key -> bundle name, limit
    'det2': [(f'{C}mx17_det1_det2_overnight_6-17-26__longer_run__mx17_2', 'det2_base', 5000),
             (f'{C}mx17_det2_det3_overnight_6-22-26__long_run__mx17_2', 'det2_base', 5000),
             (f'{C}mx17_det2_det3_overnight_6-22-26__short_run__mx17_2', 'det2_base', 0)],
    'det3': [(f'{C}mx17_det2_det3_overnight_6-22-26__long_run__mx17_3', 'det3_base', 5000),
             (f'{C}mx17_det2_det3_overnight_6-22-26__short_run__mx17_3', 'det3_base', 0),
             (f'{C}mx17_det3_test_6-22-26__short_run__mx17_3', 'det3_base', 0),
             (f'{C}mx17_det3_det4_overnight_6-23-26__long_run__mx17_3', 'det3_v32', 5000),
             (f'{C}mx17_det3_det4_overnight_6-23-26__longer_run__mx17_3', 'det3_v32', 5000),
             (f'{C}mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3', 'det3_base', 5000),
             (f'{C}mx17_det3_saturday_scan_6-27-26__drift_scan_resist_490V_drift_900V__mx17_3', 'det3_base', 0),
             (f'{C}mx17_det3_saturday_scan_6-27-26__drift_scan_resist_490V_drift_1100V__mx17_3', 'det3_base', 0)]
            + [(f'{C}mx17_det3_saturday_scan_6-27-26__hv_scan_resist_{v}V_drift_1000V__mx17_3', 'det3_base', 0)
               for v in range(425, 526, 10)]
            + [(f'{C}mx17_det3_saturday_scan_6-27-26__hv_scan2_resist_{v}V_drift_1000V__mx17_3', 'det3_base', 0)
               for v in range(460, 521, 10)],
    'det4': [(f'{C}mx17_det3_det4_overnight_6-23-26__long_run__mx17_4', 'det4_base', 5000),
             (f'{C}mx17_det3_det4_overnight_6-23-26__longer_run__mx17_4', 'det4_base', 5000),
             (f'{C}mx17_det3_det4_overnight_6-23-26__short_run__mx17_4', 'det4_base', 0)],
    'det6': [(f'{C}mx17_det6_det7_overnight_6-26-26__longer_run__mx17_6', 'det6_base', 5000),
             (f'{C}mx17_det6_det7_overnight_6-26-26__short_run__mx17_6', 'det6_base', 0),
             (f'{C}mx17_det6_det7_hv_scan_6-26-26__resist_det6_495V_det7_495V_drift_700V__mx17_6', 'det6_base', 0)],
    'det7': [(f'{C}mx17_det6_det7_overnight_6-26-26__longer_run__mx17_7', 'det7_base', 5000),
             (f'{C}mx17_det6_det7_overnight_6-26-26__short_run__mx17_7', 'det7_base', 0),
             (f'{C}mx17_det6_det7_hv_scan_6-26-26__resist_det6_495V_det7_495V_drift_700V__mx17_7', 'det7_base', 0)],
}
ARM_DETS = ['det3', 'det4', 'det6']
#: model-free column duration on det3's drift scan (v(E) with no M3 and no angle: the field coupling of item 1).
#: 100 and 300 V are left out: their column (> 3 us) overruns the 1.92 us window.
COLSUM_EXTRA = [(f'{C}mx17_det3_saturday_scan_6-27-26__drift_scan_resist_490V_drift_{v}V__mx17_3', 'det3_base')
                for v in (500, 700, 900, 1100)]


def edit(src, dst, fn, note):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    p = os.path.join(dst, 'bundle.json')
    b = json.load(open(p))
    fn(b)
    b.setdefault('provenance', {})['fa_study'] = note
    json.dump(b, open(p, 'w'), indent=1)


def bundles(out):
    os.makedirs(out, exist_ok=True)
    for det in GOLD:
        g = os.path.join(LOCAL, det, 'calib_bundle')
        lat = os.path.join(LOCAL, 'ws_t0', 'bundles', det)

        def s0c(b):
            b['hyper']['sigma_p0_x'] = b['hyper']['sigma_p0_y'] = S0C
            b['hyper']['sigma_p0_prof_x'] = []; b['hyper']['sigma_p0_prof_y'] = []
        edit(g, os.path.join(out, f'{det}_base'), lambda b: None, 'FA: promoted bundle, unchanged')
        edit(g, os.path.join(out, f'{det}_s0c'), s0c, f'FA item 2: common sigma0 {S0C} mm, no per-event choice')
        edit(lat, os.path.join(out, f'{det}_pin'), lambda b: None, 'FA item 2: ws_t0 latency bundle (t0 pinned)')
        edit(lat, os.path.join(out, f'{det}_pins0c'), s0c, f'FA item 2: t0 pinned + common sigma0 {S0C} mm')
    edit(os.path.join(LOCAL, 'det3', 'calib_bundle'), os.path.join(out, 'det3_v32'),
         lambda b: b.__setitem__('v_drift', 32.0), 'FA item 3: det3 bundle with v 32.0 for the 600 V-drift rows')


def jobs(code):
    L = []
    for det, key in GOLD.items():
        L.append(f'{code} colsum {key} {EOSB}/{det}_base')
    for key, b in COLSUM_EXTRA:
        L.append(f'{code} colsum {key} {EOSB}/{b}')
    for det in ARM_DETS:
        for arm in ('base', 's0c', 'pin', 'pins0c'):
            L.append(f'{code} arm:{arm} {GOLD[det]} {EOSB}/{det}_{arm}')
    for det, rows in ROWS.items():
        for key, b, lim in rows:
            L.append(f'{code} reco {key} {EOSB}/{b} {lim}')
    return L


if __name__ == '__main__':
    if sys.argv[1] == '--bundles-only':
        bundles(sys.argv[2]); print('bundles ->', sys.argv[2])
    else:
        L = jobs(sys.argv[1])
        open('fa_jobs.txt', 'w').write('\n'.join(L) + '\n'); print(len(L), 'jobs -> fa_jobs.txt')
