#!/usr/bin/env python3
"""arms_summary.py [it] -- W12 protocol arms on one iteration's caches: v, kernel constants, reference-plane offsets,
training chi2 and held-out resolution per chamber, against the timing v.  Writes arms_<it>.json."""
import json, os, sys
it = sys.argv[1] if len(sys.argv) > 1 else 'it0'
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
DETS = [('det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2', 39.3, 1.2),
        ('det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3', 35.8, 0.6),
        ('det4', 'mx17_det4_day_6-24-26/long_run/mx17_4', 33.0, 0.7),
        ('det6', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6', 25.9, 0.8),
        ('det7', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7', 36.8, 1.1)]
ARMS = ['fit_pinned', 'profA', 'profB', 'pinDZ', 'timeD']
out = {}
print(f"{'det':5s} {'arm':10s} {'v':>6s} {'dv/vt':>7s} {'s0x':>6s} {'s0y':>6s} {'Drc_y':>8s} {'dz_x':>6s} {'dz_y':>6s} "
      f"{'chi2':>11s} {'s68x':>6s} {'s68y':>6s} {'biasx':>6s} {'biasy':>6s}")
for det, rel, vt, ve in DETS:
    for arm in ARMS:
        d = os.path.join(A, rel, 'wft', 'w12', it, arm)
        try:
            a = json.load(open(os.path.join(d, 'arm.json')))
            e = json.load(open(os.path.join(d, 'eval.json')))['summary']
        except OSError:
            print(f'{det:5s} {arm:10s} (missing)')
            continue
        h, dz = a['hyper'], a.get('ref_dz', {}) or {}
        out[f'{det}/{arm}'] = dict(v=a['v'], timing_v=vt, timing_err=ve, hyper=h, ref_dz=dz, chi2=a['chi2'],
                                   converged=a.get('converged'), eval=e)
        print(f"{det:5s} {arm:10s} {a['v']:6.2f} {100*(a['v']/vt-1):+6.1f}% {h['sigma_p0_x']:6.3f} {h['sigma_p0_y']:6.3f} "
              f"{h['rc_D_y']:8.2e} {dz.get('dz_x', 0):+6.2f} {dz.get('dz_y', 0):+6.2f} {a['chi2']:11.4e} "
              f"{e['x']['s68_deg']:6.3f} {e['y']['s68_deg']:6.3f} {e['x']['bias_deg']:+6.3f} {e['y']['bias_deg']:+6.3f}")
    print(f"{det:5s} {'timing':10s} {vt:6.2f} ± {ve}")
json.dump(out, open(f'arms_{it}.json', 'w'), indent=1)
