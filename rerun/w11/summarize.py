#!/usr/bin/env python3
"""summarize.py [tag ...] -- W11 table: per chamber, plain (cm0) vs masked (cm5) common mode, same events and fit.

Constants of the RC-kernel fit (sigma_p0_x, sigma_p0_y, rc_D_y, v), the training chi2 (cold objective, same 180
events: comparable between cm0 and cm5 only as a ratio, since the inputs differ), and the held-out evaluation
(angle s68, bias, position s68, chi2/dof median, far and runaway fractions).  Writes w11_summary.json."""
import glob, json, os, sys

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
DETS = [('det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2'),
        ('det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3'),
        ('det4', 'mx17_det4_day_6-24-26/long_run/mx17_4'),
        ('det6', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6'),
        ('det7', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7')]
tags = sys.argv[1:] or ['w11p', 'w11q']
out = {}
for tag in tags:
    print(f'\n=== {tag}')
    print(f"{'det':5s} {'cm':4s} {'s0x':>6s} {'s0y':>6s} {'Drc_y':>8s} {'v':>6s} {'chi2':>11s} | "
          f"{'s68x':>6s} {'s68y':>6s} {'biasx':>6s} {'biasy':>6s} {'posx':>6s} {'posy':>6s} {'c2x':>6s} {'c2y':>6s} {'farx':>5s} {'fary':>5s}")
    for det, rel in DETS:
        for cm in ('cm0', 'cm5'):
            d = os.path.join(A, rel, 'wft', 'w11', f'{tag}_{cm}')
            try:
                arm = json.load(open(os.path.join(d, 'arm.json')))
                ev = json.load(open(os.path.join(d, 'eval.json')))['summary']
            except OSError:
                print(f'{det:5s} {cm:4s} (missing)')
                continue
            h = arm['hyper']
            out[f'{tag}/{det}/{cm}'] = dict(hyper=h, v=arm['v'], chi2=arm['chi2'], chi2_seed=arm['chi2_seed'],
                                           converged=arm.get('converged'), eval=ev)
            print(f"{det:5s} {cm:4s} {h['sigma_p0_x']:6.3f} {h['sigma_p0_y']:6.3f} {h['rc_D_y']:8.2e} {arm['v']:6.2f} "
                  f"{arm['chi2']:11.4e} | {ev['x']['s68_deg']:6.3f} {ev['y']['s68_deg']:6.3f} {ev['x']['bias_deg']:+6.3f} "
                  f"{ev['y']['bias_deg']:+6.3f} {ev['x']['pos_s68_mm']:6.3f} {ev['y']['pos_s68_mm']:6.3f} "
                  f"{ev['x']['chi2_median']:6.2f} {ev['y']['chi2_median']:6.2f} {ev['x']['frac_far']:5.3f} {ev['y']['frac_far']:5.3f}")
json.dump(out, open('w11_summary.json', 'w'), indent=1)
