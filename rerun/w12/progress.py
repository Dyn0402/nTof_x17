#!/usr/bin/env python3
"""progress.py -- W12 convergence table: per chamber and iteration, v, kernel constants, held-out resolution, and the
waveform alignment's offset from the hits alignment (z, rotation, translation)."""
import glob, json, os
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
DETS = [('det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2', 39.3),
        ('det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3', 35.8),
        ('det4', 'mx17_det4_day_6-24-26/long_run/mx17_4', 33.0),
        ('det6', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6', 25.9),
        ('det7', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7', 36.8)]
out = {}
for det, rel, vt in DETS:
    W = os.path.join(A, rel, 'wft')
    hits = json.load(open(os.path.join(A, rel, 'alignment_tpc_veto50', 'alignment.json')))
    seed = json.load(open(os.path.join(W, 'w11', 'w11q_cm0', 'arm.json')))
    print(f"\n{det}  timing v {vt}   seed (W11 profiled): v {seed['v']:.2f}")
    rows = []
    for it in sorted(glob.glob(os.path.join(W, 'w12', 'it*')), key=lambda p: int(p.rsplit('it', 1)[1])):
        try:
            arm = json.load(open(os.path.join(it, 'fit_pinned', 'arm.json')))
            ev = json.load(open(os.path.join(it, 'fit_pinned', 'eval.json')))['summary']
            al = json.load(open(os.path.join(it, 'alignment', 'alignment.json')))
        except OSError:
            continue
        h = arm['hyper']
        d = {k: al[k] - hits[k] for k in ('z_x', 'z_y', 'theta_deg', 'x_offset', 'y_offset')}
        r = dict(it=os.path.basename(it), v=arm['v'], s0x=h['sigma_p0_x'], s0y=h['sigma_p0_y'], drc=h['rc_D_y'],
                 chi2=arm['chi2'], s68x=ev['x']['s68_deg'], s68y=ev['y']['s68_deg'], d_align=d)
        rows.append(r)
        print(f"  {r['it']}: v {r['v']:6.2f}  s0 {r['s0x']:.3f}/{r['s0y']:.3f}  Drc {r['drc']:.2e}  chi2 {r['chi2']:.4e}  "
              f"s68 {r['s68x']:.3f}/{r['s68y']:.3f}  | wft-hits align: dz {d['z_x']:+.1f}/{d['z_y']:+.1f} mm "
              f"dtheta {d['theta_deg']:+.2f} dxy {d['x_offset']:+.2f}/{d['y_offset']:+.2f}")
    out[det] = dict(timing_v=vt, seed_v=seed['v'], iterations=rows)
json.dump(out, open('w12_progress.json', 'w'), indent=1)
