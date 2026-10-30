#!/usr/bin/env python3
"""ov07_check.py [--pass-dir D] [--out results/ov07.json]

Why run60 flush ov07 has x quality_ok 65 % (the other plateaus 83-86 %): the trigger rate per minute of run_60
overnight_05..08 from the decoded FEU 3 timestamps (the timestamp counter is ~10 ns; one sub-run = one acquisition, its
first/last event mapped to the dream_daq.log span), and the pass-1 ov07 table split at the beam's return.
Read-only on Analysis_pass1.  Runs on lxplus (EOS).
"""
import argparse, glob, json, os
import numpy as np

RUNS = '/eos/experiment/ntof/data/x17/p2_sps_july/runs/run_60/'
PASS = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps'
SPAN_S = 1805.0          # overnight_07: started 00:52:13, finished 01:22:18 (dream_daq.log)
T_BEAM_BACK = 91170.0    # wall clock [s of day 0] at which the beam returns in ov07 (first minute above 1 kHz)


def main():
    import uproot, pandas as pd
    ap = argparse.ArgumentParser(); ap.add_argument('--pass-dir', default=PASS)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'ov07.json'))
    a = ap.parse_args()
    res = {'rate_per_min': {}}
    for s in ('overnight_05', 'overnight_06', 'overnight_07', 'overnight_08'):
        fs = sorted(glob.glob(RUNS + s + '/decoded_root_fa80c06/*datrun*_03.root'))
        ts = np.concatenate([uproot.open(f)['nt'].arrays(['timestamp'], library='np')['timestamp'] for f in fs]).astype(float)
        t = (ts - ts[0]) * SPAN_S / (ts[-1] - ts[0])
        h, _ = np.histogram(t, bins=30, range=(0, SPAN_S))
        res['rate_per_min'][s] = dict(files=len(fs), events=int(len(ts)), per_min=h.tolist())
        print(s, len(fs), 'files', len(ts), 'events; per minute:', h.tolist())
    d = pd.read_parquet(os.path.join(a.pass_dir, 'run60_flush__ov07', 'events.parquet'))
    for name, m in (('beam_off', d.t_wall < T_BEAM_BACK), ('beam_on', d.t_wall >= T_BEAM_BACK)):
        s = d[m]
        res[name] = dict(n=int(len(s)), t_wall=[float(s.t_wall.min()), float(s.t_wall.max())],
                         x_quality_ok=float(s.x_quality_ok.mean()), y_quality_ok=float(s.y_quality_ok.mean()),
                         x_n_strips_med=float(s.x_n_strips.median()), y_n_strips_med=float(s.y_n_strips.median()),
                         x_q_sum_med=float(s.x_q_sum.median()))
        print(name, res[name])
    others = []
    for k in sorted(glob.glob(os.path.join(a.pass_dir, 'run60_flush__ov*'))):
        t = pd.read_parquet(os.path.join(k, 'events.parquet'), columns=['t_wall', 'x_quality_ok', 'y_quality_ok'])
        others.append(dict(key=os.path.basename(k), span_s=float(t.t_wall.max() - t.t_wall.min()),
                           x_quality_ok=float(t.x_quality_ok.mean()), y_quality_ok=float(t.y_quality_ok.mean())))
    res['plateaus'] = others
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
