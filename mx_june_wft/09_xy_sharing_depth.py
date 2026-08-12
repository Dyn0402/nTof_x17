#!/usr/bin/env python3
"""
09_xy_sharing_depth.py — model-free X/Y sharing-mechanism measurement (T1.2).

The resistive strips run along y, so the Y view can get genuine resistive (RC)
sharing and the X view cannot — its ±1 sharing should be drift diffusion (F6).
The two mechanisms differ in TIME, not just amplitude, and the difference is
sharpest where no model is needed:

  * diffusion — the neighbour receives real charge, prompt-template-shaped,
    a fraction growing with drift depth u; when the drift column ends the
    neighbour signal must fall WITH the central strip;
  * RC transport — the neighbour sees a low-passed copy of the central
    charge; it lags the central rise and keeps discharging AFTER the column
    ends (tail past the end, tau ~ hundreds of ns).

So: near-vertical tracks (the whole column on one strip), average the central
and ±1-neighbour waveforms aligned to the central strip's half-max crossing,
and look at the neighbour/central ratio through and past the column end.
Rising ratio after the end = RC; tracking fall = diffusion.

    ../.venv/bin/python mx_june_wft/09_xy_sharing_depth.py sat_det3
Outputs: <OUT_BASE>/wft/sharing_depth/{sharing_depth.json, sharing_depth.png}
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

TAN_MAX = 0.05          # near-vertical: the column stays on one strip
AMP_MIN = 300.0         # central-strip peak, ADC — clean signal
SAT_ADC = 3550.0        # exclude saturated centrals (clipped shape)
SNS = 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = get_config(args.run_key)
    cache = args.cache or os.path.join(cfg.OUT_BASE, 'wft', 'bench_cache_ftst.pkl')
    out_dir = args.out or cfg.out_dir('wft', 'sharing_depth')
    os.makedirs(out_dir, exist_ok=True)
    with open(cache, 'rb') as f:
        events = pickle.load(f)['events']

    # aligned-average accumulators per plane: fine 10 ns grid via sub-sample
    # shift of each event to its central half-max crossing. Two selections:
    # 'all' (any bright central) mixes impact-position geometry into the ±1
    # ratio; 'centered' (left/right neighbours symmetric -> the track hit the
    # middle of the strip) removes the impact-position mixture and leaves
    # transport + diffusion.
    GRID = np.arange(-180.0, 1750.0, 10.0)
    acc = {(p, s): dict(c=np.zeros_like(GRID), n1=np.zeros_like(GRID),
                        n2=np.zeros_like(GRID), w=np.zeros_like(GRID), nev=0)
           for p in ('x', 'y') for s in ('all', 'centered')}
    CENTERED_ASYM = 0.25

    for eid, ev in events.items():
        t = ev['truth']
        for plane in ('x', 'y'):
            tan = t.get(f'tan_{plane}')
            if not np.isfinite(tan) or abs(tan) > TAN_MAX:
                continue
            for P in ev['wins'].get(plane) or []:
                W = np.asarray(P['W'], float)
                if W.shape[0] < 5:
                    continue
                amax = W.max(axis=1)
                ci = int(np.argmax(amax))
                if ci < 2 or ci > W.shape[0] - 3:
                    continue        # need ±2 neighbours in the window
                a = amax[ci]
                if a < AMP_MIN or W[ci].max() >= SAT_ADC:
                    continue
                # sub-sample half-max crossing of the central strip
                wc = W[ci]
                ipk = int(np.argmax(wc))
                t50 = None
                for k in range(1, ipk + 1):
                    if wc[k] >= 0.5 * a > wc[k - 1]:
                        t50 = SNS * (k - 1 + (0.5 * a - wc[k - 1])
                                     / (wc[k] - wc[k - 1]))
                        break
                if t50 is None:
                    continue
                ts = np.arange(W.shape[1]) * SNS - t50
                sels = ['all']
                al, ar = amax[ci - 1], amax[ci + 1]
                if al + ar > 0 and abs(al - ar) / (al + ar) < CENTERED_ASYM:
                    sels.append('centered')
                for s in sels:
                    A = acc[(plane, s)]
                    for name, wv in (('c', wc),
                                     ('n1', 0.5 * (W[ci - 1] + W[ci + 1])),
                                     ('n2', 0.5 * (W[ci - 2] + W[ci + 2]))):
                        A[name] += np.interp(GRID, ts, wv / a, left=0, right=0)
                    A['w'] += np.interp(GRID, ts, np.ones_like(ts),
                                        left=0, right=0)
                    A['nev'] += 1
                break               # one (brightest) cluster per plane

    out = {'run_key': args.run_key, 'tan_max': TAN_MAX, 'amp_min': AMP_MIN,
           'centered_asym': CENTERED_ASYM}
    fig, axs = plt.subplots(2, 4, figsize=(22, 9))
    for j, sel in enumerate(('all', 'centered')):
        for i, plane in enumerate(('x', 'y')):
            A = acc[(plane, sel)]
            n = np.maximum(A['w'], 1.0)
            c, n1, n2 = A['c'] / n, A['n1'] / n, A['n2'] / n
            key = f'{plane}_{sel}'
            out[key] = dict(n_events=int(A['nev']))
            col = 2 * j + i
            ax = axs[0, col]
            ax.plot(GRID, c, label='central (norm.)')
            ax.plot(GRID, n1, label='±1 mean')
            ax.plot(GRID, n2, label='±2 mean')
            ax.axhline(0, color='gray', lw=0.6)
            ax.set_title(f'{plane} [{sel}]: n={A["nev"]:,}')
            ax.set_xlabel('t − t50(central) [ns]')
            ax.legend(fontsize=8)

            ax = axs[1, col]
            ok = c > 0.02
            ax.plot(GRID[ok], (n1 / np.maximum(c, 1e-9))[ok],
                    label='±1 / central')
            ax.plot(GRID[ok], (n2 / np.maximum(c, 1e-9))[ok],
                    label='±2 / central')
            ax.set_ylim(-0.02, 0.8)
            ax.axhline(0, color='gray', lw=0.6)
            ax.set_title(f'{plane} [{sel}]: neighbour/central vs time')
            ax.set_xlabel('t − t50(central) [ns]')
            ax.legend(fontsize=8)
            out[key]['grid_ns'] = GRID[ok].tolist()
            out[key]['ratio1'] = (n1 / np.maximum(c, 1e-9))[ok].round(4).tolist()
            out[key]['ratio2'] = (n2 / np.maximum(c, 1e-9))[ok].round(4).tolist()
            out[key]['central'] = c[ok].round(4).tolist()
    fig.suptitle(f'{args.run_key}: sharing mechanism vs time — RC keeps '
                 'rising past the column end, diffusion falls with it; '
                 '[centered] isolates the transport copy')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sharing_depth.png'), dpi=130)
    with open(os.path.join(out_dir, 'sharing_depth.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for sel in ('all', 'centered'):
        print(f"n {sel}: x {acc[('x', sel)]['nev']}, y {acc[('y', sel)]['nev']}")
    print(f'wrote {out_dir}/sharing_depth.png')


if __name__ == '__main__':
    main()
