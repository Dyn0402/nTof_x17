#!/usr/bin/env python3
"""
05_beam_kernel_xcheck.py — the H4 beam kernel observables, measured on the
June cosmic bench with the SAME estimators.

The SPS run_71 RAW campaign measured det4's charge-spreading kernel
model-independently (sps_beam_test_26/analysis/robust_waveforms.py):
per-event leading strip, ±d neighbour traces, event-wise ±1 peak-time shift,
peak-amplitude and window-area ratios. Those numbers are drift- and
gain-invariant, so they are properties of the resistive layer and must
reproduce on the bench if the layer is the same layer.

This script applies the identical estimator to the wft calibration cache
(waveform windows along the M3 corridor) restricted to near-normal tracks
(|tan| < 0.10, the bench's stand-in for the beam's normal incidence), and
prints/saves the side-by-side. It needs no calibration bundle and no forward
model — it is a raw-waveform comparison.

    ../.venv/bin/python mx_june_wft/05_beam_kernel_xcheck.py g_det4 \
        [--beam-lib <robust_library npz>] [--tan-max 0.10] [--q0 400,3000]

Output: <det>/wft/beam_xcheck/{xcheck.json, xcheck_overlay.png, table}
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
          os.path.join(REPO, 'cosmic_bench_analysis')):
    if p not in sys.path:
        sys.path.insert(0, p)

BEAM_LIB_DEFAULT = ('/media/dylan/data/x17/sps_run53_det4_check/staging/run_71/'
                    'reanalysis_clean_cmmasked/robust_library_run71_raw.npz')
SNS = 60.0
NREL = 12            # peak-aligned stack half-width (bench window is 32 samples)


def parabolic_peak(w, i):
    """Sub-sample peak position by parabola through (i-1, i, i+1)."""
    if i <= 0 or i >= len(w) - 1:
        return float(i)
    a, b, c = w[i - 1], w[i], w[i + 1]
    den = a - 2 * b + c
    return i + (0.5 * (a - c) / den if den < 0 else 0.0)


def measure_plane(events, plane, tan_max, q0lo, q0hi):
    """Beam-style estimator on the cache windows of one plane."""
    shifts_raw, shifts_par, pk_ratio, area_ratio = [], [], {}, {}
    aligned = {}
    n_used = 0
    for ev in events.values():
        if plane not in ev or abs(ev[f'tan_{plane}']) > tan_max:
            continue
        P = ev[plane]
        W = np.asarray(P['W'], np.float64)
        ns = W.shape[1]
        peaks = W.max(axis=1)
        lead = int(np.argmax(peaks))
        q0 = peaks[lead]
        if not (q0lo <= q0 <= q0hi):
            continue
        ipk = int(np.argmax(W[lead]))
        if ipk < 3 or ipk > ns - 4:
            continue                      # peak too close to the window edge
        n_used += 1
        c_par = parabolic_peak(W[lead], ipk)
        for d in (-3, -2, -1, 0, 1, 2, 3):
            j = lead + d
            if j < 0 or j >= W.shape[0]:
                continue
            w = W[j]
            jpk = int(np.argmax(w))
            if d != 0 and abs(d) == 1:
                shifts_raw.append((jpk - ipk) * SNS)
                shifts_par.append((parabolic_peak(w, jpk) - c_par) * SNS)
            pk_ratio.setdefault(d, []).append(w.max() / q0)
            # window-area ratio over the rel range the bench window supports
            lo, hi = max(0, ipk - 5), min(ns, ipk + 20)
            a0 = W[lead][lo:hi].sum()
            if a0 > 0:
                area_ratio.setdefault(d, []).append(w[lo:hi].sum() / a0)
            # peak-aligned normalised stack
            cols = np.arange(ipk - NREL, ipk + NREL + 1)
            ok = (cols >= 0) & (cols < ns)
            row = np.full(2 * NREL + 1, np.nan)
            row[ok] = w[cols[ok]] / q0
            aligned.setdefault(d, []).append(row)
    out = dict(n_events=n_used,
               dtpk_med_raw=float(np.median(shifts_raw)) if shifts_raw else np.nan,
               dtpk_med_par=float(np.median(shifts_par)) if shifts_par else np.nan,
               n_shift=len(shifts_raw))
    for d in sorted(pk_ratio):
        out[f'pk_{d:+d}'] = float(np.median(pk_ratio[d]))
        out[f'area_{d:+d}'] = float(np.median(area_ratio.get(d, [np.nan])))
        out[f'n_{d:+d}'] = len(pk_ratio[d])
    stacks = {d: np.nanmedian(np.array(v), axis=0) for d, v in aligned.items()
              if len(v) >= 20}
    return out, stacks


def beam_numbers(lib_path):
    """The beam-side numbers, from the run_71 clean library, matched-window."""
    z = np.load(lib_path)
    t_rel = z['t_rel']
    out = {}
    for lab in ('raw700', 'raw450', 'raw275'):
        for v in ('x', 'y'):
            sh = np.concatenate([z[f'dtpk_{lab}_{v}_{d:+d}']
                                 for d in (1, -1)
                                 if f'dtpk_{lab}_{v}_{d:+d}' in z])
            if len(sh) == 0:
                continue
            e = {'dtpk_med': float(np.median(sh)), 'n_shift': int(len(sh))}
            # matched-window area ratios from the peak-aligned median stacks:
            # same rel range as the bench estimator (-300 ns .. +1140 ns)
            sel = (t_rel >= -5 * SNS) & (t_rel < 20 * SNS)
            a0 = None
            for d in (0, 1, -1, 2, -2, 3, -3):
                k = f'alm_{lab}_{v}_{d:+d}'
                if k not in z:
                    continue
                tr = z[k]
                area = np.nansum(tr[sel])
                pk = np.nanmax(tr)
                if d == 0:
                    a0 = area
                e[f'stackpk_{d:+d}'] = float(pk)
                if a0:
                    e[f'area_{d:+d}'] = float(area / a0)
            out[f'{lab}_{v}'] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--beam-lib', default=BEAM_LIB_DEFAULT)
    ap.add_argument('--tan-max', type=float, default=0.10)
    ap.add_argument('--q0', default='400,3000')
    args = ap.parse_args()
    q0lo, q0hi = (float(x) for x in args.q0.split(','))

    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(args.run_key)
    cache = os.path.join(cfg.OUT_BASE, 'wft', 'calib_work', 'calib_cache.pkl')
    events = pickle.load(open(cache, 'rb'))
    print(f'{len(events)} cached events, tan_max={args.tan_max}, '
          f'q0 gate {q0lo:.0f}-{q0hi:.0f} ADC')

    bench, stacks = {}, {}
    for plane in ('x', 'y'):
        bench[plane], stacks[plane] = measure_plane(
            events, plane, args.tan_max, q0lo, q0hi)

    beam = beam_numbers(args.beam_lib) if os.path.exists(args.beam_lib) else {}

    out_dir = cfg.out_dir('wft', 'beam_xcheck')

    print('\n=== bench (June cosmics, near-normal, beam estimator) ===')
    for plane in ('x', 'y'):
        b = bench[plane]
        print(f"  {plane.upper()}: n={b['n_events']:4d}  "
              f"±1 shift median {b['dtpk_med_raw']:+.0f} ns "
              f"(parabolic {b['dtpk_med_par']:+.1f} ns, n={b['n_shift']})")
        for d in (1, -1, 2, -2):
            if f'pk_{d:+d}' in b:
                # stack-based numbers: same estimator as the beam library's
                # alm_ stacks (median-aligned trace, then peak / area) --
                # per-event medians are systematically higher because the
                # neighbour peaks are not time-locked to the central one
                st = stacks[plane].get(d)
                st0 = stacks[plane].get(0)
                spk = float(np.nanmax(st)) if st is not None else np.nan
                sar = (float(np.nansum(st) / np.nansum(st0))
                       if st is not None and st0 is not None else np.nan)
                print(f"      d={d:+d}: ev-peak {b[f'pk_{d:+d}']:.3f}  "
                      f"ev-area {b[f'area_{d:+d}']:.3f}  "
                      f"stack-pk {spk:.3f}  stack-area {sar:.3f}  "
                      f"(n={b[f'n_{d:+d}']})")
                b[f'stackpk_{d:+d}'] = spk
                b[f'stackarea_{d:+d}'] = sar
    with open(os.path.join(out_dir, 'xcheck.json'), 'w') as f:
        json.dump(dict(bench=bench, beam=beam,
                       config=dict(run_key=args.run_key, tan_max=args.tan_max,
                                   q0=[q0lo, q0hi], beam_lib=args.beam_lib)),
                  f, indent=1)

    print('\n=== beam (run_71 RAW clean library, matched window) ===')
    for k, e in beam.items():
        line = f"  {k}: ±1 shift median {e['dtpk_med']:+.0f} ns (n={e['n_shift']})"
        for d in (1, -1, 2, -2):
            if f'stackpk_{d:+d}' in e:
                line += f"  | d={d:+d} pk {e[f'stackpk_{d:+d}']:.3f} area {e.get(f'area_{d:+d}', np.nan):.3f}"
        print(line)

    # overlay figure, Y view
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        z = np.load(args.beam_lib)
        t_rel_beam = z['t_rel']
        t_rel_bench = (np.arange(2 * NREL + 1) - NREL) * SNS
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
        for ax, d in zip(axes, (0, 1, -1)):
            if d in stacks['y']:
                ax.plot(t_rel_bench, stacks['y'][d], 'o-', ms=3,
                        label='bench cosmics (Y)')
            k = f'alm_raw450_y_{d:+d}'
            if k in z.files:
                ax.plot(t_rel_beam, z[k], 's--', ms=2.5,
                        label='beam run_71 450V (Y)')
            ax.axvline(0, color='0.8', lw=0.7)
            ax.set_title(f'd = {d:+d}')
            ax.set_xlabel('t − t_peak(central) [ns]')
            ax.set_xlim(-720, 1440)
            ax.legend(fontsize=8)
        axes[0].set_ylabel('amplitude / central peak')
        fig.suptitle(f'{args.run_key}: peak-aligned median traces, '
                     f'bench vs beam')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'xcheck_overlay.png'), dpi=130)
        print(f'\nwrote {out_dir}/xcheck_overlay.png')
    except Exception as e:
        print(f'(overlay figure skipped: {e})')


if __name__ == '__main__':
    main()
