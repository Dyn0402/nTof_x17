#!/usr/bin/env python3
"""beam_score.py -- beam-side check of the t0 pin on run_145 (plan §6 Q2): latency on the beam trigger and capsule pointing.

    beam_score.py [--dir /media/ucla/work/ntof_pass1/t0pin/beam] [--out beam_results.json]

1. Latency: the pin was measured on beam-off cosmics (t0 + 10 ns * ftst, median per view).  On beam the trigger is
   different (PS + singles), so the free fit's t0 + 10*ftst of gated, x-y paired beam tracks is compared with the
   cosmic C.  A shift larger than the pin's sigma (15 ns) would bias pinned beam fits.
2. Capsule pointing (rerun/ntof/capsule_point.py) per configuration on the same triggers: tracks whose line passes
   the capsule axis within 10 / 15 / 25 mm.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import capsule_point
RC = '/media/ucla/work/ntof_pass1/runs/run_145/run_config.json'
LAT = '/media/ucla/work/ntof_pass1/bundles/latency_{arm}_r149.json'


def merged(d, lab, arm, tmp):
    fs = sorted(glob.glob(f'{d}/{lab}/run_145/*/{arm}/*.candidates.parquet'))
    if not fs:
        return None, 0
    out = os.path.join(tmp, f'r145_{arm}_{lab}')
    parts = []
    for f in fs:          # event ids restart per sub-run: offset them so (sub-run, event) stays unique
        c = pd.read_parquet(f)
        c['event_id'] = c['event_id'] + 10_000_000 * int(f.split('/stat090_')[1][:4])
        parts.append(c)
    pd.concat(parts, ignore_index=True).to_parquet(out + '.candidates.parquet')
    return out + '.parquet', len(fs)


def robust(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    m = np.median(x)
    return dict(n=int(len(x)), median=float(m), median_err=float(1.2533 * 1.4826 * np.median(np.abs(x - m)) / np.sqrt(len(x))),
                sigma_mad=float(1.4826 * np.median(np.abs(x - m))), q16=float(np.percentile(x, 16)), q84=float(np.percentile(x, 84)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='/media/ucla/work/ntof_pass1/t0pin/beam')
    ap.add_argument('--out', default=None); a = ap.parse_args()
    tmp = os.path.join(a.dir, 'merged'); os.makedirs(tmp, exist_ok=True)
    res = {}
    for arm in 'AC':
        lat = json.load(open(LAT.format(arm=arm)))
        for lab in sorted(os.path.basename(p) for p in glob.glob(f'{a.dir}/*') if os.path.basename(p) != 'merged'):
            path, n = merged(a.dir, lab, arm, tmp)
            if path is None:
                continue
            s, T = capsule_point.score(path, arm, RC, lab)
            s.pop('bins', None)
            s['n_acq'] = n
            cand = pd.read_parquet(path.replace('.parquet', '.candidates.parquet'))
            g = cand[cand.track_gated.astype(bool)] if 'track_gated' in cand else cand
            for p, pl in (('x', 0), ('y', 1)):
                sel = g[g.plane == pl] if g.plane.dtype != object else g[g.plane == p]
                s[f'lat_{p}'] = robust(sel.t0 + 10.0 * sel.ftst)
                s[f'lat_{p}_cosmic_C'] = float(lat[p]['C'])
                # prompt only: within +-300 ns of the cosmic latency (the flat pedestal is unrelated tracks)
                pr = sel[np.abs(sel.t0 + 10.0 * sel.ftst - lat[p]['C']) < 300]
                s[f'lat_{p}_prompt'] = robust(pr.t0 + 10.0 * pr.ftst)
            res[f'{arm}_{lab}'] = s
            print(arm, lab, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                             if k.startswith(('n_', 'frac_point', 'dca_med'))})
            for p in 'xy':
                print('   ', p, 'cosmic C', round(s[f'lat_{p}_cosmic_C'], 1), 'beam all', {k: round(v, 1) for k, v in s[f'lat_{p}'].items()},
                      '\n      prompt', {k: round(v, 1) for k, v in s[f'lat_{p}_prompt'].items()})
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
