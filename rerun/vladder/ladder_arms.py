import json, os, sys, glob
import numpy as np
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1'
DETS = [('det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2', 39.3),
        ('det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3', 35.8),
        ('det4', 'mx17_det4_day_6-24-26/long_run/mx17_4', 33.0),
        ('det6', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6', 25.9),
        ('det7', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7', 36.8)]
out = {}
for det, rel, vt in DETS:
    for it in ('it0', 'it1'):
        for d in sorted(glob.glob(f'{A}/{rel}/wft/w12/{it}/*/eval.json')):
            arm = d.split('/')[-2]
            e = json.load(open(d))
            b = json.load(open(os.path.join(os.path.dirname(d), 'bundle', 'bundle.json')))
            v = b['v_drift']; w0 = b.get('w0', {}) or {}; kw = b.get('kw', {}) or {}
            row = dict(v_bundle=v)
            for p in ('x', 'y'):
                R = [r[p] for r in e['events'].values() if p in r]
                tan = np.array([r['tan'] for r in R]); tr = np.array([r['tr'] for r in R])
                w = (tan * kw.get(p, 1.0) * v + w0.get(p, 0.0))  # um/ns
                ok = np.abs(np.degrees(np.arctan(tan)) - np.degrees(np.arctan(tr))) < 10
                w, tr = w[ok], tr[ok]
                # robust slope: iterative LS with 3-sigma (s68) clipping
                m = np.ones(len(w), bool)
                for _ in range(5):
                    s, c = np.polyfit(tr[m], w[m], 1)
                    res = w - (s * tr + c); sg = np.percentile(np.abs(res[m]), 68.27)
                    m = np.abs(res) < 3 * sg
                bins = {}
                for lo, hi in ((0.1, 0.25), (0.25, 0.45), (0.45, 1.0)):
                    q = m & (np.abs(tr) >= lo) & (np.abs(tr) < hi)
                    bins[f'{lo}-{hi}'] = float(np.median(w[q] / tr[q])) if q.sum() > 10 else None
                q = m & (np.abs(tr) > 0.1)
                row[p] = dict(v_ladder=float(s), icpt=float(c), n=int(m.sum()), kw=kw.get(p, 1.0), w0=w0.get(p, 0.0),
                              ratio_med=float(np.median(w[q] / tr[q])), bins=bins,
                              pos=float(np.median(w[q & (tr > 0)] / tr[q & (tr > 0)])),
                              neg=float(np.median(w[q & (tr < 0)] / tr[q & (tr < 0)])))
            out[f'{det}/{it}/{arm}'] = row
            print(f"{det} {it} {arm:10s} v_bundle {v:5.2f} | X ladder {row['x']['v_ladder']:5.2f} (kw {row['x']['kw']:.3f}) "
                  f"bins {[round(x,1) if x else None for x in row['x']['bins'].values()]} +/- {row['x']['pos']:.1f}/{row['x']['neg']:.1f} | "
                  f"Y {row['y']['v_ladder']:5.2f} (kw {row['y']['kw']:.3f}) bins {[round(x,1) if x else None for x in row['y']['bins'].values()]} "
                  f"+/- {row['y']['pos']:.1f}/{row['y']['neg']:.1f} | timing {vt}")
json.dump(out, open(sys.argv[1], 'w'), indent=1)
