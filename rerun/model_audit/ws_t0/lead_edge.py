#!/usr/bin/env python3
"""lead_edge.py [--base M] -- model-free timing of the charge column on each bench chamber (workstream D).

From the raw waveforms of the calibration cache (no fit, no template, no v): per plane, the EARLIEST 30 % constant-
fraction crossing among the strips with a clear pulse (the charge ionised next to the mesh, which arrives first
whatever v is), and the LATEST pulse peak (the charge from the cathode).  Every time is phase-corrected with the
trigger's fine timestamp, t + 10 ns * ftst.  If the trigger latency is the same for all chambers, t_lead must agree
across chambers up to per-FEU cable offsets; the fitted t0 is compared against it.

Rows written to ws_t0/lead_edge.json (per det/view: medians and robust widths); per-event arrays in lead_edge.npz.
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd

DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
FEU = {'det2': (6, 8), 'det3': (7, 8), 'det4': (6, 8), 'det6': (3, 4), 'det7': (6, 8)}
SNS = 60.0
NSIG = 8.0          # strip must peak above NSIG x its noise
CF = 0.3


def cf_time(w, frac=CF):
    """first crossing of frac x peak, linearly interpolated [ns]; nan if the pulse starts in sample 0."""
    pk = int(np.argmax(w)); thr = frac * w[pk]
    above = np.where(w[:pk + 1] >= thr)[0]
    if not len(above):
        return np.nan
    i = above[0]
    if i == 0:
        return np.nan
    return SNS * (i - 1 + (thr - w[i - 1]) / max(w[i] - w[i - 1], 1e-9))


def plane_times(P):
    W = np.asarray(P['W'], float); nz = np.asarray(P['noise'], float)
    pk = W.max(axis=1)
    good = pk > NSIG * np.maximum(nz, 1.0)
    if good.sum() < 2:
        return np.nan, np.nan, 0
    tl = [cf_time(W[i]) for i in np.where(good)[0]]
    tl = np.array([t for t in tl if np.isfinite(t)])
    tp = SNS * np.argmax(W[good], axis=1)
    if not len(tl):
        return np.nan, np.nan, int(good.sum())
    return float(tl.min()), float(tp.max()), int(good.sum())


def rsig(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    q = np.percentile(a, [16, 84]); return 0.5 * (q[1] - q[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/home/dylan/x17/cosmic_bench/model_audit')
    a = ap.parse_args()
    out, arrs = {}, {}
    for det in DETS:
        C = pickle.load(open(f'{a.base}/ws_t0/caches/{det}.pkl', 'rb'))
        ev = pd.read_parquet(f'{a.base}/{det}/final_physmask/events.parquet').set_index('event_id')
        out[det] = {}
        for vi, p in enumerate('xy'):
            R = []
            for eid, e in C.items():
                if p not in e or e.get(f'ftst_{p}') is None or int(e[f'ftst_{p}']) < 0:
                    continue
                tl, tt, ng = plane_times(e[p])
                ph = 10.0 * int(e[f'ftst_{p}'])
                t0c = (ev.at[eid, f'{p}_t0'] + ph) if eid in ev.index and np.isfinite(ev.at[eid, f'{p}_t0']) else np.nan
                R.append((eid, tl + ph, tt + ph, ng, float(e[f'tan_{p}']), t0c, int(e[f'ftst_{p}'])))
            R = np.array(R, float); arrs[f'{det}_{p}'] = R
            ok = np.isfinite(R[:, 1]) & (R[:, 3] >= 3) & (np.abs(R[:, 4]) < 0.3)
            d = R[ok]
            lad = pd.Series(d[:, 1] - 10 * d[:, 6]).groupby(d[:, 6]).median()
            slope = float(np.polyfit(lad.index.values, lad.values, 1)[0])
            both = np.isfinite(d[:, 5])
            out[det][p] = dict(feu=FEU[det][vi], n=int(ok.sum()),
                               t_lead=float(np.median(d[:, 1])), t_lead_rsig=float(rsig(d[:, 1])),
                               t_last_peak=float(np.median(d[:, 2])), t_last_rsig=float(rsig(d[:, 2])),
                               span=float(np.median(d[:, 2] - d[:, 1])),
                               t0c_minus_lead=float(np.median(d[both, 5] - d[both, 1])),
                               t0c_minus_lead_rsig=float(rsig(d[both, 5] - d[both, 1])),
                               ftst_slope_uncorrected=slope)
            o = out[det][p]
            print(f'{det} {p} FEU{o["feu"]} n={o["n"]:4d}  t_lead {o["t_lead"]:6.1f} ({o["t_lead_rsig"]:4.1f})  '
                  f'last peak {o["t_last_peak"]:6.1f} ({o["t_last_rsig"]:4.1f})  span {o["span"]:6.1f}  '
                  f't0c - lead {o["t0c_minus_lead"]:+6.1f} ({o["t0c_minus_lead_rsig"]:4.1f})  ftst slope {slope:+.1f}')
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, 'lead_edge.json'), 'w'), indent=1)
    np.savez(f'{a.base}/ws_t0/lead_edge.npz', **arrs)


if __name__ == '__main__':
    main()
