#!/usr/bin/env python3
"""vbox.py --out-dir D [--per-plateau 1000] [--t-min KEY=SECONDS ...] <registry key> [...]

Drift velocity for SPS rows that have no v (pass-1 review P3), by the model-free column-duration route of ws_spsv
(rerun/model_audit/ws_spsv/REPORT.md): the strip-sum waveform of a view is a box of length T = gap / v convolved with
the electronics template, whatever the track angle, so v = 28.8 mm / T.  Per row: the row's det4 events through the
pass's cache builder (sps_build_cache.build_cache, same baseline as the pass), then zsstack.py, the estimator ws_spsv
quotes: the box (x template, charge gradient) fitted to the STACKED strip sum (T_box), with the flat box (T_flat) and
the 50 % width beside it.  Per-event box fits are not used: on ZS rows they scatter 0.6-2.3 us (suppressed tails).
Only valid where the column fits in the 64 x 60 ns window (ws_spsv: above ~170 V/cm); below, T is a lower bound.

--t-min KEY=S drops a row's events before wall-clock S (run60 ov07: the beam came back at ~91170 s; before that the
triggers are beam-off, see README §iii).  Writes <out-dir>/<key>.pkl caches and <out-dir>/vbox.json.
"""
import argparse, csv, json, os, pickle, subprocess, sys

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
BUNDLE = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_bundle'
GAP_MM = 28.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+'); ap.add_argument('--out-dir', required=True); ap.add_argument('--per-plateau', type=int, default=1000)
    ap.add_argument('--t-min', nargs='*', default=[]); ap.add_argument('--bundle', default=BUNDLE)
    a = ap.parse_args()
    import sps_build_cache as sbc
    os.makedirs(a.out_dir, exist_ok=True)
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
    tmin = {k: float(v) for k, v in (s.split('=') for s in a.t_min)}
    caches = []
    for key in a.keys:
        row = reg[key]; evs = {}
        for sub in row['subruns'].split():
            need = a.per_plateau - len(evs)
            if need <= 0:
                break
            got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=need if key not in tmin else 4000,
                                  only=[row['plateau']], verbose=False)
            for eid, ev in got.get(row['plateau'], {}).items():
                if ev.get('t_wall', 0) >= tmin.get(key, -1e18) and len(evs) < a.per_plateau:
                    evs[eid] = ev
        p = os.path.join(a.out_dir, f'{key}.pkl'); pickle.dump(evs, open(p, 'wb'), protocol=4); caches.append(p)
        print(f'{key}: {len(evs)} events', flush=True)
    out = os.path.join(a.out_dir, 'zsstack.json')
    subprocess.run([sys.executable, os.path.join(REPO, 'rerun', 'model_audit', 'ws_spsv', 'zsstack.py'), '--bundle', a.bundle,
                    '--out', out] + caches, check=True)
    B = json.load(open(out)); res = {}
    for key in a.keys:
        r = {}
        for v in 'xy':
            q = B.get(key, {}).get(v)
            if q:
                r[v] = {k: q.get(k) for k in ('n', 'T_box', 'T_flat', 'T_edge', 'T_50', 'grad', 'box_t0')}
                r[v]['v_box'] = GAP_MM * 1e3 / q['T_box'] if q.get('T_box') else None
        res[key] = dict(field_Vcm=float(reg[key]['field_Vcm']), gas=reg[key]['gas'], mount=reg[key]['mount'], **r)
        print(key, {v: (r[v]['T_box'], r[v]['T_flat'], r[v]['v_box'] and round(r[v]['v_box'], 2)) for v in r}, flush=True)
    json.dump(res, open(os.path.join(a.out_dir, 'vbox.json'), 'w'), indent=1)
    for p in caches:
        os.remove(p)


if __name__ == '__main__':
    main()
