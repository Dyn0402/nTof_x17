#!/usr/bin/env python3
"""Forward-fit drift-velocity vs drift HV across the Saturday scan.

Per scan subrun: build a mini waveform cache (long-run alignment + subrun
rays/event-cache), then scan v on ref-pinned fits (v2 hypers fixed, t0 free,
NNLS charge) -> chi2(v) minimum = forward-fit v at that HV.
"""
import os, sys, glob, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa')
from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles
from common.Mx17StripMap import RunConfig

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
import forward_model2 as fm2

CFG = get_config('sat_det3')
BASE = fm2.BASE
hj = json.load(open(os.path.join(BASE, 'hyper_v2.json')))
H0 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
ARUN = os.path.dirname(os.path.dirname(CFG.OUT_BASE.rstrip('/')))  # .../<run>
SUBRUNS = ['drift_scan_resist_490V_drift_%dV' % v for v in
           (300, 500, 700, 900, 1100)]  # 100V likely too slow/truncated; try later
V_GRID = dict((s, None) for s in SUBRUNS)
V_EXPECT = {300: 24.0, 500: 30.0, 700: 33.0, 900: 35.5, 1100: 37.5}

align_json = os.path.join(CFG.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json')
best = cm.load_alignment(align_json)
rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
det = rc.get_detector(CFG.DET_NAME)
pos_map = {}
for feu, axis in ((7, 0), (8, 1)):
    p = np.full(512, np.nan)
    for ch in range(512):
        hit = det.map_hit(feu, ch)
        if hit is not None and hit[axis] is not None:
            p[ch] = hit[axis]
    pos_map[feu] = p

PAD = 5.0
N_PED = 300


def build_mini_cache(subrun):
    out_pkl = os.path.join(BASE, f'wfcache_{subrun.split("drift_")[-1]}.pkl')
    if os.path.exists(out_pkl):
        return pickle.load(open(out_pkl, 'rb'))
    sub_dir = os.path.join(CFG.BASE_PATH, CFG.RUN, subrun)
    m3dir = sub_dir + '/m3_tracking_root_v2/'
    if not os.path.isdir(m3dir):
        m3dir = sub_dir + '/m3_tracking_root/'
    cache_pkl = os.path.join(ARUN, subrun, 'mx17_3', 'cache',
                             'event_results_veto50.pkl')
    results = pickle.load(open(cache_pkl, 'rb'))
    rays = M3RefTracking(m3dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    events = {}
    for r in results:
        if not (r.has_x and r.has_y):
            continue
        if not np.isfinite(r.radial_residual_mm) or r.radial_residual_mm > 10.0:
            continue
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, best)
        events[int(r.event_id)] = dict(
            eid=int(r.event_id), ref_mesh_x=float(r.ref_mesh_x_mm),
            ref_mesh_y=float(r.ref_mesh_y_mm), tan_x=float(tx), tan_y=float(ty))
    # corridor channels: use widest v in scan for z range
    for ev in events.values():
        for plane, feu in (('x', 7), ('y', 8)):
            p0 = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
            tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
            a = p0 - 3.0 * abs(tn)
            b = p0 + 33.0 * abs(tn)
            lo, hi = min(a, b) - PAD, max(a, b) + PAD
            pm = pos_map[feu]
            ch = np.where((pm >= lo) & (pm <= hi))[0]
            o = np.argsort(pm[ch])
            ev[plane] = dict(ch=ch[o].astype(np.int16),
                             pos=pm[ch][o].astype(np.float32))
    for feu, plane in ((7, 'x'), (8, 'y')):
        fs = sorted(glob.glob(os.path.join(sub_dir, 'decoded_root',
                                           f'*_{feu:02d}.root')))
        for fn in fs:
            t = uproot.open(fn)['nt']
            eids_all = t.arrays(['eventId'], library='np')['eventId']
            a0 = t.arrays(['amplitude'], entry_stop=N_PED, library='np')['amplitude']
            stack = np.stack([a.reshape(32, 512) for a in a0]).astype(np.float32)
            ped = np.median(stack, axis=(0, 1))
            sub0 = stack - ped[None, None, :]
            cms = np.median(sub0.reshape(N_PED, 32, 8, 64), axis=3)
            sub0 -= np.repeat(cms, 64, axis=2)
            sig = 1.4826 * np.median(np.abs(sub0), axis=(0, 1))
            want = np.where(np.isin(eids_all,
                                    np.fromiter(events.keys(), np.int64)))[0]
            for lo_i in range(0, len(want), 400):
                idx = want[lo_i:lo_i + 400]
                arr = t.arrays(['eventId', 'amplitude'],
                               entry_start=int(idx[0]), entry_stop=int(idx[-1]) + 1,
                               library='np')
                base_i = int(idx[0])
                for i in idx:
                    j = i - base_i
                    eid = int(arr['eventId'][j])
                    if eid not in events:
                        continue
                    wfm = arr['amplitude'][j].reshape(32, 512).astype(np.float32) - ped
                    cms2 = np.median(wfm.reshape(32, 8, 64), axis=2)
                    wfm -= np.repeat(cms2, 64, axis=1)
                    ev = events[eid]
                    ch = ev[plane]['ch']
                    ev[plane]['W'] = wfm[:, ch].T.astype(np.float16)
                    ev[plane]['noise'] = sig[ch]
    events = {k: v for k, v in events.items()
              if 'W' in v.get('x', {}) and 'W' in v.get('y', {})}
    pickle.dump(events, open(out_pkl, 'wb'))
    print(f'{subrun}: {len(events)} events cached', flush=True)
    return events


def solve_t0(P, plane, p0l, wline, t0_grid):
    W, noise, pos, sat = fm2.prep_plane(P, plane)
    best_c, best_t = np.inf, t0_grid[0]
    for t0 in t0_grid:
        c, _ = fm2.chi2_plane(plane, W, noise, pos, sat, p0l, wline, float(t0), H0)
        if c < best_c:
            best_c, best_t = c, t0
    return best_c, best_t


_ev_store = {}

def ev_chi2_at_v(args):
    eid, v = args
    ev = _ev_store[eid]
    tot = 0.0
    for plane in ('x', 'y'):
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        if abs(tn) < 0.08:
            continue
        p0l = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        c, _ = solve_t0(ev[plane], plane, p0l, tn * v * 1e-3,
                        np.arange(150.0, 800.0, 30.0))
        if np.isfinite(c):
            tot += c
    return tot

if __name__ == '__main__':
    out = {}
    for subrun in SUBRUNS:
        hv = int(subrun.split('_')[-1][:-1])
        events = build_mini_cache(subrun)
        sel = [e for e, ev in events.items()
               if 0.10 < np.hypot(ev['tan_x'], ev['tan_y']) < 0.45][:250]
        _ev_store.clear(); _ev_store.update({e: events[e] for e in sel})
        vexp = V_EXPECT[hv]
        vgrid = np.arange(max(vexp - 10, 8), vexp + 11, 1.0)
        chis = []
        with ProcessPoolExecutor(max_workers=14) as pool:
            for v in vgrid:
                c = sum(pool.map(ev_chi2_at_v, [(e, v) for e in sel], chunksize=8))
                chis.append(c)
                print(f'  {subrun} v={v:.0f}: chi2 {c:.4e}', flush=True)
        chis = np.array(chis)
        j = int(np.argmin(chis))
        # parabolic min
        if 0 < j < len(vgrid) - 1:
            a, b, c_ = chis[j - 1], chis[j], chis[j + 1]
            vmin = vgrid[j] + 0.5 * (a - c_) / (a - 2 * b + c_) * (vgrid[1] - vgrid[0])
        else:
            vmin = vgrid[j]
        out[hv] = dict(v=float(vmin), n=len(sel),
                       vgrid=vgrid.tolist(), chi2=chis.tolist())
        print(f'== {subrun}: v_fit = {vmin:.1f} um/ns (n={len(sel)}) ==', flush=True)
        json.dump(out, open(os.path.join(BASE, 'drift_scan_v.json'), 'w'), indent=1)
    print('saved drift_scan_v.json')
