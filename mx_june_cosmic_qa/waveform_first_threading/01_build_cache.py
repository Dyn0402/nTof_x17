#!/usr/bin/env python3
"""
wf0_build_cache.py — one-time extraction of waveform windows around the M3
reference track for the det3 Saturday long run (sat_det3).

From-zero threading study: for every matched muon (both planes, radial
residual < 10 mm), store the pedestal/CNS-subtracted 32-sample waveforms of
all strips within ±PAD mm of the reference-track corridor, plus the reference
geometry (raw-strip-frame anchor + rotated tangents) and the production
fit/hits for later comparison.

Output: <Analysis>/waveform_first/wfcache.pkl
"""
import os, sys, glob, pickle
import numpy as np

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa')
from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles
from common.Mx17StripMap import RunConfig

CFG = get_config('sat_det3')
VETO = 50
RES_CUT_MM = 10.0
PAD_MM = 5.0          # corridor half-width around ref track
Z_LO, Z_HI = -3.0, 33.0   # depth range for corridor projection [mm]
N_PED_EVENTS = 300
CHUNK = 400
SAMPLE_NS = 60.0

OUT_DIR = CFG.out_dir('waveform_first')
OUT_PKL = os.path.join(OUT_DIR, 'wfcache.pkl')

# ---------- reference chain (identical to production) ----------
cache = os.path.join(CFG.out_dir('cache'), f'event_results_veto{VETO}.pkl')
align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
results = pickle.load(open(cache, 'rb'))
best = cm.load_alignment(align_json)
rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
xang, _, anum = get_xy_angles(rays.ray_data)
xang = best.ref_x_sign * np.array(xang)
cm.attach_reference_positions(results, rays, best, xang, anum)

events = {}
for r in results:
    if not (r.has_x and r.has_y):
        continue
    if not np.isfinite(r.radial_residual_mm) or r.radial_residual_mm > RES_CUT_MM:
        continue
    if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
        continue
    tx, ty = cm._rotate_ref_tangents(r, best)
    events[int(r.event_id)] = dict(
        eid=int(r.event_id),
        ref_mesh_x=float(r.ref_mesh_x_mm), ref_mesh_y=float(r.ref_mesh_y_mm),
        tan_x=float(tx), tan_y=float(ty),
        ref_x_al=float(r.ref_x_mm), ref_y_al=float(r.ref_y_mm),
        radial_residual=float(r.radial_residual_mm),
        prod=dict(
            det_x=float(r.det_x_mm), det_y=float(r.det_y_mm),
            slope_x=float(r.x_fit.slope_ns_per_mm), slope_y=float(r.y_fit.slope_ns_per_mm),
            t0_x=float(r.x_fit.earliest_time_ns), t0_y=float(r.y_fit.earliest_time_ns),
        ),
    )
print(f'{len(events):,} matched events selected')

# ---------- strip position maps ----------
rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
det = rc.get_detector(CFG.DET_NAME)
pos_map = {}
for feu, axis in ((CFG.MX17_FEU_X, 0), (CFG.MX17_FEU_Y, 1)):
    p = np.full(512, np.nan)
    for ch in range(512):
        hit = det.map_hit(feu, ch)
        if hit is not None and hit[axis] is not None:
            p[ch] = hit[axis]
    pos_map[feu] = p

plane_of_feu = {CFG.MX17_FEU_X: 'x', CFG.MX17_FEU_Y: 'y'}

# corridor channels per event/plane
def corridor_channels(ev, plane):
    feu = CFG.MX17_FEU_X if plane == 'x' else CFG.MX17_FEU_Y
    p0 = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
    tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
    a = p0 + Z_LO * tn
    b = p0 + Z_HI * tn
    lo, hi = min(a, b) - PAD_MM, max(a, b) + PAD_MM
    pm = pos_map[feu]
    ch = np.where((pm >= lo) & (pm <= hi))[0]
    o = np.argsort(pm[ch])
    return ch[o].astype(np.int16), pm[ch][o].astype(np.float32)

for ev in events.values():
    for plane in ('x', 'y'):
        ch, ps = corridor_channels(ev, plane)
        ev[plane] = dict(ch=ch, pos=ps)

# ---------- production hits for the selected events ----------
hits_fs = sorted(glob.glob(os.path.join(CFG.combined_hits_dir, '*_feu-combined_hits.root')))
import pandas as pd
hf = uproot.concatenate(
    [f'{f}:hits' for f in hits_fs],
    expressions=['eventId', 'feu', 'channel', 'amplitude', 'time', 'significance'],
    library='pd')
hf = hf[hf['feu'].isin(CFG.MX17_FEUS) & hf['eventId'].isin(events)]
for (eid, feu), g in hf.groupby(['eventId', 'feu']):
    p = plane_of_feu[feu]
    events[int(eid)][p]['hits'] = dict(
        ch=g['channel'].to_numpy().astype(np.int16),
        amp=g['amplitude'].to_numpy().astype(np.float32),
        time=g['time'].to_numpy().astype(np.float32),
        sig=g['significance'].to_numpy().astype(np.float32))
print('production hits attached')

# ---------- stream decoded waveforms ----------
noise = {}   # (feu, subrun) -> per-channel MAD sigma
for feu in CFG.MX17_FEUS:
    p = plane_of_feu[feu]
    fs = sorted(glob.glob(os.path.join(
        CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root', f'*_{feu:02d}.root')))
    print(f'FEU {feu} ({p}): {len(fs)} files')
    for fn in fs:
        t = uproot.open(fn)['nt']
        eids_all = t.arrays(['eventId'], library='np')['eventId']
        a0 = t.arrays(['amplitude'], entry_stop=N_PED_EVENTS, library='np')['amplitude']
        stack = np.stack([a.reshape(32, 512) for a in a0]).astype(np.float32)
        ped = np.median(stack, axis=(0, 1))
        # post-CNS noise estimate from ped events
        sub = stack - ped[None, None, :]
        cms = np.median(sub.reshape(N_PED_EVENTS, 32, 8, 64), axis=3)
        sub -= np.repeat(cms, 64, axis=2)
        sig = 1.4826 * np.median(np.abs(sub), axis=(0, 1))
        subrun = os.path.basename(fn).split('_datrun_')[1].split('_')[1]
        noise[(feu, subrun)] = sig.astype(np.float32)
        want_mask = np.isin(eids_all, np.fromiter(events.keys(), dtype=np.int64))
        want_idx = np.where(want_mask)[0]
        for lo in range(0, len(want_idx), CHUNK):
            idx = want_idx[lo:lo + CHUNK]
            arr = t.arrays(['eventId', 'amplitude', 'ftst'],
                           entry_start=int(idx[0]), entry_stop=int(idx[-1]) + 1,
                           library='np')
            base = int(idx[0])
            for i in idx:
                j = i - base
                eid = int(arr['eventId'][j])
                if eid not in events:
                    continue
                wfm = arr['amplitude'][j].reshape(32, 512).astype(np.float32) - ped
                cms2 = np.median(wfm.reshape(32, 8, 64), axis=2)
                wfm -= np.repeat(cms2, 64, axis=1)
                ev = events[eid]
                ch = ev[p]['ch']
                ev[p]['W'] = wfm[:, ch].T.astype(np.float16)   # (nstrip, 32)
                ev[p]['noise'] = noise[(feu, subrun)][ch]
                ev[f'ftst_{p}'] = int(arr['ftst'][j])
        print(f'  {os.path.basename(fn)} done')

# drop events missing either plane's waveforms
events = {k: v for k, v in events.items() if 'W' in v['x'] and 'W' in v['y']}
print(f'{len(events):,} events with both waveform windows')

meta = dict(sample_ns=SAMPLE_NS, pad_mm=PAD_MM, z_range=(Z_LO, Z_HI),
            veto=VETO, res_cut_mm=RES_CUT_MM,
            align=dict(z_x=best.z_x, z_y=best.z_y, theta_deg=best.theta_deg),
            ref_sigma_raw=cm.ref_sigma_raw_frame(best, 0.206, 0.242),
            pos_map={f: pos_map[f] for f in CFG.MX17_FEUS})
with open(OUT_PKL, 'wb') as f:
    pickle.dump(dict(meta=meta, events=events), f, protocol=4)
print('wrote', OUT_PKL, f'{os.path.getsize(OUT_PKL)/1e6:.0f} MB')
