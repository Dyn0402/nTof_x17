#!/usr/bin/env python3
"""build_table.py [dets...] -- one per-event table per golden row for the v-uncertainty / w0 study (ws_v_w0).

Joins the finalize reconstruction (final_physmask/events.parquet), the M3 rays (evttime, Chi2X/Y, NClus, the
M3-frame tangents) and the finalize alignment (reference mesh positions in the raw strip frame, the reference
tangents rotated into it exactly as 03_angles does).  Writes <root>/ws_v_w0/<det>.parquet.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import awkward as ak                                     # noqa: E402
import cosmic_micro_tpc_analysis as cm                   # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles   # noqa: E402
from wft import compat                                   # noqa: E402

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
KEYS = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}


def build(det, key=None, table=None, alignment=None, out=None, m3dir=None):
    d = os.path.join(ROOT, det, 'final_physmask')
    cfg = get_config(key or KEYS[det])
    params = cm.load_alignment(alignment or os.path.join(d, 'alignment', 'alignment.json'))
    df = pd.read_parquet(table or os.path.join(d, 'events.parquet')).drop_duplicates('event_id', keep=False)
    results = compat.as_event_results(df)
    rays = M3RefTracking(m3dir or cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    rd = rays.ray_data
    cols = [k for k in ['evn', 'evttime', 'Chi2X', 'Chi2Y', 'NClusX', 'NClusY'] if k in ak.fields(rd)]
    m3 = pd.DataFrame({k: ak.to_numpy(rd[k]) for k in cols}).drop_duplicates('evn', keep=False).set_index('evn')
    rows = []
    for r in results:
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, params)
        rows.append(dict(event_id=int(r.event_id), tan_ref_x=tx, tan_ref_y=ty, m3_tan_x=r.ref_tan_theta_x,
                         m3_tan_y=r.ref_tan_theta_y, ref_mesh_x=r.ref_mesh_x_mm, ref_mesh_y=r.ref_mesh_y_mm))
    ref = pd.DataFrame(rows).set_index('event_id')
    t = df.set_index('event_id').join(ref, how='inner').join(m3, how='left')
    t['theta_align_deg'] = params.theta_deg
    out = out or os.path.join(ROOT, 'ws_v_w0', f'{det}.parquet')
    t.to_parquet(out)
    print(det, len(t), 'events with a reference ->', out)


# det3 rows of the dry-run vrefit (field / mounting tests of w0): name -> (run key for the config, table, alignment,
# M3 dir).  6-25 has no qa_config key; its config is only used for the M3 directory, which is given explicitly.
DRY = os.path.join(ROOT, 'ws_v_w0', 'dryrun')
BENCH = '/home/dylan/x17/cosmic_bench/det3'
EXTRA = {
    'det3_d700': (f'{DRY}/d700/vrefit/events1.parquet', f'{DRY}/d700/vrefit/alignment0/alignment.json',
                  f'{BENCH}/mx17_det3_saturday_scan_6-27-26/drift_scan_resist_490V_drift_700V/m3_tracking_root_v2/'),
    'det3_d500': (f'{DRY}/d500/vrefit/events0.parquet', f'{DRY}/d500/vrefit/alignment0/alignment.json',
                  f'{BENCH}/mx17_det3_saturday_scan_6-27-26/drift_scan_resist_490V_drift_500V/m3_tracking_root_v2/'),
    'det3_day625': (f'{DRY}/day625/vrefit/events0.parquet', f'{DRY}/day625/vrefit/alignment0/alignment.json',
                    f'{BENCH}/mx17_det3_day_6-25-26/long_run/m3_tracking_root/'),
}


if __name__ == '__main__':
    for det in (sys.argv[1:] or list(KEYS)):
        if det in EXTRA:
            tab, al, m3 = EXTRA[det]
            build('det3', key='sat_det3', table=tab, alignment=al, m3dir=m3, out=os.path.join(ROOT, 'ws_v_w0', f'{det}.parquet'))
        else:
            build(det)
