#!/usr/bin/env python3
"""classify.py <det> -- failure-mode class for every outlier event (and the controls, as a check of the classifier).

Outlier: in either view, the production selection (ok, n_dropped <= 2) with a ladder-scaled angle residual > 5 deg
or a raw-frame position residual > 3 mm, OR a fit rejected for competing clusters (n_dropped > 2).

Per view (waveform features, wffeat.py). Footprint = the reference segment through the 30 mm gap
[ref_mesh, ref_mesh + 30 tan_ref], +-2.5 mm:
  outside   the footprint leaves the measured active area (common/mx17_active_area.TRUE_ACTIVE_BY_DET)
  edge      it comes within EDGE mm of the active edge
  nocharge  no strip on the footprint above SNR 10
  extra     charge on the footprint AND another cluster (> 15 % of the largest) > 5 mm off it
  wide      one cluster wider than the footprint by > WIDE mm (a delta ray / shower riding on the track)
  sat       >= 3 strips on the ADC rail
Event class, first match wins (verified by eye on random displays of every class, see the report):
  REJECTED_OK  rejected only by the competing-cluster cut (n_dropped > 2) while the fit itself agrees with M3
             (|dθ| <= 5 deg, |res| <= 3 mm): not a failure; the cut is conservative
  OUTSIDE    the M3 track misses the active area: not a detector or reference failure (acceptance)
  EDGE       an outlier view's footprint is within 20 mm of the active edge of its own coordinate (drift-field
             distortion zone, edge_response.py) or within 8 mm of the other coordinate's edge (truncated cluster)
  HW_KNOWN   known hardware: det7 X card 1 (asterisk), det4 dead / low-gain amplification bands
  MX17_DEAD  no charge on the footprint and nothing anywhere in an outlier view: MX17 did not see the track
  M3_ELSEWHERE  MX17 has one clean particle in both views and none of it on the footprint in either: the M3 track
             is not the particle MX17 saw
  POS_TAIL   an outlier view holds one clean cluster with the RIGHT angle (|dθ| < 3°) but > 3 mm from the M3
             point, and the fit sits on that cluster. First read as an M3 pointing failure; the two-chamber test
             (sixpt.py, 6-26 night) puts the error on the MX17 chamber, never on an M3 layer, and an M3 track with
             4 clusters and chi2 < 1 mm^2 cannot mis-point by > ~1 mm at the DUT. So: the far tail of the MX17
             position response. `why` keeps which single M3 doublet the MX17 point happens to agree with
  MULTI      extra charge in an outlier view: a second particle / shower; the fit took the other one
  DELTA      one cluster far wider than a single track: a delta ray (or shower) attached to the track
  ONE_VIEW_ELSEWHERE  charge elsewhere in one view, on the footprint in the other
  FIT_SAT    one clean cluster on the footprint, saturated, fit off -> reconstruction / saturation model
  FIT        one clean cluster on the footprint, unsaturated, fit off -> reconstruction
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO]
from olib import BASE, load  # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402

DETNAME = {'det2': 'mx17_2', 'det3': 'mx17_3', 'det4': 'mx17_4', 'det6': 'mx17_6', 'det7': 'mx17_7'}
EDGE, EDGE_OWN, WIDE, MATCH, SNR_ON, BAND = 8.0, 20.0, 8.0, 3.0, 10.0, 2.5
ORDER = ['REJECTED_OK', 'OUTSIDE', 'EDGE', 'HW_KNOWN', 'MX17_DEAD', 'M3_ELSEWHERE', 'POS_TAIL', 'MULTI', 'DELTA', 'ONE_VIEW_ELSEWHERE',
         'FIT_SAT', 'FIT']


def outlier_ids(det):
    d = load(det); out = np.zeros(len(d), bool)
    for v in 'xy':
        out |= d[f'use_{v}'] & ((d[f'dl_{v}'].abs() > 5) | (d[f'rres_{v}'].abs() > 3))
        out |= d[f'{v}_ok'] & (d[f'{v}_n_dropped'] > 2)
    return d.event_id[out].tolist(), d.event_id[~out].tolist()


def table(det):
    d = load(det)
    F = pd.read_parquet(os.path.join(BASE, f'wffeat_{det}.parquet'))
    M = pd.read_parquet(os.path.join(BASE, f'm3feat_{det}.parquet'))
    x = d.merge(F, on='event_id').merge(M, on='event_id', how='left')
    act = TRUE_ACTIVE_BY_DET[DETNAME[det]]
    out = np.zeros(len(x), bool)
    for v in 'xy':
        ov = x[f'use_{v}'] & ((x[f'dl_{v}'].abs() > 5) | (x[f'rres_{v}'].abs() > 3))
        ov |= x[f'{v}_ok'] & (x[f'{v}_n_dropped'] > 2)
        out |= ov; x[f'out_{v}'] = ov
        lo = np.minimum(x[f'ref_mesh_{v}'], x[f'ref_mesh_{v}'] + 30 * x[f'ref_tan_{v}'])
        hi = np.maximum(x[f'ref_mesh_{v}'], x[f'ref_mesh_{v}'] + 30 * x[f'ref_tan_{v}'])
        x[f'outside_{v}'] = (lo < act[v][0] - 1) | (hi > act[v][1] + 1)
        x[f'edge_{v}'] = (lo < act[v][0] + EDGE) | (hi > act[v][1] - EDGE)
        # the view's OWN coordinate: the drift-field distortion zone near the cage walls (edge_response.py: the
        # angle response is off within ~20 mm of an edge on every chamber that has no passivation there)
        x[f'edgeown_{v}'] = (lo < act[v][0] + EDGE_OWN) | (hi > act[v][1] - EDGE_OWN)
        x[f'noch_{v}'] = x[f'w{v}_snr_ref'] < SNR_ON
        x[f'extra_{v}'] = ~x[f'noch_{v}'] & (x[f'w{v}_ncl_far'] >= 1)
        x[f'wide_{v}'] = ~x[f'noch_{v}'] & ~x[f'extra_{v}'] & (x[f'w{v}_wid1'] > (hi - lo) + 2 * BAND + WIDE)
        x[f'sat_{v}'] = x[f'w{v}_nsatstrip'] >= 3
    x['outlier'] = out
    # chamber-specific, already-known hardware: det7 X card 1 (M1 badly seated, asterisk; ch 0-255 = x < 199.3 mm);
    # det4 dead / low-gain amplification bands (median charge along x < 30 % of the 90th percentile, live_det4)
    x['hw_x'] = False; x['hw_y'] = False
    if det == 'det7':
        x['hw_x'] = np.minimum(x.ref_mesh_x, x.ref_mesh_x + 30 * x.ref_tan_x) < 199.3
    if det == 'det4':
        L = pd.read_parquet(os.path.join(BASE, 'an', 'live_det4.parquet'))
        x = x.merge(L, on='event_id', how='left'); x['hw_x'] = x['hw_y'] = x.live < 0.3
    x['outside_any'] = x.outside_x | x.outside_y
    x['edge_any'] = x.edge_x | x.edge_y
    return x


def doublets(r):
    """The four single-doublet points at the DUT (raw frame), each at mid-gap depth."""
    pts = []
    for v in 'xy':
        for k in ('up', 'dn'):
            s = r.get(f'm3_{k}_{v}')
            if isinstance(s, str):
                ux, uy = json.loads(s)
                pts.append((f'{k}_{v}', ux + 15 * r['ref_tan_x'], uy + 15 * r['ref_tan_y']))
    return pts


def m3_explains(r, px, py):
    """Does another M3 cluster combination, or one doublet alone, point at (px, py) (mid-gap, raw frame)?"""
    best = (np.inf, '')
    for v in 'xy':
        for a in json.loads(r.get(f'm3_alt_{v}') or '[]'):
            c2, ax, ay, tx, ty = a
            dd = max(abs(ax + 15 * tx - px), abs(ay + 15 * ty - py))
            if dd < best[0]:
                best = (dd, f'alt_{v} chi2 {c2:.2f}')
    for k, ux, uy in doublets(r):
        dd = max(abs(ux - px), abs(uy - py))
        if dd < best[0]:
            best = (dd, f'doublet {k}')
    return best


def classify_row(r):
    ov = [v for v in 'xy' if r[f'out_{v}']]
    fine = all(abs(r[f'dl_{v}']) <= 5 and abs(r[f'rres_{v}']) <= 3 for v in ov if r[f'{v}_ok'])
    if fine and all(r[f'{v}_ok'] for v in ov):
        return 'REJECTED_OK', ''
    if r['outside_any']:
        return 'OUTSIDE', ''
    if any(r[f'edge_{v}'] or r[f'edgeown_{v}'] for v in ov):
        return 'EDGE', ('own coordinate < 20 mm' if any(r[f'edgeown_{v}'] for v in ov) else 'other coordinate < 8 mm')
    if any(r[f'hw_{v}'] for v in ov):
        return 'HW_KNOWN', ''
    if any(r[f'noch_{v}'] and r[f'w{v}_ncl'] == 0 for v in ov):
        return 'MX17_DEAD', ''
    clean = {v: (r[f'w{v}_ncl'] <= 1) and not r[f'wide_{v}'] for v in 'xy'}
    if r['noch_x'] and r['noch_y'] and clean['x'] and clean['y'] and r['wx_ncl'] == 1 and r['wy_ncl'] == 1:
        dd, why = m3_explains(r, r['wx_pos1'], r['wy_pos1'])
        return 'M3_ELSEWHERE', (why if dd < MATCH else 'unexplained by M3 clusters')
    for v in ov:
        o = 'y' if v == 'x' else 'x'
        if (clean[v] and r[f'w{v}_ncl'] == 1 and abs(r[f'dl_{v}']) < 3 and abs(r[f'rres_{v}']) > 3
                and r[f'h{v}_dfit'] < 1.0 and not r[f'noch_{o}']):
            # the MX17 point in raw coordinates: the fitted view + the reference in the other view
            p = {v: r[f'{v}_p0'] + 15 * r[f'tl_{v}'], o: r[f'ref_mesh_{o}'] + 15 * r[f'ref_tan_{o}']}
            dd, why = m3_explains(r, p['x'], p['y'])
            return 'POS_TAIL', (why if dd < 1.5 else f"no M3 combination within 1.5 mm (best {dd:.1f})")
    if any(r[f'extra_{v}'] or r[f'noch_{v}'] and r[f'w{v}_ncl'] > 1 for v in ov) or \
            any(r[f'{v}_ok'] and r[f'{v}_n_dropped'] > 2 for v in ov):
        return 'MULTI', ''
    if any(r[f'wide_{v}'] for v in ov):
        return 'DELTA', ''
    if r['noch_x'] or r['noch_y']:
        return 'ONE_VIEW_ELSEWHERE', ''
    if any(r[f'sat_{v}'] for v in ov):
        return 'FIT_SAT', ''
    return 'FIT', ''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('det'); a = ap.parse_args()
    x = table(a.det)
    cl = [classify_row(r) for r in x.to_dict('records')]
    x['cls'] = [c[0] for c in cl]; x['why'] = [c[1] for c in cl]
    x.to_parquet(os.path.join(BASE, f'classified_{a.det}.parquet'))
    o = x[x.outlier]; n = len(load(a.det))
    print(f'{a.det}: {len(o)} outlier events of {n} ({100 * len(o) / n:.1f} %)')
    print(pd.DataFrame({'n': o.cls.value_counts(), '% of events': (o.cls.value_counts() / n * 100).round(2)}).reindex(ORDER).to_string())
    for c in ('POS_TAIL', 'M3_ELSEWHERE'):
        print(c, o[o.cls == c].why.str.split(' chi2').str[0].value_counts().to_string())
    print('controls:', x[~x.outlier].cls.value_counts().to_dict())


if __name__ == '__main__':
    main()
