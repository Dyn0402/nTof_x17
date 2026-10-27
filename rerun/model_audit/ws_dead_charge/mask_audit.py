"""The promoted golden bundle masks against the cross-dataset classes (f12_table.json): what each masked strip is, and
which detector-side faults (flagged under different electronics) the mask leaves out.  Active area: X 0-398.6, Y 18-380
(common/mx17_active_area.py)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, 'f12_table.json')))
C = np.arange(512); BPOS = np.round(49.92 * (C // 64) + (63 - C % 64) * 0.78, 2)
out = {}
for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
    b = json.load(open(os.path.expanduser(f'~/x17/cosmic_bench/model_audit/{det}/calib_bundle/bundle.json')))
    for v in 'xy':
        t = T[f'{det}_{v}']; cls = {r['pos']: r['cls'] for r in t['positions']}
        lo, hi = (0.0, 398.6) if v == 'x' else (18.0, 380.0)
        mask = sorted(float(BPOS[c]) for c in b['dead'][v])
        mc = {}
        for p in mask:
            mc.setdefault(cls.get(p, 'not flagged on its own data by flank/census of other rows'), []).append(p)
        det_side = [r['pos'] for r in t['positions'] if r['cls'] == 'detector' and lo <= r['pos'] <= hi]
        missing = [p for p in det_side if p not in mask]
        # census-dead (M3 crossing, ~20 crossings per channel) that the flank test on the SAME hits finds healthy (f > 0.6
        # with >= 20 flank events): a census statistics artefact, or a strip that fires with its neighbours but not on the
        # mapped M3 crossing (mis-seated pin, sharing from neighbours)
        key = b['run_key']
        cen = json.load(open(os.path.join(HERE, '..', '..', 'channel_layer', f'{det}_{key}.json')))
        fl = json.load(open(os.path.join(HERE, 'census', f'flank_{det}_{key}.json')))[v]
        healthy = [int(c) for c in cen[v]['dead'] if fl['f'][c] > 0.6 and max(fl['n1'][c], fl['n2'][c]) >= 20]
        out[f'{det}_{v}'] = dict(n_mask=len(mask), mask_by_class={k: len(x) for k, x in mc.items()},
                                 detector_side_in_active=len(det_side), detector_side_not_masked=missing,
                                 census_dead=len(cen[v]['dead']), census_dead_flank_healthy=healthy)
        print(f'   census dead {len(cen[v]["dead"])}, of which flank-healthy on the same hits {len(healthy)}: {healthy}')
        print(f'{det} {v}: mask {len(mask)} ->', {k: len(x) for k, x in mc.items()}, f'| detector-side in active area {len(det_side)}, '
              f'not in the mask {len(missing)}: {missing[:30]}')
json.dump(out, open(os.path.join(HERE, 'mask_audit.json'), 'w'), indent=1)
