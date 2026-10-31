#!/usr/bin/env python3
"""probe.py <key> <out.pkl>: rebuild one SPS row's cache exactly as sps_reco does (3000 events in the plateau) and keep,
per event, the baseline-subtracted Y samples of channels 356-395 (around the oscillating ch 372/373) with the record's
seeds and windows.  Read-only diagnostic for the residual 155-160 mm Y windows (rerun/sps_audit/README.md §7)."""
import csv, os, pickle, sys
import numpy as np

CODE = os.environ['WFT_REPO']
sys.path[:0] = [CODE, os.path.join(CODE, 'rerun', 'model_audit')]
import sps_build_cache as sbc   # noqa: E402

key, outp = sys.argv[1], sys.argv[2]
n_want = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
row = {r['key']: r for r in csv.DictReader(open(os.path.join(CODE, 'rerun', 'sps_registry.csv')))}[key]
orig = sbc.event_record


def wrapped(ch, sm, am, raw, nsmp, *a, **k):
    rec = orig(ch, sm, am, raw, nsmp, *a, **k)
    if rec is not None:
        am2 = sbc._baseline(ch, sm, am, raw)
        m = (ch >= 356) & (ch <= 395)
        rec['_y'] = (ch[m].astype(np.int16), sm[m].astype(np.int16), am2[m].astype(np.float32))
    return rec


sbc.event_record = wrapped
tan = 0.0 if row['mount'] == 'flat' else float(np.tan(np.radians(float(row['mount'].split()[0]))))
evs = {}
for sub in row['subruns'].split():
    got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=n_want - len(evs), tan=tan, only=[row['plateau']],
                          verbose=False).get(row['plateau'], {})
    evs.update(got)
    if len(evs) >= n_want:
        break
slim = {e: dict(seed_x=r['seed_ch_x'], seed_y=r['seed_ch_y'], ref_y=r['ref_mesh_y'], ref_x=r['ref_mesh_x'], t_wall=r['t_wall'],
                y=r['_y'], wy_ch=r['y']['ch'], wy=r['y']['W'], wx_ch=r['x']['ch'], wx=r['x']['W']) for e, r in evs.items()}
pickle.dump(dict(evs=slim, hot={str(k): v for k, v in sbc.HOT_LAST.items()}), open(outp, 'wb'))
print(key, len(slim), dict(sbc.HOT_LAST))
