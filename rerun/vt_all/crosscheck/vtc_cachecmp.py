#!/usr/bin/env python3
"""vtc_cachecmp.py -- byte-compare of rerun/model_audit/sps_build_cache.py across vt_all's two edits (CROSSCHECK §4).

    build    vtc_cachecmp.py build <builder.py> <sps key> <out.pkl> [--chunk N]
             imports THAT file as the builder (everything else -- datasets, det4_sps_map -- from this checkout) and
             builds the row's cache exactly as sps_reco.py does (subruns in order, per_plateau 3000, only=[plateau],
             half_win 16, the mount's tan); pickles {events, HOT_LAST}.
    compare  vtc_cachecmp.py compare <a.pkl> <b.pkl>
             same event keys, every record field equal (arrays: dtype, shape and values; np.array_equal), same HOT_LAST.

The two builders: 15900b39 (the SPS audit's seed fix, before vt_all) and 514f7aac (+ 4eb667c1 CHUNK, + b3cc38a6
close-after-read and skip-chunks-outside-open-plateaus).  The seed rule is the same in both.
"""
import csv
import importlib.util
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def build(builder, key, out, chunk=None):
    os.environ['WFT_REPO'] = REPO
    spec = importlib.util.spec_from_file_location('sps_build_cache', builder)
    sbc = importlib.util.module_from_spec(spec)
    sys.modules['sps_build_cache'] = sbc
    spec.loader.exec_module(sbc)
    if chunk and hasattr(sbc, 'CHUNK'):
        sbc.CHUNK = chunk
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[key]
    tan = 0.0 if row['mount'] == 'flat' else float(np.tan(np.radians(float(row['mount'].split()[0]))))
    evs = {}
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=3000 - len(evs), half_win=16, tan=tan,
                              only=[row['plateau']], verbose=False).get(row['plateau'], {})
        evs.update(got)
        print(f'{sub}: {len(got)} ({len(evs)})', flush=True)
        if len(evs) >= 3000:
            break
    hot = {str(k): sorted(int(c) for c in v) for k, v in sbc.HOT_LAST.items()}
    pickle.dump(dict(events=evs, hot=hot, builder=builder), open(out, 'wb'), protocol=4)
    import resource
    print(f'{key}: {len(evs)} events, hot {hot}, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB')


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b))


def compare(pa, pb):
    A, B = pickle.load(open(pa, 'rb')), pickle.load(open(pb, 'rb'))
    ea, eb = A['events'], B['events']
    ka, kb = set(ea), set(eb)
    diff = [k for k in sorted(ka & kb) if not same(ea[k], eb[k])]
    print(f'{os.path.basename(pa)} vs {os.path.basename(pb)}: {len(ka)} / {len(kb)} events, {len(ka - kb)} only in A, '
          f'{len(kb - ka)} only in B, {len(diff)} records differ; hot {A["hot"] == B["hot"]} ({A["hot"]})')
    print('pickled events byte-identical:', pickle.dumps(ea, protocol=4) == pickle.dumps(eb, protocol=4))
    return not (ka ^ kb) and not diff and A['hot'] == B['hot']


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        ch = int(sys.argv[sys.argv.index('--chunk') + 1]) if '--chunk' in sys.argv else None
        build(sys.argv[2], sys.argv[3], sys.argv[4], ch)
    else:
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
