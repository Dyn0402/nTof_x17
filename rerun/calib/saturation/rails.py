#!/usr/bin/env python3
"""rails.py -- the per-sample ADC rails of every window of a held-out cache, from the raw decoded_root.

The cache windows (wft.calibrate.build_cache) hold W = raw - ped - CM (pedestal per channel over the file's first 300
events, plain 64-channel block median per sample).  The ADC is 12 bit, so a sample's TOP rail is 4095 - ped - CM and
its BOTTOM rail is 0 - ped - CM, both per channel and per sample (the CM moves them).  The cache does not carry them.

    rails.py DET [--cache C] [--out O]     -> O (default $CLAUDE_JOB_DIR/tmp/satur/rails_<det>.pkl)

Output: {eid: {plane: dict(raw=int16 (n, ns), ped=float32 (n,), cm=float32 (n, ns))}} for the cache's windows, and a
check that raw - ped - cm reproduces the cache W (printed; refuses to write above 3 ADC: det4's cache was built on an
older decode whose pedestal differs by <= 1.75 ADC; all other chambers are bit-exact).  Consumers take the rails
RELATIVE TO THE CACHE W -- top = W + (4095 - raw), bottom = W - raw -- so they are exact whatever the pedestal vintage
(rail_arrays below).
Golden rows (rerun/model_audit/make_gate_dag.py GOLDEN): det3 sat_det3, det2 o22_long_det2, det4 g_det4,
det6 g_det6_long, det7 g_det7_long.
"""
import argparse, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
GOLDEN = {'det3': 'sat_det3', 'det2': 'o22_long_det2', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
TMP = os.path.join(os.environ.get('CLAUDE_JOB_DIR', os.path.expanduser('~/.claude/jobs/c68d6c6a')), 'tmp', 'satur')
CACHE = os.path.join(os.environ.get('CLAUDE_JOB_DIR', os.path.expanduser('~/.claude/jobs/c68d6c6a')), 'tmp', 't15',
                     'caches', '{}.pkl')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('det'); ap.add_argument('--cache', default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from qa_config import get_config
    from wft import io as wio
    cache = a.cache or CACHE.format(a.det)
    out = a.out or os.path.join(TMP, f'rails_{a.det}.pkl')
    C = pickle.load(open(cache, 'rb'))
    cfg = get_config(GOLDEN[a.det])
    nblk = 512 // wio.CNS_BLOCK
    R, err = {}, []
    for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        want = {e for e, ev in C.items() if plane in ev}
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            if not want:
                break
            r = wio.FeuReader(f)
            idx = np.flatnonzero(np.isin(r.event_ids, np.fromiter(want, dtype=np.int64)))
            for lo in range(0, len(idx), 400):
                blk = idx[lo:lo + 400]
                arr = r.tree.arrays(['eventId', 'amplitude'], entry_start=int(blk[0]), entry_stop=int(blk[-1]) + 1,
                                    library='np')
                for i in blk:
                    j = i - int(blk[0]); eid = int(arr['eventId'][j]); amp = arr['amplitude'][j]
                    if eid not in want or len(amp) != r.n_sample * 512:
                        continue
                    raw = amp.reshape(-1, 512).astype(np.float32)
                    sub = raw - r.ped
                    cms = np.median(sub.reshape(raw.shape[0], nblk, wio.CNS_BLOCK), axis=2)   # (ns, nblk)
                    P = C[eid][plane]; ch = np.asarray(P['ch'], int)
                    cm = cms[:, ch // wio.CNS_BLOCK].T
                    rec = raw[:, ch].T - r.ped[ch][:, None] - cm
                    if rec.shape != np.asarray(P['W']).shape:
                        continue
                    err.append(float(np.abs(rec - P['W']).max()))
                    R.setdefault(eid, {})[plane] = dict(raw=raw[:, ch].T.astype(np.int16), ped=r.ped[ch].astype(np.float32),
                                                        cm=cm.astype(np.float32))
                    want.discard(eid)
        print(f'{a.det} {plane}: {sum(1 for e in R.values() if plane in e)} windows, {len(want)} not found')
    err = np.array(err)
    print(f'{a.det}: max |raw - ped - cm - W| = {err.max():.4f} ADC (p99 {np.percentile(err, 99):.4f})')
    if err.max() > 3.0:
        raise SystemExit('rebuild does not reproduce the cache W -- not writing')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pickle.dump(R, open(out, 'wb'), protocol=4)
    print(out)


def rail_arrays(W, raw):
    """(top, bottom) rail of every sample of a window, in the units of the cache W (pedestal + CM subtracted)."""
    W = np.asarray(W, np.float64); raw = np.asarray(raw, np.float64)
    return W + (4095.0 - raw), W - raw


if __name__ == '__main__':
    main()
