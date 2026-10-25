#!/usr/bin/env python3
"""run_bench.py <run_key> --feu-x F --feu-y F [--subs ...] --out-dir DIR

Accumulate the pogo-check pair statistics (accumulate.PairStats) for both planes of one bench chamber from
hits_root (QA only).  Writes <out-dir>/<run_key>_<plane>.npz, which `pogo_check.py` turns into verdicts.

    ../../.venv/bin/python run_bench.py g_det7_long --feu-x 6 --feu-y 8 --subs long_run longer_run --out-dir out
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
import accumulate as A


def plane(cfg, subs, feu, pos):
    import uproot
    files = [f for s in subs for f in sorted(glob.glob(os.path.join(cfg.BASE_PATH, cfg.RUN, s, 'hits_root',
                                                                     f'*_{feu:02d}_hits.root')))]
    if not files:
        raise SystemExit(f'no hits_root files for FEU {feu} in {subs}')
    rms = uproot.open(files[0])['pedestals'].arrays(library='np')['rms'].astype(float)
    ps = A.PairStats(A.strip_index_from_positions(pos))
    for f in files:
        a = uproot.open(f)['hits'].arrays(['eventId', 'channel', 'amplitude', 'significance', 'saturated'], library='np')
        o = np.argsort(a['eventId'], kind='stable'); a = {k: v[o] for k, v in a.items()}
        e = a['eventId']; b = np.flatnonzero(np.diff(e)) + 1
        for s0, s1 in zip(np.r_[0, b], np.r_[b, len(e)]):
            if a['saturated'][s0:s1].any():
                continue
            g = a['significance'][s0:s1] > A.SIG_MIN
            ps.add(a['channel'][s0:s1][g].astype(int), a['amplitude'][s0:s1][g])
    return ps, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key'); ap.add_argument('--feu-x', type=int, required=True); ap.add_argument('--feu-y', type=int, required=True)
    ap.add_argument('--subs', nargs='*', default=None); ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.io import strip_position_map
    cfg = get_config(a.run_key); pm = strip_position_map(cfg)
    subs = a.subs or [cfg.SUB_RUN]
    os.makedirs(a.out_dir, exist_ok=True)
    for p, feu in (('x', a.feu_x), ('y', a.feu_y)):
        ps, rms = plane(cfg, subs, feu, np.asarray(pm[feu], float))
        out = os.path.join(a.out_dir, f'{a.run_key}_{p}.npz')
        ps.save(out, rms=rms, pos=np.asarray(pm[feu], float), run_key=a.run_key, detector=cfg.DET_NAME,
                plane=p, feu=feu, subs=np.array(subs))
        print(a.run_key, p, 'FEU', feu, 'events', ps.nev, '->', out)


if __name__ == '__main__':
    main()
