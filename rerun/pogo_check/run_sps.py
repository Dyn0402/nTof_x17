#!/usr/bin/env python3
"""run_sps.py [--npz wf_run71_raw_det4only.npz] --out-dir DIR

Pogo-check pair statistics for det4 at H4 (run_71 RAW, FEU 03; QA only). At H4 one FEU carried both planes,
detector connectors x_3..x_6 on Dream 0..3 and y_3..y_6 on Dream 4..7, every plug inverted
(sps_beam_test_26/det4_sps_assessment/det4_sps_map.py). The pogo layout is defined in M1 channel numbering, so each
FEU channel is renumbered to the bench-equivalent M1 channel, (connector - 1) * 64 + strip-in-connector, and the
plane is a 512-channel array with connectors 3-6 populated (cards 1 and 2 both covered: 3-4 on card 1, 5-6 on
card 2). This assumes det4's own M1 cards were on the chamber at H4; if the beam readout bypassed them, a pad short
is impossible and the check reads CLEAN by construction.
Hits: per event and channel the waveform maximum; fires at > 5 sigma (noise 10 ADC, RAW_RUN71_PHYSICS); events with
a sample > 3500 ADC are dropped as saturated.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
import accumulate as A  # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, CONNECTOR, STRIP_IN_CONNECTOR  # noqa: E402

NOISE, SAT = 10.0, 3500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/media/dylan/data/x17/sps_run53_det4_check/staging/run_71/wf_run71_raw_det4only.npz')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    Z = np.load(a.npz, allow_pickle=True)
    ev, ch, amp = Z['ev'], Z['ch'].astype(int), Z['amp']
    bench = (CONNECTOR - 1) * 64 + STRIP_IN_CONNECTOR                 # FEU channel -> M1 channel of its plane
    planes = {}
    for p in 'xy':
        pos = np.full(512, np.nan); m = VIEW == p
        pos[bench[m]] = POSITION_MM[m]
        planes[p] = (A.PairStats(A.strip_index_from_positions(pos)), pos)
    # per event and channel: the maximum sample
    key = ev * 512 + ch
    o = np.argsort(key, kind='stable'); key, amp = key[o], amp[o]
    u, st = np.unique(key, return_index=True)
    mx = np.maximum.reduceat(amp, st); e_u, c_u = u // 512, u % 512
    b = np.flatnonzero(np.diff(e_u)) + 1
    for s0, s1 in zip(np.r_[0, b], np.r_[b, len(e_u)]):
        c, A_ = c_u[s0:s1], mx[s0:s1]
        if (A_ > SAT).any():
            continue
        g = A_ > A.SIG_MIN * NOISE
        for p in 'xy':
            k = g & (VIEW[c] == p)
            planes[p][0].add(bench[c[k]], A_[k])
    for p, (ps, pos) in planes.items():
        out = os.path.join(a.out_dir, f'sps_run71_det4_{p}.npz')
        ps.save(out, rms=np.full(512, NOISE), pos=pos, run_key='sps_run71', detector='mx17_4', plane=p, feu=3,
                subs=np.array(['cfg_gain4.5_peaktime50']))
        print('sps_run71', p, 'events', ps.nev, '->', out)


if __name__ == '__main__':
    main()
