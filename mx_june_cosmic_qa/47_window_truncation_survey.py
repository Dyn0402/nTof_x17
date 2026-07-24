#!/usr/bin/env python3
"""Careful window-truncation / mis-timing study across the June cosmic fleet.

Key question: are pulses (in NORMAL muon events, sparks separated) truncated by
the 32-sample x 60 ns DREAM window in a way that corrupts timing or charge?

Per waveform (candidate = peak > 5 sigma post ped-sub+CNS):
  lead_trunc : above threshold already at s0  (rise unseen -> timing corrupted)
  tail_trunc : still above threshold at s31   (tail clipped -> integral clipped)
  peak_edge_hi: argmax >= 29                  (peak itself at window end -> amp+time bad)
  peak_edge_lo: argmax <= 1
Event class: spark if >= 50 candidate strips in the FEU, else normal.
"""
import numpy as np
import uproot

NSAMP, NCH = 32, 512
NEV = 1200
NPED = 250

FILES = [
    ('det3wk X f7 1000V', '/home/dylan/x17/cosmic_bench/det3/mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/decoded_root/MX17_long_run_p2_det1_sanity_check_datrun_260628_01H34_000_07.root'),
    ('det3wk Y f8 1000V', '/home/dylan/x17/cosmic_bench/det3/mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/decoded_root/MX17_long_run_p2_det1_sanity_check_datrun_260628_01H34_000_08.root'),
    ('det2/3 622 f6 1000V', '/home/dylan/x17/cosmic_bench/det2_det3/mx17_det2_det3_overnight_6-22-26/longer_run/decoded_root/MX17_longer_run_datrun_260622_20H20_000_06.root'),
    ('det2/3 622 f8 1000V', '/home/dylan/x17/cosmic_bench/det2_det3/mx17_det2_det3_overnight_6-22-26/longer_run/decoded_root/MX17_longer_run_datrun_260622_20H20_000_08.root'),
    ('det6 X f3 700V', '/home/dylan/x17/cosmic_bench/det6_det7/mx17_det6_det7_overnight_6-26-26/long_run/decoded_root/MX17_long_run_datrun_260626_06H03_000_03.root'),
    ('det6 Y f4 700V', '/home/dylan/x17/cosmic_bench/det6_det7/mx17_det6_det7_overnight_6-26-26/long_run/decoded_root/MX17_long_run_datrun_260626_06H03_000_04.root'),
    ('det7 X f6 700V', '/home/dylan/x17/cosmic_bench/det6_det7/mx17_det6_det7_overnight_6-26-26/long_run/decoded_root/MX17_long_run_datrun_260626_06H03_000_06.root'),
    ('det7 Y f8 700V', '/home/dylan/x17/cosmic_bench/det6_det7/mx17_det6_det7_overnight_6-26-26/long_run/decoded_root/MX17_long_run_datrun_260626_06H03_000_08.root'),
    ('det4 X f6 900V', '/home/dylan/x17/cosmic_bench/det4_day/mx17_det4_day_6-24-26/long_run/decoded_root/MX17_long_run_datrun_260624_12H22_000_06.root'),
    ('det4 Y f8 900V', '/home/dylan/x17/cosmic_bench/det4_day/mx17_det4_day_6-24-26/long_run/decoded_root/MX17_long_run_datrun_260624_12H22_000_08.root'),
]


def cns(w):
    for b in range(0, NCH, 64):
        w[..., b:b + 64] -= np.median(w[..., b:b + 64], axis=-1, keepdims=True)
    return w


def dense_events(path, nmax):
    t = uproot.open(path)['nt']
    arr = t.arrays(['channel', 'sample', 'amplitude'], entry_stop=min(nmax, t.num_entries), library='np')
    for ch, s, a in zip(arr['channel'], arr['sample'], arr['amplitude']):
        if len(a) != NSAMP * NCH:
            continue
        w = np.zeros((NSAMP, NCH), np.float32)
        w[s, ch] = a
        yield w


print(f'{"file":22s} {"nrm ev":>6s} {"spk%":>5s} | normal-event waveforms:  '
      f'{"n":>6s} {"lead%":>6s} {"tail%":>6s} {"pkHi%":>6s} {"pkLo%":>6s} '
      f'{"pk med":>6s} {"pk p95":>6s} {"rise med":>8s} | {"spark tail%":>11s}')

for label, path in FILES:
    # self-contained pedestals: median baseline, MAD sigma post-CNS
    ped = []
    for w in dense_events(path, NPED):
        ped.append(w)
    ped = np.array(ped)
    base = np.median(ped, axis=(0, 1))
    pedc = cns(ped - base[None, None, :])
    sig = 1.4826 * np.median(np.abs(pedc - np.median(pedc, axis=(0, 1))), axis=(0, 1))
    sig[sig <= 0] = np.inf
    thr = 5.0 * sig

    stats = dict(nev=0, nspark=0, n=0, lead=0, tail=0, pkhi=0, pklo=0, sp_n=0, sp_tail=0)
    pks, rises = [], []
    for w in dense_events(path, NEV):
        w = cns(w - base[None, :])
        pk = w.max(axis=0)
        cand = np.where(pk > thr)[0]
        if len(cand) == 0:
            continue
        if len(cand) >= 50:
            stats['nspark'] += 1
            for c in cand:
                stats['sp_n'] += 1
                stats['sp_tail'] += w[-1, c] > thr[c]
            continue
        stats['nev'] += 1
        for c in cand:
            wf = w[:, c]
            m = int(np.argmax(wf))
            stats['n'] += 1
            stats['lead'] += wf[0] > thr[c]
            stats['tail'] += wf[-1] > thr[c]
            stats['pkhi'] += m >= 29
            stats['pklo'] += m <= 1
            pks.append(m)
            l = m
            while l > 0 and wf[l] > thr[c]:
                l -= 1
            rises.append(l)
    n = max(stats['n'], 1)
    pks = np.array(pks); rises = np.array(rises)
    ntot = stats['nev'] + stats['nspark']
    print(f'{label:22s} {stats["nev"]:6d} {stats["nspark"]/max(ntot,1)*100:5.1f} |'
          f' {stats["n"]:24d} {stats["lead"]/n*100:6.2f} {stats["tail"]/n*100:6.2f}'
          f' {stats["pkhi"]/n*100:6.2f} {stats["pklo"]/n*100:6.2f}'
          f' {np.median(pks) if len(pks) else -1:6.0f} {np.percentile(pks,95) if len(pks) else -1:6.0f}'
          f' {np.median(rises) if len(rises) else -1:8.0f} |'
          f' {stats["sp_tail"]/max(stats["sp_n"],1)*100:11.1f}')
