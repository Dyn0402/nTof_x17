#!/usr/bin/env python3
"""Follow-ups: (1) what are lead-truncated waveforms in normal events?
(2) estimated charge fraction lost to tail clipping, from average pulse shape."""
import numpy as np, uproot
NSAMP, NCH = 32, 512

def cns(w):
    for b in range(0, NCH, 64):
        w[..., b:b+64] -= np.median(w[..., b:b+64], axis=-1, keepdims=True)
    return w

def dense_events(path, nmax, start=0):
    t = uproot.open(path)['nt']
    arr = t.arrays(['channel','sample','amplitude'], entry_start=start, entry_stop=min(start+nmax, t.num_entries), library='np')
    for ch, s, a in zip(arr['channel'], arr['sample'], arr['amplitude']):
        if len(a) != NSAMP*NCH: continue
        w = np.zeros((NSAMP, NCH), np.float32); w[s, ch] = a
        yield w

for label, path in [
  ('det3wk X f7', '/home/dylan/x17/cosmic_bench/det3/mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/decoded_root/MX17_long_run_p2_det1_sanity_check_datrun_260628_01H34_000_07.root'),
  ('det3wk Y f8', '/home/dylan/x17/cosmic_bench/det3/mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/decoded_root/MX17_long_run_p2_det1_sanity_check_datrun_260628_01H34_000_08.root')]:
    ped = np.array(list(dense_events(path, 250)))
    base = np.median(ped, axis=(0,1))
    pedc = cns(ped - base[None,None,:])
    sig = 1.4826*np.median(np.abs(pedc - np.median(pedc, axis=(0,1))), axis=(0,1))
    sig[sig <= 0] = np.inf; thr = 5.0*sig

    # average normalized shape from CLEAN pulses peaking mid-window (peak 8..14)
    shapes = []; lead_examples = []; peak_hist = np.zeros(NSAMP)
    n_show = 0
    for w in dense_events(path, 1200):
        w = cns(w - base[None,:])
        pk = w.max(axis=0); cand = np.where(pk > thr)[0]
        if len(cand) == 0 or len(cand) >= 50: continue
        for c in cand:
            wf = w[:,c]; m = int(np.argmax(wf))
            peak_hist[m] += 1
            if wf[0] > thr[c] and n_show < 6 and pk[c] > 10*sig[c]:
                lead_examples.append((pk[c]/sig[c], m, wf.copy())); n_show += 1
            if 8 <= m <= 14 and wf[0] < thr[c] and wf[-1] < thr[c] and pk[c] > 20*sig[c]:
                shapes.append(np.roll(wf, 11-m)/pk[c])
    shapes = np.array(shapes)
    avg = shapes.mean(axis=0)
    # cumulative charge fraction vs samples-after-peak (peak at idx 11)
    pos = np.clip(avg, 0, None)
    cum = np.cumsum(pos)/pos.sum()
    # expected lost tail fraction for a pulse peaking at sample m: charge beyond (31-m) after peak
    tail_after = 31 - np.arange(NSAMP)          # samples available after peak
    lost = np.zeros(NSAMP)
    for m in range(NSAMP):
        idx = min(11 + tail_after[m], NSAMP-1)
        lost[m] = 1.0 - cum[idx]
    exp_loss = (peak_hist*lost).sum()/peak_hist.sum()
    print(f'\n== {label}: clean shapes n={len(shapes)}')
    print('avg shape (peak=1, peak at idx 11):')
    print(np.array2string(avg, precision=2, suppress_small=True, max_line_width=200))
    print(f'charge fraction after peak: +5samp {1-cum[16]:.2f}, +10samp {1-cum[21]:.2f}, +15samp {1-cum[26]:.2f}, +20samp {1-cum[31]:.2f}')
    print(f'expected charge lost to window end, averaged over observed peak-position dist: {exp_loss*100:.1f}%')
    print('lead-truncated examples (amp_sigma, argmax):')
    np.set_printoptions(precision=0, suppress=True, linewidth=220)
    for z, m, wf in lead_examples[:4]:
        print(f'  {z:5.0f} sigma, peak@s{m}: {wf}')
