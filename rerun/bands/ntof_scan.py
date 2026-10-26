#!/usr/bin/env python3
"""ntof_scan.py -- item (e): the noise bands in the n_TOF data (runs on lxplus, LCG_105, Python 3.9).

    python ntof_scan.py <run_dir> <out.npz> [N_SUBRUNS] [FILES_PER_SUBRUN] [NMAX_PER_FILE]

For each sub-run of the run (the first N_SUBRUNS, evenly spread), the first FILES_PER_SUBRUN file indices, all eight
FEUs: per trigger and FEU the same quantities as ../satstudy/bands.py on the bench -- pedestal = median of the first
300 events, noise = 1.4826 MAD after a 64-channel block-median common mode, a channel "offset" when |window mean after
CM| > 5 sigma and its RMS about that mean < 2 sigma, a band = >= 8 offset channels on one 64-channel chip -- plus the
count of channels above 6 sigma (spark measure) and, per FEU, per-channel offset counts and the signed offset sum over
band events (the spatial profile).

n_TOF chamber -> FEU (run_config, July/August): A (det3) x3/y4, B (det2) x5/y6, C (det6) x7/y8, D (det7) x1/y2;
connector k of a chamber on FEU connector k, as on the bench, so chip c = connector c+1 on both.
"""
import glob, os, sys
import numpy as np
import uproot

OFF_SIG, FLAT_RMS, BLK, NPED = 5.0, 2.0, 64, 300


def ped_noise(t, ns):
    a = t.arrays(['amplitude'], entry_stop=NPED, library='np')['amplitude']
    st = np.stack([x.reshape(ns, 512) for x in a if len(x) == ns * 512]).astype(np.float32)
    ped = np.median(st, axis=(0, 1)); sub = st - ped
    cm = np.median(sub.reshape(len(st), ns, 8, BLK), axis=3); sub -= np.repeat(cm, BLK, axis=2)
    return ped, (1.4826 * np.median(np.abs(sub), axis=(0, 1))).astype(np.float32)


def scan_file(path, nmax):
    t = uproot.open(path)['nt']
    a0 = t.arrays(['amplitude'], entry_stop=50, library='np')['amplitude']
    ns = int(np.bincount([len(x) // 512 for x in a0]).argmax())
    ped, noise = ped_noise(t, ns)
    E, TS, BAND, NSIG = [], [], [], []
    cnt = np.zeros(512); ssum = np.zeros(512); nb = 0
    stop = min(t.num_entries, nmax)
    for lo in range(0, stop, 2000):
        A = t.arrays(['eventId', 'timestamp', 'amplitude'], entry_start=lo, entry_stop=min(lo + 2000, stop), library='np')
        for e, ts, amp in zip(A['eventId'], A['timestamp'], A['amplitude']):
            if len(amp) != ns * 512:
                continue
            w = amp.reshape(ns, 512).astype(np.float32) - ped
            cm = np.median(w.reshape(ns, 8, BLK), axis=2)
            wc = w - np.repeat(cm, BLK, axis=1)
            mc, sd = wc.mean(0), wc.std(0)
            off = (np.abs(mc) > OFF_SIG * noise) & (sd < FLAT_RMS * noise)
            b = off.reshape(8, 64).sum(1) >= 8
            E.append(int(e)); TS.append(int(ts)); BAND.append(b); NSIG.append(int((wc.max(0) > 6 * noise).sum()))
            if b.any():
                nb += 1; cnt += off; ssum += np.where(off, mc, 0)
    return dict(eid=np.array(E), ts=np.array(TS), band=np.array(BAND), nsig=np.array(NSIG), cnt=cnt, ssum=ssum,
                nband=nb, noise=noise, ns=ns)


def main(run_dir, out, nsub=6, nfile=1, nmax=4000):
    nsub, nfile, nmax = int(nsub), int(nfile), int(nmax)
    subs = sorted(d for d in glob.glob(os.path.join(run_dir, '*')) if os.path.isdir(os.path.join(d, 'decoded_root')))
    if len(subs) > nsub:
        subs = [subs[i] for i in np.linspace(0, len(subs) - 1, nsub).astype(int)]
    res = {}
    for s in subs:
        files = sorted(glob.glob(os.path.join(s, 'decoded_root', '*_01.root')))[:nfile]
        for f1 in files:
            tag = os.path.basename(s) + '|' + os.path.basename(f1)[:-8]
            for feu in range(1, 9):
                f = f1[:-7] + '%02d.root' % feu
                if not os.path.exists(f):
                    continue
                try:
                    d = scan_file(f, nmax)
                except Exception as ex:          # noqa: BLE001  (report and go on)
                    print('FAIL', f, ex, flush=True); continue
                for k, v in d.items():
                    res['%s|%d|%s' % (tag, feu, k)] = np.asarray(v)
                print(tag, feu, len(d['eid']), 'band %.3f' % (np.asarray(d['band']).any(1).mean() if len(d['eid']) else -1), flush=True)
    np.savez_compressed(out, **res)


if __name__ == '__main__':
    main(*sys.argv[1:])
