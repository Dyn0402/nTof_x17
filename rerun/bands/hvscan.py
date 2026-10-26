#!/usr/bin/env python3
"""hvscan.py -- item (f): noise-band rate against the amplification (resist) and drift HV, from the June scans.

    python hvscan.py scan      # every subrun of the scans below -> $BANDS_OUT/hvscan_<scan>.npz (compact, per trigger)
    python hvscan.py summary   # -> $BANDS_OUT/hvscan.json

A field-dependent band rate points at the chamber (gain / discharges), a flat one at pickup.  The band definition is
exactly ../satstudy/bands.py's (per channel: |window mean after CM| > 5 sigma and RMS about it < 2 sigma; a band = >= 8
such channels on one 64-channel chip), with the pedestal and noise of wft.io.FeuReader (first 300 events of each file).

Scans (all local, ~/x17/cosmic_bench):
  det3_resist  det3/mx17_det3_saturday_scan_6-27-26  hv_scan_resist_* + hv_scan2_resist_* (425-525 V, drift 1000 V), FEU 7/8
  det3_drift   same run, drift_scan_resist_490V_drift_* (100-1100 V at 490 V)
  det6, det7   det6_det7/mx17_det6_det7_hv_scan_6-26-26 (both chambers stepped together, 400-500 V, drift 700 V),
               det6 = FEU 3/4, det7 = FEU 6/8
"""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO]
OUTD = os.environ.get('BANDS_OUT', os.path.expanduser('~/x17/cosmic_bench/satstudy/bands')); os.makedirs(OUTD, exist_ok=True)
BENCH = os.path.expanduser('~/x17/cosmic_bench')
OFF_SIG, FLAT_RMS, BLK = 5.0, 2.0, 64
SCANS = {
    'det3_resist': (f'{BENCH}/det3/mx17_det3_saturday_scan_6-27-26', 'hv_scan*_resist_*', 7, 8),
    'det3_drift': (f'{BENCH}/det3/mx17_det3_saturday_scan_6-27-26', 'drift_scan_*', 7, 8),
    'det3_long': (f'{BENCH}/det3/mx17_det3_saturday_scan_6-27-26', 'long_run_*', 7, 8),
    'det6': (f'{BENCH}/det6_det7/mx17_det6_det7_hv_scan_6-26-26', 'resist_*', 3, 4),
    'det7': (f'{BENCH}/det6_det7/mx17_det6_det7_hv_scan_6-26-26', 'resist_*', 6, 8),
}
NMAX = 6000          # triggers per subrun (the long run: first 6000 only, as one more point at 490 V)


def scan_feu(files, nmax=NMAX):
    import uproot
    from wft import io as wio
    E, TS, BAND, NSIG, NOFF = [], [], [], [], []
    for f in files:
        r = wio.FeuReader(f); t = uproot.open(f)['nt']; n = r.noise
        stop = min(r.n_entries, nmax - len(E))
        for lo in range(0, stop, 2000):
            a = t.arrays(['eventId', 'timestamp', 'amplitude'], entry_start=lo, entry_stop=min(lo + 2000, stop), library='np')
            for e, ts, amp in zip(a['eventId'], a['timestamp'], a['amplitude']):
                if len(amp) != r.n_sample * 512:
                    continue
                w = amp.reshape(-1, 512).astype(np.float32) - r.ped
                cm = np.median(w.reshape(w.shape[0], 512 // BLK, BLK), axis=2)
                wc = w - np.repeat(cm, BLK, axis=1)
                mc, sd = wc.mean(0), wc.std(0)
                off = (np.abs(mc) > OFF_SIG * n) & (sd < FLAT_RMS * n)
                E.append(int(e)); TS.append(int(ts))
                BAND.append(off.reshape(8, 64).sum(1) >= 8); NOFF.append(off.sum())
                NSIG.append(int((wc.max(0) > 6 * n).sum()))
        if len(E) >= nmax:
            break
    return dict(eid=np.array(E), ts=np.array(TS), band=np.array(BAND), nsig=np.array(NSIG), noff=np.array(NOFF))


def scan(which=None):
    for name, (run, pat, fx, fy) in SCANS.items():
        if which and name != which:
            continue
        res = {}
        for sub in sorted(glob.glob(os.path.join(run, pat))):
            tag = os.path.basename(sub); d = {}
            for v, feu in (('x', fx), ('y', fy)):
                files = sorted(glob.glob(os.path.join(sub, 'decoded_root', f'*_{feu:02d}.root')))
                if not files:
                    break
                d[v] = scan_feu(files)
            if len(d) < 2:
                continue
            for v in 'xy':
                for k, a in d[v].items():
                    res[f'{tag}|{v}|{k}'] = a
            print(name, tag, len(d['x']['eid']), flush=True)
        np.savez_compressed(os.path.join(OUTD, f'hvscan_{name}.npz'), **res)


def volts(tag, name):
    if name.startswith('det3'):
        m = re.search(r'resist_(\d+)V_drift_(\d+)V', tag); return int(m.group(1)), int(m.group(2))
    m = re.search(rf'{name}_(\d+)V', tag); d = re.search(r'drift_(\d+)V', tag)
    return int(m.group(1)), int(d.group(1))


def summary():
    out = {}
    for name in SCANS:
        p = os.path.join(OUTD, f'hvscan_{name}.npz')
        if not os.path.exists(p):
            continue
        Z = np.load(p); tags = sorted({k.split('|')[0] for k in Z.files}); rows = []
        for tag in tags:
            ex, ey = Z[f'{tag}|x|eid'], Z[f'{tag}|y|eid']
            com, ix, iy = np.intersect1d(ex, ey, return_indices=True)
            bx = Z[f'{tag}|x|band'][ix].any(1); by = Z[f'{tag}|y|band'][iy].any(1)
            ns = Z[f'{tag}|x|nsig'][ix] + Z[f'{tag}|y|nsig'][iy]
            sp = ns > 150; b = bx | by; nb = int(b.sum()); n = len(com)
            rv, dv = volts(tag, name)
            rows.append(dict(sub=tag, resist=rv, drift=dv, n=n, p_band=nb / n, err=float(np.sqrt(max(nb, 1)) / n),
                             p_band_x=float(bx.mean()), p_band_y=float(by.mean()), p_spark=float(sp.mean()),
                             p_band_nospark=float(b[~sp].mean()) if (~sp).any() else None,
                             chips_x=Z[f'{tag}|x|band'][ix].mean(0).round(4).tolist(),
                             chips_y=Z[f'{tag}|y|band'][iy].mean(0).round(4).tolist()))
        out[name] = rows
        for r in rows:
            print(name, r['resist'], r['drift'], r['n'], 'band %.3f +- %.3f' % (r['p_band'], r['err']), 'spark %.3f' % r['p_spark'],
                  'band|nospark %.3f' % (r['p_band_nospark'] or 0))
    json.dump(out, open(os.path.join(OUTD, 'hvscan.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'scan': scan, 'summary': summary}[sys.argv[1]](*sys.argv[2:])
