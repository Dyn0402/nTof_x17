"""Time-base transport monitor: plastic / liquid flash time vs PKUP, per run.

PSS and LIQ are never gated, so they record the true flash in EVERY run -- which
makes them the only way to test whether the flash time base is stable across the
campaign (the walls can only be measured in the seven divert-off runs).

Estimator: the EARLIEST hit above `THR` ADC inside dt in [-3000,-500] ns of the
PKUP pulse, per (bunch, channel).  Max-amplitude does not work in the late
(post-FIFO) runs, where the flash region is full of large pile-up hits.
"""
import os, sys, glob, numpy as np, uproot

OUT = sys.argv[1]
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 3_000_000
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 1
THR = 25000
TREES = ['PSSA', 'PSSB', 'PSSC', 'PSSD', 'LIQA', 'LIQB', 'LIQC', 'LIQD']
ALWAYS = {224356, 224357, 224358, 224359, 224360, 224464, 224466}

allruns = sorted(int(os.path.basename(p)[3:-5])
                 for p in glob.glob('/eos/experiment/ntof/processing/official/done/run*.root')
                 if os.path.basename(p)[3:-5].isdigit()
                 and 224345 <= int(os.path.basename(p)[3:-5]) <= 224600)
runs = sorted(set(allruns[::STEP]) | (ALWAYS & set(allruns)))
print(f"{len(runs)} runs", flush=True)


def pkup_map(f):
    a = f['PKUP'].arrays(['BunchNumber', 'tof', 'amp'], library='np')
    b, t, am = a['BunchNumber'], a['tof'], a['amp']
    o = np.lexsort((-am, b)); b, t = b[o], t[o]
    fi = np.ones(len(b), bool); fi[1:] = b[1:] != b[:-1]
    b, t = b[fi], t[fi]
    m = np.abs(t - np.median(t)) < 200
    return dict(zip(b[m].tolist(), t[m].tolist()))


done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        if ln.split(',')[0].isdigit():
            done.add(int(ln.split(',')[0]))
with open(OUT, 'a' if done else 'w') as fh:
    if not done:
        fh.write('run,' + ','.join(f'{t},{t}_sig,{t}_n' for t in TREES) + '\n')
    for r in runs:
        if r in done:
            continue
        try:
            f = uproot.open(f"/eos/experiment/ntof/processing/official/done/run{r}.root")
            keys = [k.split(';')[0] for k in f.keys(recursive=False)]
        except Exception:
            fh.write(f"{r}" + ",,,"*len(TREES) + "\n"); fh.flush(); continue
        if 'PKUP' not in keys or f['PKUP'].num_entries < 50:
            fh.write(f"{r}" + ",,,"*len(TREES) + "\n"); fh.flush(); continue
        pk = pkup_map(f)
        cells = []
        for t in TREES:
            if t not in keys or f[t].num_entries < 1000:
                cells += ['', '', '']; continue
            a = f[t].arrays(['BunchNumber', 'detn', 'tof', 'amp'], entry_stop=CAP, library='np')
            ref = np.array([pk.get(int(x), np.nan) for x in a['BunchNumber']])
            dt = a['tof'] - ref
            sel = np.isfinite(dt) & (dt > -3000) & (dt < -500) & (a['amp'] > THR)
            if sel.sum() < 30:
                cells += ['', '', '']; continue
            key = a['BunchNumber'][sel] * 100 + a['detn'][sel]
            v = dt[sel]
            o = np.lexsort((v, key)); v, key = v[o], key[o]     # earliest per bunch+channel
            fi = np.ones(len(v), bool); fi[1:] = key[1:] != key[:-1]
            v = v[fi]
            m = np.median(v); core = v[np.abs(v - m) < 60]
            if len(core) < 30:
                cells += ['', '', '']; continue
            cells += [f"{core.mean():.2f}",
                      f"{1.4826*np.median(np.abs(core-np.median(core))):.1f}", str(len(core))]
        fh.write(f"{r}," + ",".join(cells) + "\n"); fh.flush()
        print(r, cells[:6], flush=True)
print("DONE", flush=True)
