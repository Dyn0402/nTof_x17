"""Campaign-wide divert-state scan: for every official processed X17 run,
decide whether the SiPM-wall blanking gate was ON or OFF."""
import os, sys, glob, numpy as np, uproot

OUT = sys.argv[1]
runs = sorted(int(os.path.basename(p)[3:-5])
              for p in glob.glob('/eos/experiment/ntof/processing/official/done/run*.root')
              if os.path.basename(p)[3:-5].isdigit()
              and 224264 <= int(os.path.basename(p)[3:-5]) <= 224600)
print(f"{len(runs)} runs in range", flush=True)

def mode(a, step=5):
    h, e = np.histogram(a, bins=np.arange(0, 25000, step))
    return e[h.argmax()]

with open(OUT, 'w') as fh:
    fh.write("run,state,nbunch,dA,dB,dC,dD,ampA,ampB,ampC,ampD,agree\n")
    for r in runs:
        try:
            f = uproot.open(f"/eos/experiment/ntof/processing/official/done/run{r}.root")
            keys = [k.split(';')[0] for k in f.keys(recursive=False)]
        except Exception as e:
            fh.write(f"{r},OPENFAIL,,,,,,,,,,\n"); fh.flush()
            print(r, "OPENFAIL", flush=True); continue
        if 'WALA' not in keys or f['WALA'].num_entries < 1000:
            nb = f['index'].num_entries if 'index' in keys else -1
            fh.write(f"{r},NO_WAL,{nb},,,,,,,,,\n"); fh.flush()
            print(r, "NO_WAL", flush=True); continue
        nb = f['index'].num_entries if 'index' in keys else -1
        m, amp, tfl = {}, {}, {}
        for t in ('WALA', 'WALB', 'WALC', 'WALD'):
            if t not in keys or f[t].num_entries < 1000:
                m[t] = amp[t] = np.nan; continue
            a = f[t].arrays(['tflash', 'tof', 'amp', 'BunchNumber'], entry_stop=300000, library='np')
            m[t] = mode(a['tflash'])
            sel = np.abs(a['tof'] - a['tflash']) < 60
            amp[t] = np.median(a['amp'][sel]) if sel.sum() > 20 else np.nan
            # per-bunch tflash for the agreement metric
            b, tf = a['BunchNumber'], a['tflash']
            o = np.argsort(b, kind='stable'); b, tf = b[o], tf[o]
            fi = np.ones(len(b), bool); fi[1:] = b[1:] != b[:-1]
            tfl[t] = dict(zip(b[fi].tolist(), tf[fi].tolist()))
        ref = m['WALB'] if np.isfinite(m['WALB']) else m['WALD']
        d = {t: m[t] - ref for t in m}
        agree = np.nan
        if len(tfl) == 4:
            common = sorted(set(tfl['WALA']) & set(tfl['WALB']) & set(tfl['WALC']) & set(tfl['WALD']))
            if common:
                v = np.array([[tfl[t][k] for t in ('WALA', 'WALB', 'WALC', 'WALD')] for k in common])
                agree = float(np.mean((v.max(1) - v.min(1)) < 60))
        # OFF  = all four walls tag the same feature AND the flash is big
        state = 'OFF' if (agree > 0.5 and np.nanmedian(list(amp.values())) > 5000) else 'ON'
        fh.write(f"{r},{state},{nb},{d['WALA']:.0f},{d['WALB']:.0f},{d['WALC']:.0f},{d['WALD']:.0f},"
                 f"{amp['WALA']:.0f},{amp['WALB']:.0f},{amp['WALC']:.0f},{amp['WALD']:.0f},{agree:.3f}\n")
        fh.flush()
        print(f"{r} {state} agree={agree:.2f} d={d['WALA']:.0f}/{d['WALC']:.0f}/{d['WALD']:.0f} amp={amp['WALB']:.0f}", flush=True)
print("DONE", flush=True)
