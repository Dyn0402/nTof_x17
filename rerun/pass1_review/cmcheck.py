# Common-mode noise on RAW run_71 (and ZS run_70 for contrast), after sps_build_cache._baseline.
import sys, glob, os, numpy as np, uproot
sys.path.insert(0, os.path.expanduser('~/x17rerun/code_b5a29af/rerun/model_audit'))
import sps_build_cache as sbc
from det4_sps_map import VIEW
RUNS = sbc.RUNS
def check(run, sub, raw, nev=300):
    f = sorted(glob.glob(f'{RUNS}/{run}/{sub}/decoded_root_fa80c06/*_datrun_*_03.root'))[len(sorted(glob.glob(f'{RUNS}/{run}/{sub}/decoded_root_fa80c06/*_datrun_*_03.root')))//2]
    a = uproot.open(f)['nt'].arrays(['sample', 'channel', 'amplitude'], entry_stop=nev, library='np')
    raw_rms, cm_rms, res_rms, nsig = [], [], [], []
    for j in range(nev):
        ch = a['channel'][j].astype(int); sm = a['sample'][j].astype(int); am = a['amplitude'][j].astype(np.float32)
        b = sbc._baseline(ch, sm, am, raw)
        present = am != 0
        ch, sm, b = ch[present], sm[present], b[present]
        # signal strips: peak > 100 ADC; quiet strips: peak < 60 on every sample and not within 3 of a signal strip
        pk = {c: b[ch == c].max() for c in np.unique(ch)}
        sig = {c for c, p in pk.items() if p > 100}
        quiet = np.array([c for c in pk if all(abs(c - s) > 3 for s in sig)])
        nsig.append(len(sig))
        if len(quiet) < 20: continue
        q = np.isin(ch, quiet)
        chip = ch // 64
        for (cp, s) in set(zip(chip[q], sm[q])):
            k = q & (chip == cp) & (sm == s)
            if k.sum() < 16: continue
            v = b[k]; cm = np.median(v)
            raw_rms.append(np.std(v)); cm_rms.append(cm); res_rms.append(np.std(v - cm))
    cm_rms = np.array(cm_rms)
    print(f'{run} {sub} raw={raw}: n strips >100 ADC p50 {np.median(nsig):.0f} | quiet-strip noise per chip-sample: '
          f'sigma {np.median(raw_rms):.1f} ADC; common mode |cm| p50 {np.median(np.abs(cm_rms)):.1f}, rms {np.std(cm_rms):.1f}; '
          f'after cm subtraction sigma {np.median(res_rms):.1f} ADC')
check('run_71', 'cfg_gain4.5_peaktime50', True)
for sub in sorted(os.listdir(f'{RUNS}/run_70'))[:1]:
    check('run_70', sub, False)
