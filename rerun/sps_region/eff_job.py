#!/usr/bin/env python3
"""eff_job.py --run R --subrun S --group G --key K --out O.parquet [--feu3-dir decoded_root_fa80c06] [--nsamp 64]

det4 efficiency denominator for the small-region analysis (sps_region/README.md).  rerun/sps_audit/tel_job.py keeps only
the telescope tracks that have a det4 record (seed >= 50 ADC in BOTH views), so it has no denominator.  Here EVERY
single-cluster four-view uRWELL track of one FEU file group is kept, and det4 is summarised per view without any fit:
the up-to-4 brightest local maxima of the per-strip peak amplitude (good strips only, after the chain's baseline and
common-mode, the final hot/oscillator/coupled rules of the frozen SPS archive), their positions and a +-2-strip
amplitude-weighted centroid.  Efficiency is decided offline (a maximum >= 50 ADC within +-2 mm of the telescope
prediction), so the alignment is the same one the resolution uses.  No fit, so it is fast; run on condor in the frozen
archive (WFT_REPO).
"""
import argparse, csv, glob, os, sys
import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'rerun', 'sps_audit'),
                os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc                                         # noqa: E402
from sps_build_cache import POSITION_MM, VIEW, PITCH_MM               # noqa: E402
import tel_job as tj                                                  # noqa: E402

NMAX = 4
CHUNK = 5000


def summarise(ch, sm, am, raw, bad_base, couple):
    am = sbc._baseline(ch, sm, am, raw)
    pk = np.zeros(512, np.float32); np.maximum.at(pk, ch, am)
    bad = bad_base.copy()
    for o, cs in couple.items():
        if pk[o] > sbc.COUPLE_ON_ADC:
            bad[cs] = True
    pk[bad] = 0.0
    out = {}
    for v in 'xy':
        idx = np.flatnonzero(VIEW == v)
        order = np.argsort(POSITION_MM[idx]); idx = idx[order]
        p = pk[idx]; pos = POSITION_MM[idx]
        loc = np.flatnonzero((p >= 50.0) & (p >= np.r_[0, p[:-1]]) & (p >= np.r_[p[1:], 0]))
        loc = loc[np.argsort(-p[loc])][:NMAX]
        amp = np.full(NMAX, np.nan, np.float32); cen = np.full(NMAX, np.nan, np.float32)
        for i, j in enumerate(loc):
            s = slice(max(j - 2, 0), j + 3); w = np.clip(p[s], 0, None)
            amp[i] = p[j]; cen[i] = float((w * pos[s]).sum() / max(w.sum(), 1e-9))
        out[v] = (amp, cen, int((p >= 50.0).sum()))
    return out


def main():
    import pandas as pd, uproot
    ap = argparse.ArgumentParser()
    for k in ('--run', '--subrun', '--group', '--key', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--feu3-dir', default='decoded_root_fa80c06'); ap.add_argument('--nsamp', type=int, default=64)
    a = ap.parse_args()
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    raw = bool(int(reg['raw']))
    sub_dir = os.path.join(sbc.RUNS, a.run, a.subrun)
    f3dir = a.feu3_dir if os.path.isabs(a.feu3_dir) else os.path.join(sub_dir, a.feu3_dir)
    f3 = sorted(glob.glob(os.path.join(f3dir, f'*_datrun_*_{a.group}_03.root')))[0]
    chf = sorted(glob.glob(os.path.join(sub_dir, 'combined_hits_root', f'*_{a.group}_feu-combined_hits.root')))[0]
    tel, _ = tj.telescope_zs(chf)
    single = np.all([tel[f'{p}_n'] == 1 for p in ('fx', 'fy', 'bx', 'by')], axis=0)
    tel = {k: v[single] for k, v in tel.items()}
    ti = {int(e): i for i, e in enumerate(tel['eid'])}
    hot, nprobe, couple = tj.hot_probe(f3, raw, a.nsamp)
    bad = sbc.SEED_BAD | hot
    print(f'{a.run} {a.subrun} {a.group}: {len(ti)} single-cluster telescope tracks; hot {list(np.flatnonzero(hot))}', flush=True)
    files = sorted(glob.glob(os.path.join(f3dir, '*_datrun_*_03.root')))
    clock = tj.wall_clock(a.run, a.subrun, files)[f3]
    t_lo, t_hi = float(reg['t_lo_s']) + 20, float(reg['t_hi_s']) - 20
    rows = {}
    t = uproot.open(f3)['nt']
    for lo in range(0, t.num_entries, CHUNK):
        arr = t.arrays(['eventId', 'timestamp', 'sample', 'channel', 'amplitude'], entry_start=lo,
                       entry_stop=min(lo + CHUNK, t.num_entries), library='np')
        for j, eid in enumerate(arr['eventId']):
            eid = int(eid)
            if eid not in ti:
                continue
            tw = clock[0] + (float(arr['timestamp'][j]) - clock[1]) * clock[2]
            if not (t_lo <= tw < t_hi):
                continue
            ch = arr['channel'][j].astype(int)
            o = dict(eid=eid, t_wall=tw, d4_present=True)
            if len(ch):
                s = summarise(ch, arr['sample'][j].astype(int), arr['amplitude'][j].astype(np.float32), raw, bad, couple)
                for v in 'xy':
                    amp, cen, nab = s[v]
                    o[f'{v}_nabove'] = nab
                    for i in range(NMAX):
                        o[f'{v}_a{i}'] = float(amp[i]); o[f'{v}_c{i}'] = float(cen[i])
            rows[eid] = o
    t.file.close()
    out = []
    for eid, i in ti.items():
        o = rows.get(eid)
        if o is None:
            continue          # outside the plateau (or no feu-3 event: counted below)
        o.update({k: v[i] for k, v in tel.items() if k != 'eid'})
        o.update(group=a.group, key=a.key)
        out.append(o)
    df = pd.DataFrame(out)
    df.to_parquet(a.out, index=False)
    print(f'wrote {a.out}: {len(df)} tracks in the plateau (of {len(ti)})', flush=True)


if __name__ == '__main__':
    main()
