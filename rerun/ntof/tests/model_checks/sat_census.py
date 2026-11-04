#!/usr/bin/env python3
"""sat_census.py -- model check 2a: the ADC distribution near the rail on n_TOF data, per chamber and view.

    sat_census.py --run 145 --subrun stat090_0000 [--tags 2] [--base /media/ucla/work/ntof_pass1/runs] --out J.json

The fit sees W = raw - ped - CM (wft.io: pedestal = per-channel median of the file's first 300 events, CM = per-sample
median of each 64-channel block).  The 12-bit DREAM ADC clips at raw 4095, so a sample's true rail in W units is
4095 - ped - CM, per channel AND per sample (rerun/calib/saturation/DECISION_2026-10-29.md K4).  The pass censors
every sample with W >= sat_adc (bundle field, 3550).  For each threshold S in THRESH this counts, over every sample of
every event of the acquisitions read:

    (samples in a 64-channel block whose CM is above CM_COH_HI = +500 ADC or below CM_COH_LO = -1000 ADC -- whole-chip
    saturation / floor, K7 -- are counted apart as clipped_coherent and left out of everything else.  A NEGATIVE CM of
    a few hundred ADC is common in blocks carrying a large signal at n_TOF and is kept: it moves the rail up, to
    4095 - ped - CM ~ 4050-4100 in W units.  Blocks with more than MAX_BLOCK_CLIP = 16 clipped channels at a sample are
    coherent too, and channels clipping in > 2 % of the events (stuck/hot) are excluded per file)
    clipped        raw >= 4095 (truly saturated samples)
    missed(S)      clipped samples with W < S  -> fitted as exact data below their truth
    censored_ok(S) unclipped samples (raw < 4095) with W >= S -> real data thrown away as a lower bound

and histograms raw (3000..4095) and W (3000..4100), plus the per-channel rail 4095 - ped.
Only the arm FEUs of rerun/ntof/ntof_reco.py FEUS are read.  No reconstruction, no seeding.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)

FEUS = dict(A=(3, 4), B=(5, 6), C=(7, 8), D=(1, 2))
THRESH = (3400, 3500, 3550, 3600, 3650, 3700, 3750, 3800)
RAIL = 4095
CM_COH_HI, CM_COH_LO = 500.0, -1000.0     # block CM outside [LO, HI] = a coherent chip event, excluded below
MAX_BLOCK_CLIP = 16    # more clipped channels than this in one 64-channel block at one sample = coherent too
EV_LO = 3500.0         # events with any non-coherent sample >= this are listed (events_hi)
STUCK_FRAC = 0.02      # a channel clipping (outside coherent blocks) in > 2 % of events is stuck/hot, excluded
HB = np.arange(3000, 4101, 10)


def census_file(path):
    import uproot
    from wft import io as wio
    r = wio.FeuReader(path)
    ns = r.n_sample
    nblk = 512 // wio.CNS_BLOCK
    out = dict(n_events=0, n_samples=0, clipped=0, clipped_events=0,
               missed={s: 0 for s in THRESH}, censored_ok={s: 0 for s in THRESH},
               h_raw=np.zeros(len(HB) - 1, np.int64), h_w_unclipped=np.zeros(len(HB) - 1, np.int64),
               h_w_clipped=np.zeros(len(HB) - 1, np.int64), rail_ped=(RAIL - r.ped).tolist(),
               rail_sample_clipped=[])
    t = r.tree

    def chunks():
        for lo in range(0, r.n_entries, 2000):
            arr = t.arrays(['amplitude', 'eventId'], entry_start=lo, entry_stop=min(lo + 2000, r.n_entries),
                           library='np')
            ok = [k for k, x in enumerate(arr['amplitude']) if len(x) == ns * 512]
            a = [arr['amplitude'][k] for k in ok]
            eids = arr['eventId'][ok]
            if not a:
                continue
            raw = np.stack(a).reshape(len(a), ns, 512).astype(np.float32)
            sub = raw - r.ped[None, None, :]
            cm = np.median(sub.reshape(len(a), ns, nblk, wio.CNS_BLOCK), axis=3)
            cmf = np.repeat(cm, wio.CNS_BLOCK, axis=2)
            clip0 = raw >= RAIL
            nblk_clip = np.repeat(clip0.reshape(len(a), ns, nblk, wio.CNS_BLOCK).sum(3), wio.CNS_BLOCK, axis=2)
            # coherent chip events (K7): the CM-subtracted W is not data
            coh = (cmf > CM_COH_HI) | (cmf < CM_COH_LO) | (nblk_clip > MAX_BLOCK_CLIP)
            yield raw, sub - cmf, cmf, clip0, coh, eids

    # pass 1: channels that clip outside coherent events in > STUCK_FRAC of the events are stuck/hot, not signal
    nclip_ch, nev = np.zeros(512), 0
    for raw, W, cmf, clip0, coh, eids in chunks():
        nclip_ch += (clip0 & ~coh).any(axis=1).sum(0)
        nev += len(raw)
    stuck = nclip_ch > STUCK_FRAC * max(nev, 1)
    out['stuck_channels'] = np.flatnonzero(stuck).tolist()
    for raw, W, cmf, clip0, coh, eids in chunks():
        coh = coh | stuck[None, None, :]
        out['clipped_coherent'] = out.get('clipped_coherent', 0) + int((clip0 & coh).sum())
        out['coherent_events'] = out.get('coherent_events', 0) + int((clip0 & coh).any(axis=(1, 2)).sum())
        clip = clip0 & ~coh
        out['n_events'] += len(raw)
        out['n_samples'] += raw.size
        out['clipped'] += int(clip.sum())
        out['clipped_events'] += int(clip.any(axis=(1, 2)).sum())
        hi = (W >= 3000) & ~coh
        out['h_raw'] += np.histogram(raw[(raw >= 3000) & ~coh], HB)[0]
        out['h_w_unclipped'] += np.histogram(W[hi & ~clip], HB)[0]
        out['h_w_clipped'] += np.histogram(W[clip], HB)[0]
        Wc = W[clip]
        out.setdefault('cm_clipped', []).extend(np.round(cmf[clip][:: max(1, int(clip.sum()) // 2000)], 1).tolist())
        if Wc.size:
            out['rail_sample_clipped'].extend(np.round(Wc[:: max(1, Wc.size // 2000)], 1).tolist())
        Wu = W[hi & ~clip]
        # events with a non-coherent sample at/above EV_LO: the only events a threshold in [EV_LO, rail) can change
        out.setdefault('events_hi', []).extend(eids[((W >= EV_LO) & ~coh).any(axis=(1, 2))].astype(int).tolist())
        for s_ in THRESH:
            out['missed'][s_] += int((Wc < s_).sum())
            out['censored_ok'][s_] += int((Wu >= s_).sum())
    return out


def merge(rs):
    m = dict(n_events=0, n_samples=0, clipped=0, clipped_events=0, clipped_coherent=0, coherent_events=0, missed={s: 0 for s in THRESH},
             censored_ok={s: 0 for s in THRESH}, h_raw=0, h_w_unclipped=0, h_w_clipped=0, rail_ped=[],
             rail_sample_clipped=[])
    for r in rs:
        for k in ('n_events', 'n_samples', 'clipped', 'clipped_events', 'clipped_coherent', 'coherent_events', 'h_raw', 'h_w_unclipped', 'h_w_clipped'):
            m[k] = m[k] + r[k]
        for k in ('missed', 'censored_ok'):
            for s in THRESH:
                m[k][s] += r[k][s]
        m['rail_ped'] += r['rail_ped']
        m.setdefault('stuck_channels', []).append(r['stuck_channels'])
        m.setdefault('events_hi', []).extend(r.get('events_hi', []))
        m['rail_sample_clipped'] += r['rail_sample_clipped']
    rp = np.array(m['rail_ped'])
    rs_ = np.array(m['rail_sample_clipped']) if m['rail_sample_clipped'] else np.array([np.nan])
    m['rail_ped_pct'] = {p: float(np.percentile(rp, p)) for p in (1, 5, 16, 50, 84, 95, 99)}
    m['rail_sample_pct'] = {p: float(np.nanpercentile(rs_, p)) for p in (1, 5, 16, 50, 84, 95, 99)}
    cmc = np.array([c for r in rs for c in r.get('cm_clipped', [])] or [np.nan])
    m['cm_clipped_pct'] = {p: float(np.nanpercentile(cmc, p)) for p in (1, 5, 16, 50, 84, 95, 99)}
    m['frac_missed'] = {s: m['missed'][s] / max(m['clipped'], 1) for s in THRESH}
    for k in ('h_raw', 'h_w_unclipped', 'h_w_clipped'):
        m[k] = np.asarray(m[k]).tolist()
    del m['rail_sample_clipped']
    del m['rail_ped']
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--subrun', required=True)
    ap.add_argument('--tags', type=int, default=2)
    ap.add_argument('--arms', default='ABCD')
    ap.add_argument('--base', default='/media/ucla/work/ntof_pass1/runs')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from wft import io as wio
    res = dict(run=a.run, subrun=a.subrun, thresholds=list(THRESH), hist_edges=HB.tolist(), arms={})
    for arm in a.arms:
        for plane, feu in zip('xy', FEUS[arm]):
            files = wio.subrun_files(a.base, f'run_{a.run}', a.subrun, feu)[:a.tags]
            m = merge([census_file(f) for f in files])
            m['files'] = [os.path.basename(f) for f in files]
            res['arms'][f'{arm}{plane}'] = m
            print(f'{arm}{plane} feu {feu}: {m["n_events"]} ev, clipped {m["clipped"]} samples in '
                  f'{m["clipped_events"]} ev (coherent {m["coherent_events"]} ev); rail(4095-ped) p5/50/95 {m["rail_ped_pct"][5]:.0f}/'
                  f'{m["rail_ped_pct"][50]:.0f}/{m["rail_ped_pct"][95]:.0f}; clipped-W p5/50/95 '
                  f'{m["rail_sample_pct"][5]:.0f}/{m["rail_sample_pct"][50]:.0f}/{m["rail_sample_pct"][95]:.0f}; '
                  f'missed 3550 {m["frac_missed"][3550]:.3f} 3700 {m["frac_missed"][3700]:.3f}; censored-ok 3550 '
                  f'{m["censored_ok"][3550]} 3700 {m["censored_ok"][3700]}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
