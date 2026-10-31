#!/usr/bin/env python3
"""tel_gain.py --run R --subrun S --keys k1,k2 --out O.parquet -- det4's response on every clean uRWELL telescope track
of one SPS sub-run, for a gain curve with a BEAM denominator (Paper I §4, rerun/paper1_ops/README.md).

Why: at SPS only events that cross det4's zero-suppression threshold reach the pass table, so the charge quantiles of the
reconstructed events (sps_gain.py) measure the threshold, not the gain: run66's charge moves ~10 % over a 200 V resist
scan while the event yield halves (sps_beam_test_26/analysis/EXTRACTION_2026-08-05b.md §9).  A gain estimator has to
rank the crossings det4 MISSED below the ones it saw.  The telescope supplies those crossings.

Per event of the sub-run's combined_hits (banco ZS hits; feu 1 = both uRWELL planes, feu 3 = det4):
  telescope   rerun/sps_audit/tel_job.telescope_zs: a cluster in all four uRWELL plane-views; kept here only when every
              plane-view has exactly one cluster (strict, as the resolution sample)
  det4        from the feu-3 decoded file (banco's combined_hits carry no det4 hits after run_55): per strip its peak
              sample, the strips over the hits estimator's cut (amp > 60 ADC, tel_job.DET4_AMP_MIN): per view the summed
              hit amplitude q_<v>, the peak hit p_<v>, the hit count n_<v>, the hits at or above the 3550 rule nsat_<v>;
              the pass's oscillating / coupled channels (BAD below) left out.  No strip = 0 (missed).  Where combined_hits
              do carry det4 (run_55) the same numbers from them are kept as h_* (a cross-check of the two inputs).
  plateau     the registry row whose [t_lo + 20 s, t_hi - 20 s] holds the event's wall-clock time (sps_build_cache's
              per-acquisition clock, read from the feu-3 decoded files); events outside every plateau are dropped.
Hits are allowed here: this is charge/QA, not geometry (CLAUDE.md, RECONSTRUCTION_BASIS.md).
"""
import argparse, csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'rerun', 'sps_audit'),
                os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc   # noqa: E402
import tel_job as TJ            # noqa: E402

SAT = 3550.0
#: left out of det4's sums: the pass's seed-bad oscillators (372, 510) and their in-phase copies (370-377, 508-511;
#: rerun/sps_junk/README.md); all sit at Y ~155 / ~250 mm, away from the beam spot's det4 X band
BAD = sbc.SEED_BAD.copy(); BAD[370:378] = True; BAD[508:512] = True


def det4_per_event(hits, eids):
    """Per telescope event (sorted eids): summed amplitude, peak, count and saturated count per det4 view."""
    out = {f'{k}_{v}': np.zeros(len(eids)) for k in ('q', 'p', 'n', 'nsat') for v in 'xy'}
    if hits is None or len(hits['ev']) == 0:
        return out
    ch = hits['ch'].astype(int); amp = hits['amp'].astype(float)
    good = ~BAD[ch]
    ev, ch, amp = hits['ev'][good], ch[good], amp[good]
    idx = np.searchsorted(eids, ev)
    ok = (idx < len(eids)) & (eids[np.clip(idx, 0, len(eids) - 1)] == ev)
    idx, ch, amp = idx[ok], ch[ok], amp[ok]
    for v in 'xy':
        k = sbc.VIEW[ch] == v
        np.add.at(out[f'q_{v}'], idx[k], amp[k])
        np.maximum.at(out[f'p_{v}'], idx[k], amp[k])
        np.add.at(out[f'n_{v}'], idx[k], 1)
        np.add.at(out[f'nsat_{v}'], idx[k], (amp[k] >= SAT).astype(float))
    return out


def det4_decoded(f3, eids):
    """The same per-view numbers from the feu-3 decoded file (banco's combined_hits carry no det4 hits on runs 56-71):
    per strip the peak sample, strips with a peak > DET4_AMP_MIN (the hits estimator's cut) summed, BAD left out."""
    import uproot
    out = {f'{k}_{v}': np.zeros(len(eids)) for k in ('q', 'p', 'n', 'nsat') for v in 'xy'}
    pos = {int(e): i for i, e in enumerate(eids)}
    t = uproot.open(f3)['nt']
    for lo in range(0, t.num_entries, TJ.CHUNK):
        arr = t.arrays(['eventId', 'channel', 'amplitude'], entry_start=lo,
                       entry_stop=min(lo + TJ.CHUNK, t.num_entries), library='np')
        for j, e in enumerate(arr['eventId']):
            i = pos.get(int(e))
            if i is None:
                continue
            ch = arr['channel'][j].astype(int); am = arr['amplitude'][j].astype(float)
            if not len(ch):
                continue
            pk = np.zeros(512); np.maximum.at(pk, ch, am)
            pk[BAD] = 0.0
            for v in 'xy':
                s = pk[(sbc.VIEW == v) & (pk > TJ.DET4_AMP_MIN)]
                out[f'q_{v}'][i] = s.sum(); out[f'p_{v}'][i] = s.max() if len(s) else 0.0
                out[f'n_{v}'][i] = len(s); out[f'nsat_{v}'][i] = float((s >= SAT).sum())
    t.file.close()
    return out


def eid_wall(run, subrun, files):
    """{eventId: wall-clock s} over the sub-run's feu-3 decoded files (sps_build_cache's calibration)."""
    import uproot
    clock = TJ.wall_clock(run, subrun, files)
    E, T = [], []
    for f in files:
        a = uproot.open(f)['nt'].arrays(['eventId', 'timestamp'], library='np')
        c = clock[f]
        E.append(a['eventId'].astype(np.int64)); T.append(c[0] + (a['timestamp'].astype(float) - c[1]) * c[2])
    E, T = np.concatenate(E), np.concatenate(T)
    o = np.argsort(E)
    return E[o], T[o]


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--subrun', required=True)
    ap.add_argument('--keys', required=True, help='registry keys (plateaus) of this sub-run, comma separated')
    ap.add_argument('--feu3-dir', default='decoded_root_fa80c06')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
    rows = [reg[k] for k in a.keys.split(',')]
    sub = os.path.join(sbc.RUNS, a.run, a.subrun)
    f3 = sorted(glob.glob(os.path.join(sub, a.feu3_dir, '*_datrun_*_03.root')))
    E, T = eid_wall(a.run, a.subrun, f3)
    parts = []
    for chf in sorted(glob.glob(os.path.join(sub, 'combined_hits_root', '*_feu-combined_hits.root'))):
        tel, hits = TJ.telescope_zs(chf)
        strict = np.all([tel[f'{d}{v}_n'] == 1 for d in 'fb' for v in 'xy'], axis=0)
        eids = tel['eid'].astype(np.int64)
        D = pd.DataFrame({k: v for k, v in tel.items()})
        D['strict'] = strict
        # primary: the decoded feu-3 file (every run); the combined_hits det4 hits only as a cross-check (h_*)
        g = os.path.basename(chf).split('_feu-combined')[0].rsplit('_', 1)[-1]
        f3g = [f for f in f3 if f.endswith(f'_{g}_03.root')]
        for k, v in det4_per_event(hits, eids).items():
            D['h_' + k] = v
        if f3g:
            for k, v in det4_decoded(f3g[0], eids).items():
                D[k] = v
        else:
            print(f'  no feu-3 decoded file for group {g}: det4 numbers from combined_hits only', flush=True)
            for k in [c for c in D.columns if c.startswith('h_')]:
                D[k[2:]] = D[k]
        i = np.clip(np.searchsorted(E, eids), 0, len(E) - 1)
        D['t_wall'] = np.where(E[i] == eids, T[i], np.nan)
        D['file'] = os.path.basename(chf)
        D['key'] = ''
        for r in rows:
            m = (D.t_wall >= float(r['t_lo_s']) + 20) & (D.t_wall < float(r['t_hi_s']) - 20)
            D.loc[m, 'key'] = r['key']
        D = D[D.key != '']
        parts.append(D)
        print(f'{os.path.basename(chf)}: {len(tel["eid"])} four-view, {int(strict.sum())} strict, {len(D)} in plateaus', flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(a.out)
    print(out.groupby('key').agg(n=('eid', 'size'), strict=('strict', 'sum'), det=('n_x', lambda s: float((s > 0).mean()))))


if __name__ == '__main__':
    main()
