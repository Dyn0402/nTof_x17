#!/usr/bin/env python3
"""
framing_compare.py — where does the drift column sit inside the readout window,
on the bench and at n_TOF?

The window ablation (`run_bench.py --variant trunc<N>`) emulates the n_TOF
readout by cropping bench windows. That crop is only meaningful if the *signal*
lands at the same place in the cropped frame as it does at n_TOF: the beam runs
use a different DREAM `latency` AND a different trigger G&D delay, so the frame
origin is not a known offset -- it has to be measured on both sides and matched.

Measured here, per detector and plane, from combined_hits (a legitimate hits
use: which sample a pulse peaked in, not track geometry):

  onset  = per-event EARLIEST max_sample of the track cluster  (prompt, near-mesh)
  edge   = per-event LATEST   max_sample of the track cluster  (deep, near-cathode)
  span   = edge - onset                                        (full-gap drift)
  ceil   = fraction of clusters whose edge is in the last sample bin

Clusters are the production seed: per-plane relative significance floor, 12 mm
gap clustering, largest cluster, >= MIN_STRIPS strips.

    ../../.venv/bin/python mx_june_wft/bench/framing_compare.py
    ... --bench-key sat_det3 --beam-run run_79 --beam-subrun stat090_0000
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from wft import seed as wseed                                   # noqa: E402

MIN_STRIPS = 5          # a real column, not a single-strip deposit
GAP_MM = wseed.GAP_THRESHOLD_MM
SIG_FLOOR = wseed.SIG_REL_FLOOR
BEAM_BASE = '/media/dylan/data/x17/beam_july/runs'
# beam detector -> (bench detector, FEU x, FEU y); det_labels.py + run_config
BEAM_DETS = {'A': ('mx17_3', 3, 4), 'B': ('mx17_2', 5, 6),
             'C': ('mx17_6', 7, 8), 'D': ('mx17_7', 1, 2)}
PCT = [1, 5, 25, 50, 75, 95, 99]


def strip_positions(run_config_path, det_name, feus):
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(run_config_path, os.path.join(REPO, 'mx17_m1_map.csv'))
    det = rc.get_detector(det_name)
    out = {}
    for feu, axis in zip(feus, (0, 1)):
        p = np.full(512, np.nan)
        for ch in range(512):
            h = det.map_hit(feu, ch)
            if h is not None and h[axis] is not None:
                p[ch] = h[axis]
        out[feu] = p
    return out


def _rank_corr(a, b):
    """Spearman correlation without scipy."""
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def cluster_stats(df, pos_map, n_sample, busy_strips=None, amp_min=0.0):
    """Per (event, plane) column statistics from the largest seed cluster.

    Also records the quality handles needed to tell a real micro-TPC column
    from the ringing-block artifact documented in
    `mx_july_beam_qa/DRIFT_WINDOW_ANALYSIS.md` §1a (fixed channel blocks that
    ring late after a saturated event and masquerade as late clusters):
    strip count, peak amplitude, and `ladder` = |rank correlation between strip
    position and peak sample|, which is ~1 for an inclined column and ~0 for a
    block of channels ringing together.
    """
    if amp_min > 0:
        # An ABSOLUTE amplitude cut, applied identically on both sides, is the
        # only way to compare runs processed with different analyzer versions:
        # pre-2026-07-24 files have no `significance`, so the relative floor is
        # a no-op there and the two hit populations are otherwise incomparable.
        df = df[df['amplitude'] > amp_min]
    df = wseed.apply_significance_floor(df, SIG_FLOOR)
    # A channel can carry SEVERAL hits in one event at beam (pileup, and the
    # ringing that follows a saturated event). Keep the largest pulse per
    # channel: a second, later pulse on a strip that is already in the cluster
    # would otherwise be read as the column's deep edge.
    df = df.sort_values('amplitude').drop_duplicates(['eventId', 'feu',
                                                      'channel'], keep='last')
    rows = []
    for (eid, feu), g in df.groupby(['eventId', 'feu'], sort=False):
        if busy_strips is not None and len(g) > busy_strips:
            rows.append(dict(eventId=eid, feu=feu, busy=True))
            continue
        ch = g['channel'].to_numpy().astype(int)
        pos = pos_map[feu][ch]
        s = wseed.seed_plane(pos, ch, g['amplitude'].to_numpy(),
                             gap_mm=GAP_MM, min_strips=MIN_STRIPS)
        if s is None:
            continue
        gg = g.set_index('channel').loc[s.channels]
        m = gg['max_sample'].to_numpy()
        p = pos_map[feu][np.asarray(s.channels, dtype=int)]
        ok = np.isfinite(m) & np.isfinite(p)
        m, p = m[ok], p[ok]
        if len(m) < MIN_STRIPS:
            continue
        a_all = gg['amplitude'].to_numpy()[ok]
        rows.append(dict(eventId=eid, feu=feu, busy=False, n=len(m),
                         onset=float(m.min()), edge=float(m.max()),
                         span=float(m.max() - m.min()),
                         pos=float((p * a_all).sum() / a_all.sum())
                         if a_all.sum() > 0 else float(np.mean(p)),
                         amp=float(gg['amplitude'].max()),
                         ladder=abs(_rank_corr(p, m)),
                         ceil=bool(m.max() >= n_sample - 1.5)))
    return pd.DataFrame(rows)


def report(tag, res, n_sample, clean=False):
    good = res[~res['busy']] if 'busy' in res else res
    good = good.dropna(subset=['onset'])
    if clean and len(good):
        # a real inclined column: a monotone position<->time ladder, a
        # plausible strip count, and a pulse well above the noise
        good = good[(good['ladder'] > 0.7) & good['n'].between(5, 25)
                    & (good['amp'] > 300)]
    if not len(good):
        print(f'{tag:16s} no clusters')
        return None
    o = np.percentile(good['onset'], PCT)
    e = np.percentile(good['edge'], PCT)
    s = np.percentile(good['span'], PCT)
    print(f'{tag:16s} n={len(good):6,d}  window={n_sample} smp'
          f'  busy={res["busy"].mean()*100 if "busy" in res else 0:4.1f}%')
    print(f'{"":16s}   onset p5/p50/p95 = {o[1]:5.1f} /{o[3]:5.1f} /{o[5]:5.1f}'
          f'   edge p50/p95/p99 = {e[3]:5.1f} /{e[5]:5.1f} /{e[6]:5.1f}'
          f'   span p50/p95 = {s[3]:5.1f} /{s[5]:5.1f}'
          f'   at ceiling {good["ceil"].mean()*100:5.2f}%')
    return dict(n=int(len(good)), n_sample=int(n_sample),
                onset={str(p): float(v) for p, v in zip(PCT, o)},
                edge={str(p): float(v) for p, v in zip(PCT, e)},
                span={str(p): float(v) for p, v in zip(PCT, s)},
                ceiling_frac=float(good['ceil'].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench-key', default='sat_det3')
    ap.add_argument('--beam-run', default='run_79')
    ap.add_argument('--beam-subrun', default='stat090_0000')
    ap.add_argument('--beam-dets', default='A,B,C,D')
    ap.add_argument('--files', type=int, default=1,
                    help='hits files per side (1 is plenty for percentiles)')
    ap.add_argument('--busy-strips', type=int, default=120,
                    help='beam busy/flash veto: skip planes with more hits')
    ap.add_argument('--amp-min', type=float, default=0.0,
                    help='absolute hit amplitude cut [ADC], applied to BOTH '
                         'sides -- required when comparing runs processed with '
                         'different analyzer versions')
    ap.add_argument('--out', default=os.path.join(HERE, 'framing.json'))
    args = ap.parse_args()

    cols = ['eventId', 'feu', 'channel', 'amplitude', 'significance', 'max_sample']
    summary = {}

    def read_hits(paths):
        """Older runs (pre-2026-07-24 analyzer) have no `significance` branch;
        the relative floor is then a no-op and the threshold is whatever the
        analyzer applied. Comparisons ACROSS analyzer versions are therefore
        threshold-dependent -- compare within a run, not between."""
        out = []
        for p in paths:
            t = uproot.open(p)['hits']
            have = [c for c in cols if c in t.keys()]
            if 'significance' not in have:
                print(f'  note: {os.path.basename(p)} has no `significance` '
                      f'(old analyzer) -- relative floor not applied')
            out.append(t.arrays(have, library='pd'))
        return pd.concat(out)

    # ---------------------------------------------------------------- bench
    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(args.bench_key)
    files = sorted(f for f in glob.glob(cfg.combined_hits_dir + '*.root')
                   if '_datrun_' in os.path.basename(f))[:args.files]
    pos = strip_positions(cfg.run_config_path, cfg.DET_NAME, cfg.MX17_FEUS)
    df = read_hits(files)
    df = df[df['feu'].isin(cfg.MX17_FEUS)]
    with open(cfg.run_config_path) as f:
        n_bench = int(json.load(f)['dream_daq_info']['n_samples_per_waveform'])
    res = cluster_stats(df, pos, n_bench, amp_min=args.amp_min)
    res.to_parquet(os.path.join(HERE, f'framing_clusters_{args.bench_key}.parquet'))
    for plane, feu in zip('xy', cfg.MX17_FEUS):
        summary[f'bench:{args.bench_key}:{plane}'] = report(
            f'{args.bench_key} {plane}', res[res['feu'] == feu], n_bench)
        summary[f'bench:{args.bench_key}:{plane}:clean'] = report(
            f'  ...clean {plane}', res[res['feu'] == feu], n_bench, clean=True)
    del df, res

    # ----------------------------------------------------------------- beam
    rcfg = os.path.join(BEAM_BASE, args.beam_run, 'run_config.json')
    with open(rcfg) as f:
        n_beam = int(json.load(f)['dream_daq_info']['n_samples_per_waveform'])
    # --beam-subrun may be a glob: run_55's resist scan puts one hits file in
    # each of 27 sub-runs, so a single sub-run has far too few gap-crossing
    # columns to compare chambers. Sub-runs of the scan differ in RESIST HV
    # (gain) only -- the drift field, which is what a column length measures,
    # is the same in all of them.
    files = sorted(glob.glob(os.path.join(
        BEAM_BASE, args.beam_run, args.beam_subrun, 'combined_hits_root',
        '*_feu-combined_hits.root')))[:args.files]
    print(f'[beam] {len(files)} hits file(s) matching {args.beam_subrun}')
    df_all = read_hits(files)
    for letter in args.beam_dets.split(','):
        det_name, fx, fy = BEAM_DETS[letter.strip()]
        beam_det = f'mx17_{letter.strip()}'
        pos = strip_positions(rcfg, beam_det, [fx, fy])
        res = cluster_stats(df_all[df_all['feu'].isin((fx, fy))], pos, n_beam,
                            busy_strips=args.busy_strips,
                            amp_min=args.amp_min)
        res.to_parquet(os.path.join(
            HERE, f'framing_clusters_{args.beam_run}_{beam_det}.parquet'))
        for plane, feu in zip('xy', (fx, fy)):
            summary[f'beam:{beam_det}:{plane}'] = report(
                f'{args.beam_run} {beam_det}{plane}',
                res[res['feu'] == feu], n_beam)
            summary[f'beam:{beam_det}:{plane}:clean'] = report(
                f'  ...clean {plane}', res[res['feu'] == feu], n_beam,
                clean=True)

    # ------------------------------------------------------------ the crop
    print('\ncrop offsets to emulate the beam framing on the bench '
          '(bench_onset_p5 - beam_onset_p5):')
    for k, v in summary.items():
        if not k.startswith('beam:') or v is None:
            continue
        parts = k.split(':')                      # beam:<det>:<plane>[:clean]
        plane, suffix = parts[2], (':' + parts[3] if len(parts) > 3 else '')
        b = summary.get(f'bench:{args.bench_key}:{plane}{suffix}')
        if b is None:
            continue
        off = b['onset']['5'] - v['onset']['5']
        print(f'  {k:22s} onset {v["onset"]["5"]:4.1f} vs bench '
              f'{b["onset"]["5"]:4.1f}  ->  crop start {off:+.1f} '
              f'({int(round(off))} samples), keep {v["n_sample"]}')
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=1)
    print('\nwrote', args.out)


if __name__ == '__main__':
    main()
