#!/usr/bin/env python3
"""
beam_window_loss.py — how much SIGNAL does the 20-sample readout cut off?

`framing_compare.py` measures the *incidence* of truncation (what fraction of
columns still have a hit in the last sample bin). That is not a charge loss.
This measures the charge, on the beam waveforms themselves, so the gas and the
drift field are included by construction -- no bench extrapolation.

Per detector and plane, over clean micro-TPC columns (the `framing_compare`
selection: significance floor, 12 mm gap clustering, >= 5 strips, ladder
correlation > 0.7, peak > 300 ADC, largest pulse per channel):

  1. the stacked cluster-summed waveform, normalised per event -- where the
     column's charge actually sits in the frame;
  2. `edge_frac`  = charge in the last sample / total in-window charge;
  3. `live_at_end`= fraction of columns whose last-sample charge is still above
     20 % of that column's peak, i.e. visibly still rising or falling;
  4. `clip_loss`  = the charge missing because pulses run off the end, from a
     BEAM-measured single-pulse template: a strip peaking at sample p keeps
     only the template's integral up to the window end, so its missing area is
     known once the template is known.

  (4) counts charge whose pulse STARTED inside the window. Charge arriving so
  late that its pulse never peaks in-window is invisible to any in-window
  method and is NOT included -- see the caveat printed with the results.

    ../../.venv/bin/python mx_june_wft/bench/beam_window_loss.py
    ... --run run_79 --subrun stat090_0000 --tags 000,001 --dets A,B,C,D
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

from wft import io as wio                                       # noqa: E402
from wft import seed as wseed                                   # noqa: E402
import framing_compare as fc                                    # noqa: E402

BEAM_BASE = '/media/dylan/data/x17/beam_july/runs'
TMPL_GRID = np.arange(-300, 900, 5.0)     # ns, relative to the pulse's t50
TMPL_MIN_AMP = 300.0
LIVE_FRAC = 0.20


def clean_columns(df, pos_map, feu, n_sample, busy_strips):
    """{eventId: channel array} for the clean columns of one plane."""
    d = df[df['feu'] == feu]
    d = wseed.apply_significance_floor(d, fc.SIG_FLOOR)
    d = d.sort_values('amplitude').drop_duplicates(['eventId', 'channel'],
                                                   keep='last')
    out = {}
    for eid, g in d.groupby('eventId', sort=False):
        if len(g) > busy_strips:
            continue
        ch = g['channel'].to_numpy().astype(int)
        pos = pos_map[feu][ch]
        s = wseed.seed_plane(pos, ch, g['amplitude'].to_numpy(),
                             gap_mm=fc.GAP_MM, min_strips=fc.MIN_STRIPS)
        if s is None:
            continue
        gg = g.set_index('channel').loc[s.channels]
        m = gg['max_sample'].to_numpy()
        p = pos_map[feu][np.asarray(s.channels, dtype=int)]
        ok = np.isfinite(m) & np.isfinite(p)
        if ok.sum() < fc.MIN_STRIPS:
            continue
        lad = abs(fc._rank_corr(p[ok], m[ok]))
        if not (lad > 0.7 and 5 <= ok.sum() <= 25
                and gg['amplitude'].max() > 300):
            continue
        out[int(eid)] = np.asarray(s.channels, dtype=int)[ok]
    return out


def t50(w):
    ipk = int(np.argmax(w))
    a = w[ipk]
    for k in range(1, ipk + 1):
        if w[k] >= 0.5 * a > w[k - 1]:
            return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
    return np.nan


def analyse_plane(reader, columns, n_sample, sample_ns=60.0):
    """One plane: stacked profile, edge charge, and the template it needs."""
    prof = np.zeros(n_sample)
    n_prof = 0
    edge_frac, live, peaks, tot_amp = [], [], [], []
    tmpl_acc, strips = [], []
    for eid, ftst, wfm in reader.iter_events(set(columns)):
        ch = columns[eid]
        W = wfm[ch]                                   # (n_strip, n_sample)
        c = W.sum(axis=0)
        tot = c[c > 0].sum()
        if tot <= 0:
            continue
        prof += c / tot
        n_prof += 1
        edge_frac.append(float(max(c[-1], 0.0) / tot))
        live.append(bool(c[-1] > LIVE_FRAC * c.max()))
        peaks.append(int(np.argmax(c)))
        # per-strip peak positions + amplitudes, for the clipping estimate
        for w in W:
            a = float(w.max())
            if a <= 0:
                continue
            strips.append((int(np.argmax(w)), a))
            # template candidates: a COMPLETE pulse, peak early enough that its
            # whole fall is inside the window (fall to baseline is ~7 samples)
            if a > TMPL_MIN_AMP and 2 <= np.argmax(w) <= n_sample - 8:
                c50 = t50(w)
                if np.isfinite(c50):
                    tt = (np.arange(n_sample) - c50) * sample_ns
                    tmpl_acc.append(np.interp(TMPL_GRID, tt, w / a,
                                              left=np.nan, right=np.nan))
        tot_amp.append(float(tot))
    if n_prof == 0:
        return None
    tmpl = (np.nan_to_num(np.nanmedian(np.array(tmpl_acc), axis=0))
            if tmpl_acc else None)
    return dict(prof=prof / n_prof, n=n_prof, edge_frac=np.array(edge_frac),
                live=np.array(live), peaks=np.array(peaks),
                strips=np.array(strips), tmpl=tmpl, n_tmpl=len(tmpl_acc))


def clip_loss(res, n_sample, sample_ns=60.0):
    """Charge lost because pulses run off the end of the window, from the
    beam-measured template: a strip peaking at sample p keeps the template's
    integral up to (n_sample - 1 - p) samples past its peak."""
    t = res['tmpl']
    if t is None or not np.isfinite(t).any():
        return np.nan, np.nan
    t = np.clip(t, 0, None)
    i_full = float(t.sum())
    if i_full <= 0:
        return np.nan, np.nan
    ipk = int(np.argmax(t))

    def kept(p):
        end_ns = (n_sample - 1 - p) * sample_ns
        j = np.searchsorted(TMPL_GRID, TMPL_GRID[ipk] + end_ns)
        return float(t[:j].sum()) / i_full

    vis = miss = 0.0
    for p, a in res['strips']:
        f = min(1.0, max(0.0, kept(int(p))))
        vis += a * f
        miss += a * (1.0 - f)
    return miss / (vis + miss), i_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='run_79')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--tags', default='000')
    ap.add_argument('--dets', default='A,B,C,D')
    ap.add_argument('--busy-strips', type=int, default=120)
    ap.add_argument('--out', default=os.path.join(HERE, 'beam_window_loss.json'))
    args = ap.parse_args()

    run_dir = os.path.join(BEAM_BASE, args.run)
    rcfg = os.path.join(run_dir, 'run_config.json')
    with open(rcfg) as f:
        cj = json.load(f)
    n_sample = int(cj['dream_daq_info']['n_samples_per_waveform'])
    sample_ns = float(cj['dream_daq_info']['sample_period'])
    cols = ['eventId', 'feu', 'channel', 'amplitude', 'significance', 'max_sample']

    summary = {}
    for letter in args.dets.split(','):
        letter = letter.strip()
        _bench, fx, fy = fc.BEAM_DETS[letter]
        det = f'mx17_{letter}'
        pos_map = fc.strip_positions(rcfg, det, [fx, fy])
        for plane, feu in (('x', fx), ('y', fy)):
            acc = None
            for tag in args.tags.split(','):
                hits = glob.glob(os.path.join(
                    run_dir, args.subrun, 'combined_hits_root',
                    f'*_{tag.strip()}_feu-combined_hits.root'))
                dec = glob.glob(os.path.join(
                    run_dir, args.subrun, 'decoded_root',
                    f'*_{tag.strip()}_{feu:02d}.root'))
                if not hits or not dec:
                    print(f'  {det}{plane} tag {tag}: files missing, skipped')
                    continue
                df = uproot.open(hits[0])['hits'].arrays(cols, library='pd')
                colmap = clean_columns(df, pos_map, feu, n_sample,
                                       args.busy_strips)
                del df
                if not colmap:
                    continue
                r = analyse_plane(wio.FeuReader(dec[0]), colmap, n_sample,
                                  sample_ns)
                if r is None:
                    continue
                if acc is None:
                    acc = r
                else:
                    acc['prof'] = ((acc['prof'] * acc['n'] + r['prof'] * r['n'])
                                   / (acc['n'] + r['n']))
                    acc['n'] += r['n']
                    for k in ('edge_frac', 'live', 'peaks'):
                        acc[k] = np.concatenate([acc[k], r[k]])
                    acc['strips'] = np.vstack([acc['strips'], r['strips']])
                    if r['n_tmpl'] > acc['n_tmpl']:
                        acc['tmpl'], acc['n_tmpl'] = r['tmpl'], r['n_tmpl']
            if acc is None:
                continue
            loss, _ = clip_loss(acc, n_sample, sample_ns)
            p = acc['prof'] / acc['prof'].sum()
            summary[f'{det}:{plane}'] = dict(
                n=int(acc['n']), n_tmpl=int(acc['n_tmpl']),
                edge_frac_mean=float(np.mean(acc['edge_frac'])),
                edge_frac_p90=float(np.percentile(acc['edge_frac'], 90)),
                last2_frac=float(p[-2:].sum()),
                live_at_end=float(np.mean(acc['live'])),
                clip_loss=float(loss),
                profile=[float(v) for v in p])
            s = summary[f'{det}:{plane}']
            print(f'{det}{plane}: n={s["n"]:5d}  charge in last sample '
                  f'{100*s["edge_frac_mean"]:5.2f} % (p90 '
                  f'{100*s["edge_frac_p90"]:5.2f})  last 2 samples '
                  f'{100*s["last2_frac"]:5.2f} %  still live at end '
                  f'{100*s["live_at_end"]:5.1f} %  clipped-pulse loss '
                  f'{100*s["clip_loss"]:5.2f} %   [template n={s["n_tmpl"]}]')
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=1)
    print('\nCAVEAT: `clipped-pulse loss` is the charge missing from pulses '
          'that DID peak inside the window.\nCharge arriving late enough that '
          'its pulse never peaks in-window is invisible to any in-window\n'
          'measurement and is not included here -- its incidence is bounded '
          'by the ceiling fractions in framing.json.')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
