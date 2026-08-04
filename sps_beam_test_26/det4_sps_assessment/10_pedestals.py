#!/usr/bin/env python3
"""
10_pedestals.py — are the dead bands disconnected strips, or dead amplification?

08 ruled out *dead channels* (every strip fires) and connector periodicity, but a
partially-seated connector or a cold solder joint is not a dead channel — it is a
channel whose strip is no longer attached to the preamp input. That has a sharp,
unmistakable pedestal signature and it does not need any beam:

    a strip that is disconnected loses its load capacitance, so the preamp sees
    less input and its **pedestal noise drops**, sharply, at the channel where
    the contact fails. A *resistive* strip that has lost its HV feed leaves the
    readout strip exactly where it was — same capacitance, same pedestal.

So: pedestal RMS flat across a dead band  -> the readout is fine, the
amplification is not (broken resistive strip / mesh).
   pedestal RMS stepping down over a dead band -> readout, and the whole gain
   story is an artefact.

Reads the pedestal run taken for this data (`MX17_pedestals_pedthr_260623_18H43`,
which sits in the 6-24 run's raw_daq_data), computes per-channel mean, raw RMS
and CNS RMS (median of each 64-channel block subtracted per sample, the repo
convention), and overlays the live/dead bands from 04.

    ../../.venv/bin/python sps_beam_test_26/det4_sps_assessment/10_pedestals.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths                # noqa: E402
setup_paths()
import matplotlib                                            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import uproot                                                # noqa: E402
from common.Mx17StripMap import Mx17StripMap                 # noqa: E402

NCH, NSAMP = 512, 32
PED_GLOB = ('/home/dylan/x17/cosmic_bench/det4_day/mx17_det4_day_6-24-26/'
            'long_run/raw_daq_data/MX17_pedestals_pedthr_*_{feu:02d}.root')


def cns(w):
    """Common-noise subtraction, per 64-channel block (repo convention)."""
    w = w.copy()
    for b in range(0, NCH, 64):
        w[..., b:b + 64] -= np.median(w[..., b:b + 64], axis=-1, keepdims=True)
    return w


def frames(path, nmax=400):
    t = uproot.open(path)['nt']
    arr = t.arrays(['channel', 'sample', 'amplitude'],
                   entry_stop=min(nmax, t.num_entries), library='np')
    out = []
    for ch, s, a in zip(arr['channel'], arr['sample'], arr['amplitude']):
        if len(a) != NSAMP * NCH:
            continue
        w = np.full((NSAMP, NCH), np.nan, np.float32)
        w[s, ch] = a
        if np.isnan(w).any():
            continue
        out.append(w)
    return np.array(out)                     # (events, samples, channels)


def ped_stats(feu):
    paths = sorted(glob.glob(PED_GLOB.format(feu=feu)))
    if not paths:
        raise SystemExit(f'no pedestal file for FEU {feu}')
    w = frames(paths[0])
    mean = w.reshape(-1, NCH).mean(axis=0)
    raw_rms = w.reshape(-1, NCH).std(axis=0)
    wc = cns(w)
    cns_rms = wc.reshape(-1, NCH).std(axis=0)
    return dict(path=paths[0], n_frames=int(len(w)), mean=mean,
                raw_rms=raw_rms, cns_rms=cns_rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='g_det4')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    cfg = get_config(args.key)
    fx, fy = cfg.MX17_FEUS
    sm = Mx17StripMap(cfg.MAP_CSV_PATH)
    posx = np.array([sm.lookup('x', *Mx17StripMap.feu_channel_to_connector(c))[0]
                     for c in range(NCH)])
    posy = np.array([sm.lookup('y', *Mx17StripMap.feu_channel_to_connector(c))[1]
                     for c in range(NCH)])
    bands = json.load(open(os.path.join(args.out,
                                        f'stripes_{args.key}.json')))['bands_mm']

    px, py = ped_stats(fx), ped_stats(fy)
    inb = np.zeros(NCH, bool)
    for a, b in bands:
        inb |= (posx >= a) & (posx <= b)

    def summarise(p, sel_a, sel_b, la, lb):
        return {la: dict(n=int(sel_a.sum()),
                         raw_rms=float(np.median(p['raw_rms'][sel_a])),
                         cns_rms=float(np.median(p['cns_rms'][sel_a])),
                         mean=float(np.median(p['mean'][sel_a]))),
                lb: dict(n=int(sel_b.sum()),
                         raw_rms=float(np.median(p['raw_rms'][sel_b])),
                         cns_rms=float(np.median(p['cns_rms'][sel_b])),
                         mean=float(np.median(p['mean'][sel_b])))}

    rep = dict(run_key=args.key,
               pedestal_files=dict(x=os.path.basename(px['path']),
                                   y=os.path.basename(py['path'])),
               n_frames=dict(x=px['n_frames'], y=py['n_frames']),
               Xplane_live_vs_dead=summarise(px, inb, ~inb, 'live_band', 'dead_band'),
               Yplane_same_selection=summarise(py, inb, ~inb, 'live_band', 'dead_band'))
    # ratio is the number that matters: a disconnected strip drops its noise
    lv = rep['Xplane_live_vs_dead']
    rep['cns_rms_dead_over_live_Xplane'] = (lv['dead_band']['cns_rms']
                                            / lv['live_band']['cns_rms'])
    rep['raw_rms_dead_over_live_Xplane'] = (lv['dead_band']['raw_rms']
                                            / lv['live_band']['raw_rms'])
    # per-connector, for the "is a whole connector badly seated" question
    conn = {}
    for k in range(8):
        s = slice(k * 64, (k + 1) * 64)
        conn[k + 1] = dict(x_mm=[float(posx[s].min()), float(posx[s].max())],
                           x_cns_rms=float(np.median(px['cns_rms'][s])),
                           x_raw_rms=float(np.median(px['raw_rms'][s])),
                           x_mean=float(np.median(px['mean'][s])),
                           y_cns_rms=float(np.median(py['cns_rms'][s])),
                           live_frac=float(inb[s].mean()))
    rep['per_connector'] = conn
    # channels whose noise is anomalously low = candidate disconnects
    lowthr = 0.5 * np.median(px['cns_rms'])
    rep['n_channels_below_half_median_noise'] = dict(
        x=int((px['cns_rms'] < lowthr).sum()),
        y=int((py['cns_rms'] < 0.5 * np.median(py['cns_rms'])).sum()))
    with open(os.path.join(args.out, f'pedestals_{args.key}.json'), 'w') as f:
        json.dump(rep, f, indent=1)

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    o = np.argsort(posx)
    a = axs[0]
    a.plot(posx[o], px['cns_rms'][o], lw=.9, color='#0072b2', label='CNS noise')
    a.plot(posx[o], px['raw_rms'][o], lw=.9, color='#999999', label='raw noise')
    for aa, bb in bands:
        a.axvspan(aa, bb, color='#009e73', alpha=.15)
    for k in range(1, 8):
        a.axvline(k * 64 * 0.78, color='k', ls='--', lw=1)
    a.set_ylabel('pedestal RMS [ADC]')
    a.set_yscale('log')
    a.legend(fontsize=8)
    a.set_title(f'{args.key} X plane (FEU {fx}) — pedestal noise vs strip position '
                f'(green = live gain bands, dashed = connectors)')
    a.grid(alpha=.3, which='both')

    a = axs[1]
    a.plot(posx[o], px['mean'][o], lw=.9, color='#d55e00')
    for aa, bb in bands:
        a.axvspan(aa, bb, color='#009e73', alpha=.15)
    for k in range(1, 8):
        a.axvline(k * 64 * 0.78, color='k', ls='--', lw=1)
    a.set_ylabel('pedestal mean [ADC]')
    a.grid(alpha=.3)
    a.set_title('pedestal baseline — a floating input also shifts this')

    a = axs[2]
    oy = np.argsort(posy)
    a.plot(posy[oy], py['cns_rms'][oy], lw=.9, color='#0072b2', label='CNS noise')
    a.plot(posy[oy], py['raw_rms'][oy], lw=.9, color='#999999', label='raw noise')
    a.set_yscale('log')
    a.set_xlabel('detector-local coordinate [mm]')
    a.set_ylabel('pedestal RMS [ADC]')
    a.legend(fontsize=8)
    a.set_title(f'Y plane (FEU {fy}) control, vs its own coordinate')
    a.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'pedestals_{args.key}.png'), dpi=115)

    print(json.dumps({k: v for k, v in rep.items() if k != 'per_connector'}, indent=1))
    print('\nper connector:')
    print(f'{"conn":>5}{"X range":>16}{"X cns":>8}{"X raw":>8}{"X mean":>9}'
          f'{"Y cns":>8}{"live frac":>11}')
    for k, v in conn.items():
        print(f'{k:5d}{v["x_mm"][0]:7.0f}-{v["x_mm"][1]:<8.0f}{v["x_cns_rms"]:8.2f}'
              f'{v["x_raw_rms"]:8.1f}{v["x_mean"]:9.0f}{v["y_cns_rms"]:8.2f}'
              f'{v["live_frac"]:11.2f}')


if __name__ == '__main__':
    main()
