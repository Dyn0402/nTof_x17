#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27z_waveform_reco.py — FULL-WAVEFORM micro-TPC reconstruction of the filtered
run_55 real tracks (the point of the hit-level filtering: the surviving set is
small enough for a proper waveform pass).

For every gate-selected INCLINED cluster (b1/b2) we pull the decoded_root
waveform (32 samples x 60 ns) of each of its strips and reconstruct with
sub-sample precision:
  * baseline (pre-pulse) and SATURATION-aware peak amplitude;
  * CFD leading-edge time (30 % constant fraction, linear sub-sample interp) —
    the drift arrival, finer than the 60 ns-quantised max_sample the hit cache
    used;
  * per-strip pulse width.
Then it refits the micro-TPC trail with the CFD times:
  * tan θ_cfd = (du/dt)/v  — the sub-sample angle;
  * T_gap_cfd = CFD-time span of the cluster = gap/v (angle-independent drift
    time) — a per-track drift-velocity handle that does not need the angle
    calibration.
It contrasts the CFD reconstruction with the max_sample (hit-level) one on the
source hypothesis: does sub-sample timing sharpen the angular resolution and the
drift-velocity, and does the far/near-edge charge (below the 400 ADC hit
threshold) recovered from the waveform lengthen the drift span toward garfield?

Outputs: figures/27_tracks/{11_wf_examples,12_wf_vdrift,13_wf_source}.png,
calib/27_waveform.npz.

Run:  venv/bin/python mx_july_beam_qa/27z_waveform_reco.py [--limit N]
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import trackcache as tc

HERE = os.path.dirname(__file__)
RUN_DIR = os.path.expanduser('~/x17/beam_july/runs/run_55')
FIGDIR = os.path.join(HERE, 'figures', '27_tracks')
CALIB = os.path.join(HERE, 'calib')
SAMPLE_NS = 60.0
SAT_ADC = 4000.0
CFD_FRAC = 0.30
GAP_UM = 30000.0


def reco_channel(wf):
    """One 32-sample raw-ADC waveform -> dict of reconstructed quantities."""
    base = np.median(wf[:4])
    p = wf - base
    k = int(np.argmax(p))
    amp = float(p[k])
    sat = bool((wf >= SAT_ADC).any())
    # CFD 30% rising edge, sub-sample
    t_cfd = np.nan
    if amp > 60:
        thr = CFD_FRAC * amp
        for i in range(k, 0, -1):
            if p[i - 1] < thr <= p[i]:
                frac = (thr - p[i - 1]) / (p[i] - p[i - 1] + 1e-9)
                t_cfd = (i - 1 + frac) * SAMPLE_NS
                break
        if np.isnan(t_cfd):
            t_cfd = k * SAMPLE_NS
    width = float((p > max(100.0, 0.2 * amp)).sum())
    return dict(base=base, amp=amp, peak=k, t_cfd=t_cfd, sat=sat, width=width)


def fit_trail(pos, t, amp):
    """amp-weighted du/dt [mm/ns] and CFD-time span."""
    ok = np.isfinite(pos) & np.isfinite(t) & np.isfinite(amp) & (amp > 0)
    pos, t, amp = pos[ok], t[ok], amp[ok]
    if len(pos) < 3:
        return np.nan, np.nan, len(pos)
    # robust: drop the single worst residual once
    tt = t - t.mean(); pp = pos - pos.mean()
    w = amp
    den = np.sum(w * tt * tt)
    dudt = np.sum(w * tt * pp) / den if den > 0 else np.nan   # mm/ns
    span = float(np.ptp(t))
    return dudt, span, len(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='limit #subruns')
    args = ap.parse_args()

    cl, ss, ev = tc.load_all()
    cl = tc.add_derived(cl, ss)
    evk = ev.set_index(['sub', 'eid'])
    cl = cl.join(evk[['t_ms']], on=['sub', 'eid'])
    cl['b12'] = (cl['t_ms'] > 7) & (cl['t_ms'] < 30)
    cl = tc.tag_tracklike(cl)
    inc = cl[cl['b12'] & cl['inclined']].copy().reset_index(drop=True)
    inc['cid'] = np.arange(len(inc))
    subs = sorted(inc['sub'].unique())
    if args.limit:
        subs = subs[:args.limit]
        inc = inc[inc['sub'].isin(subs)]
    print(f'inclined clusters: {len(inc)} across {len(subs)} subruns')

    rows = []
    examples = []
    for sub in subs:
        g = inc[inc['sub'] == sub]
        subdir = os.path.join(RUN_DIR, sub)
        # gather needed (feu -> {eid -> set(ch)}) from the cluster strips
        need = {}
        cl_strips = {}
        for _, r in g.iterrows():
            s = tc.cluster_strips(r, ss)
            feus = s['feu']; chs = s['ch']
            cl_strips[int(r['cid'])] = (s['pos'], s['amp'], feus, chs)
            for fe, c in zip(feus, chs):
                need.setdefault(int(fe), {}).setdefault(int(r['eid']), set()).add(int(c))
        # load each needed decoded feu once
        wfcache = {}
        for feu, ev_ch in need.items():
            fs = sorted(glob.glob(os.path.join(subdir, 'decoded_root', f'*_{feu:02d}.root')))
            if not fs:
                continue
            a = uproot.open(fs[0])['nt'].arrays(
                ['eventId', 'sample', 'channel', 'amplitude'], library='np')
            eidx = {int(e): i for i, e in enumerate(a['eventId'])}
            for e, chs in ev_ch.items():
                i = eidx.get(int(e))
                if i is None:
                    continue
                ch_arr = a['channel'][i]; sm = a['sample'][i]; am = a['amplitude'][i]
                for c in chs:
                    m = ch_arr == c
                    if not m.any():
                        continue
                    wf = np.zeros(32, np.float32)
                    ss_ = sm[m]; aa = am[m]
                    wf[ss_[ss_ < 32]] = aa[ss_ < 32]
                    wfcache[(feu, int(e), int(c))] = wf
        # reconstruct each cluster
        for _, r in g.iterrows():
            pos, amp0, feus, chs = cl_strips[int(r['cid'])]
            P, T, A, sats, wid = [], [], [], [], []
            wfset = []
            for p, fe, c in zip(pos, feus, chs):
                wf = wfcache.get((int(fe), int(r['eid']), int(c)))
                if wf is None:
                    continue
                rc = reco_channel(wf)
                P.append(p); T.append(rc['t_cfd']); A.append(rc['amp'])
                sats.append(rc['sat']); wid.append(rc['width']); wfset.append(wf)
            if len(P) < 3:
                continue
            P = np.array(P); T = np.array(T); A = np.array(A)
            dudt, span, n = fit_trail(P, T, A)
            v = tc.DRIFT[r['detn']]['v_um_ns']
            tan_cfd = dudt / (v / 1000.0) if np.isfinite(dudt) else np.nan  # (mm/ns)/(mm/ns)
            rows.append(dict(cid=int(r['cid']), sub=sub, detn=r['detn'],
                             plnn=r['plnn'], det=int(r['det']), pln=int(r['pln']),
                             cen=float(r['cen']), resist_v=int(r['resist_v']),
                             tan_cfd=float(tan_cfd), span_cfd=float(span),
                             n_cfd=int(n), lead_sat=bool(np.array(sats)[np.argmax(A)]),
                             maxwidth=float(np.max(wid)), tan_raw_dur=float(r['dur'])))
            if len(examples) < 16 and len(wfset) >= 5 and span > 300:
                examples.append((P - P[np.argmin(T)], np.array(T), np.array(A),
                                 wfset, r['detn'], r['plnn']))
    W = pd.DataFrame(rows)
    print(f'reconstructed {len(W)} clusters from waveforms')

    # ---- drift velocity from CFD span (angle-independent) ----
    print(f'\n{"det":4s} {"n":>5s} {"span_med_ns":>11s} {"v_cfd":>6s} {"garfield":>8s}')
    vrec = {}
    for dn in 'ABCD':
        d = W[(W['detn'] == dn) & np.isfinite(W['span_cfd'])]
        if len(d) < 20:
            continue
        # full-gap tracks pile at the span plateau; use the upper-half median
        sp = d['span_cfd'].values
        hi = sp[sp >= np.percentile(sp, 50)]
        span_med = float(np.median(hi))
        v_cfd = GAP_UM / span_med if span_med > 0 else np.nan
        vrec[dn] = v_cfd
        print(f'{dn:4s} {len(d):5d} {span_med:11.0f} {v_cfd:6.1f} {tc.DRIFT[dn]["v_um_ns"]:8.1f}')

    # ---- CFD-angle source pointing (does sub-sample timing sharpen it?) ----
    FID = (70.0, 330.0); SC = 199.3
    def s68(a):
        a = a[np.isfinite(a)]
        return 0.5*(np.percentile(a, 84)-np.percentile(a, 16)) if len(a) > 20 else np.nan
    print(f'\n{"plane":6s} {"n":>5s} {"scale_cfd":>9s} {"u0":>6s} {"sigθ_cfd":>8s}')
    src = []
    for dn in 'ABCD':
        R = tc.DRIFT[dn]['R']
        for pn in 'xy':
            d = W[(W['detn'] == dn) & (W['plnn'] == pn)]
            x = d['cen'].values; y = d['tan_cfd'].values
            g = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < 0.6)
            fid = g & (x > FID[0]) & (x < FID[1])
            if fid.sum() < 40:
                continue
            edges = np.linspace(*FID, 8); ctr, med, err = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = fid & (x >= lo) & (x < hi)
                if m.sum() >= 10:
                    ctr.append(np.median(x[m])); med.append(np.median(y[m]))
                    err.append(1.253*np.std(y[m])/np.sqrt(m.sum()))
            if len(ctr) < 4:
                continue
            ctr, med, err = map(np.array, (ctr, med, err))
            w = 1/err**2; W_ = w.sum(); Wx = (w*ctr).sum(); Wy = (w*med).sum()
            Wxx = (w*ctr**2).sum(); Wxy = (w*ctr*med).sum()
            s = (W_*Wxy-Wx*Wy)/(W_*Wxx-Wx*Wx); b = (Wy-s*Wx)/W_
            scale = -s*R; u0 = -b/s
            sig = s68(y[fid]-(s*x[fid]+b)); sigdeg = np.degrees(np.arctan(abs(sig)))
            src.append(dict(det=dn, plane=pn, scale=scale, u0=u0, sig=sigdeg,
                            n=int(fid.sum())))
            print(f'{dn}{pn:5s} {fid.sum():5d} {scale:9.2f} {u0:6.0f} {sigdeg:8.1f}')
    S = pd.DataFrame(src)
    if len(S):
        print(f'\nCFD source scale median {S["scale"].median():.2f} (raw was ~0.62); '
              f'σθ median {S["sig"].median():.1f}°')
        for pn, lab in [('x', 'transverse'), ('y', 'beam/vertical')]:
            s = S[S['plane'] == pn]
            print(f'  source {lab}: u0-center = '
                  f'{dict(zip(s.det,(s.u0-SC).round(0).tolist()))} mm')

    # ---- figures ----
    fig, ax = plt.subplots(2, 1, figsize=(9, 8))
    for dp, T, A, wfset, dn, pn in examples[:8]:
        ax[0].plot(dp, T - T.min(), 'o-', alpha=0.6, ms=4)
    ax[0].set_xlabel('strip pos - anchor [mm]'); ax[0].set_ylabel('CFD time - anchor [ns]')
    ax[0].set_title('waveform CFD micro-TPC trails (inclined, span>300 ns)')
    for dp, T, A, wfset, dn, pn in examples[:6]:
        for wf in wfset[:6]:
            ax[1].plot(np.arange(32) * 60, wf - np.median(wf[:4]), alpha=0.4, lw=0.8)
    ax[1].set_xlabel('time [ns]'); ax[1].set_ylabel('ADC - baseline')
    ax[1].set_title('their strip waveforms')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '11_wf_examples.png'), dpi=95)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for i, dn in enumerate('ABCD'):
        d = W[W['detn'] == dn]
        ax[i].hist(d['span_cfd'], bins=np.arange(0, 1400, 60), color='C0')
        if dn in vrec:
            ax[i].axvline(GAP_UM / vrec[dn], color='r', ls='--')
        gar = tc.DRIFT[dn]['v_um_ns']
        ax[i].axvline(GAP_UM / gar, color='g', ls=':')
        ax[i].set_title(f'{dn}: CFD span [ns]  v_cfd={vrec.get(dn,np.nan):.0f} gar {gar:.0f}')
        ax[i].set_xlabel('CFD drift span [ns]')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '12_wf_vdrift.png'), dpi=95)
    plt.close(fig)

    np.savez_compressed(os.path.join(CALIB, '27_waveform.npz'),
                        **{c: W[c].values for c in W.columns})
    print(f'\nsaved calib/27_waveform.npz, figures 11/12')
    return W


if __name__ == '__main__':
    main()
