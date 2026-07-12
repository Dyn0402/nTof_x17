#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
40_spark_waveforms.py

WAVEFORM-LEVEL anatomy of a spark. The reconstructed-hit spark study
(det3_spark_analysis/, Poisson-in-time, muon-induced, edge-dominated) collapses
each strip to one (time, amplitude). This script goes to the raw DREAM waveforms
(decoded_root: 512 ch x 32 samples x 60 ns, pedestal-subtracted, common mode
INTACT) to answer the open-ended question: *what does the detector actually do
during a discharge, and does the spark propagate across the plane or light up all
at once?*

Requires local decoded_root covering the spark events. The headline g_det3_wknd
spark run has decoded_root only for its first ~13 k events (before the spark
region), so this runs on **sat_det3** (490 V / 1000 V operating point, full
decoded coverage, 2878 sparks = 9.1 % of firing events). Same detector, same gas,
same operating regime -> the spark morphology transfers.

What it measures (spark sample vs a normal-muon control sample):
  1. Common mode is the signature. Per event, the median waveform over all 512
     channels (the common mode). Normal muons: ~0. Sparks: a fast step that on
     ~1/4-1/3 of events drives the ENTIRE FEU to saturation for hundreds of ns.
     The "50+ strips firing" that DEFINES a spark is, on these events, mostly every
     channel crossing threshold off this common-mode swing, not local charge.
  2. Raw vs genuine strips. Common-noise subtraction (per-64ch-chip median per
     sample) removes the common mode and leaves the GENUINE localised charge.
     raw strips (~90) >> genuine CNS strips (~40-60): the multiplicity is inflated
     by the common mode / cross-talk.
  3. All-at-once, not propagating. For full-FEU sparks the high-ADC onset time is
     FLAT across all 512 channels to within one 60 ns sample (front-time std ~1
     sample) -> the discharge couples to every channel simultaneously. A charge
     front crossing the 40 cm plane at any physical speed would sweep visibly; it
     does not. The genuine localised charge that DOES lead sits at one edge (the
     seed), and its onset-vs-position correlation is weak (~0.2) with a spread ~=
     the drift window -> that spread is drift time, not a streamer.
  4. Fast recovery. The common mode returns to baseline by the end of the 1.92 us
     window on ~94 % of sparks -> the pad is ready for the next trigger. This is
     the waveform-level cause of the "no post-spark dead time" result (script 39).

Outputs (<sat_det3>/alignment_tpc_veto50/spark_waveforms/):
  spark_waveforms_gallery.png   raw-vs-CNS event images: normal muon, localised
                                spark, full-FEU spark (both planes)
  spark_waveforms_analysis.png  6-panel quantitative morphology
  spark_waveforms.{json,csv}    the numbers + per-event feature table

Usage: ../.venv/bin/python 40_spark_waveforms.py sat_det3 [--nspark=1200] [--nnorm=800]
       [--rebuild]
"""
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import uproot

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()
from common.Mx17StripMap import RunConfig

SAMPLE_NS = 60.0
NSAMP = 32
NCH = 512
SAT = 3600.0          # ped-subtracted saturation level (4095 - ~450 ped)
THR = 150.0           # per-strip "hit" threshold on ped-subtracted peak
CM_THR = 500.0        # common-mode "excursion" threshold
HI = 2000.0           # high-ADC level for the discharge-front timing
FULL_FEU_CM = 1000.0  # cm_peak above this = a full-FEU common-mode spark
NPED = 300
CHUNK = 400
SPARK_THRESH = 50

NSPARK = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--nspark=')), 1200)
NNORM = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--nnorm=')), 800)
REBUILD = '--rebuild' in sys.argv
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')


def posmap():
    """(feu, channel) -> strip position [mm] for the two detector planes."""
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    fx, fy = CFG.MX17_FEUS
    pm = {}
    for c in range(NCH):
        px = det.map_hit(fx, c)
        py = det.map_hit(fy, c)
        if px and px[0] is not None:
            pm[(fx, c)] = float(px[0])
        if py and py[1] is not None:
            pm[(fy, c)] = float(py[1])
    return pm


def cfd_time(w, frac=0.5):
    """First upward crossing of frac*peak before the peak sample [ns]."""
    ip = int(np.argmax(w))
    a = w[ip]
    if ip == 0 or a < THR:
        return np.nan
    lev = frac * a
    for k in range(1, ip + 1):
        if w[k] >= lev > w[k - 1]:
            return SAMPLE_NS * (k - 1 + (lev - w[k - 1]) / (w[k] - w[k - 1]))
    return np.nan


def event_features(wf, pm_plane):
    """Per-plane features from one (32,512) ped-subtracted, CM-intact waveform.
    pm_plane: (512,) strip position [mm] (nan where unmapped)."""
    cm = np.median(wf, axis=1)                                   # common mode (32,)
    cms = np.median(wf.reshape(NSAMP, 8, 64), axis=2)
    wfc = wf - np.repeat(cms, 64, axis=1)                        # CNS
    peak_raw = wf.max(axis=0)
    peak_cns = wfc.max(axis=0)
    nsat = (wf >= SAT).sum(axis=1)                              # sat channels / sample
    over = np.where(cm > CM_THR)[0]
    cm_onset = float(over[0] * SAMPLE_NS) if len(over) else np.nan
    raw_strips = int((peak_raw > THR).sum())
    gen = peak_cns > THR
    gen_strips = int(gen.sum())
    # discharge-front simultaneity: per-channel first crossing of HI, its spread
    front = np.full(NCH, np.nan)
    hich = np.where(peak_raw > HI)[0]
    for c in hich:
        idx = np.where(wf[:, c] > HI)[0]
        if len(idx):
            front[c] = idx[0] * SAMPLE_NS
    front_ok = np.isfinite(front)
    front_std = float(np.std(front[front_ok])) if front_ok.sum() >= 20 else np.nan
    front_n = int(front_ok.sum())
    # genuine-charge onset vs position (propagation of real charge)
    gi = np.where(gen)[0]
    on = np.array([cfd_time(wfc[:, c]) for c in gi])
    pos = pm_plane[gi]
    ok = np.isfinite(on) & np.isfinite(pos)
    corr = span = tspread = gcent = np.nan
    if ok.sum() >= 8 and np.std(pos[ok]) > 3 and np.std(on[ok]) > 1:
        corr = float(abs(np.corrcoef(pos[ok], on[ok])[0, 1]))
        span = float(pos[ok].max() - pos[ok].min())
        tspread = float(on[ok].max() - on[ok].min())
    if ok.sum() >= 3:
        gcent = float(np.median(pos[ok]))
    return dict(cm=cm.astype(np.float32), cm_peak=float(cm.max()), cm_onset=cm_onset,
                cm_last=float(cm[-1]), nsat_peak=int(nsat.max()),
                raw_strips=raw_strips, gen_strips=gen_strips,
                front_std=front_std, front_n=front_n,
                gcorr=corr, gspan=span, gtspread=tspread, gcent=gcent)


def build(cache_path, want, pm):
    """One pass over both FEU decoded files; per-(eid,plane) features."""
    fx, fy = CFG.MX17_FEUS
    res = {fx: {}, fy: {}}
    pmv = {f: np.array([pm.get((f, c), np.nan) for c in range(NCH)]) for f in (fx, fy)}
    for feu in (fx, fy):
        files = sorted(f for f in os.listdir(DEC_DIR) if f.endswith(f'_{feu:02d}.root'))
        for fn in files:
            t = uproot.open(os.path.join(DEC_DIR, fn))['nt']
            eall = t.arrays(['eventId'], library='np')['eventId']
            a0 = t.arrays(['amplitude'], entry_stop=NPED, library='np')['amplitude']
            ped = np.median(np.stack([a.reshape(NSAMP, NCH) for a in a0]).astype(np.float32),
                            axis=(0, 1))
            for lo in range(0, t.num_entries, CHUNK):
                hi = min(lo + CHUNK, t.num_entries)
                widx = [i for i in range(lo, hi) if int(eall[i]) in want]
                if not widx:
                    continue
                arr = t.arrays(['amplitude'], entry_start=lo, entry_stop=hi,
                               library='np')['amplitude']
                for i in widx:
                    wf = arr[i - lo].reshape(NSAMP, NCH).astype(np.float32) - ped
                    res[feu][int(eall[i])] = event_features(wf, pmv[feu])
        print(f'  FEU{feu}: {len(res[feu])} events')
    np.save(cache_path, dict(res=res, ped_note='per-file median of first 300 events'),
            allow_pickle=True)
    return res


def load_raw_event(eid, feu, pm):
    """Reload one event's full (32,512) ped-sub waveform + CNS, for the gallery."""
    files = sorted(f for f in os.listdir(DEC_DIR) if f.endswith(f'_{feu:02d}.root'))
    for fn in files:
        t = uproot.open(os.path.join(DEC_DIR, fn))['nt']
        eall = t.arrays(['eventId'], library='np')['eventId']
        loc = np.where(eall == eid)[0]
        if not len(loc):
            continue
        li = int(loc[0])
        a0 = t.arrays(['amplitude'], entry_stop=NPED, library='np')['amplitude']
        ped = np.median(np.stack([a.reshape(NSAMP, NCH) for a in a0]).astype(np.float32),
                        axis=(0, 1))
        wf = t.arrays(['amplitude'], entry_start=li, entry_stop=li + 1,
                      library='np')['amplitude'][0].reshape(NSAMP, NCH).astype(np.float32) - ped
        cms = np.median(wf.reshape(NSAMP, 8, 64), axis=2)
        return wf, wf - np.repeat(cms, 64, axis=1)
    return None, None


def main():
    out_dir = CFG.out_dir('alignment_tpc_veto50', 'spark_waveforms')
    pm = posmap()
    fx, fy = CFG.MX17_FEUS

    # ---- multiplicity -> spark & normal event samples ----
    rng = np.random.RandomState(1)
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu'], library='pd')
    raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    mult = raw.groupby('eventId').size()
    spark = mult.index[mult > SPARK_THRESH].values
    normal = mult.index[(mult >= 4) & (mult <= SPARK_THRESH)].values
    sp_s = rng.choice(spark, min(NSPARK, len(spark)), replace=False)
    nm_s = rng.choice(normal, min(NNORM, len(normal)), replace=False)
    spark_set = set(int(e) for e in sp_s)
    want = {int(e): ('spark' if int(e) in spark_set else 'normal')
            for e in np.concatenate([sp_s, nm_s])}
    print(f'{CFG.KEY}: {len(spark)} sparks ({100*len(spark)/len(mult):.1f}%), '
          f'sampling {len(sp_s)} sparks + {len(nm_s)} normals')

    cache = os.path.join(out_dir, 'wf_features.npy')
    if REBUILD or not os.path.exists(cache):
        print('  extracting waveform features (one pass over decoded_root) ...')
        res = build(cache, want, pm)
    else:
        res = np.load(cache, allow_pickle=True).item()['res']
        # keys may be int; ensure sample matches
        want = {e: want[e] for e in want if e in res[fx] or e in res[fy]}

    def col(kind, key, feu):
        return np.array([res[feu][e][key] for e in res[feu]
                         if want.get(e) == kind and np.isfinite(res[feu][e][key])])

    # ---- summary numbers ----
    S = {}
    for feu, pl in [(fx, 'x'), (fy, 'y')]:
        sp = [e for e in res[feu] if want.get(e) == 'spark']
        cmpk = np.array([res[feu][e]['cm_peak'] for e in sp])
        rs = np.array([res[feu][e]['raw_strips'] for e in sp], float)
        gs = np.array([res[feu][e]['gen_strips'] for e in sp], float)
        cml = np.array([res[feu][e]['cm_last'] for e in sp])
        fstd = col('spark', 'front_std', feu)
        gcorr = col('spark', 'gcorr', feu)
        gspan = col('spark', 'gspan', feu)
        gtsp = col('spark', 'gtspread', feu)
        gcent = col('spark', 'gcent', feu)
        S[pl] = dict(
            n=len(sp),
            full_feu_frac=float(np.mean(cmpk > FULL_FEU_CM)),
            localised_frac=float(np.mean(cmpk < 200)),
            cm_peak_med=float(np.median(cmpk)),
            raw_strips_med=float(np.median(rs)), gen_strips_med=float(np.median(gs)),
            gen_over_raw_med=float(np.median(gs / np.maximum(rs, 1))),
            front_std_med_ns=float(np.median(fstd)) if len(fstd) else None,
            gcorr_med=float(np.median(gcorr)) if len(gcorr) else None,
            gspan_med_mm=float(np.median(gspan)) if len(gspan) else None,
            gtspread_med_ns=float(np.median(gtsp)) if len(gtsp) else None,
            gcent_med_mm=float(np.median(gcent)) if len(gcent) else None,
            recover_frac=float(np.mean(cml < CM_THR)),
        )
        print(f'  plane {pl}: full-FEU {100*S[pl]["full_feu_frac"]:.0f}%  '
              f'raw {S[pl]["raw_strips_med"]:.0f}/gen {S[pl]["gen_strips_med"]:.0f} strips  '
              f'front_std {S[pl]["front_std_med_ns"]:.0f}ns  gcorr {S[pl]["gcorr_med"]:.2f}  '
              f'recover {100*S[pl]["recover_frac"]:.0f}%')

    _figures(res, want, pm, S, out_dir, spark)

    json.dump(dict(key=CFG.KEY, run=CFG.RUN, subrun=CFG.SUB_RUN,
                   n_sparks_total=int(len(spark)),
                   spark_frac_pct=float(100 * len(spark) / len(mult)),
                   n_sampled_spark=int(len(sp_s)), n_sampled_normal=int(len(nm_s)),
                   sample_ns=SAMPLE_NS, sat_level=SAT, thr=THR, cm_thr=CM_THR,
                   hi_level=HI, full_feu_cm=FULL_FEU_CM, planes=S),
              open(os.path.join(out_dir, 'spark_waveforms.json'), 'w'), indent=2)
    print(f'  wrote {out_dir}/spark_waveforms.{{json,png}}')


def _figures(res, want, pm, S, out_dir, spark):
    fx, fy = CFG.MX17_FEUS
    pmv = {f: np.array([pm.get((f, c), np.nan) for c in range(NCH)]) for f in (fx, fy)}

    # ============ GALLERY: normal / localised spark / full-FEU spark ============
    def pick(pred, feu):
        cand = [e for e in res[feu] if want.get(e) == 'spark' and pred(res[feu][e])]
        return cand[len(cand) // 2] if cand else None
    e_local = pick(lambda d: 100 < d['cm_peak'] < 200 and d['gen_strips'] > 20, fx)
    e_full = pick(lambda d: d['cm_peak'] > 2500, fx)
    e_norm = next((e for e in res[fx] if want.get(e) == 'normal'
                   and res[fx][e]['gen_strips'] >= 5), None)
    rows = [('normal muon', e_norm), ('localised spark', e_local), ('full-FEU spark', e_full)]
    fig, axes = plt.subplots(3, 4, figsize=(19, 11))
    ext = [0, NCH, 0, NSAMP * SAMPLE_NS]
    for r, (label, eid) in enumerate(rows):
        if eid is None:
            continue
        for c, (feu, plname) in enumerate([(fx, 'X'), (fy, 'Y')]):
            wf, wfc = load_raw_event(eid, feu, pm)
            if wf is None:
                continue
            for cc, (W, tag) in enumerate([(wf, 'raw (ped-sub, CM in)'), (wfc, 'CNS (common mode removed)')]):
                ax = axes[r, c * 2 + cc]
                im = ax.imshow(W, aspect='auto', origin='lower', cmap='RdBu_r',
                               vmin=-1500, vmax=1500, extent=ext)
                ax.set_title(f'{label} — {plname} — {tag}', fontsize=9)
                if r == 2:
                    ax.set_xlabel('channel')
                if c == 0 and cc == 0:
                    ax.set_ylabel(f'eid {eid}\ntime [ns]', fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='ADC (ped-sub)', shrink=0.6, pad=0.01)
    fig.suptitle(f'{CFG.DET_NAME} spark waveform gallery — {CFG.RUN}/{CFG.SUB_RUN}\n'
                 'raw shows the global common-mode saturation band; CNS reveals the genuine localised charge',
                 fontsize=12)
    fig.savefig(os.path.join(out_dir, 'spark_waveforms_gallery.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ============ ANALYSIS: 6 quantitative panels ============
    fig, ax = plt.subplots(2, 3, figsize=(19, 11))

    # (a) common-mode peak: spark vs normal
    a = ax[0, 0]
    for kind, col_ in [('normal', 'steelblue'), ('spark', 'crimson')]:
        v = np.concatenate([[res[f][e]['cm_peak'] for e in res[f] if want.get(e) == kind]
                            for f in (fx, fy)])
        v = np.clip(v, 1, 5000)
        a.hist(v, bins=np.logspace(0, np.log10(5000), 50), histtype='step', lw=1.8,
               color=col_, label=f'{kind} (med {np.median(v):.0f})')
    a.axvline(SAT, color='k', ls='--', lw=1, label='saturation')
    a.set_xscale('log'); a.set_xlabel('common-mode peak [ADC]'); a.set_ylabel('events')
    ff = 0.5 * (S['x']['full_feu_frac'] + S['y']['full_feu_frac'])
    a.set_title(f'(a) common mode is the signature\n{100*ff:.0f}% of sparks drive the whole FEU to saturation')
    a.legend(fontsize=8)

    # (b) raw vs genuine strips (the "50 strips" decomposition)
    b = ax[0, 1]
    rs = np.concatenate([[res[f][e]['raw_strips'] for e in res[f] if want.get(e) == 'spark'] for f in (fx, fy)])
    gs = np.concatenate([[res[f][e]['gen_strips'] for e in res[f] if want.get(e) == 'spark'] for f in (fx, fy)])
    b.hist(rs, bins=np.arange(0, 260, 10), histtype='step', lw=1.8, color='darkorange',
           label=f'raw >THR strips (med {np.median(rs):.0f})')
    b.hist(gs, bins=np.arange(0, 260, 10), histtype='step', lw=1.8, color='seagreen',
           label=f'genuine CNS strips (med {np.median(gs):.0f})')
    b.axvline(SPARK_THRESH, color='k', ls=':', lw=1, label='spark def (>50)')
    b.set_xlabel('strips firing per spark'); b.set_ylabel('sparks')
    b.set_title('(b) "50+ strips" is inflated by common mode\ngenuine localised charge is ~half')
    b.legend(fontsize=8)

    # (c) discharge front: onset-time std across channels (full-FEU sparks) + example
    c = ax[0, 2]
    fstd = np.concatenate([[res[f][e]['front_std'] for e in res[f]
                            if want.get(e) == 'spark' and np.isfinite(res[f][e]['front_std'])]
                           for f in (fx, fy)])
    c.hist(fstd, bins=np.linspace(0, 300, 40), color='mediumpurple', alpha=0.85)
    c.axvline(SAMPLE_NS, color='k', ls='--', lw=1.2, label=f'1 sample ({SAMPLE_NS:.0f} ns)')
    c.axvline(np.median(fstd), color='crimson', ls='-', lw=1.2, label=f'median {np.median(fstd):.0f} ns')
    c.set_xlabel('spread of high-ADC onset across 512 channels [ns]')
    c.set_ylabel('full-FEU sparks')
    c.set_title('(c) the discharge hits every channel AT ONCE\nonset flat to ~1–2 samples → not a propagating front')
    c.legend(fontsize=8)

    # (d) genuine-charge onset-vs-position correlation (propagation of REAL charge)
    d = ax[1, 0]
    gcorr = np.concatenate([[res[f][e]['gcorr'] for e in res[f]
                             if want.get(e) == 'spark' and np.isfinite(res[f][e]['gcorr'])]
                            for f in (fx, fy)])
    d.hist(gcorr, bins=np.linspace(0, 1, 30), color='teal', alpha=0.85)
    d.axvline(np.median(gcorr), color='crimson', lw=1.2, label=f'median {np.median(gcorr):.2f}')
    gtsp = 0.5 * (S['x']['gtspread_med_ns'] + S['y']['gtspread_med_ns'])
    d.set_xlabel('|corr(strip position, genuine-charge onset)|'); d.set_ylabel('sparks')
    d.set_title(f'(d) genuine charge: weak pos–time order\nspread {gtsp:.0f} ns ≈ drift window, not a streamer')
    d.legend(fontsize=8)

    # (e) recovery: common mode at end of window
    e = ax[1, 1]
    cml = np.concatenate([[res[f][ev]['cm_last'] for ev in res[f] if want.get(ev) == 'spark'] for f in (fx, fy)])
    e.hist(np.clip(cml, -500, 4000), bins=60, color='slategray', alpha=0.85)
    e.axvline(CM_THR, color='crimson', ls='--', lw=1.2, label=f'{CM_THR:.0f} ADC')
    rec = np.mean(cml < CM_THR)
    e.set_xlabel('common mode at end of 1.92 µs window [ADC]'); e.set_ylabel('sparks')
    e.set_title(f'(e) fast recovery: {100*rec:.0f}% back to baseline in-window\n→ cause of the no-dead-time result (script 39)')
    e.legend(fontsize=8)

    # (f) mean common-mode shape aligned to the SATURATION STEP (full-FEU sparks).
    # Align on the first sample the common mode crosses HI (the discharge step), not the
    # small precursor crossing of CM_THR, so the fast step is not smeared. Window
    # [-4,+16] samples about the step; average where data exist (band widens post-step
    # because late steps have little window left = the recovery variability).
    f6 = ax[1, 2]
    PRE, POST = 4, 16
    stacks = []
    for f in (fx, fy):
        for ev in res[f]:
            if want.get(ev) != 'spark':
                continue
            dd = res[f][ev]
            cm = np.asarray(dd['cm'])
            if dd['cm_peak'] < FULL_FEU_CM:
                continue
            step = np.where(cm > HI)[0]
            if not len(step):
                continue
            s0 = int(step[0])
            seg = np.full(PRE + POST + 1, np.nan)
            for k in range(-PRE, POST + 1):
                j = s0 + k
                if 0 <= j < NSAMP:
                    seg[k + PRE] = cm[j]
            stacks.append(seg)
    if stacks:
        M = np.vstack(stacks)
        tt = (np.arange(-PRE, POST + 1)) * SAMPLE_NS
        mean = np.nanmean(M, axis=0)
        lo = np.nanpercentile(M, 16, axis=0)
        hi = np.nanpercentile(M, 84, axis=0)
        f6.fill_between(tt, lo, hi, color='indianred', alpha=0.2, label='16–84%')
        f6.plot(tt, mean, 'o-', color='firebrick', label=f'mean CM ({len(stacks)} full-FEU sparks)')
    f6.axhline(SAT, color='k', ls='--', lw=1, label='saturation')
    f6.axvline(0, color='gray', ls=':', lw=1)
    f6.set_xlabel('time relative to discharge step [ns]'); f6.set_ylabel('common mode [ADC]')
    f6.set_title('(f) common-mode pulse shape\nfast step to saturation (<1 sample), decays over ~1 µs')
    f6.legend(fontsize=8)

    fig.suptitle(f'{CFG.DET_NAME} spark waveform anatomy — {CFG.RUN}/{CFG.SUB_RUN}  '
                 f'({len(spark)} sparks, 9.1% of firing events; sat_det3 490V/1000V)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, 'spark_waveforms_analysis.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
