#!/usr/bin/env python3
"""
quick.py -- the quick pass over the flash_tail summaries: average waveforms and
baseline recovery per scintillator, ALWAYS split by beam-pulse intensity.

    python -m ntof_scint_recovery.quick [--summaries DIR]

Reads every ``summary_<run>.npz`` (flash_summary.py, made on lxplus and copied
to ``out_dir('summaries')``) and writes ``out_dir('quick')``:

``curves.npz``          per (detector, class): median / 16 / 84 % baseline shift and
                        mean signal against time since the flash, pooled over
                        channels and bunches; also per channel
``recovery.csv``        per (detector, channel or 'all', class): the recovery times
``recovery_by_run.csv`` the same per run, for the run-to-run spread
``bunches.csv``         one row per bunch: run, class, PulseIntensity, PKUP amplitude

Intensity class: from PulseIntensity when the run was processed (< 1e11 'none',
< 6.5e12 parasitic, else dedicated).  For the runs n_TOF never processed, from the
PKUP flash amplitude in the product, cut at the midpoint of the two classes'
medians measured on the processed bunches (they separate cleanly: ~420 vs ~840 mV).

Only LIVE channel-bunches enter (flash peak > LIVE_MV): some runs had whole
families unpowered.

Pooled curves and recovery.csv use the production configuration only (runs >=
PRODUCTION_FROM, see EPOCHS); recovery_by_run.csv has every run.

Recovery time at a tolerance X: the first time after which the median baseline
shift stays inside +-X mV out to 10 ms.  X = 3 mV (a tenth of the lowest wall
trigger threshold) and 1 mV (~1.5x the rms noise).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from ntof_scint_recovery.common import out_dir

DETS = ('WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD', 'LIQA', 'LIQB', 'LIQC', 'LIQD')
CLASSES = ('par', 'ded')
TOLS = (3.0, 1.0)
#: the only runs with the SiPM-wall divert (blanking) off: the walls see the whole flash
DIVERT_OFF = (224356, 224357, 224358, 224359, 224360, 224464, 224466)
#: Commissioning steps, found in this data (recovery_by_run: steps at these run
#: numbers on every detector of a family; cause not identified in the logs we have):
#: 224355 walls 0.25 -> 0.55 ms; 224401 walls -> 0.83 ms, plastics 12 -> 31 us,
#: liquids appear; 224450 (14 July) plastics 31 -> 114 us, liquids 20 -> 8 us.
#: From 224450 the recovery is flat to the end of the campaign, so the headline
#: numbers use that configuration only -- pooling epochs would quote a time that
#: belongs to none of them.  First run of each epoch:
EPOCHS = (('commissioning 1', 0), ('commissioning 2', 224355), ('commissioning 3', 224401),
          ('production', 224450))
PRODUCTION_FROM = 224450
#: divert-off runs compared only with divert-on runs of their own epoch
DIVERT_PAIRS = {'e2': ((224356, 224357, 224358, 224359, 224360), (224355, 224361, 224400)),
                'prod': ((224464, 224466), (224450, 10**9))}
T_MAX_US = 1e4
MIN_BUNCHES = 5
#: a channel saw the flash if its mean signal peaks above this in the first 60 us
LIVE_MV = 100.0


def load(sdir: Path) -> dict:
    keys = ('run', 'raw_index', 'bunch', 'ppp', 'has_official', 'pk_amp', 'shift', 'mean', 'nblk', 'lin',
            'noise_mV')
    acc = {k: [] for k in keys}
    te = le = ch = None
    for f in sorted(sdir.glob('summary_*.npz')):
        z = np.load(f)
        if 'shift' not in z.files or not len(z['bunch']):
            continue
        te, le, ch = z['t_edges_us'], z['lin_edges_us'], z['channels']
        for k in keys:
            acc[k].append(z[k])
    D = {k: np.concatenate(v) for k, v in acc.items()}
    D.update(t_edges_us=te, lin_edges_us=le, channels=ch)
    return D


def classify(D: dict) -> np.ndarray:
    ppp, pk = D['ppp'].astype(float), D['pk_amp'].astype(float)
    cls = np.full(len(ppp), 'none', object)
    ok = np.isfinite(ppp)
    cls[ok & (ppp >= 1e11) & (ppp < 6.5e12)] = 'par'
    cls[ok & (ppp >= 6.5e12)] = 'ded'
    mp = np.nanmedian(pk[cls == 'par']) if (cls == 'par').any() else 420.
    md = np.nanmedian(pk[cls == 'ded']) if (cls == 'ded').any() else 840.
    cut = 0.5 * (mp + md)
    nb = ~ok
    cls[nb & (pk >= 0.3 * mp) & (pk < cut)] = 'par'
    cls[nb & (pk >= cut)] = 'ded'
    D['pk_cut'] = cut
    return cls


def t_recover(tc, y, tol, tmax=T_MAX_US):
    m = (tc < tmax) & np.isfinite(y)
    t, v = tc[m], y[m]
    if not len(t):
        return np.nan
    bad = np.nonzero(np.abs(v) > tol)[0]
    if not len(bad):
        return float(t[0])
    return float(t[bad[-1] + 1]) if bad[-1] + 1 < len(t) else np.nan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--summaries', type=Path, default=out_dir('summaries'))
    a = ap.parse_args(argv)
    D = load(a.summaries)
    cls = classify(D)
    te = D['t_edges_us']
    tc = np.sqrt(te[:-1] * te[1:])
    chans = [tuple(c) for c in D['channels']]
    S = D['shift'].astype(np.float32)               # (bunch, ch, t)
    M = D['mean'].astype(np.float32)
    L = D['lin'].astype(np.float32)
    # a channel-bunch counts only if the channel was LIVE: it saw the flash.  Whole
    # families were unpowered in some runs (walls in 224377/405/528/529, PSS+LIQ in
    # 224455/468: flash peaks of 0-5 mV), and a dead channel "recovers" at once.
    has_sc = np.isfinite(S).any(axis=2) & (np.nanmax(L, axis=2) > LIVE_MV)
    out = out_dir('quick')
    curves = {'tc_us': tc, 't_edges_us': te, 'lin_edges_us': D['lin_edges_us'], 'channels': np.array(chans)}
    rows, rows_run = [], []
    runs = D['run']
    for di, det in enumerate(DETS):
        cis = [i for i, c in enumerate(chans) if c[0] == di]
        for c in CLASSES:
            sel = (cls == c) & (runs >= PRODUCTION_FROM)
            # pooled over channels: stack (bunch*channel, t)
            stk = np.concatenate([S[sel][:, i][has_sc[sel][:, i]] for i in cis]) if cis else np.zeros((0, len(tc)))
            mstk = np.concatenate([M[sel][:, i][has_sc[sel][:, i]] for i in cis]) if cis else np.zeros((0, len(tc)))
            lstk = np.concatenate([L[sel][:, i][has_sc[sel][:, i]] for i in cis]) if cis else np.zeros((0, L.shape[2]))
            n = len(stk)
            key = f'{det}_{c}'
            if n < MIN_BUNCHES:
                continue
            with np.errstate(all='ignore'):
                q = np.nanpercentile(stk, [16, 50, 84], axis=0)
            curves[key + '_shift'] = q
            curves[key + '_mean'] = np.nanmean(mstk, axis=0)
            curves[key + '_lin'] = np.nanmean(lstk, axis=0)
            curves[key + '_n'] = n
            r = dict(det=det, ch='all', cls=c, n=n)
            for tol in TOLS:
                r[f't{tol:g}mV_us'] = t_recover(tc, q[1], tol)
            rows.append(r)
            for i in cis:
                s = S[sel][:, i][has_sc[sel][:, i]]
                if len(s) < MIN_BUNCHES:
                    continue
                with np.errstate(all='ignore'):
                    med = np.nanmedian(s, axis=0)
                curves[f'{det}{chans[i][1]}_{c}_shift50'] = med
                r = dict(det=det, ch=chans[i][1], cls=c, n=len(s))
                for tol in TOLS:
                    r[f't{tol:g}mV_us'] = t_recover(tc, med, tol)
                rows.append(r)
            # per run, pooled over the detector's channels -- EVERY epoch
            for run in np.unique(runs[cls == c]):
                rs = (cls == c) & (runs == run)
                s = np.concatenate([S[rs][:, i][has_sc[rs][:, i]] for i in cis])
                if len(s) < MIN_BUNCHES:
                    continue
                with np.errstate(all='ignore'):
                    med = np.nanmedian(s, axis=0)
                r = dict(run=int(run), det=det, cls=c, n=len(s))
                for tol in TOLS:
                    r[f't{tol:g}mV_us'] = t_recover(tc, med, tol)
                rows_run.append(r)
    # the walls with the divert off, against divert-on runs of the SAME epoch
    for ep, (offr, (lo_on, *rest)) in DIVERT_PAIRS.items():
        if ep == 'e2':
            on_mask = np.isin(runs, (224355,)) | ((runs >= 224361) & (runs < 224401))
        else:
            on_mask = (runs >= PRODUCTION_FROM) & ~np.isin(runs, DIVERT_OFF)
        off_mask = np.isin(runs, offr)
        for di, det in enumerate(DETS[:4]):
            cis = [i for i, c in enumerate(chans) if c[0] == di]
            for c in CLASSES:
                for tag, sub in (('divoff', off_mask), ('divon', on_mask)):
                    sel = (cls == c) & sub
                    stk = np.concatenate([S[sel][:, i][has_sc[sel][:, i]] for i in cis])
                    if len(stk) < MIN_BUNCHES:
                        continue
                    with np.errstate(all='ignore'):
                        q = np.nanpercentile(stk, [16, 50, 84], axis=0)
                    lstk = np.concatenate([L[sel][:, i][has_sc[sel][:, i]] for i in cis])
                    k = f'{det}_{c}_{tag}_{ep}'
                    curves[k + '_shift'] = q
                    curves[k + '_lin'] = np.nanmean(lstk, axis=0)
                    curves[k + '_n'] = len(stk)
                    rows.append(dict(det=det, ch=f'{tag}_{ep}', cls=c, n=len(stk),
                                     **{f't{t:g}mV_us': t_recover(tc, q[1], t) for t in TOLS}))
    np.savez_compressed(out / 'curves.npz', **curves)
    for name, rr in (('recovery.csv', rows), ('recovery_by_run.csv', rows_run)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
    with open(out / 'bunches.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'raw_index', 'bunch', 'cls', 'ppp', 'has_official', 'pk_amp_mV', 'n_live'])
        for i in range(len(cls)):
            w.writerow([int(D['run'][i]), int(D['raw_index'][i]), int(D['bunch'][i]), cls[i],
                        float(D['ppp'][i]), int(D['has_official'][i]), round(float(D['pk_amp'][i]), 2),
                        int(has_sc[i].sum())])
    nc = {c: int((cls == c).sum()) for c in ('par', 'ded', 'none')}
    print(f'{len(cls)} bunches {nc}, {len(np.unique(runs))} runs, PKUP cut {D["pk_cut"]:.0f} mV -> {out}')
    for r in rows:
        if r['ch'] == 'all':
            print(f"  {r['det']} {r['cls']}: n={r['n']:5d}  3 mV {r['t3mV_us']:9.1f} us   1 mV {r['t1mV_us']:9.1f} us")
    return 0


if __name__ == '__main__':
    sys.exit(main())
