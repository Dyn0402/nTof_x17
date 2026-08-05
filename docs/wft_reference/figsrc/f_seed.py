#!/usr/bin/env python3
"""
Part V figures — from a plane to an event: seeding from hits, cutting the
window, choosing between candidate clusters, and using the X/Y coincidence.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import wftdoc as K
from wftdoc import C, save

from wft import model as wm, seed as ws, reco as wr, io as wio

CAL = None


def setup():
    global CAL
    CAL = K.install()
    return CAL


def load_hits(cfg, max_files=1):
    import uproot
    import pandas as pd
    files = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                   if f.endswith('.root') and '_datrun_' in f)[:max_files]
    df = uproot.concatenate(
        [f'{cfg.combined_hits_dir}{f}:hits' for f in files],
        expressions=['eventId', 'feu', 'channel', 'amplitude', 'significance'],
        library='pd')
    return df[df['feu'].isin(cfg.MX17_FEUS)]


# ------------------------------------------------------------- 1. seeding
def fig_seeding(cfg, hits, pos_maps):
    """The significance floor and the spatial clustering, on a real event with
    coherent noise in it."""
    feu = cfg.MX17_FEU_X
    pm = pos_maps[feu]
    g = hits[hits['feu'] == feu]
    # an event where the floor actually does something
    cand = None
    for eid, gp in g.groupby('eventId'):
        if not (8 <= len(gp) <= 40):
            continue
        sig = gp['significance'].to_numpy()
        ch_ = gp['channel'].to_numpy().astype(int)
        kept = sig >= ws.SIG_REL_FLOOR * sig.max()
        if kept.sum() >= len(gp) - 2 or kept.sum() < 4:
            continue
        s = ws.seed_candidates(pm[ch_[kept]], ch_[kept],
                               gp['amplitude'].to_numpy()[kept],
                               n_candidates=ws.N_CANDIDATES)
        if len(s) >= 2:
            cand = (int(eid), gp)
            break
    if cand is None:
        print('[seed] no illustrative event found'); return None
    eid, gp = cand
    ch = gp['channel'].to_numpy().astype(int)
    pos = pm[ch]
    amp = gp['amplitude'].to_numpy()
    sig = gp['significance'].to_numpy()
    keep = sig >= ws.SIG_REL_FLOOR * sig.max()
    seeds = ws.seed_candidates(pos[keep], ch[keep], amp[keep],
                               n_candidates=ws.N_CANDIDATES)
    print(f'[seed] event {eid}: {len(gp)} hits -> {keep.sum()} past the floor '
          f'-> {len(seeds)} candidate clusters '
          f'{[s.n_strips for s in seeds]}')

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    ax = axs[0]
    ax.scatter(pos, sig, s=26, color=C['grey'], label='all hits')
    ax.scatter(pos[keep], sig[keep], s=26, color=C['blue'],
               label='past the floor')
    ax.axhline(ws.SIG_REL_FLOOR * sig.max(), color=C['red'], ls='--',
               label=f'{ws.SIG_REL_FLOOR:.0%} of the strongest strip')
    ax.set_yscale('log')
    ax.set_xlabel('strip position [mm]'); ax.set_ylabel('significance')
    ax.set_title(f'event {eid}: the relative significance floor', loc='left')
    ax.legend(fontsize=7.5)

    ax = axs[1]
    ax.scatter(pos[keep], amp[keep], s=30, color=C['blue'])
    p = np.sort(pos[keep])
    brk = np.where(np.diff(p) > ws.GAP_THRESHOLD_MM)[0]
    for b in brk:
        ax.axvspan(p[b], p[b + 1], color=C['red'], alpha=0.12)
    ax.set_xlabel('strip position [mm]'); ax.set_ylabel('amplitude [ADC]')
    ax.set_title(f'spatial clustering: gaps > {ws.GAP_THRESHOLD_MM:.0f} mm '
                 'split clusters', loc='left')

    ax = axs[2]
    for i, s in enumerate(seeds):
        sp = pm[s.channels]
        ax.scatter(sp, np.full_like(sp, i), s=34,
                   color=[C['blue'], C['orange'], C['red']][i % 3],
                   label=f'candidate {i+1}: {s.n_strips} strips, '
                         f'{s.amp_sum:.0f} ADC')
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f'#{i+1}' for i in range(len(seeds))])
    ax.set_xlabel('strip position [mm]')
    ax.set_title(f'{len(seeds)} candidates handed to the fit\n'
                 '(ranked by strip count — the fit re-ranks them)', loc='left')
    ax.legend(fontsize=7)
    save(fig, 'seeding')
    return eid


# ------------------------------------------------------- 2. window padding
def fig_window_pad():
    """Why the window must extend at least 2 strips past the seed: that is how
    far the sharing kernel reaches, and the model has to see where the shared
    charge went."""
    evs = K.calib_events()
    e = evs[1663]
    P = K.trim_window(e['x'], pad=6)
    W = np.asarray(P['W'], float)
    pos = np.asarray(P['pos'], float)
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    amp = W.max(axis=1)
    frac = amp / amp.sum()
    core = amp > 0.10 * amp.max()
    ci = np.where(core)[0]

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axs[0]
    ax.bar(pos, 100 * frac, width=0.7, color=[C['blue'] if c else C['orange']
                                              for c in core])
    ax.set_yscale('log')
    ax.set_xlabel('strip position [mm]')
    ax.set_ylabel('% of the window\'s peak amplitude sum')
    ax.set_title('blue: the seed cluster.  orange: strips the seed misses\n'
                 'but the sharing kernel puts charge on', loc='left')

    # the population answer: refit 120 events at each pad width
    ax = axs[1]
    pads = np.array([0, 1, 2, 3, 5])
    sig = []
    for p in pads:
        d = []
        for eid in sorted(evs)[:120]:
            ee = evs[eid]
            if 'x' not in ee or abs(ee['tan_x']) < 0.10:
                continue
            Pp = K.trim_window(ee['x'], pad=6)
            Wp = np.asarray(Pp['W'], float)
            pp = np.asarray(Pp['pos'], float)
            a = Wp.max(axis=1)
            c = np.where(a > 0.10 * a.max())[0]
            lo = max(0, c.min() - p); hi = min(len(pp) - 1, c.max() + p)
            Q = dict(W=Wp[lo:hi + 1], pos=pp[lo:hi + 1],
                     noise=np.asarray(Pp['noise'])[lo:hi + 1],
                     ch=np.asarray(Pp['ch'])[lo:hi + 1])
            if Q['W'].shape[1] != wm.NSAMP:
                wm.set_nsamp(Q['W'].shape[1])
            f = wr.fit_plane(Q, 'x', CAL)
            if f is None:
                continue
            d.append(np.degrees(np.arctan(f.tan_theta))
                     - np.degrees(np.arctan(ee['tan_x'])))
        d = np.array(d)
        sig.append(1.4826 * np.median(np.abs(d - np.median(d))))
        print(f'[seed] pad {p}: n={len(d)} angle sigma {sig[-1]:.3f} deg')
    ax.plot(pads, sig, 'o-', color=C['orange'], ms=7)
    ax.axvline(2, color=C['red'], ls=':', label='kernel reach (±2 strips)')
    ax.set_xlabel('pad_strips')
    ax.set_ylabel(r'angle $\sigma$ vs reference [deg]')
    ax.set_title('measured on 61 planes: the fit needs to see\n'
                 'where the shared charge went', loc='left')
    ax.legend(fontsize=7.5)
    save(fig, 'window_pad')


# ------------------------------------------------- 3. candidate selection
def fig_candidates(cfg, hits, pos_maps):
    """A real event with two competing clusters, both fitted, and the rule that
    picks between them."""
    feu = cfg.MX17_FEU_X
    pm = pos_maps[feu]
    g = hits[hits['feu'] == feu]
    target = None
    for eid, gp in g.groupby('eventId'):
        ch = gp['channel'].to_numpy().astype(int)
        sig = gp['significance'].to_numpy()
        keep = sig >= ws.SIG_REL_FLOOR * sig.max()
        if keep.sum() < 6:
            continue
        s = ws.seed_candidates(pm[ch[keep]], ch[keep],
                               gp['amplitude'].to_numpy()[keep],
                               n_candidates=3)
        if len(s) >= 2 and s[1].n_strips >= 3:
            target = (int(eid), s)
            break
    if target is None:
        print('[seed] no multi-cluster event in this file'); return
    eid, seeds = target

    fx = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu)
    rdr = wio.FeuReader(fx[0])
    W = None
    for e_, ftst, wfm in rdr.iter_events({eid}):
        W = wfm
    if W is None:
        print('[seed] waveforms not in the first file'); return

    fits, wins = [], []
    for s in seeds:
        win = wio.extract_window(W, rdr.noise, pm, s.channels, 3)
        if win is None:
            continue
        Q = dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch)
        if np.asarray(Q['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(Q['W']).shape[1])
        f = wr.fit_plane(Q, 'x', CAL)
        if f is None:
            continue
        pl, dchi = wr._candidate_score(Q, 'x', f)
        fits.append((f, pl, dchi, s))
        wins.append(Q)
    if len(fits) < 2:
        print('[seed] fewer than two fittable candidates'); return
    best = max(range(len(fits)), key=lambda i: (int(fits[i][1]), fits[i][2]))
    print(f'[seed] event {eid} candidates: ' +
          ', '.join(f'{s.n_strips} strips u_end={f.q_uend:.0f} '
                    f'tan={f.tan_theta:+.3f} plausible={pl} '
                    f'dchi2={dchi:.3g}'
                    for f, pl, dchi, s in fits) +
          f'  -> picks #{best+1}')

    n = len(fits)
    fig, axs = plt.subplots(1, n + 1, figsize=(3.3 * (n + 1), 3.5))
    for i, (Q, (f, pl, dchi, s)) in enumerate(zip(wins, fits)):
        ax = axs[i]
        Wq = np.asarray(Q['W'], float)
        pq = np.asarray(Q['pos'], float)
        ax.imshow(Wq, aspect='auto', origin='lower', cmap='magma',
                  extent=[0, wm.NSAMP * .06, pq[0] - .39, pq[-1] + .39],
                  interpolation='nearest')
        ax.set_title(f'candidate {i+1}{"  ← chosen" if i == best else ""}\n'
                     f'{s.n_strips} strips, tan {f.tan_theta:+.3f}\n'
                     f'$u_{{end}}$ {f.q_uend:.0f} ns, plausible={pl}',
                     loc='left', fontsize=8.5)
        ax.set_xlabel('time [µs]')
        if i == 0:
            ax.set_ylabel('strip position [mm]')
        ax.grid(False)
    ax = axs[-1]
    lab = [f'#{i+1}' for i in range(n)]
    ax.bar(lab, [f[2] for f in fits],
           color=[C['green'] if i == best else C['grey'] for i in range(n)])
    ax.set_ylabel(r'$\chi^2_{\rm null} - \chi^2_{\rm fit}$')
    ax.set_title('the rule: plausible first,\nthen best explained charge',
                 loc='left')
    save(fig, 'candidates')


# --------------------------------------------------------------- 4. dt_xy
def fig_dt_xy():
    cal = CAL
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ks = sorted(cal.dt_xy)
    ax.bar([str(k) for k in ks], [cal.dt_xy[k] for k in ks], width=0.5,
           color=C['blue'])
    for k in ks:
        ax.text(str(k), cal.dt_xy[k], f'{cal.dt_xy[k]:+.1f} ns',
                ha='center', va='bottom' if cal.dt_xy[k] > 0 else 'top',
                fontsize=9, color=K.CHROME)
    ax.axhline(0, color=K.CHROME, lw=0.8)
    ax.set_xlabel('fine-timestamp difference  ftst(X) − ftst(Y)')
    ax.set_ylabel(r'measured $t_{0x} - t_{0y}$ [ns]')
    ax.set_title('the two FEUs are not synchronous — the offset is measured\n'
                 'per ftst difference and used by the pair selector',
                 loc='left')
    ax.set_ylim(min(cal.dt_xy.values()) * 1.4, 4)
    print('[seed] dt_xy:', cal.dt_xy)
    save(fig, 'dt_xy')


def main():
    setup()
    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(K.RUN_KEY)
    pos_maps = wio.strip_position_map(cfg)
    hits = load_hits(cfg)
    print(f'[seed] {len(hits):,} hits loaded')
    fig_seeding(cfg, hits, pos_maps)
    fig_window_pad()
    fig_candidates(cfg, hits, pos_maps)
    fig_dt_xy()


if __name__ == '__main__':
    main()
