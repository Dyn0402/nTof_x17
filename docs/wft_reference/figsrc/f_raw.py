#!/usr/bin/env python3
"""
Part I figures — what the DAQ actually records, and the three corrections that
turn it into something fittable: pedestal, common-mode, noise.

All from `sat_det3` decoded_root, read exactly the way `wft/io.py` reads it.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

import wftdoc as K
from wftdoc import C, save


def load_raw(feu=7, n_ped=300, n_show=4000, run_key=None):
    """Raw (un-pedestal-subtracted) amplitude blocks straight from the tree,
    plus the pedestal / CNS / noise that wft.io derives from them."""
    import uproot
    from wft import io as wio
    from qa_config import get_config

    cfg = get_config(run_key) if run_key else K.cfg()
    files = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu)
    path = files[0]
    tree = uproot.open(path)['nt']
    n = min(n_show, tree.num_entries)
    arr = tree.arrays(['eventId', 'amplitude', 'ftst'], entry_stop=n, library='np')
    lens = np.array([len(a) // 512 for a in arr['amplitude']])
    ns = int(np.bincount(lens).argmax())
    raw = np.stack([a.reshape(ns, 512) for a, l in zip(arr['amplitude'], lens)
                    if l == ns]).astype(np.float32)          # (nev, nsamp, 512)
    eids = np.array([e for e, l in zip(arr['eventId'], lens) if l == ns])

    ped = np.median(raw[:n_ped], axis=(0, 1))                # (512,)
    sub = raw - ped[None, None, :]
    nblk = 512 // 64
    cms = np.median(sub.reshape(len(sub), ns, nblk, 64), axis=3)   # (nev, nsamp, 8)
    cor = sub - np.repeat(cms, 64, axis=2)

    noise_raw = 1.4826 * np.median(np.abs(sub[:n_ped]), axis=(0, 1))
    noise_cns = 1.4826 * np.median(np.abs(cor[:n_ped]), axis=(0, 1))
    return dict(path=path, eids=eids, raw=raw, sub=sub, cor=cor, cms=cms,
                ped=ped, noise_raw=noise_raw, noise_cns=noise_cns, ns=ns)


def brightest(d, lo=800, hi=3300, nstrip_min=6):
    """An event with a healthy, unsaturated, multi-strip cluster."""
    amp = d['cor'].max(axis=1)                               # (nev, 512)
    peak = amp.max(axis=1)
    nst = (amp > 100).sum(axis=1)
    ok = np.where((peak > lo) & (peak < hi) & (nst >= nstrip_min) &
                  (nst < 30))[0]
    if len(ok) == 0:
        ok = np.argsort(peak)[::-1][:1]
    return int(ok[len(ok) // 2])


# --------------------------------------------------------------------- plots
def fig_raw_event(d, i):
    ns = d['ns']
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6),
                            gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axs[0]
    im = ax.imshow(d['raw'][i].T, aspect='auto', origin='lower', cmap='magma',
                   extent=[0, ns * 60 / 1000, 0, 512], interpolation='nearest')
    ax.set_xlabel('time within the DAQ window [µs]')
    ax.set_ylabel('FEU channel')
    ax.set_title('raw ADC, all 512 channels of one FEU, one event', loc='left')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('raw ADC', color=K.CHROME)
    cb.ax.tick_params(colors=K.CHROME)
    cb.outline.set_edgecolor(K.CHROME)

    ax = axs[1]
    ch_hot = int(np.argmax(d['cor'][i].max(axis=0)))
    for ch, col, lab in ((ch_hot, C['blue'], f'ch {ch_hot} (on the track)'),
                         ((ch_hot + 60) % 512, C['grey'],
                          f'ch {(ch_hot+60)%512} (quiet)')):
        ax.plot(np.arange(ns) * 0.06, d['raw'][i][:, ch], color=col, label=lab,
                marker='o', ms=2.5)
    ax.set_xlabel('time [µs]')
    ax.set_ylabel('raw ADC')
    ax.set_title('two channels, raw', loc='left')
    ax.legend()
    save(fig, 'raw_event')


def fig_pedestal(d):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2),
                            gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axs[0]
    ax.plot(d['ped'], color=C['blue'], lw=1.0)
    for b in range(1, 8):
        ax.axvline(b * 64, color=C['grey'], lw=0.6, alpha=0.5)
    ax.set_xlabel('FEU channel')
    ax.set_ylabel('pedestal [ADC]')
    ax.set_title('per-channel pedestal — median over the first 300 events '
                 '(grey lines: 64-channel Dream chip blocks)', loc='left')
    ax.set_xlim(0, 512)

    ax = axs[1]
    ax.hist(d['ped'], bins=60, color=C['blue'], alpha=0.85)
    ax.set_xlabel('pedestal [ADC]')
    ax.set_ylabel('channels')
    ax.set_title(f'spread: {d["ped"].min():.0f}–{d["ped"].max():.0f} ADC',
                 loc='left')
    save(fig, 'pedestal')


def fig_cns(d, i):
    ns = d['ns']
    t = np.arange(ns) * 0.06
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.3))

    ax = axs[0]
    for b in range(8):
        ax.plot(t, d['cms'][i][:, b], lw=1.2, alpha=0.9,
                color=plt.cm.viridis(b / 7))
    ax.set_xlabel('time [µs]')
    ax.set_ylabel('block common mode [ADC]')
    ax.set_title('common mode of each 64-channel block, one event', loc='left')

    ax = axs[1]
    ch = int(np.argmax(d['cor'][i].max(axis=0)))
    quiet = (ch + 100) % 512
    ax.plot(t, d['sub'][i][:, quiet], color=C['grey'], marker='o', ms=2.5,
            label='pedestal-subtracted')
    ax.plot(t, d['cor'][i][:, quiet], color=C['blue'], marker='o', ms=2.5,
            label='+ common-mode subtracted')
    ax.axhline(0, color=K.CHROME, lw=0.7)
    ax.set_xlabel('time [µs]')
    ax.set_ylabel('ADC')
    ax.set_title(f'a channel with no signal (ch {quiet})', loc='left')
    ax.legend()

    ax = axs[2]
    ax.plot(t, d['sub'][i][:, ch], color=C['grey'], marker='o', ms=2.5,
            label='pedestal-subtracted')
    ax.plot(t, d['cor'][i][:, ch], color=C['blue'], marker='o', ms=2.5,
            label='+ common-mode subtracted')
    ax.axhline(0, color=K.CHROME, lw=0.7)
    ax.set_xlabel('time [µs]')
    ax.set_ylabel('ADC')
    ax.set_title(f'a channel on the track (ch {ch})', loc='left')
    ax.legend()
    save(fig, 'common_mode')


def fig_noise(d, d2=None):
    """det3's FEU 7 is a quiet board; det4's FEU 6 is the pathological one the
    recipe was written for. Both are shown — CNS is cheap insurance on the
    first and indispensable on the second."""
    n = 2 if d2 is None else 3
    fig, axs = plt.subplots(1, n, figsize=(4.4 * n, 3.2))
    ax = axs[0]
    ax.plot(d['noise_raw'], color=C['grey'], lw=1.0, label='before CNS')
    ax.plot(d['noise_cns'], color=C['blue'], lw=1.0, label='after CNS')
    ax.set_xlabel('FEU channel')
    ax.set_ylabel(r'noise $\sigma$ [ADC]  (1.4826 × MAD)')
    ax.set_title('det3 FEU 7 (X): a quiet board', loc='left')
    ax.set_xlim(0, 512)
    ax.legend()

    ax = axs[1]
    ax.hist(d['noise_raw'], bins=np.linspace(0, 25, 50), color=C['grey'],
            alpha=0.8, label=f'before  (med {np.median(d["noise_raw"]):.1f})')
    ax.hist(d['noise_cns'], bins=np.linspace(0, 25, 50), color=C['blue'],
            alpha=0.8, label=f'after  (med {np.median(d["noise_cns"]):.1f})')
    ax.set_xlabel(r'$\sigma$ [ADC]')
    ax.set_ylabel('channels')
    ax.legend()
    ax.set_title('det3: 8.9 → 7.4 ADC, a modest gain', loc='left')

    if d2 is not None:
        ax = axs[2]
        ax.plot(d2['noise_raw'], color=C['grey'], lw=1.0, label='before CNS')
        ax.plot(d2['noise_cns'], color=C['blue'], lw=1.0, label='after CNS')
        ax.set_yscale('log')
        ax.set_xlabel('FEU channel')
        ax.set_ylabel(r'noise $\sigma$ [ADC]')
        ax.set_xlim(0, 512)
        ax.legend()
        ax.set_title(f'det4 FEU 6: {np.median(d2["noise_raw"]):.0f} → '
                     f'{np.median(d2["noise_cns"]):.1f} ADC — '
                     'the case the recipe exists for', loc='left')
    save(fig, 'noise')


def fig_window(d, i):
    """The fit's actual input: a small window of strips × samples."""
    from wft import io as wio
    cfg = K.cfg()
    pos_maps = wio.strip_position_map(cfg)
    pm = pos_maps[7]
    amp = d['cor'][i].max(axis=0)
    ch_hot = int(np.argmax(amp))
    # strips ordered by position around the hot one
    order = np.argsort(pm)
    order = order[np.isfinite(pm[order])]
    rank = {int(c): k for k, c in enumerate(order)}
    r0 = rank[ch_hot]
    sel = order[max(0, r0 - 8):r0 + 9]
    W = d['cor'][i][:, sel].T                        # (nstrip, nsamp)
    pos = pm[sel]
    ns = d['ns']
    t = np.arange(ns) * 0.06

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0),
                            gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axs[0]
    im = ax.imshow(W, aspect='auto', origin='lower', cmap='magma',
                   extent=[0, ns * 0.06, pos[0] - 0.39, pos[-1] + 0.39],
                   interpolation='nearest')
    ax.set_xlabel('time [µs]')
    ax.set_ylabel('strip position [mm]')
    ax.set_title('the fit window: strips × samples', loc='left')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('ADC', color=K.CHROME)
    cb.ax.tick_params(colors=K.CHROME)
    cb.outline.set_edgecolor(K.CHROME)

    ax = axs[1]
    off = 0.9 * np.nanmax(W)
    for j in range(len(sel)):
        ax.plot(t, W[j] + j * off * 0.35, lw=1.3,
                color=plt.cm.viridis(j / max(len(sel) - 1, 1)))
        ax.text(t[-1] + 0.03, j * off * 0.35, f'{pos[j]:.1f}',
                fontsize=7, color=K.CHROME, va='center')
    ax.set_xlabel('time [µs]')
    ax.set_yticks([])
    ax.set_ylabel('strip (offset), labelled by position [mm]')
    ax.set_title('the same window as a waveform stack — the ladder is visible',
                 loc='left')
    ax.set_xlim(0, t[-1] + 0.35)
    save(fig, 'fit_window')
    return sel, pos, W


def main():
    print('[raw] reading decoded_root ...')
    d = load_raw()
    print(f'[raw] {d["path"]}')
    print(f'[raw] {len(d["raw"])} events, {d["ns"]} samples')
    i = brightest(d)
    print(f'[raw] display event index {i} (eventId {d["eids"][i]})')
    fig_raw_event(d, i)
    fig_pedestal(d)
    fig_cns(d, i)
    try:
        d4 = load_raw(feu=6, n_show=400, run_key='g_det4')
    except Exception as exc:
        print('[raw] det4 comparison unavailable:', exc)
        d4 = None
    fig_noise(d, d4)
    fig_window(d, i)
    print(f'[raw] noise median before/after CNS: '
          f'{np.median(d["noise_raw"]):.1f} / {np.median(d["noise_cns"]):.2f} ADC')


if __name__ == '__main__':
    main()
