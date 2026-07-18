"""19b_triples_figs.py — figures + tables from the 19 triples cache:

figures/19_triples/
  pss_mip_spectra.png   per-PMT triple-tagged plastic spectra per HV step (mV),
                        double sideband-subtracted, smoothed-mode markers
  pss_mip_vs_v.png      plastic MIP mode vs V per PMT, power-law fit, nominal-V
                        extrapolation; comparison with 12b coincident-median n
  wall_in_triples.png   wall spectrum in triples vs in wall-plastic pairs
  liq_spectra.png       LIQ amplitude spectra per arm (inclusive, double-sub)
  liq_position_map.png  median LIQ amp over (wall group x bar) x vertical bin
  liq_timing.png        WAL-LIQ dt per wall channel + PSS-LIQ dt per bar

Also writes calib/liq_offsets_<run>.json (per wall-channel and per-bar LIQ dt
peaks — new file, old consumers untouched) and prints the MIP mode tables.

Usage: python 19b_triples_figs.py [run_stem]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224489'
OUT = BASE / 'figures' / '19_triples'
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(BASE / 'cache' / f'19_triples_{RUN_STEM}.npz')
FAC = json.loads((BASE / 'calib' / f'adc_to_mv_{RUN_STEM}.json').read_text())['factors']
AE, LAE, LRE = d['amp_edges'], d['lamp_edges'], d['lr_edges']
CEN, LCEN = np.sqrt(AE[:-1] * AE[1:]), np.sqrt(LAE[:-1] * LAE[1:])
SWP, SLQ = float(d['s_wp']), float(d['s_lq'])
LABELS = list(d['step_labels'])
DTC = 0.5 * (d['dt_edges'][:-1] + d['dt_edges'][1:])
# scan-step voltages parsed from labels 's1_1600V' etc.
VOLT = {l: (int(l.split('_')[1][:-1]) if l.startswith('s') and l[1].isdigit() else None)
        for l in LABELS}
SCAN_STEPS = [l for l in LABELS if VOLT[l]]
ARM_COLORS = dict(zip('ABCD', plt.cm.tab10.colors))
NOMINAL_V = {'PSSA1': 1325, 'PSSA2': 1275, 'PSSB1': 1325, 'PSSB2': 1300,
             'PSSC1': 1300, 'PSSC2': 1300, 'PSSD1': 1300, 'PSSD2': 1300}

KERN = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
KERN /= KERN.sum()

RB = 6                                     # display rebin: 300 -> 50 log bins
CEN_RB = np.exp(np.log(CEN)[:len(CEN) // RB * RB].reshape(-1, RB).mean(axis=1))
DLOG = np.log10(AE[-1] / AE[0]) / (len(AE) - 1)   # fine log-bin width [dex]


def rebin(spec, rb=RB):
    n = len(spec) // rb
    return spec[:n * rb].reshape(n, rb).sum(axis=1)


def shift_log(spec, shift_dex):
    """Translate a counts-per-log-bin spectrum by shift_dex (gain scaling is a
    pure translation in log amplitude; linear interp on fractional bins)."""
    x = np.arange(len(spec), dtype=float)
    return np.interp(x - shift_dex / DLOG, x, spec, left=0, right=0)


def dsub(h):
    """(wp, liq, nbins) -> double sideband-subtracted spectrum."""
    return h[0, 0] - SLQ * h[0, 1] - SWP * h[1, 0] + SWP * SLQ * h[1, 1]


def smoothed_mode(sub, cen, min_amp=150, min_n=400):
    sm = np.convolve(sub, KERN, mode='same')
    m = cen > min_amp
    return float(cen[m][np.argmax(sm[m])]) if sub.sum() > min_n else np.nan


def pss_sub(i_step, ai, b):
    return dsub(d['pss_amp'][i_step, ai, :, :, b])


# ---------------------------------------------------------------- MIP spectra
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
cmap = plt.cm.viridis
modes = {}                       # (PMT, label) -> mode ADC
for ai, st in enumerate('ABCD'):
    for b in range(2):
        ax = axes[b][ai]
        pmt = f'PSS{st}{b + 1}'
        f_mv = FAC[f'PSS{st}'][str(b + 1)]
        for l in SCAN_STEPS + ['end_nominal']:
            i = LABELS.index(l)
            sub = pss_sub(i, ai, b)
            v = VOLT[l]
            modes[(pmt, l)] = smoothed_mode(sub, CEN)   # numbers: fine hist
            # display only a readable subset; all steps enter the aligned fig
            if v is not None and v not in (1600, 1500, 1400, 1300):
                continue
            col = cmap((v - 1250) / 400) if v else 'crimson'
            ax.plot(CEN_RB * f_mv, rebin(sub), color=col, lw=1.4 if v else 2.4,
                    label=(f'{v} V' if v else 'nominal'), alpha=0.9)
        ax.set_xscale('log')
        ax.set_title(pmt)
        ax.grid(alpha=0.3, which='both')
        ax.axhline(0, color='k', lw=0.5)
        if b == 1:
            ax.set_xlabel('plastic amplitude [mV]')
        if ai == 0:
            ax.set_ylabel('triples / bin (dbl sideband-sub)')
axes[0][0].legend(fontsize=7, ncol=2)
fig.suptitle(f'{RUN_STEM}: triple-tagged (WALxPSSxLIQ) plastic spectra per HV '
             'step (rebinned x6)', y=0.995)
fig.tight_layout()
fig.savefig(OUT / 'pss_mip_spectra.png', dpi=140)
plt.close(fig)

# ------------------------------------------------------- coincident-median n
# gain exponent per PMT from the 12 cache (coincident sideband-subtracted
# medians, ~100x the triple statistics) — used to transport the high-V MIP
# modes to nominal HV instead of refitting n from a handful of noisy modes
d12 = np.load(BASE / 'cache' / f'12_hvscan_{RUN_STEM}.npz')
CEN12 = np.sqrt(d12['amp_edges'][:-1] * d12['amp_edges'][1:])
SB12 = float(d12['sb_scale'])
L12 = list(d12['step_labels'])
V12 = d12['step_volts']


def n_coinc(ai, b):
    v, m = [], []
    for i, l in enumerate(L12):
        if V12[i] <= 0:
            continue
        sub = np.clip(d12['pss_mip'][i, ai, 0, b]
                      - SB12 * d12['pss_mip'][i, ai, 1, b], 0, None)
        c = np.cumsum(sub)
        if c[-1] > 200:
            v.append(V12[i])
            m.append(CEN12[np.searchsorted(c, c[-1] / 2)])
    return np.polyfit(np.log(v), np.log(m), 1)[0]


# ---------------------------------------------------------------- mode vs V
V_MIN_FIT = 1400          # below this the MIP sits in the threshold turn-on
fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
fit_n = {}
mip_nominal = {}
print('\nPlastic MIP mode [ADC] vs step (triple-tagged, dbl-subtracted):')
hdr = ' '.join(f'{l.split("_")[1]:>7s}' for l in SCAN_STEPS) + '  nominal'
print(f'{"PMT":8s} {hdr}')
for ai, st in enumerate('ABCD'):
    for b in range(2):
        pmt = f'PSS{st}{b + 1}'
        f_mv = FAC[f'PSS{st}'][str(b + 1)]
        ax = axes[b][ai]
        v = np.array([VOLT[l] for l in SCAN_STEPS], float)
        m = np.array([modes[(pmt, l)] for l in SCAN_STEPS], float)
        ok = np.isfinite(m)
        row = ' '.join(f'{x:7.0f}' if np.isfinite(x) else f'{"-":>7s}' for x in m)
        print(f'{pmt:8s} {row}  {modes[(pmt, "end_nominal")]:7.0f}')
        ax.plot(v[ok], m[ok] * f_mv, 'o', color=ARM_COLORS[st], ms=7)
        use = ok & (v >= V_MIN_FIT)
        if use.sum() >= 2:
            n_g = n_coinc(ai, b)
            fit_n[pmt] = n_g
            vn = NOMINAL_V[pmt]
            # transport each trusted mode to nominal V with the 12-cache n
            trans = m[use] * (vn / v[use]) ** n_g
            mip_nom = float(np.exp(np.mean(np.log(trans))))
            spread = float(np.std(np.log(trans)))
            mip_nominal[pmt] = (mip_nom, spread)
            vv = np.geomspace(1150, 1650, 40)
            ax.plot(vv, mip_nom * (vv / vn) ** n_g * f_mv, '-', lw=1,
                    color=ARM_COLORS[st], alpha=0.7)
            ax.plot(vn, mip_nom * f_mv, '*', ms=14, color='crimson',
                    label=f'nominal: {mip_nom * f_mv:.1f} mV '
                          f'(±{100 * spread:.0f}%)')
            mode_nom = modes[(pmt, 'end_nominal')]
            if np.isfinite(mode_nom):
                ax.plot(vn, mode_nom * f_mv, 'x', ms=9, color='gray',
                        label='measured nominal')
            ax.text(0.05, 0.9, f'{pmt}  n = {n_g:.2f} (coinc-med)',
                    transform=ax.transAxes)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7, loc='lower right')
        if b == 1:
            ax.set_xlabel('PMT bias [V]')
        if ai == 0:
            ax.set_ylabel('plastic MIP mode [mV]')
fig.suptitle(f'{RUN_STEM}: plastic MIP mode vs HV (power-law fits, '
             'star = extrapolation to nominal)', y=0.995)
fig.tight_layout()
fig.savefig(OUT / 'pss_mip_vs_v.png', dpi=140)
plt.close(fig)

# ------------------------------------------------- gain-aligned stacked MIP
# The decisive existence test: scaling each step's spectrum to its
# nominal-V-equivalent amplitude (translation in log amp by n*log10(Vnom/V))
# must ALIGN a physical peak across steps while the fixed-ADC threshold edge
# moves. Summing the aligned steps then gives one high-statistics spectrum.
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
for ai, st in enumerate('ABCD'):
    for b in range(2):
        pmt = f'PSS{st}{b + 1}'
        f_mv = FAC[f'PSS{st}'][str(b + 1)]
        ax = axes[b][ai]
        if pmt not in fit_n:
            continue
        n_g, vn = fit_n[pmt], NOMINAL_V[pmt]
        total = np.zeros(len(CEN))
        for l in SCAN_STEPS:
            v = VOLT[l]
            if v < 1300:                    # deep-threshold steps add noise only
                continue
            sub = pss_sub(LABELS.index(l), ai, b)
            al = shift_log(sub, n_g * np.log10(vn / v))
            total += al
            ax.plot(CEN_RB * f_mv, rebin(al), lw=0.8, alpha=0.65,
                    color=plt.cm.viridis((v - 1250) / 400), label=f'{v} V')
        ax.plot(CEN_RB * f_mv, rebin(total) / 3, lw=2.4, color='crimson',
                label='sum / 3')
        # dashed marker: the adopted linear-space MPV from 19d when available
        # (the log-binned mode plotted here sits above the linear MPV — see 19d)
        cal_p = BASE / 'calib' / f'pss_mip_calib_{RUN_STEM}.json'
        try:
            mip = json.loads(cal_p.read_text())['pmts'][pmt]['mip_mv'] / f_mv
        except (FileNotFoundError, KeyError):
            mip = mip_nominal[pmt][0]
        ax.axvline(mip * f_mv, color='k', lw=1.2, ls='--',
                   label=f'MIP (MPV) {mip * f_mv:.1f} mV')
        ax.set_xscale('log')
        ax.set_title(f'{pmt}  (n = {n_g:.2f})')
        ax.grid(alpha=0.3, which='both')
        ax.axhline(0, color='k', lw=0.5)
        if b == 1:
            ax.set_xlabel('nominal-V-equivalent amplitude [mV]')
        if ai == 0:
            ax.set_ylabel('aligned triples / bin')
        if ai == 0 and b == 0:
            ax.legend(fontsize=6, ncol=2)
fig.suptitle(f'{RUN_STEM}: gain-aligned triple spectra — steps pile up at a '
             'common peak (dashed: adopted MIP) while the threshold edge moves',
             y=0.995)
fig.tight_layout()
fig.savefig(OUT / 'pss_mip_aligned.png', dpi=140)
plt.close(fig)

# ------------------------------------------------------- wall MIP in triples
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True)
for ai, st in enumerate('ABCD'):
    ax = axes[ai]
    f_mv = float(np.mean([FAC[f'WAL{st}'][str(c + 1)] for c in range(8)]))
    h = d['wal_amp'][:, ai].sum(axis=0)                  # sum steps
    trip = dsub(h)
    # wall-plastic-pair-only spectrum: liq columns summed = wp sig - swp*side
    pair = (h[0].sum(axis=0) - SWP * h[1].sum(axis=0))
    for spec, lbl, col in ((pair / max(pair.sum(), 1), 'wall-plastic pairs', 'gray'),
                           (trip / max(trip.sum(), 1), 'triples (+LIQ)', ARM_COLORS[st])):
        ax.plot(CEN * f_mv, np.convolve(spec, KERN, 'same'), color=col, label=lbl)
    ax.set_xscale('log')
    ax.set_title(f'WAL{st}')
    ax.grid(alpha=0.3, which='both')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=8)
    ax.set_xlabel('wall amplitude [mV]')
fig.suptitle('wall spectrum: adding the LIQ tag does not move the wall MIP '
             '(normalized, smoothed)')
fig.tight_layout()
fig.savefig(OUT / 'wall_in_triples.png', dpi=140)
plt.close(fig)

# ------------------------------------------------------------- LIQ spectra
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True)
print('\nLIQ triple-tagged spectra (inclusive): median / mode [ADC and mV]')
for ai, st in enumerate('ABCD'):
    ax = axes[ai]
    f_mv = FAC[f'LIQ{st}']['1']
    sub = dsub(d['liq_amp'][:, ai].sum(axis=0))
    sm = np.convolve(sub, KERN, 'same')
    ax.plot(LCEN * f_mv, sm, color=ARM_COLORS[st])
    c = np.cumsum(np.clip(sub, 0, None))
    med = LCEN[np.searchsorted(c, c[-1] / 2)] if c[-1] > 100 else np.nan
    mode = smoothed_mode(sub, LCEN, min_amp=60)
    print(f'LIQ{st}: median {med:6.0f} ADC = {med * f_mv:5.1f} mV   '
          f'mode {mode:6.0f} ADC = {mode * f_mv:5.1f} mV   n={sub.sum():.0f}')
    ax.axvline(med * f_mv, color='k', lw=0.8, ls='--', label=f'median {med * f_mv:.1f} mV')
    ax.set_xscale('log')
    ax.set_title(f'LIQ{st}')
    ax.grid(alpha=0.3, which='both')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=8)
    ax.set_xlabel('LIQ amplitude [mV]')
fig.suptitle('LIQ amplitude in triples (double sideband-subtracted, smoothed)')
fig.tight_layout()
fig.savefig(OUT / 'liq_spectra.png', dpi=140)
plt.close(fig)

# --------------------------------------------------------- position map (T6)
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
print('\nLIQ median amp [mV] vs position (rows: vertical ln(At/Ab) bins '
      'bottom-heavy first, i.e. bottom->top; cols: group1L..group4L | '
      'group1R..group4R):')
pos_med = np.full((4, 8, len(LRE) - 1), np.nan)
for ai, st in enumerate('ABCD'):
    f_mv = FAC[f'LIQ{st}']['1']
    for b in range(2):
        for g in range(4):
            for vb in range(len(LRE) - 1):
                sub = dsub(d['liq_pos'][ai, :, :, g, b, vb])
                c = np.cumsum(np.clip(sub, 0, None))
                if c[-1] > 150:
                    pos_med[ai, b * 4 + g, vb] = LCEN[np.searchsorted(c, c[-1] / 2)] * f_mv
    ax = axes[ai]
    im = ax.imshow(pos_med[ai].T, origin='upper', aspect='auto', cmap='viridis')
    ax.set_title(f'LIQ{st}')
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'{b}g{g + 1}' for b in 'LR' for g in range(4)], fontsize=7)
    ax.set_yticks(range(len(LRE) - 1))
    ax.set_yticklabels([f'{LRE[i]:.1f}..{LRE[i + 1]:.1f}' for i in range(len(LRE) - 1)],
                       fontsize=7)
    ax.set_xlabel('plastic bar x wall group (horiz.)')
    if ai == 0:
        ax.set_ylabel('ln(A_top/A_bot) bin (negative = hit near bottom)')
    fig.colorbar(im, ax=ax, label='median LIQ amp [mV]')
    print(f'LIQ{st}:')
    for vb in range(len(LRE) - 1):
        print('   ' + ' '.join(f'{pos_med[ai, h, vb]:6.1f}'
                               if np.isfinite(pos_med[ai, h, vb]) else f'{"-":>6s}'
                               for h in range(8)))
fig.suptitle('LIQ gain vs position (triple-tagged, dbl-subtracted median)')
fig.tight_layout()
fig.savefig(OUT / 'liq_position_map.png', dpi=140)
plt.close(fig)

# ------------------------------------------------------------- timing (T7)
def peak_of(hh, cen):
    base = np.median(hh)
    sub = hh - base
    ipk = np.argmax(sub)
    win = np.abs(cen - cen[ipk]) <= 10
    w = np.clip(sub[win], 0, None)
    if w.sum() < 100:
        return np.nan, np.nan, 0.0
    mu = np.average(cen[win], weights=w)
    rms = np.sqrt(np.average((cen[win] - mu) ** 2, weights=w))
    return float(mu), float(rms), float(w.sum())


fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
liq_off = {}
print('\nLIQ timing peaks:')
for ai, st in enumerate('ABCD'):
    ax = axes[0][ai]
    liq_off[st] = {}
    for c in range(8):
        h = d['walliq_dt'][ai, c]
        mu, rms, n = peak_of(h, DTC)
        liq_off[st][f'WAL{st}{c + 1}_LIQ{st}1'] = {
            'offset_ns': round(mu, 2) if np.isfinite(mu) else None,
            'rms_ns': round(rms, 2) if np.isfinite(rms) else None,
            'n_excess': int(n)}
        ax.plot(DTC, h / max(h.max(), 1), lw=1, label=f'ch{c + 1} {mu:+.1f}')
    ax.set_title(f'WAL{st} - LIQ{st}')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    ax = axes[1][ai]
    for b in range(2):
        h = d['pssliq_dt'][ai, b]
        mu, rms, n = peak_of(h, DTC)
        liq_off[st][f'PSS{st}{b + 1}_LIQ{st}1'] = {
            'offset_ns': round(mu, 2) if np.isfinite(mu) else None,
            'rms_ns': round(rms, 2) if np.isfinite(rms) else None,
            'n_excess': int(n)}
        ax.plot(DTC, h / max(h.max(), 1), lw=1.2, label=f'bar{b + 1} {mu:+.1f}')
    ax.set_title(f'PSS{st} - LIQ{st}')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlabel('dt = t_det - t_liq [ns]')
fig.suptitle('LIQ timing: dt peaks per wall channel (top) and plastic bar '
             '(bottom), late hits')
fig.tight_layout()
fig.savefig(OUT / 'liq_timing.png', dpi=140)
plt.close(fig)

meta = {'run': RUN_STEM,
        'definition': 'dt = t_det - t_liq; subtract offset_ns to center at 0',
        'sample': 'late (tof - tflash > 0.1 ms) LIQ hits, good bunches, '
                  'no third-detector requirement',
        'stations': liq_off}
out = BASE / 'calib' / f'liq_offsets_{RUN_STEM}.json'
out.write_text(json.dumps(meta, indent=2))
print(f'LIQ offsets -> {out}')
for st in 'ABCD':
    for k, v in liq_off[st].items():
        print(f'  {k:18s} {str(v["offset_ns"]):>8s} rms {str(v["rms_ns"]):>6s} '
              f'n {v["n_excess"]}')

print(f'\nFigures -> {OUT}')
if mip_nominal:
    print('\nPlastic MIP at nominal HV (high-V modes transported with '
          'coincident-median n):')
    print(f'{"PMT":8s} {"V_nom":>6s} {"n":>6s} {"MIP[ADC]":>9s} {"MIP[mV]":>8s} '
          f'{"spread":>7s}')
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            pmt = f'PSS{st}{b + 1}'
            if pmt not in mip_nominal:
                continue
            f_mv = FAC[f'PSS{st}'][str(b + 1)]
            mip, spr = mip_nominal[pmt]
            print(f'{pmt:8s} {NOMINAL_V[pmt]:6d} {fit_n[pmt]:6.2f} {mip:9.0f} '
                  f'{mip * f_mv:8.1f} {100 * spr:6.0f}%')
