"""
15_pipeline_figs.py — Step-by-step visualisation of the sideband-subtraction
pipeline that turns raw wall-plastic coincidences into clean SiPM-wall MIP peaks.

Everything is reconstructed from existing caches (no reprocessing of the 13 GB
run file):
  * calibrated dt' = t_wall - t_pss - offset is rebuilt from the per-(wall ch x
    bar) raw dt histograms in cache/03_offsets_hists_<run>.npz, shifting each
    channel-pair by its stored offset so the true peak sits at 0 ns.
  * the signal-window and sideband-window amplitude spectra are read straight
    from cache/07_mip_<run>.npz (h[0]=|dt'|<=8 ns, h[1]=+20..+120 ns).

Pipeline, per wall channel:
    signal window   |dt'| <= 8 ns          -> amp spectrum  = MIP + accidentals
    sideband window +20..+120 ns           -> amp spectrum  = accidentals only
    true coincidences = signal - (16/100) * sideband        -> MIP peak

Representative clean channel: WALB6.  Null contrast: WALA2 (arm A, accidental
dominated -> subtraction leaves no peak).

Usage: python 15_pipeline_figs.py [run_stem]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '15_pipeline'
OUT.mkdir(parents=True, exist_ok=True)

W_SIG = 8.0
SB_LO, SB_HI = 20.0, 120.0
SB_SCALE = (2 * W_SIG) / (SB_HI - SB_LO)          # = 0.16

# palette (matches 05_sideband_diagram)
C_SIG = '#fb923c'      # signal window / on-peak
C_SIGL = '#c2410c'     # signal dark
C_SIDE = '#3b82f6'     # sideband
C_SIDEL = '#1d4ed8'
C_TRUE = '#c2410c'     # subtracted true coincidences
C_LINE = '#1e3a5f'
C_ACC = '#9ca3af'      # accidental level

d3 = np.load(BASE / 'cache' / f'03_offsets_hists_{RUN_STEM}.npz')
DT_EDGES = d3['dt_edges']
DT_CEN = 0.5 * (DT_EDGES[:-1] + DT_EDGES[1:])
OFFS = json.loads((BASE / 'calib' / f'time_offsets_{RUN_STEM}.json').read_text())['stations']

d7 = np.load(BASE / 'cache' / f'07_mip_{RUN_STEM}.npz')
AMP_EDGES = d7['amp_edges']
AMP_CEN = np.sqrt(AMP_EDGES[:-1] * AMP_EDGES[1:])
FAC = json.loads((BASE / 'calib' / f'adc_to_mv_{RUN_STEM}.json').read_text())['factors']

KERN = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
KERN /= KERN.sum()


def dtcal(st, wc):
    """Calibrated dt' distribution for one wall channel (summed over both bars)."""
    h = d3[st]
    out = np.zeros(len(DT_CEN))
    for pc in range(2):
        off = OFFS[st][f'WAL{st}{wc + 1}_PSS{st}{pc + 1}']['offset_ns']
        out += np.roll(h[wc, pc], -int(round(off)))
    return out


def amp_spectra(st, wc):
    """(signal, sideband, scaled_sideband, subtracted) amp spectra, wall channel."""
    h = d7[f'{st}_wal']
    sig, sb = h[0, wc], h[1, wc]
    return sig, sb, SB_SCALE * sb, sig - SB_SCALE * sb


def peak_mv(sub, fmv):
    sm = np.convolve(sub, KERN, mode='same')
    m = AMP_CEN > 150
    return AMP_CEN[m][np.argmax(sm[m])] * fmv


# ---------------------------------------------------------------- step 1: time
def step1_time(st, wc):
    fmv = FAC[f'WAL{st}'][str(wc + 1)]
    hc = dtcal(st, wc)
    acc = hc[(DT_CEN >= SB_LO) & (DT_CEN <= SB_HI)].mean()     # accidentals / ns

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.step(DT_CEN, hc, where='mid', color=C_LINE, lw=1.3, zorder=5)

    sig_m = np.abs(DT_CEN) <= W_SIG
    sb_m = (DT_CEN >= SB_LO) & (DT_CEN <= SB_HI)
    ax.fill_between(DT_CEN[sig_m], 0, hc[sig_m], step='mid', color=C_SIG,
                    alpha=0.85, lw=0, zorder=3, label=f'signal window  |dt\'| $\\leq$ {W_SIG:.0f} ns')
    ax.fill_between(DT_CEN[sb_m], 0, hc[sb_m], step='mid', color=C_SIDE,
                    alpha=0.75, lw=0, zorder=3,
                    label=f'sideband  +{SB_LO:.0f}..+{SB_HI:.0f} ns  (accidentals)')
    ax.axhline(acc, color=C_ACC, ls='--', lw=1.3, zorder=4,
               label=f'accidental level ({acc:,.0f}/ns)')

    ax.set_xlim(-60, 140)
    ax.set_ylim(0, hc.max() * 1.12)
    ax.set_xlabel("dt' = t(wall) $-$ t(plastic) $-$ offset   [ns]")
    ax.set_ylabel('coincident pairs / ns')
    ax.set_title(f"Step 1 — time selection   (WAL{st}{wc + 1}, hits >0.1 ms after $\\gamma$-flash)")
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.2)
    ax.annotate('true coincidences\nsit under the peak\n(MIP band, dt\'$\\approx$0)',
                xy=(0, hc.max() * 0.62), xytext=(-55, hc.max() * 0.80),
                fontsize=9.5, color=C_SIGL,
                arrowprops=dict(arrowstyle='->', color=C_SIGL))
    ax.annotate('flat in dt\' $\\Rightarrow$ random\nwall+plastic pile-ups',
                xy=(80, acc), xytext=(60, hc.max() * 0.55), fontsize=9.5,
                color=C_SIDEL, arrowprops=dict(arrowstyle='->', color=C_SIDEL))
    ax.annotate('satellite bump\n$\\Rightarrow$ sideband is\npositive side only',
                xy=(-50, hc[np.argmin(np.abs(DT_CEN + 50))]),
                xytext=(-58, hc.max() * 0.30), fontsize=8.5, color='#374151',
                arrowprops=dict(arrowstyle='->', color='#374151'))
    fig.tight_layout()
    fig.savefig(OUT / 'step1_time.png', dpi=150)
    plt.close(fig)


# ------------------------------------------------- steps 2-5: amplitude domain
def _amp_axis(ax, logx=True):
    if logx:
        ax.set_xscale('log')
        ax.set_xlim(1.0, 3000)
    ax.set_xlabel('coincidence amplitude   [mV]')
    ax.grid(alpha=0.2, which='both')


def step2_signal(st, wc):
    fmv = FAC[f'WAL{st}'][str(wc + 1)]
    sig, sb, sb_s, sub = amp_spectra(st, wc)
    x = AMP_CEN * fmv
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.step(x, sig, where='mid', color=C_SIGL, lw=1.4)
    ax.fill_between(x, 0, sig, step='mid', color=C_SIG, alpha=0.55, lw=0)
    _amp_axis(ax)
    ax.set_ylabel('pairs / bin')
    ax.set_title(f"Step 2 — on-peak amplitude spectrum  (signal window, WAL{st}{wc + 1})")
    ax.annotate('MIP bump\n(through-going tracks)', xy=(37, sig[np.argmin(np.abs(x - 37))]),
                xytext=(80, sig.max() * 0.72), fontsize=9.5, color=C_SIGL,
                arrowprops=dict(arrowstyle='->', color=C_SIGL))
    ax.annotate('low-amplitude accidentals\nstill buried in here',
                xy=(4, sig[np.argmin(np.abs(x - 4))]),
                xytext=(1.4, sig.max() * 0.45), fontsize=9.5, color=C_ACC,
                arrowprops=dict(arrowstyle='->', color=C_ACC))
    fig.tight_layout()
    fig.savefig(OUT / 'step2_signal.png', dpi=150)
    plt.close(fig)


def step3_sideband(st, wc):
    fmv = FAC[f'WAL{st}'][str(wc + 1)]
    sig, sb, sb_s, sub = amp_spectra(st, wc)
    x = AMP_CEN * fmv
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.step(x, sb, where='mid', color=C_SIDEL, lw=1.2, alpha=0.6,
            label='raw sideband (100 ns wide)')
    ax.step(x, sb_s, where='mid', color=C_SIDEL, lw=1.6,
            label=f'sideband $\\times$ {SB_SCALE:.2f}  (= accidentals in a 16 ns window)')
    ax.fill_between(x, 0, sb_s, step='mid', color=C_SIDE, alpha=0.5, lw=0)
    _amp_axis(ax)
    ax.set_ylabel('pairs / bin')
    ax.set_title(f"Step 3 — sideband = shape of the accidental background  (WAL{st}{wc + 1})")
    ax.legend(loc='upper right', fontsize=9)
    ax.annotate('no MIP bump —\nfalling accidental shape',
                xy=(37, sb_s[np.argmin(np.abs(x - 37))]),
                xytext=(70, sb_s.max() * 0.6), fontsize=9.5, color=C_SIDEL,
                arrowprops=dict(arrowstyle='->', color=C_SIDEL))
    fig.tight_layout()
    fig.savefig(OUT / 'step3_sideband.png', dpi=150)
    plt.close(fig)


def step4_overlay(st, wc):
    fmv = FAC[f'WAL{st}'][str(wc + 1)]
    sig, sb, sb_s, sub = amp_spectra(st, wc)
    x = AMP_CEN * fmv
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.fill_between(x, sb_s, sig, where=sig >= sb_s, step='mid', color=C_TRUE,
                    alpha=0.30, lw=0, label='excess = true coincidences')
    ax.step(x, sig, where='mid', color=C_SIGL, lw=1.5, label='signal window (on-peak)')
    ax.step(x, sb_s, where='mid', color=C_SIDEL, lw=1.5,
            label=f'scaled sideband (accidentals)')
    _amp_axis(ax)
    ax.set_ylabel('pairs / bin')
    ax.set_title(f"Step 4 — overlay: the two windows share the same accidental floor  (WAL{st}{wc + 1})")
    ax.legend(loc='upper right', fontsize=9)
    ax.annotate('curves coincide at low amp\n$\\Rightarrow$ accidentals cancel exactly',
                xy=(3, sig[np.argmin(np.abs(x - 3))]),
                xytext=(1.3, sig.max() * 0.5), fontsize=9.5, color='#374151',
                arrowprops=dict(arrowstyle='->', color='#374151'))
    fig.tight_layout()
    fig.savefig(OUT / 'step4_overlay.png', dpi=150)
    plt.close(fig)


def step5_subtracted(st, wc):
    fmv = FAC[f'WAL{st}'][str(wc + 1)]
    sig, sb, sb_s, sub = amp_spectra(st, wc)
    wid = np.diff(AMP_EDGES) * fmv
    x = AMP_CEN * fmv
    dens = sub / wid                                  # hits / mV
    pk = peak_mv(sub, fmv)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axhline(0, color=C_ACC, lw=0.9)
    ax.fill_between(x, 0, dens, step='mid', color=C_TRUE, alpha=0.55, lw=0)
    ax.step(x, dens, where='mid', color=C_SIGL, lw=1.5)
    ax.axvline(pk, color=C_TRUE, ls=':', lw=1.4)
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=min(0, dens[x < 120].min() * 1.1))
    ax.set_xlabel('coincidence amplitude   [mV]')
    ax.set_ylabel('true-coincidence hits / mV')
    ax.grid(alpha=0.2)
    ax.set_title(f"Step 5 — subtracted spectrum: clean SiPM-wall MIP peak  (WAL{st}{wc + 1})")
    ax.annotate(f'MIP peak\n$\\approx$ {pk:.0f} mV', xy=(pk, dens[np.argmin(np.abs(x - pk))]),
                xytext=(pk + 25, dens[x < 120].max() * 0.75), fontsize=11, color=C_TRUE,
                arrowprops=dict(arrowstyle='->', color=C_TRUE))
    fig.tight_layout()
    fig.savefig(OUT / 'step5_subtracted.png', dpi=150)
    plt.close(fig)


# ------------------------------------------------------ one-figure overview
def overview(st, wc):
    fmv = FAC[f'WAL{st}'][str(wc + 1)]
    hc = dtcal(st, wc)
    acc = hc[(DT_CEN >= SB_LO) & (DT_CEN <= SB_HI)].mean()
    sig, sb, sb_s, sub = amp_spectra(st, wc)
    x = AMP_CEN * fmv
    wid = np.diff(AMP_EDGES) * fmv
    pk = peak_mv(sub, fmv)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))

    ax = axes[0, 0]
    ax.step(DT_CEN, hc, where='mid', color=C_LINE, lw=1.2)
    sig_m, sb_m = np.abs(DT_CEN) <= W_SIG, (DT_CEN >= SB_LO) & (DT_CEN <= SB_HI)
    ax.fill_between(DT_CEN[sig_m], 0, hc[sig_m], step='mid', color=C_SIG, alpha=0.85, lw=0)
    ax.fill_between(DT_CEN[sb_m], 0, hc[sb_m], step='mid', color=C_SIDE, alpha=0.7, lw=0)
    ax.axhline(acc, color=C_ACC, ls='--', lw=1.1)
    ax.set_xlim(-60, 140)
    ax.set_ylim(0, hc.max() * 1.1)
    ax.set_xlabel("dt'  [ns]")
    ax.set_ylabel('pairs / ns')
    ax.set_title('(a) time: signal (orange) vs sideband (blue)')
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.step(x, sig, where='mid', color=C_SIGL, lw=1.4, label='signal window')
    ax.step(x, sb_s, where='mid', color=C_SIDEL, lw=1.4, label='scaled sideband')
    ax.fill_between(x, sb_s, sig, where=sig >= sb_s, step='mid', color=C_TRUE, alpha=0.25, lw=0)
    ax.set_xscale('log')
    ax.set_xlim(1, 3000)
    ax.set_xlabel('amplitude [mV]')
    ax.set_ylabel('pairs / bin')
    ax.set_title('(b) amplitude: on-peak vs accidentals')
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.2, which='both')

    ax = axes[1, 0]
    ax.step(x, sig - sb_s, where='mid', color=C_TRUE, lw=1.4)
    ax.fill_between(x, 0, sig - sb_s, step='mid', color=C_SIG, alpha=0.4, lw=0)
    ax.axhline(0, color=C_ACC, lw=0.9)
    ax.set_xscale('log')
    ax.set_xlim(1, 3000)
    ax.set_xlabel('amplitude [mV]')
    ax.set_ylabel('true pairs / bin')
    ax.set_title('(c) subtracted (log axis): accidentals gone below $\\sim$10 mV')
    ax.grid(alpha=0.2, which='both')

    ax = axes[1, 1]
    dens = sub / wid
    ax.axhline(0, color=C_ACC, lw=0.9)
    ax.fill_between(x, 0, dens, step='mid', color=C_TRUE, alpha=0.55, lw=0)
    ax.step(x, dens, where='mid', color=C_SIGL, lw=1.5)
    ax.axvline(pk, color=C_TRUE, ls=':', lw=1.3)
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=min(0, dens[x < 120].min() * 1.1))
    ax.set_xlabel('amplitude [mV]')
    ax.set_ylabel('true hits / mV')
    ax.set_title(f'(d) subtracted (linear): MIP peak $\\approx$ {pk:.0f} mV')
    ax.grid(alpha=0.2)

    fig.suptitle(f'{RUN_STEM}  WAL{st}{wc + 1}: sideband-subtraction pipeline '
                 f'(signal $-$ {SB_SCALE:.2f}$\\times$sideband)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / 'pipeline_overview.png', dpi=150)
    plt.close(fig)


# --------------------------------------------------- null-channel contrast
def null_contrast(st, wc):
    fmv = FAC[f'WAL{st}'][str(wc + 1)]
    sig, sb, sb_s, sub = amp_spectra(st, wc)
    x = AMP_CEN * fmv
    wid = np.diff(AMP_EDGES) * fmv
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    ax.step(x, sig, where='mid', color=C_SIGL, lw=1.5, label='signal window')
    ax.step(x, sb_s, where='mid', color=C_SIDEL, lw=1.5, label='scaled sideband')
    ax.set_xscale('log')
    ax.set_xlim(1, 3000)
    ax.set_xlabel('amplitude [mV]')
    ax.set_ylabel('pairs / bin')
    ax.set_title(f'(a) WAL{st}{wc + 1}: signal $\\approx$ sideband everywhere')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, which='both')
    ax = axes[1]
    ax.axhline(0, color=C_ACC, lw=0.9)
    ax.fill_between(x, 0, sub / wid, step='mid', color='#6b7280', alpha=0.5, lw=0)
    ax.step(x, sub / wid, where='mid', color='#374151', lw=1.4)
    ax.set_xlim(0, 120)
    ax.set_xlabel('amplitude [mV]')
    ax.set_ylabel('true hits / mV')
    ax.set_title('(b) subtracted: no MIP peak — subtraction cannot invent one')
    ax.grid(alpha=0.2)
    fig.suptitle(f'{RUN_STEM}  WAL{st}{wc + 1} (arm {st}): accidental-dominated null case', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / 'null_contrast.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    ST, WC = 'B', 5          # WALB6 — clean representative MIP channel
    step1_time(ST, WC)
    step2_signal(ST, WC)
    step3_sideband(ST, WC)
    step4_overlay(ST, WC)
    step5_subtracted(ST, WC)
    overview(ST, WC)
    null_contrast('A', 1)    # WALA2 — accidental-dominated null
    print(f'Figures -> {OUT}')
