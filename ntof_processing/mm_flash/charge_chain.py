#!/usr/bin/env python3
"""From ADC codes to charge on one strip: the conversion, the cable, the spread.

Produces the equations' numbers, the cable-attenuation correction and the
per-bunch charge RMS for run 224709 (MMA = strip 32 of MX17 detector A, cable Y8).

Writes results_chain.json and four figures.
"""
import csv
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = pathlib.Path('/media/dylan/data/x17/ntof_mm_flash')
FIG = HERE / 'figures'

# ---- the conversion chain, run 224709 ---------------------------------------
FULL_SCALE_MV = 5043.7915          # MODH channel record
N_CODES = 65536                    # ADCrange, 16 bits signed
LSB_MV = FULL_SCALE_MV / N_CODES   # mV per count
DT_NS = 1.0                        # 1 GS/s
R_TERM = 50.0                      # ohm
ZS_FILL = -32768
FLASH_LO, FLASH_HI = 11_000, 20_000
BASE_LO, BASE_HI = 1_000, 10_000   # same width, before the flash

# ---- RG-58C/U ----------------------------------------------------------------
# alpha(dB/m) = A*sqrt(f_MHz) + B*f_MHz, fitted to the catalogue attenuation
# (1.6 / 5.3 / 12.5 / 18.4 dB per 100 m at 1 / 10 / 50 / 100 MHz).
A_SKIN = 0.01676
B_DIEL = 1.64e-4
# DC loop resistance: centre 0.0358 ohm/m + braid 0.0098 ohm/m (Belden 8262).
R_LOOP_CU = 0.0456                 # solid/stranded copper centre
R_LOOP_CCS = 0.165                 # copper-clad steel centre, worst case
LENGTHS = (10.0, 20.0)

# detector geometry, for the single-strip statement
N_STRIPS = 512
STRIP_PITCH_MM = 398.58 / 512


def alpha_db(f_mhz, length_m):
    return (A_SKIN * np.sqrt(f_mhz) + B_DIEL * f_mhz) * length_m


def plateaus():
    rows = []
    with open(DATA / 'hv_plateaus_224709.csv') as f:
        for r in csv.DictReader(f):
            rows.append(dict(start=int(r['start_s']), end=int(r['end_s']),
                             drift=int(r['A_drift_V']), resist=int(r['A_resist_V'])))
    return rows


def main():
    d = np.load(DATA / 'mm_224709.npz')
    stats = d['stats']
    ev = stats[:, 0].astype(int)
    base = stats[:, 1]
    flash = d['flash'].astype(np.int32)
    dev_ct = base[:, None] - flash[ev]
    dev_ct[flash[ev] <= ZS_FILL] = 0.0
    dev_mv = dev_ct * LSB_MV

    wall = d['wall']
    secs = (wall[:, 0] // 10000) * 3600 + ((wall[:, 0] // 100) % 100) * 60 + wall[:, 0] % 100
    secs_ev = secs[ev]

    pk = np.zeros(len(d['bunch']))
    pk[d['pkup'][:, 0].astype(int)] = d['pkup'][:, 1]
    pk_ev = pk[ev]
    valid = pk_ev > 1000
    split = 0.5 * (np.percentile(pk_ev[valid], 10) + np.percentile(pk_ev[valid], 90))
    dedicated = valid & (pk_ev > split)
    parasitic = valid & ~dedicated

    # ---- charge, per bunch ---------------------------------------------------
    # Q = (dt/R) * sum(dV)  ->  pC
    k_pc_per_mv_ns = 1e-3 * 1e-9 / R_TERM * 1e12          # mV*ns -> pC
    q_pc = dev_mv[:, FLASH_LO:FLASH_HI].sum(axis=1) * DT_NS * k_pc_per_mv_ns
    q_noise = dev_mv[:, BASE_LO:BASE_HI].sum(axis=1) * DT_NS * k_pc_per_mv_ns
    fc_per_count_ns = LSB_MV * 1e-3 * 1e-9 / R_TERM * 1e15

    res = dict(
        conversion=dict(full_scale_mV=FULL_SCALE_MV, n_codes=N_CODES,
                        lsb_uV=LSB_MV * 1000, dt_ns=DT_NS, R_ohm=R_TERM,
                        fC_per_count_ns=fc_per_count_ns,
                        pC_per_mV_ns=k_pc_per_mv_ns,
                        window_ns=[FLASH_LO, FLASH_HI]),
        geometry=dict(n_strips=N_STRIPS, pitch_mm=STRIP_PITCH_MM,
                      note='MMA is ONE strip of detector A; the flash deposits '
                           'over the whole 40x40 cm active area'),
    )

    # ---- cable ---------------------------------------------------------------
    cable = dict(model='RG-58C/U', alpha_fit='alpha[dB/m] = %.5f*sqrt(f_MHz) + %.2e*f_MHz'
                 % (A_SKIN, B_DIEL), lengths_m=list(LENGTHS), table=[])
    for f in (0.1, 1.0, 3.0, 10.0, 30.0, 100.0):
        row = dict(f_MHz=f)
        for L in LENGTHS:
            a = alpha_db(f, L)
            row[f'dB_{L:.0f}m'] = float(a)
            row[f'amp_{L:.0f}m'] = float(10 ** (-a / 20))
        cable['table'].append(row)
    # charge (f -> 0) loss: skin effect vanishes, only the series DC resistance
    # of the matched line survives, exp(-R_loop*L/(2*Z0))
    cable['charge_loss_pct'] = {}
    for L in LENGTHS:
        for lbl, rloop in (('copper', R_LOOP_CU), ('copper-clad steel', R_LOOP_CCS)):
            loss = 1 - np.exp(-rloop * L / (2 * R_TERM))
            cable['charge_loss_pct'][f'{L:.0f}m {lbl}'] = float(100 * loss)
    cable['charge_correction_applied'] = float(
        np.exp(R_LOOP_CU * 20.0 / (2 * R_TERM)))       # 20 m copper, the nominal
    res['cable'] = cable

    # ---- de-attenuation of the measured mean pulse ---------------------------
    work = (700, 540)
    p = next(x for x in plateaus() if (x['drift'], x['resist']) == work)
    m = (secs_ev >= p['start'] + 45) & (secs_ev <= p['end']) & dedicated
    mean_mv = dev_mv[m].mean(axis=0)
    seg = mean_mv[10_000:25_000].copy()
    n = len(seg)
    freq = np.fft.rfftfreq(n, d=DT_NS * 1e-9) / 1e6      # MHz
    X = np.fft.rfft(seg)
    deatt = {}
    roll = np.exp(-(freq / 50.0) ** 2)      # tame the inverse above 50 MHz
    # the reference carries the SAME roll-off, so the ratio isolates the cable
    seg_ref = np.fft.irfft(X * roll, n)
    for L in LENGTHS:
        H = 10 ** (-alpha_db(np.maximum(freq, 1e-6), L) / 20)
        rec = np.fft.irfft(X / H * roll, n)
        deatt[f'{L:.0f}m'] = dict(
            peak_measured_mV=float(seg_ref.max()),
            peak_corrected_mV=float(rec.max()),
            peak_ratio=float(rec.max() / seg_ref.max()),
            area_measured=float(seg_ref.sum()),
            area_corrected=float(rec.sum()),
            area_ratio=float(rec.sum() / seg_ref.sum()))
        if L == 20.0:
            rec20 = rec
    res['de_attenuation'] = dict(working_point=dict(drift=work[0], resist=work[1],
                                                    n_bunch=int(m.sum())), **deatt)

    # ---- charge and its RMS, per scan point ---------------------------------
    rows = []
    for p in plateaus():
        sel_t = (secs_ev >= p['start'] + 45) & (secs_ev <= p['end'])
        for cls, sel in (('dedicated', sel_t & dedicated), ('parasitic', sel_t & parasitic)):
            if sel.sum() < 5:
                continue
            q = q_pc[sel]
            # the beam itself jitters within a class; the pickup measures that,
            # so quote the charge spread against it
            i_frac = float(pk_ev[sel].std(ddof=1) / pk_ev[sel].mean())
            f_q = float(q.std(ddof=1) / q.mean())
            # remove the additive electronics term and the beam term in quadrature
            resid = f_q ** 2 - i_frac ** 2 - (q_noise.std(ddof=1) / q.mean()) ** 2
            rows.append(dict(drift=p['drift'], resist=p['resist'], cls=cls,
                             n=int(sel.sum()), mean_pC=float(q.mean()),
                             rms_pC=float(q.std(ddof=1)),
                             frac_rms=f_q, frac_rms_beam=i_frac,
                             frac_rms_noise=float(q_noise.std(ddof=1) / q.mean()),
                             frac_rms_residual=float(np.sqrt(resid)) if resid > 0 else None,
                             median_pC=float(np.median(q))))
    res['charge_rms'] = rows
    # White-noise expectation for the same integration: sigma_V * sqrt(N) * dt / R.
    # The measured value is far larger, which says the baseline wanders coherently.
    sigma_mv = float(np.median(dev_mv[:, BASE_LO:BASE_HI].std(axis=1)))
    white_pc = sigma_mv * np.sqrt(BASE_HI - BASE_LO) * DT_NS * k_pc_per_mv_ns
    res['noise_equivalent_charge_pC'] = dict(
        rms=float(q_noise.std(ddof=1)), mean=float(q_noise.mean()),
        window_ns=[BASE_LO, BASE_HI], sample_sigma_mV=sigma_mv,
        white_noise_expectation=float(white_pc),
        excess_over_white=float(q_noise.std(ddof=1) / white_pc),
        note='same integration width, taken before the flash; the excess over the '
             'white-noise expectation is coherent baseline wander')

    # ============================ figures ====================================
    # 1. the chain
    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.axis('off')
    boxes = [(0.02, 'strip 32\nDet A, cable Y8', 'induced current\n$i(t)$'),
             (0.235, '10-20 m\nRG-58 BNC', 'attenuates $H(f)$\narea preserved'),
             (0.45, '50 $\\Omega$\ntermination', '$v(t)=R\\,i(t)$'),
             (0.665, 'S014 ADC\n1 GS/s, 16 bit', '$c_i$ codes'),
             (0.88, 'charge\n$Q$', '$\\int v\\,dt / R$')]
    for x, title, sub in boxes:
        ax.add_patch(plt.Rectangle((x, .45), .155, .34, fc='#eef3f8', ec='#2f6f9f', lw=1.4,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(x + .0775, .68, title, ha='center', va='center', fontsize=9.5,
                transform=ax.transAxes, weight='bold')
        ax.text(x + .0775, .535, sub, ha='center', va='center', fontsize=8.2,
                transform=ax.transAxes, color='#40566b')
    for x in (0.185, 0.40, 0.615, 0.83):
        ax.annotate('', xy=(x + .048, .62), xytext=(x, .62), xycoords=ax.transAxes,
                    textcoords=ax.transAxes, arrowprops=dict(arrowstyle='->', lw=1.4,
                                                             color='#2f6f9f'))
    ax.text(0.5, 0.24,
            r'$\Delta V_i = (b - c_i)\,\dfrac{V_{FS}}{2^{16}}$'
            '          '
            r'$Q = \dfrac{\Delta t}{R}\sum_i \Delta V_i$'
            '          '
            r'$\dfrac{\Delta t\,V_{FS}}{R\,2^{16}} = %.3f\ \mathrm{fC\ per\ count \cdot ns}$'
            % fc_per_count_ns,
            ha='center', va='center', fontsize=12.5, transform=ax.transAxes)
    ax.text(0.5, 0.06, '$V_{FS}$ = 5043.79 mV,   $\\Delta t$ = 1 ns,   $R$ = 50 $\\Omega$,   '
                       '$b$ = per-bunch baseline (median of the first 2 $\\mu$s)',
            ha='center', va='center', fontsize=8.6, transform=ax.transAxes, color='#40566b')
    fig.tight_layout()
    fig.savefig(FIG / 'chain_diagram.png', dpi=140)
    plt.close(fig)

    # 2. integration
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    t = np.arange(len(mean_mv)) / 1000.0
    ax.plot(t, mean_mv, color='#2f6f9f', lw=1.2, label='mean signal $\\Delta V(t)$')
    ax.axvspan(FLASH_LO / 1000, FLASH_HI / 1000, color='#2f6f9f', alpha=.10,
               label='integration window 11-20 $\\mu$s')
    ax.set_xlim(10, 26)
    ax.set_xlabel('time in acquisition window ($\\mu$s)')
    ax.set_ylabel('$\\Delta V$ (mV)')
    ax.grid(alpha=.3)
    ax2 = ax.twinx()
    cum = np.cumsum(mean_mv) * DT_NS * k_pc_per_mv_ns
    ax2.plot(t, cum, color='#c0632c', lw=1.3, ls='--', label='running $\\int$ (pC)')
    ax2.set_ylabel('cumulative charge (pC)', color='#c0632c')
    ax2.tick_params(axis='y', colors='#c0632c')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right')
    ax.set_title('Detector A at 700 / 540 V, dedicated pulses (n=%d)' % m.sum())
    fig.tight_layout()
    fig.savefig(FIG / 'charge_integration.png', dpi=140)
    plt.close(fig)

    # 3. cable
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ff = np.logspace(-1, 2.3, 300)
    for L, c in zip(LENGTHS, ('#2f6f9f', '#c0632c')):
        ax[0].plot(ff, alpha_db(ff, L), color=c, lw=1.5, label=f'{L:.0f} m RG-58')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('frequency (MHz)')
    ax[0].set_ylabel('cable attenuation (dB)')
    ax[0].grid(alpha=.3, which='both')
    ax[0].legend(fontsize=8, loc='upper left')
    axs = ax[0].twinx()
    amp = np.abs(X) / np.abs(X).max()
    axs.plot(freq[1:], amp[1:], color='#6a6a6a', lw=1.0, alpha=.75)
    axs.set_ylabel('pulse amplitude spectrum (normalised)', color='#6a6a6a', fontsize=9)
    axs.set_yscale('log')
    axs.set_ylim(1e-4, 2)
    axs.tick_params(axis='y', colors='#6a6a6a')
    ax[0].set_title('Where the pulse lives vs where the cable bites')

    ts = np.arange(len(seg)) / 1000.0
    ax[1].plot(ts, seg, color='#2f6f9f', lw=1.3, label='measured at the DAQ')
    ax[1].plot(ts, rec20, color='#c0632c', lw=1.1, ls='--',
               label='de-attenuated (20 m)')
    ax[1].set_xlim(1.4, 6)
    ax[1].set_xlabel('time within the extracted segment ($\\mu$s)')
    ax[1].set_ylabel('$\\Delta V$ (mV)')
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    ax[1].set_title('Peak moves by %.1f %%, area by %.2f %%'
                    % (100 * (deatt['20m']['peak_ratio'] - 1),
                       100 * (deatt['20m']['area_ratio'] - 1)))
    fig.tight_layout()
    fig.savefig(FIG / 'cable_attenuation.png', dpi=140)
    plt.close(fig)

    # 4. charge spread
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    qd = q_pc[m]
    ax[0].hist(qd, bins=28, color='#2f6f9f', alpha=.85)
    ax[0].axvline(qd.mean(), color='k', lw=1.4, label='mean %.0f pC' % qd.mean())
    ax[0].axvline(qd.mean() + qd.std(ddof=1), color='k', ls='--', lw=1.1,
                  label='$\\pm$ RMS %.0f pC (%.1f %%)'
                        % (qd.std(ddof=1), 100 * qd.std(ddof=1) / qd.mean()))
    ax[0].axvline(qd.mean() - qd.std(ddof=1), color='k', ls='--', lw=1.1)
    ax[0].set_xlabel('flash charge on strip 32 (pC)')
    ax[0].set_ylabel('bunches')
    ax[0].set_title('Charge per bunch, 700 / 540 V, dedicated')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    for cls, c, mk in (('dedicated', '#2f6f9f', 'o'), ('parasitic', '#c0632c', 's')):
        pts = sorted([(r['resist'], 100 * r['frac_rms']) for r in rows
                      if r['cls'] == cls and r['drift'] == 700])
        ax[1].plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=c,
                   lw=1.3, ms=4, label=cls)
    ax[1].axhline(100 * res['noise_equivalent_charge_pC']['rms'] / qd.mean(),
                  color='grey', ls=':', lw=1.2,
                  label='electronics noise at this working point')
    ax[1].set_xlabel('amplification voltage (V)')
    ax[1].set_ylabel('charge RMS / mean (%)')
    ax[1].set_title('Bunch-to-bunch spread, drift 700 V')
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'charge_rms.png', dpi=140)
    plt.close(fig)

    with open(HERE / 'results_chain.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'charge_rms'}, indent=1)[:2600])


if __name__ == '__main__':
    main()
