#!/usr/bin/env python3
"""The HV-supply-current charge, measured on the SAME plateaus as the waveform scan.

The parallel package (ntof_july_analysis/flash_charge/) established the method and
its validations on July data. Here it is applied point for point to run 224709's
detector-A scan, so the two measurements share a detector, a day, a gas and 25
working points, and can be compared without transporting anything.

    Q_pulse = ( mean(imon) - median(imon) ) / f_pulse

The pulse rate comes from the n_TOF side: every bunch in 224709 carries its own
wall clock, so f_pulse is counted directly rather than taken from the beam log.

Writes results_imon.json and three figures.
"""
import csv
import datetime as dt
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = pathlib.Path('/media/dylan/data/x17/ntof_mm_flash')
FIG = HERE / 'figures'

SETTLE_S = 45
# --- strip accounting, detector A (mx17_3), from common/mx17_active_area.py ----
N_STRIPS_PER_PLANE = 512
STRIP_MAX_MM = 398.58
PITCH_MM = STRIP_MAX_MM / N_STRIPS_PER_PLANE
PASS_LO_MM, PASS_HI_MM = 18.0, 18.7          # measured Y passivation, det A
Y_LIVE_MM = STRIP_MAX_MM - PASS_LO_MM - PASS_HI_MM
N_Y_LIVE = int(round(Y_LIVE_MM / PITCH_MM))
N_X_LIVE = N_STRIPS_PER_PLANE                 # X strips all live, shortened in y
LIVE_FRACTION = Y_LIVE_MM / STRIP_MAX_MM


def secs(ts):
    t = dt.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    return t.hour * 3600 + t.minute * 60 + t.second


def load_imon():
    ts, cur, vol = [], [], []
    with open(DATA / 'imon_224709.csv') as f:
        for r in csv.DictReader(f):
            try:
                cur.append(float(r['A_resist_imon']))
                vol.append(float(r['A_resist_vmon']))
            except (TypeError, ValueError):
                continue
            ts.append(secs(r['timestamp']))
    return np.array(ts), np.array(cur), np.array(vol)


def plateaus():
    out = []
    with open(DATA / 'hv_plateaus_224709.csv') as f:
        for r in csv.DictReader(f):
            out.append(dict(start=int(r['start_s']), end=int(r['end_s']),
                            drift=int(r['A_drift_V']), resist=int(r['A_resist_V']),
                            label=r['start'][11:16]))
    return out


def main():
    t, i_ua, v = load_imon()
    d = np.load(DATA / 'mm_224709.npz')
    w = d['wall'][:, 0]
    bunch_s = (w // 10000) * 3600 + ((w // 100) % 100) * 60 + w % 100
    wf = json.load(open(HERE / 'results_709.json'))
    wfq = {(r['drift'], r['resist'], r['cls']): r for r in wf['scan']}

    rows = []
    for p in plateaus():
        lo, hi = p['start'] + SETTLE_S, p['end']
        m = (t >= lo) & (t <= hi)
        nb = int(((bunch_s >= lo) & (bunch_s <= hi)).sum())
        dur = hi - lo
        if m.sum() < 60 or nb < 5:
            continue
        leak = float(np.median(i_ua[m]))
        mean = float(np.mean(i_ua[m]))
        di = mean - leak
        f_pulse = nb / dur
        q_nc = di * 1e-6 / f_pulse * 1e9          # uA / Hz -> nC
        # bootstrap the sample mean for an error bar
        rng = np.random.default_rng(0)
        boot = [np.mean(rng.choice(i_ua[m], m.sum())) - leak for _ in range(300)]
        q_err = float(np.std(boot) * 1e-6 / f_pulse * 1e9)
        row = dict(drift=p['drift'], resist=p['resist'], label=p['label'],
                   n_imon=int(m.sum()), n_bunch=nb, duration_s=dur,
                   f_pulse_hz=f_pulse, leak_uA=leak, mean_uA=mean, di_uA=di,
                   q_nC=q_nc, q_err_nC=q_err,
                   frac_elevated=float((i_ua[m] > leak + 0.02).mean()))
        for cls in ('dedicated', 'parasitic'):
            k = (p['drift'], p['resist'], cls)
            if k in wfq:
                row[f'wf_{cls}_pC'] = wfq[k]['charge_pC']
                row[f'wf_{cls}_n'] = wfq[k]['n']
        if 'wf_dedicated_pC' in row and 'wf_parasitic_pC' in row:
            nd, npar = row['wf_dedicated_n'], row['wf_parasitic_n']
            row['wf_mix_pC'] = ((nd * row['wf_dedicated_pC'] + npar * row['wf_parasitic_pC'])
                                / (nd + npar))
        rows.append(row)

    res = dict(
        method='Q = (mean(imon) - median(imon)) / f_pulse, per plateau, det A resist channel',
        pulse_rate_source='n_TOF bunch wall clocks in run 224709 (not the beam log)',
        geometry=dict(pitch_mm=PITCH_MM, n_strips_per_plane=N_STRIPS_PER_PLANE,
                      passivation_lo_mm=PASS_LO_MM, passivation_hi_mm=PASS_HI_MM,
                      y_live_mm=Y_LIVE_MM, n_y_live=N_Y_LIVE, n_x_live=N_X_LIVE,
                      live_fraction=LIVE_FRACTION,
                      multiplier_equal_sharing=2 * N_Y_LIVE,
                      multiplier_y_only=N_Y_LIVE,
                      note='a Y-plane strip is dead if it sits under the passivation; '
                           'X strips all survive but are shortened. To scale one live '
                           'Y strip to the whole chamber, multiply by 2 x N_Y_live if '
                           'the two planes share the induced charge equally, or by '
                           'N_Y_live if it all appears on Y.',
                      strip_position=dict(
                          y_mm_connectorY8_ch32=374.40,
                          y_mm_global_strip32=24.96,
                          live_band_mm=[PASS_LO_MM, STRIP_MAX_MM - PASS_HI_MM],
                          distance_to_edge_mm=[5.5, 7.0],
                          note='"strip 32 of cable Y8" reads either as connector Y8 '
                               'channel 32 (y = 374.4 mm) or as global y-strip 32 '
                               '(y = 25.0 mm). BOTH land 5-7 mm inside a passivation '
                               'edge, so the strip is at the chamber periphery either '
                               'way - which argues against a beam-profile explanation '
                               'for the residual scale factor.')),
        plateaus=rows)

    # ---- the comparison at the shared working point -------------------------
    ref = [r for r in rows if r['drift'] == 700 and r['resist'] == 540]
    if ref:
        r0 = ref[0]
        for lbl, mult in (('equal_sharing', 2 * N_Y_LIVE), ('y_only', N_Y_LIVE),
                          ('naive_1024', 1024), ('naive_512', 512)):
            res.setdefault('working_point', {})[lbl] = dict(
                multiplier=mult,
                implied_chamber_nC=r0['wf_mix_pC'] * mult / 1e3,
                imon_nC=r0['q_nC'],
                ratio=r0['wf_mix_pC'] * mult / 1e3 / r0['q_nC'])
        res['working_point']['imon_nC'] = r0['q_nC']
        res['working_point']['waveform_strip_pC'] = r0['wf_mix_pC']
        res['working_point']['implied_n_strips'] = r0['q_nC'] * 1e3 / r0['wf_mix_pC']

    # ============================ figures ====================================
    # 1. imon over the whole scan
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot((t - t.min()) / 60, i_ua, lw=.5, color='#2f6f9f')
    for r in rows:
        ax[0].axvspan((r['start'] if 'start' in r else 0), 0, alpha=0)
    for p in plateaus():
        ax[0].axvline((p['start'] - t.min()) / 60, color='#c0632c', lw=.5, alpha=.5)
    ax[0].set_ylabel('detector A resist current (µA)')
    ax[0].set_title('The HV supply current through the detector-A scan (run 224709)')
    ax[0].grid(alpha=.3)
    ax[1].plot((t - t.min()) / 60, v, lw=1.0, color='#2e7d4f')
    ax[1].set_ylabel('amplification (V)')
    ax[1].set_xlabel('minutes from 17:10')
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / 'imon_timeseries.png', dpi=140)
    plt.close(fig)

    # 2. how the estimate is made, on one plateau
    p = next(x for x in plateaus() if (x['drift'], x['resist']) == (700, 540))
    lo, hi = p['start'] + SETTLE_S, p['end']
    m = (t >= lo) & (t <= hi)
    r0 = next(r for r in rows if r['drift'] == 700 and r['resist'] == 540)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3),
                           gridspec_kw=dict(width_ratios=[2, 1]))
    tt = t[m] - lo
    ax[0].plot(tt, i_ua[m], lw=.8, color='#2f6f9f', marker='.', ms=2.5,
               label='imon, 1 Hz')
    ax[0].axhline(r0['leak_uA'], color='#2e7d4f', lw=1.6,
                  label=f"median = leakage = {r0['leak_uA']:.3f} µA")
    ax[0].axhline(r0['mean_uA'], color='#c0632c', lw=1.6, ls='--',
                  label=f"mean = {r0['mean_uA']:.3f} µA")
    ax[0].fill_between(tt, r0['leak_uA'], r0['mean_uA'], color='#c0632c', alpha=.15)
    bs = bunch_s[(bunch_s >= lo) & (bunch_s <= hi)] - lo
    ax[0].plot(bs, np.full(len(bs), ax[0].get_ylim()[0]), '|', color='k', ms=7,
               label=f'{len(bs)} beam pulses')
    ax[0].set_xlabel('seconds into the 700 / 540 V plateau')
    ax[0].set_ylabel('current (µA)')
    ax[0].set_title('One scan point, as the monitor sees it')
    ax[0].legend(fontsize=7.5, loc='upper right')
    ax[0].grid(alpha=.3)

    ax[1].axis('off')
    txt = (f"$\\Delta I$ = mean $-$ median\n"
           f"     = {r0['mean_uA']:.4f} $-$ {r0['leak_uA']:.4f}\n"
           f"     = {r0['di_uA']*1000:.1f} nA\n\n"
           f"$f_{{pulse}}$ = {r0['n_bunch']} / {r0['duration_s']} s\n"
           f"     = {r0['f_pulse_hz']:.3f} Hz\n\n"
           f"$Q$ = $\\Delta I / f_{{pulse}}$\n"
           f"     = {r0['q_nC']:.0f} nC per pulse\n\n"
           f"({r0['n_imon']} samples, "
           f"{100*r0['frac_elevated']:.0f} % above baseline)")
    ax[1].text(0.02, 0.95, txt, va='top', ha='left', fontsize=11,
               family='monospace', transform=ax[1].transAxes)
    fig.tight_layout()
    fig.savefig(FIG / 'imon_method.png', dpi=140)
    plt.close(fig)

    # 3. full-detector comparison against the waveform. The chamber multiplier
    # comes from the board accounting (board_accounting.py): a uniform flash
    # gives one live Y comb capture x 1/2 / 465 of the chamber charge, so one
    # strip scales to the chamber by 465 / (0.85 x 0.5) = 1094.
    mult = round(N_Y_LIVE / (0.85 * 0.5))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for cls, c, mk in (('dedicated', '#2f6f9f', 'o'), ('parasitic', '#c0632c', 's')):
        pts = sorted((r['resist'], r[f'wf_{cls}_pC'] * mult / 1e3)
                     for r in rows if r['drift'] == 700 and f'wf_{cls}_pC' in r)
        ax.semilogy([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=c,
                    lw=1.3, ms=4, label=f'waveform $\\times$ {mult} (board), {cls}')
    mixpts = sorted((r['resist'], r['wf_mix_pC'] * mult / 1e3)
                    for r in rows if r['drift'] == 700 and 'wf_mix_pC' in r)
    ax.semilogy([p[0] for p in mixpts], [p[1] for p in mixpts], color='#6a6a6a',
                lw=1.1, ls=':', label=f'waveform $\\times$ {mult}, pulse mix')
    pts = sorted((r['resist'], r['q_nC'], r['q_err_nC'])
                 for r in rows if r['drift'] == 700)
    ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                yerr=[p[2] for p in pts], marker='D', color='#2e7d4f', lw=1.3, ms=4,
                label='HV supply current, whole chamber')
    imp = np.array([r['q_nC'] * 1e3 / r['wf_mix_pC'] for r in rows if 'wf_mix_pC' in r])
    ax.annotate(f'constant offset $\\times${mult/imp.mean():.1f}\n'
                f'(the local-density factor at strip 32)',
                xy=(0.05, 0.72), xycoords='axes fraction', fontsize=8.5,
                bbox=dict(fc='#f7f7f8', ec='#c0c0c0', lw=.8))
    ax.set_xlabel('detector A amplification voltage (V)')
    ax.set_ylabel('charge per beam pulse, whole chamber (nC)')
    ax.set_title('Same scan, same detector, two instruments')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'full_detector_compare.png', dpi=140)
    plt.close(fig)

    with open(HERE / 'results_imon.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'plateaus'}, indent=1))
    print(f"\n{'V':>5} {'imon nC':>9} {'wf mix pC':>10} {'implied N':>10} {'elev %':>7}")
    for r in rows:
        if r['drift'] != 700 or 'wf_mix_pC' not in r:
            continue
        print(f"{r['resist']:5d} {r['q_nC']:9.1f} {r['wf_mix_pC']:10.1f} "
              f"{r['q_nC']*1e3/r['wf_mix_pC']:10.0f} {100*r['frac_elevated']:7.0f}")


if __name__ == '__main__':
    main()
