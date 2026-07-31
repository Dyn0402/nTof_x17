#!/usr/bin/env python3
"""
09_periodicity.py — is there a 49.9 mm (one-connector) periodicity in det4's gain?

08 shows no dead channels and band edges uncorrelated with connector boundaries,
but by eye several live bands sit close to the 100/150/250/300 mm boundaries. A
64-channel = 49.92 mm periodicity would be the fingerprint of something tied to
the readout segmentation, so it is worth testing rather than eyeballing.

Two tests on the 2 mm-binned log charge profile:
  * a periodogram over trial periods 10-120 mm, to see whether 49.92 mm stands
    out at all;
  * folding at 49.92 mm, with the modulation depth compared against the same
    profile folded at 200 random nearby periods (the null: any profile with this
    much structure produces *some* modulation at any period you fold it at).

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/09_periodicity.py g_det4
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONN_MM = 64 * 0.78                    # 49.92 mm


def fold_amplitude(x, v, period, nphase=12):
    """Peak-to-peak of the phase-folded mean, normalised to the profile's rms."""
    ph = (x % period) / period
    idx = np.clip((ph * nphase).astype(int), 0, nphase - 1)
    m = np.array([v[idx == i].mean() if (idx == i).sum() > 2 else np.nan
                  for i in range(nphase)])
    return float(np.nanmax(m) - np.nanmin(m)), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key', nargs='?', default='g_det4')
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()

    d = np.load(os.path.join(args.out, f'stripes_{args.key}.npz'))
    x, med = d['c'], d['med']
    ok = np.isfinite(med) & (med > 0)
    x, v = x[ok], np.log10(med[ok])
    v = v - v.mean()

    periods = np.linspace(10, 120, 1101)
    amps = np.array([fold_amplitude(x, v, p)[0] for p in periods])
    conn_amp, conn_prof = fold_amplitude(x, v, CONN_MM)

    # null: how often does a random trial period beat the connector pitch?
    rng = np.random.default_rng(12345)
    trial = rng.uniform(30, 80, 200)
    null = np.array([fold_amplitude(x, v, p)[0] for p in trial])
    pval = float((null >= conn_amp).mean())

    best = float(periods[np.argmax(amps)])
    rep = dict(run_key=args.key,
               connector_pitch_mm=CONN_MM,
               fold_amplitude_at_connector_pitch=conn_amp,
               best_period_mm=best,
               fold_amplitude_at_best=float(amps.max()),
               null_median_amplitude=float(np.median(null)),
               p_value_random_period_beats_connector=pval,
               profile_rms_dex=float(v.std()))
    with open(os.path.join(args.out, f'periodicity_{args.key}.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    axs[0].plot(periods, amps, 'k-', lw=1)
    axs[0].axvline(CONN_MM, color='#d55e00', ls='--',
                   label=f'connector pitch {CONN_MM:.1f} mm')
    axs[0].axhline(np.median(null), color='gray', ls=':',
                   label='median of random trial periods')
    axs[0].set_xlabel('trial period [mm]')
    axs[0].set_ylabel('folded peak-to-peak [dex of charge]')
    axs[0].set_title(f'{args.key} — periodogram of the charge profile')
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=.3)

    ph = (np.arange(len(conn_prof)) + .5) / len(conn_prof) * CONN_MM
    axs[1].plot(ph, conn_prof, 'o-', color='#d55e00')
    axs[1].axhline(0, color='gray', lw=.8)
    axs[1].set_xlabel('position within a 64-channel connector block [mm]')
    axs[1].set_ylabel('mean log10 charge - mean')
    axs[1].set_title('folded at the connector pitch — flat means no connector effect')
    axs[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'periodicity_{args.key}.png'), dpi=115)


if __name__ == '__main__':
    main()
