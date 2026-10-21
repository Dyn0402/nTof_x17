#!/usr/bin/env python3
"""
gain_ref.py -- the detector's own gain against time since the flash, from the data, and
the energy scale (mV -> MeVee) per counter.

    python -m ntof_scint_recovery.gain_ref [--hists DIR]

The injection study (inject.py) adds pulses electronically, so it cannot see a gain change
in the PMT or SiPM.  Here the amplitude spectra themselves are the reference:

* A **gain change scales every quantile by the same factor**.  A change of particle mix
  changes the shape: some quantiles move, others do not.  So per time bin the quantiles
  q50, q75, q90, q95, q99 of the spectrum, each over its 1-10 ms value, are compared.
  Where they agree, the common factor is a gain; where they disagree, the mix moved.
* Populations: plastic pulses behind a wall hit (accidentals subtracted; the cleanest),
  plastic and liquid singles (template fit, > 20 mV), wall bars (geometric mean of the two ends,
  accidentals subtracted).  Median over bars (8 plastic bars, 16 wall bars).
* ``gain`` = the q99 ratio (the top of the capture-γ spectrum, least sensitive to the soft
  part of the mix and to small-pulse losses), with ``spread`` = max - min over q50..q99
  as the mix indicator.

Energy scale: the 30-32 ms window (cosmic muons, no beam) of the same data.  Plastics:
median plastic amplitude behind a wall hit = the MIP (3.45 MeVee for 20 mm, matched to the
muon calibration of ntof_calorimetry/c1, which agrees within 0-15 % per bar);
walls: the geometric mean of the two ends at the bar centre for a MIP (0.47 MeV in the 3 mm
bar, ntof_calorimetry/wp).  Liquids are left in mV (threshold-limited, no usable scale).

Writes ``out_dir('gain')/gain_quantiles.csv``, ``gain_curve.csv``, ``energy_scale.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import out_dir

Q = (0.5, 0.75, 0.9, 0.95, 0.99)
PLATEAU = (1e3, 1e4)
MIP_PLASTIC_MEVEE = 3.45
MIP_WALL_MEV = 0.47
CAL_PLASTIC = Path('/media/dylan/data/x17/calorimetry/c1/calib_plastic_e.json')


def quant(h, edges, q):
    c = np.cumsum(h)
    if c[-1] < 30:
        return np.nan
    return float(np.interp(q, np.r_[0, c / c[-1]], edges))


def curves(arrs, edges, t, pl):
    """Per counter: quantile ratios against the plateau, (n_counter, n_t, n_q)."""
    out = []
    for n in arrs:
        n = np.clip(n, 0, None)
        ref = [quant(n[pl].sum(0), edges, q) for q in Q]
        out.append([[quant(n[k], edges, q) / r for q, r in zip(Q, ref)] for k in range(len(t))])
    return np.array(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--hists', type=Path, default=out_dir('pulsehist_v3'))
    a = ap.parse_args(argv)
    from ntof_scint_recovery.coinc_analysis import load
    H = load(a.hists)
    tce, ae = H['tc_edges_us'], H['a_edges']
    tcc = np.sqrt(tce[:-1] * tce[1:])
    pl = (tcc > PLATEAU[0]) & (tcc < PLATEAU[1])
    te = H['t_edges_us']
    chans = [tuple(c) for c in H['channels']]
    rows = []
    net = H['PW_sig'] - H['pw_side_scale'] * H['PW_side']            # (cls, arm, bar, t, amp)
    BG = (H['BAR_g'] - H['BAR_g_acc']).sum(axis=4)                     # (cls, arm, bar, t, amp)
    # plastic singles on the coarse grid: sum fine bins into the coarse ones
    tf = np.sqrt(te[:-1] * te[1:])
    ic = np.digitize(tf, tce) - 1
    for cls, cname in ((0, 'par'), (1, 'ded')):
        pops = {}
        pops['plastic|wall'] = [net[cls, arm, bar] for arm in range(4) for bar in range(2)
                                if np.clip(net[cls, arm, bar], 0, None).sum() > 1000]
        pops['wall bar'] = [BG[cls, arm, g] for arm in range(4) for g in range(4)
                            if np.clip(BG[cls, arm, g], 0, None).sum() > 1e4]
        for fam_name, dets in (('plastic singles', range(4, 8)), ('liquid singles', range(8, 12))):
          sing = []
          for i, c in enumerate(chans):
            if c[0] not in dets:
                continue
            h = H['HF'][cls, i].copy()
            h[:, ae[:-1] < 20] = 0
            if h.sum() < 1e4:
                continue
            hc = np.zeros((len(tcc), h.shape[1]))
            for k in range(len(tcc)):
                hc[k] = h[ic == k].sum(axis=0)
            sing.append(hc)
          pops[fam_name] = sing
        for pname, arrs in pops.items():
            if not arrs:
                continue
            C = curves(arrs, ae, tcc, pl)
            med = np.nanmedian(C, axis=0)
            rms = np.nanstd(C, axis=0)
            for k in range(len(tcc)):
                rows.append(dict(pop=pname, cls=cname, t_us=tcc[k], t_lo_us=tce[k], t_hi_us=tce[k + 1],
                                 n_counters=len(arrs),
                                 **{f'q{int(q * 100)}': med[k, j] for j, q in enumerate(Q)},
                                 q99_rms=rms[k, -1]))
    D = pd.DataFrame(rows)
    D['spread'] = D[[f'q{int(q * 100)}' for q in Q]].max(axis=1) - D[[f'q{int(q * 100)}' for q in Q]].min(axis=1)
    o = out_dir('gain')
    D.to_csv(o / 'gain_quantiles.csv', index=False)
    G = D[['pop', 'cls', 't_us', 't_lo_us', 't_hi_us', 'q50', 'q99', 'q99_rms', 'spread']].rename(columns={'q99': 'gain'})
    G.to_csv(o / 'gain_curve.csv', index=False)

    # energy scale from the cosmic window (last coarse bin, 30-32 ms)
    cal = json.loads(CAL_PLASTIC.read_text())['bars'] if CAL_PLASTIC.exists() else {}
    erows = []
    for arm in range(4):
        for bar in range(2):
            n = np.clip(net[:, arm, bar, -1].sum(axis=0), 0, None)
            med = quant(n, ae, 0.5)
            name = f'PSS{"ABCD"[arm]}{bar + 1}'
            c = cal.get(name, {})
            erows.append(dict(counter=name, fam='PSS', cosmic_med_mV=med, n_cosmic=n.sum(),
                              mv_per_mevee=med / MIP_PLASTIC_MEVEE,
                              cal_mip_mV=c.get('mip_mpv_mv', np.nan), cal_mv_per_mevee=c.get('mv_per_mevee', np.nan)))
        for g in range(4):
            # bar centre: |dt| <= 4 ns of the dt grid (the MM calibration puts the centre within a few ns)
            dte = H['dt_edges']
            dc = 0.5 * (dte[1:] + dte[:-1])
            hh = (H['BAR_g'] - H['BAR_g_acc'])[:, arm, g, -1][:, np.abs(dc) <= 4].sum(axis=(0, 1))
            med = quant(np.clip(hh, 0, None), ae, 0.5)
            erows.append(dict(counter=f'WAL{"ABCD"[arm]} bar{g}', fam='WAL', cosmic_med_mV=med,
                              n_cosmic=np.clip(hh, 0, None).sum(), mv_per_mevee=med / MIP_WALL_MEV))
    E = pd.DataFrame(erows)
    E.to_csv(o / 'energy_scale.csv', index=False)
    pd.set_option('display.width', 250)
    print(D[D.cls == 'ded'].round(3).to_string(index=False))
    print(E.round(2).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
