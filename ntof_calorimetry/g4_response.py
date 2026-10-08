#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g4_response.py -- PLAN.md C2: what one arm's stack does with an electron or
positron of known kinetic energy, from the Geant4 single-particle runs
(`condor/submit_c2.py`, reduced by `condor/reduce_edep.py`; prompt deposits
only, MeV deposited -- no light model).

Per (particle, T, angle):
  P(plastic)        a bar takes > 0.3 MeV
  plastic deposit   median, 16/84 % (the soft-leg observable)
  missing energy    T - (gas + wall + plastic + liquid) for particles that
                    reached the plastic: the upstream + passive + escape loss,
                    the floor on any soft-leg energy resolution
  P(liquid)         the liquid takes > 1 MeV (punch-through)

    python -m ntof_calorimetry.g4_response pull     # EOS -> OUT/c2 (needs ssh lxplus)
    python -m ntof_calorimetry.g4_response          # -> OUT/c2/response.csv
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry.mip_sample import OUT  # noqa: E402

C2 = OUT / 'c2'
EOS = '/eos/experiment/ntof/data/x17/full_sim/calorimetry/c2_singles'
PLAS_MIN, LIQ_MIN = 0.3, 1.0


def pull() -> None:
    C2.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--include=*.csv.gz', '--exclude=*', f'lxplus:{EOS}/', f'{C2}/raw/'],
                   check=True)


def load() -> pd.DataFrame:
    rows = []
    for f in sorted((C2 / 'raw').glob('*MeV_th*_ph*.csv.gz')):
        if f.name.endswith('_events.csv.gz'):
            continue
        m = re.match(r'(e[mp])_([\d.]+)MeV_th(\d+)_ph(\d+)\.csv\.gz', f.name)
        p, T, th, ph = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
        n_ev = len(pd.read_csv(str(f).replace('.csv.gz', '_events.csv.gz')))
        d = pd.read_csv(f)
        # the arm the particle was aimed at is the one with the most deposit
        d['tot'] = d[['e_gas', 'e_wall', 'e_plas_L', 'e_plas_R', 'e_liq']].sum(axis=1)
        d = d.sort_values('tot', ascending=False).drop_duplicates('eventID')
        d = d.assign(particle='e-' if p == 'em' else 'e+', T=T, theta=th, phi=ph, n_events=n_ev)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def response(D: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (p, T, th, ph), g in D.groupby(['particle', 'T', 'theta', 'phi']):
        n = int(g.n_events.iloc[0])
        ep = g.e_plas_L + g.e_plas_R
        hit = ep > PLAS_MIN
        vis = g.e_gas + g.e_wall + ep + g.e_liq
        miss = (T - vis)[hit]
        stop = hit & (g.e_liq < 0.05)
        q = lambda s, x: float(np.quantile(s, x)) if len(s) else np.nan  # noqa: E731
        out.append(dict(particle=p, T=T, theta=th, phi=ph, n=n,
                        p_gas=float((g.e_gas > 0).sum() / n), p_plas=float(hit.sum() / n),
                        p_liq=float((g.e_liq > LIQ_MIN).sum() / n),
                        plas_med=q(ep[hit], .5), plas_q16=q(ep[hit], .16), plas_q84=q(ep[hit], .84),
                        wall_med=q(g.e_wall[hit], .5),
                        miss_med=q(miss, .5), miss_q16=q(miss, .16), miss_q84=q(miss, .84),
                        miss_med_stopped=q((T - vis)[stop], .5),
                        frac_stop_in_plas=float(stop.sum() / max(hit.sum(), 1))))
    return pd.DataFrame(out)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == 'pull':
        pull()
        return 0
    R = response(load())
    R.to_csv(C2 / 'response.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(R.round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
