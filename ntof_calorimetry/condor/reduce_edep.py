#!/usr/bin/env python3
"""
reduce_edep.py -- one mx17_full_sim ROOT file -> per-event energy deposits per
arm and scintillator layer (C2 single particles, C3 pairs).  Runs inside a
condor job on lxplus (LCG python with uproot), next to the file it reduces.

Output (CSV.gz, one row per event x arm with any deposit):
  eventID, armID, e_wall, e_plas_L, e_plas_R, e_liq, e_gas, n_wall_bars,
  t_plas (first plastic hit, ns)  -- MeV, PROMPT hits only (time < 1e8 ns:
  RadioactiveDecay is on and nothing in the sim cuts time; 28Al betas would
  otherwise be ~half the charged hits, CAMPAIGN_STATUS.md)
plus <out>_events.csv.gz with the EventTree truth.

    python3 reduce_edep.py <in.root> <out_prefix>
"""
import sys

import numpy as np
import pandas as pd
import uproot

T_MAX_NS = 1e8
LAYERS = {'PlasticScint': 'e_wall', 'BackScintL': 'e_plas_L', 'BackScintR': 'e_plas_R',
          'LiqScint_1': 'e_liq', 'DriftGas': 'e_gas'}


def main(path, out):
    f = uproot.open(path)
    ev = f['EventTree'].arrays(library='pd')
    ev.to_csv(out + '_events.csv.gz', index=False)
    parts = []
    for a in f['HitTree'].iterate(['eventID', 'armID', 'detType', 'edep', 'time', 'u'],
                                  step_size=2_000_000, library='np'):
        # Char[32] arrives as bytes on some uproot versions (g4_digi/extract_steps)
        dt = np.char.decode(a['detType'].astype('S32')) if a['detType'].dtype.kind in 'SO' \
            else a['detType'].astype(str)
        c = pd.DataFrame({k: a[k] for k in ('eventID', 'armID', 'edep', 'time', 'u')})
        c['detType'] = dt
        c = c[(c.time < T_MAX_NS) & c.detType.isin(list(LAYERS))]
        c = c.assign(layer=c.detType.map(LAYERS), e=c.edep * 1e-6)
        g = c.groupby(['eventID', 'armID', 'layer']).e.sum().unstack(fill_value=0.0)
        tp = c[c.layer.str.startswith('e_plas')].groupby(['eventID', 'armID']).time.min()
        nb = (c[c.layer == 'e_wall'].assign(bar=np.floor(c.u / 25.0))
              .groupby(['eventID', 'armID']).bar.nunique())
        g = g.join(tp.rename('t_plas')).join(nb.rename('n_wall_bars'))
        parts.append(g)
    D = pd.concat(parts)
    D = D.groupby(level=[0, 1]).agg({k: 'sum' for k in D.columns if k.startswith('e_')} |
                                    {'t_plas': 'min', 'n_wall_bars': 'max'})
    for k in LAYERS.values():
        if k not in D:
            D[k] = 0.0
    D.reset_index().to_csv(out + '.csv.gz', index=False, float_format='%.5g')
    print(f'{path}: {len(ev)} events, {len(D)} event-arms with deposits')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
