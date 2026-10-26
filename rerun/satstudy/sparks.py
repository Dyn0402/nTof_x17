#!/usr/bin/env python3
"""sparks.py -- what the interior "spark" rays of det3 are (budget.py: > SPARK_VETO_HITS = 50 strips above the 10 %
significance floor, the whole event dropped before the fit). Per event, from the full-plane waveforms (QA only):
  discharge     a view has > 150 strips above SNR 6, or > 100 strips swing below -10 sigma (chip-wide ringing):
                the plane itself discharged
  muon+other    in BOTH views the footprint cluster is no wider than a single track (wffeat 'wide' rule) and every
                other cluster is > 5 mm off the footprint: a clean muon plus an unrelated particle elsewhere; the
                veto counted the two together
  shower        the rest: the charge at the crossing is a wide / branching deposit (delta, shower, interaction)
Writes ~/x17/cosmic_bench/satstudy/sparks_det3.parquet and prints the split."""
import os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import satfit as S  # noqa: E402
from wffeat import feats  # noqa: E402

DIS_ON, DIS_NEG, WIDE = 150, 100, 8.0


def main():
    from wft.io import strip_position_map
    from qa_config import get_config
    cfg = get_config(S.KEY); pm = strip_position_map(cfg); pmv = {'x': pm[cfg.MX17_FEU_X], 'y': pm[cfg.MX17_FEU_Y]}
    E = pickle.load(open(os.path.join(S.OUT, 'wf_examples.pkl'), 'rb'))
    R = pd.read_parquet(os.path.expanduser('~/x17/cosmic_bench/effbudget/rays_det3.parquet')).set_index('event_id')
    rows = []
    for e in E['picks']['spark']:
        r = R.loc[e]; ev = E['wf'][e]; o = dict(event_id=e, mult=int(r.mult)); clean = True; dis = False
        for v in 'xy':
            W = ev[v].astype(np.float32); n = np.maximum(ev['noise_' + v].astype(np.float32), 3.0); pos = pmv[v]
            ok = np.isfinite(pos); snr = W.max(1) / n
            o[f'{v}_non'] = int((ok & (snr > 6)).sum()); o[f'{v}_nneg'] = int((ok & (W.min(1) / n < -10)).sum())
            dis |= o[f'{v}_non'] > DIS_ON or o[f'{v}_nneg'] > DIS_NEG
            f = feats(W, n, pos, r[f'mesh_{v}'], r[f'tan_{v}'])
            for k in ('snr_ref', 'ncl', 'ncl_far', 'wid1', 'dfoot1', 'nsatstrip'):
                o[f'{v}_{k}'] = f[k]
            lo, hi = sorted((r[f'mesh_{v}'], r[f'mesh_{v}'] + 30 * r[f'tan_{v}']))
            single = f['snr_ref'] > 10 and f['dfoot1'] == 0 and f['wid1'] <= (hi - lo) + 5 + WIDE
            # the largest cluster may be the OTHER particle: then the footprint cluster must be narrow on its own
            if f['dfoot1'] > 0 and f['snr_ref'] > 10:
                single = True
            clean &= single
        o['cls'] = 'discharge' if dis else ('muon+other' if clean else 'shower')
        rows.append(o)
    T = pd.DataFrame(rows); T.to_parquet(os.path.join(S.OUT, 'sparks_det3.parquet'))
    print(T.cls.value_counts().to_string())
    print(T.groupby('cls')[['mult', 'x_non', 'y_non', 'x_nsatstrip', 'y_nsatstrip']].median().to_string())


if __name__ == '__main__':
    main()
