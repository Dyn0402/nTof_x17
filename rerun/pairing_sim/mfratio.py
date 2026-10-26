#!/usr/bin/env python3
"""mfratio.py -- a model-free x/y charge ratio, read the same way from data and simulation waveforms, to tell
whether a sim-vs-data difference in sigma(ln q_x/q_y) is in the waveforms or in the fit.

    ../../.venv/bin/python mfratio.py data [N]                         -> mf_data_det3.parquet
    ../../.venv/bin/python mfratio.py sim EVENTS.parquet DECODED_DIR TAG -> mf_sim_<TAG>.parquet

Per event and view: the strips whose position lies within MARGIN mm of the production fit's footprint
[p0, p0 + 30 tan] (the fitted line, used only to choose the strips), summed over all 32 samples of the pedestal- and
CM-corrected waveform (FeuReader, as wft reads it), and the same sum restricted to positive samples. Also the
plane's noise (FeuReader.noise) on those strips, to compare signal-to-noise. Data: det3 golden-row clean singles
(gates_data.singles). Sim: every event with exactly one candidate per view.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
OUT = os.path.expanduser('~/x17/cosmic_bench/satstudy/pairing_sim')
MARGIN = 3.0


def measure(fp, files, pm, feus):
    """fp: DataFrame index event_id with x_p0, x_tan_theta, y_p0, y_tan_theta."""
    from wft import io as wio
    rows = {}
    want = set(int(e) for e in fp.index)
    for v in 'xy':
        p = pm[feus[v]]
        for f in files[v]:
            r = wio.FeuReader(f)
            for eid, _ft, W in r.iter_events(want & set(r.event_ids.tolist())):
                a, t = fp.loc[eid, f'{v}_p0'], fp.loc[eid, f'{v}_tan_theta']
                lo, hi = min(a, a + 30 * t) - MARGIN, max(a, a + 30 * t) + MARGIN
                sel = np.isfinite(p) & (p >= lo) & (p <= hi)
                w = W[sel].astype(np.float64)
                d = rows.setdefault(int(eid), {})
                d[f'{v}_sum'] = float(w.sum()); d[f'{v}_pos'] = float(np.clip(w, 0, None).sum())
                d[f'{v}_peak'] = float(w.max()) if w.size else 0.0
                d[f'{v}_nstr'] = int(sel.sum()); d[f'{v}_noise'] = float(np.median(r.noise[sel])) if sel.any() else 0.0
    return pd.DataFrame.from_dict(rows, orient='index')


def data(nmax=3000):
    import gates_data as G
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import io as wio
    key, w, R, _ = G.singles('det3')
    fp = pd.DataFrame({'x_p0': w.p0.x, 'x_tan_theta': w.tan_theta.x, 'y_p0': w.p0.y, 'y_tan_theta': w.tan_theta.y,
                       'x_q_sum': w.q_sum.x, 'y_q_sum': w.q_sum.y,
                       'tan_x_m3': R.tan_x, 'tan_y_m3': R.tan_y}).iloc[:nmax]
    cfg = get_config(key); pm = wio.strip_position_map(cfg)
    feus = dict(x=cfg.MX17_FEU_X, y=cfg.MX17_FEU_Y)
    files = {v: wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feus[v]) for v in 'xy'}
    M = measure(fp, files, pm, feus)
    fp.join(M, how='inner').to_parquet(os.path.join(OUT, 'mf_data_det3.parquet'))


def sim(events, decoded, tag):
    import glob
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import io as wio
    E = pd.read_parquet(events).set_index('event_id')
    E = E[E.x_ok.astype(bool) & E.y_ok.astype(bool) & (E.x_n_candidates == 1) & (E.y_n_candidates == 1)]
    cfg = get_config('sat_det3'); pm = wio.strip_position_map(cfg)
    feus = dict(x=cfg.MX17_FEU_X, y=cfg.MX17_FEU_Y)
    files = {v: sorted(glob.glob(os.path.join(decoded, f'*_{feus[v]:02d}.root'))) for v in 'xy'}
    M = measure(E, files, pm, feus)
    E[['x_p0', 'x_tan_theta', 'y_p0', 'y_tan_theta', 'x_q_sum', 'y_q_sum']].join(M, how='inner').to_parquet(
        os.path.join(OUT, f'mf_sim_{tag}.parquet'))


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    if sys.argv[1] == 'data':
        data(int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
    else:
        sim(*sys.argv[2:5])
