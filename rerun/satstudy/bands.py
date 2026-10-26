#!/usr/bin/env python3
"""bands.py -- the det3 "noise bands": groups of strips whose baseline sits flat at +50..+300 ADC (or below zero) for
the whole 1.9 us window in one event (Dylan 10-26: "figure out what they are and really characterize them").

    ../../.venv/bin/python bands.py scan [KEY [MAX]]   # every trigger of a run (default the golden row sat_det3),
                                                       # at most MAX per FEU file -> bands_scan[_KEY].npz
    ../../.venv/bin/python bands.py report   # characterisation numbers -> bands.json (+ figures in look/)

The scan keeps, per trigger and FEU, each channel's window mean before common mode (pedestal-subtracted only), the
per-sample block common mode itself, and each channel's max after CM (as wft.io.FeuReader, CM_MASK = 0), plus the
DREAM timestamp. A channel is "offset" in an event when its CM-subtracted window mean is > OFF_SIG x its pedestal
noise while its RMS about that mean stays < FLAT_RMS x noise: shifted, not pulsing.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402  (paths)

OUT = S.OUT
SCAN = os.path.join(OUT, 'bands_scan.npz')
OFF_SIG, FLAT_RMS = 5.0, 2.0
BLK = 64


def scan_path(key):
    return SCAN if key == S.KEY else SCAN.replace('.npz', f'_{key}.npz')


def scan(key=S.KEY, nmax=None):
    import uproot
    from qa_config import get_config
    from wft import io as wio
    cfg = get_config(key); res = {}; nmax = int(nmax) if nmax else None
    for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        E, TS, M0, MC, MX, CM, SD = [], [], [], [], [], [], []
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            r = wio.FeuReader(f); t = uproot.open(f)['nt']
            stop = min(r.n_entries, nmax) if nmax else r.n_entries
            for lo in range(0, stop, 2000):
                a = t.arrays(['eventId', 'timestamp', 'amplitude'], entry_start=lo,
                             entry_stop=min(lo + 2000, stop), library='np')
                for e, ts, amp in zip(a['eventId'], a['timestamp'], a['amplitude']):
                    if len(amp) != r.n_sample * 512:
                        continue
                    w = amp.reshape(-1, 512).astype(np.float32) - r.ped           # (ns, 512)
                    cm = np.median(w.reshape(w.shape[0], 512 // BLK, BLK), axis=2)  # (ns, 8)
                    wc = w - np.repeat(cm, BLK, axis=1)
                    E.append(int(e)); TS.append(int(ts))
                    M0.append(w.mean(0).astype(np.float16)); MC.append(wc.mean(0).astype(np.float16))
                    MX.append(wc.max(0).astype(np.float16)); SD.append(wc.std(0).astype(np.float16))
                    CM.append(cm.mean(0).astype(np.float16))
            res[f'noise_{v}'] = r.noise
            print(v, f, len(E), flush=True)
        res[f'feu_{v}'] = feu
        for k, arr in (('eid', E), ('ts', TS), ('m0', M0), ('mc', MC), ('mx', MX), ('sd', SD), ('cm', CM)):
            res[f'{k}_{v}'] = np.asarray(arr)
    np.savez_compressed(scan_path(key), **res)


KEYS = ('sat_det3', 'g_det3', 'g_det3_wknd', 'g_det2', 'g_det4', 'g_det6_long', 'g_det7_long')


def flags(Z, v):
    mc = Z[f'mc_{v}'].astype(float); sd = Z[f'sd_{v}'].astype(float); n = Z[f'noise_{v}']
    return (np.abs(mc) > OFF_SIG * n) & (sd < FLAT_RMS * n)


def band_events(Z, v):
    off = flags(Z, v)
    return off.reshape(len(off), 8, 64).sum(2) >= 8          # (event, chip)


def nulls(Z):
    """What the bands do NOT follow, golden row (x view = FEU 7; the y view gives the same answers)."""
    import pandas as pd
    res = {}
    b = band_events(Z, 'x').any(1) | band_events(Z, 'y').any(1)
    ts = Z['ts_x'].astype(np.int64); o = np.argsort(Z['eid_x']); ts, b = ts[o], b[o]
    # 1. the DREAM SCA cell the window was read from: (ts // 6) mod 512 in 32 bins
    c = ((ts // 6) % 512) * 32 // 512; A = np.bincount(c, minlength=32); H = np.bincount(c, b.astype(float), 32)
    res['sca_chi2_ndf'] = [float(((H - A * b.mean()) ** 2 / (A * b.mean())).sum()), 31]
    # 2. a 20 ms (mains) or any 10-32 ms period: largest Rayleigh R of the band trigger phases (DREAM tick = 10 ns)
    P = np.linspace(1.0e6, 3.2e6, 4401); tb = ts[b].astype(np.float64)
    R = np.abs(np.exp(2j * np.pi * (tb[None, :] % P[:, None]) / P[:, None]).mean(1))
    res['period_R_max'] = float(R.max()); res['period_R_random'] = float(np.sqrt(np.log(len(P)) / b.sum()))
    # 3. a discharge in the triggers before/after (> 150 strips above 6 sigma in x + y)
    nsig = sum((Z[f'mx_{v}'].astype(float) > 6 * Z[f'noise_{v}']).sum(1)[np.argsort(Z[f'eid_{v}'])] for v in 'xy')
    sp = nsig > 150
    res['p_band'] = float(b.mean())
    res['p_band_given_spark_lag'] = {str(l): float(b[np.flatnonzero(sp)[(np.flatnonzero(sp) + l >= 0) & (np.flatnonzero(sp) + l < len(b))] + l].mean())
                                     for l in (-2, -1, 0, 1, 2)}
    # 4. time since the previous trigger
    dt = np.diff(ts, prepend=ts[0]); q = np.quantile(dt[1:], [0, .1, .5, .9, 1])
    k = np.digitize(dt, q[1:-1]); res['p_band_by_dt_quartile'] = [float(b[k == i].mean()) for i in range(4)]
    # 5. HV: per-minute band rate against every imon / vmon
    Hv = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(Z_DECODED)), 'hv_monitor.csv'), parse_dates=['timestamp'])
    T = (Hv.timestamp - Hv.timestamp.iloc[0]).dt.total_seconds().to_numpy() - 29.0
    te = (ts - ts.min()) * 1e-8; m = np.digitize(te, np.arange(0, te.max() + 60, 60))
    rate = pd.Series(b).groupby(m).mean()
    res['hv_corr'] = {col: float(np.corrcoef(rate, pd.Series(np.interp(te, T, Hv[col].to_numpy(float))).groupby(m).mean())[0, 1])
                      for col in Hv.columns if col.endswith('imon') and Hv[col].std() > 0}
    # 6. in-window discharges (>= 8 strips of a chip above 3000 ADC) per chip, against the bands per chip
    for v in 'xy':
        mx = Z[f'mx_{v}'].astype(float)
        res[f'discharge_chips_{v}'] = ((mx > 3000).reshape(len(mx), 8, 64).sum(2) >= 8).mean(0).round(4).tolist()
        res[f'band_chips_{v}'] = band_events(Z, v).mean(0).round(4).tolist()
    return res


def decay(Z):
    """Do the offsets decay inside the 1.9 us window? Per band channel with |offset| > 300 ADC, the least-squares slope
    of its waveform divided by its offset [1/us]; an exponential tail of time constant tau gives -1/tau."""
    from qa_config import get_config
    from wft import io as wio
    cfg = get_config(S.KEY); res = {}
    for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        off = flags(Z, v); be = band_events(Z, v).any(1); e = Z[f'eid_{v}']
        want = set(e[be].tolist()); bo = {int(k): off[i] for i, k in enumerate(e) if k in want}; rel = []
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            for eid, ft, W in wio.FeuReader(f).iter_events(want):
                w = W[np.flatnonzero(bo[eid])]; a = w.mean(1); t = np.arange(w.shape[1]) * 0.06
                sl = ((w - a[:, None]) * (t - t.mean())).sum(1) / ((t - t.mean()) ** 2).sum()
                k = np.abs(a) > 300; rel.extend((sl[k] / a[k]).tolist())
        rel = np.array(rel)
        res[v] = dict(n=int(len(rel)), rel_slope_q={q: float(np.percentile(rel, q)) for q in (10, 25, 50, 75, 90)},
                      frac_toward_zero=float((rel < 0).mean()))
    return res


Z_DECODED = os.path.expanduser('~/x17/cosmic_bench/det3/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/decoded_root/x')


def report():
    """bands.json: rates per chamber / FEU / chip, x-y coincidence, what the bands do NOT correlate with, amplitude,
    and what they cost the golden row (detection with and without a band; truncation of band hits)."""
    import pandas as pd
    out = dict(runs={})
    for key in KEYS:
        p = scan_path(key)
        if not os.path.exists(p):
            continue
        Z = np.load(p); r = {}
        for v in 'xy':
            b = band_events(Z, v)
            r[v] = dict(feu=int(Z[f'feu_{v}']) if f'feu_{v}' in Z else (7 if v == 'x' else 8), n=int(len(b)),
                        rate=float(b.any(1).mean()), chips=b.mean(0).round(4).tolist())
        bx = pd.Series(band_events(Z, 'x').any(1), index=Z['eid_x']); by = pd.Series(band_events(Z, 'y').any(1), index=Z['eid_y'])
        j = pd.concat([bx, by], axis=1, join='inner')
        r['p_y_given_x'] = float(j[1][j[0]].mean()); r['p_x_given_y'] = float(j[0][j[1]].mean())
        out['runs'][key] = r
    Z = np.load(SCAN); g = {}
    for v in 'xy':
        off = flags(Z, v); mc = Z[f'mc_{v}'].astype(float); b = band_events(Z, v).any(1)
        A = np.abs(mc[b][off[b]])
        g[v] = dict(amp_q={q: float(np.percentile(A, q)) for q in (10, 50, 90, 99)},
                    frac_positive=float((mc[b][off[b]] > 0).mean()))
        ts = Z[f'ts_{v}'].astype(np.int64); te = (ts - ts.min()) * 1e-8
        k = np.digitize(te, np.arange(0, te.max() + 600, 600))
        g[v]['rate_per_10min'] = pd.Series(b).groupby(k).mean().round(4).tolist()
    out['golden'] = g
    out['nulls'] = nulls(Z)
    out['decay'] = decay(Z)
    # cost on the golden row
    R = pd.read_parquet(os.path.expanduser('~/x17/cosmic_bench/effbudget/rays_det3.parquet'))
    R = R[(R.region == 'interior') & ~R.not_measured].set_index('event_id')
    B = set()
    for v in 'xy':
        B |= set(Z[f'eid_{v}'][band_events(Z, v).any(1)].tolist())
    R['band'] = R.index.isin(B)
    out['cost'] = {str(bool(k)): dict(n=int(len(h)), detected=float((h.cat_xy_5 == 'detected').mean()),
                                      cats=h.cat_xy_5.value_counts(normalize=True).round(4).to_dict())
                   for k, h in R.groupby('band')}
    out['cost']['band_frac'] = float(R.band.mean())
    H = pd.read_parquet(os.path.join(OUT, 'hits_full_det3.parquet'))
    OFFS = {}
    for v, feu in (('x', 7), ('y', 8)):
        ii, cc = np.nonzero(flags(Z, v)); OFFS[feu] = set(zip(Z[f'eid_{v}'][ii].tolist(), cc.tolist()))
    H['offch'] = [(int(e), int(c)) in OFFS[int(f)] for e, c, f in zip(H.eventId, H.channel, H.feu)]
    both = H.trunc_left & H.trunc_right
    out['hits'] = dict(band_hits=int(H.offch.sum()), band_both_trunc=float(both[H.offch].mean()),
                       other_hits=int((~H.offch).sum()), other_both_trunc=float(both[~H.offch].mean()),
                       strong_other_both_trunc=float(both[~H.offch & (H.significance > 20)].mean()))
    json.dump(out, open(os.path.join(OUT, 'bands.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == '__main__':
    {'scan': scan, 'report': report}[sys.argv[1]](*sys.argv[2:])
