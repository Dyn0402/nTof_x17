#!/usr/bin/env python3
"""spark_timeline.py --key K [--out-dir D] -- whole-plane discharges of one bench pass-1 row: rate, the in-spark loss,
and the efficiency and gain as a function of the time since the previous discharge (Paper I §4; the pass-1 port of
mx_june_cosmic_qa/39_spark_deadtime.py).

Inputs (read only):
  hits    june_tests/<run>/<subrun>/combined_hits_root_rr1_cm0 (the pass's hits generation): eventId,
          trigger_timestamp_ns, feu, significance.  Hits are QA here (rates, a multiplicity), never geometry.
  rays    the Paper I loss budget's per-ray table (paper1_eff/budget/<run>/<subrun>/<det>/rays_<det>.parquet): every
          M3 recipe ray in the active area with its region and its category at R = 5 mm (cat_xy_5).
  table   Analysis_pass1/<run>/<subrun>/<det>/wft/events.parquet, for the charge of the detected rays.
Definitions (as the pass and the budget):
  vetoed     a trigger whose chamber multiplicity (strips over the per-plane significance floor SIG_REL_FLOOR, both
             views) exceeds wft.seed.SPARK_VETO_HITS = 50 -- the pass's spark veto and the budget's 'spark' class.
  discharge  a vetoed trigger with at least one chip of >= CHIP_MIN strips over DIS_ADC (band_census.py's test): a real
             avalanche breakdown.  burst = vetoed and NOT a discharge (no chip near the rail: the relative floor lets a
             low-amplitude coherent chip event or a shower through 50 strips; the known coherent chip events of det6/7).
  epoch      discharges closer than EPOCH_MS are one epoch (afterpulse trains), as 39_spark_deadtime.py.
  denominator  the budget's interior_good rays (region == interior and good_x), the efficiency session's region.
  detected   cat_xy_5 == 'detected'; in-spark = cat_xy_5 == 'spark'.
Per ray (not in a discharge): dt since the previous epoch; the efficiency and the median unsaturated charge in dt bins.
The DAQ check: the time to the next recorded trigger after a discharge against after any other trigger.
Writes <out-dir>/rows/<key>.json and <out-dir>/rays/<key>.npz.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO]
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS  # noqa: E402

BENCH = '/eos/experiment/ntof/data/x17/cosmic_bench/june_tests'
PASS = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
EFF = '/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_eff/budget'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
EPOCH_MS = 10.0
DIS_ADC, CHIP_MIN = 3000.0, 8          # rerun/effbudget/pass1/band_census.py: a chip with >= 8 strips over 3000 ADC
DT_EDGES_S = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0, np.inf])


def triggers(hits_dir, feus):
    """(eids, ts_s, mult, nchip): every trigger of the file group (any feu), this chamber's floor multiplicity, and its
    number of DREAM chips (64 channels) with >= CHIP_MIN strips above DIS_ADC -- band_census.py's discharge test."""
    import uproot
    fs = sorted(f for f in glob.glob(os.path.join(hits_dir, '*.root')) if '_datrun_' in f)
    E, T, M, C = [], [], [], []
    for f in fs:
        for a in uproot.iterate(f'{f}:hits', ['eventId', 'trigger_timestamp_ns', 'feu', 'channel', 'amplitude',
                                              'significance'], step_size='400 MB', library='pd'):
            ev = a.groupby('eventId').trigger_timestamp_ns.first()
            E.append(ev.index.to_numpy(np.int64)); T.append(ev.to_numpy(np.float64) * 1e-9)
            c = a[a.feu.isin(feus)]
            thr = c.groupby(['eventId', 'feu']).significance.transform('max') * SIG_REL_FLOOR
            M.append(c[c.significance >= thr].groupby('eventId').size())
            h = c[c.amplitude > DIS_ADC]
            n = h.groupby(['eventId', 'feu', h.channel // 64]).size()
            C.append((n >= CHIP_MIN).groupby(level=0).sum())
    eids = np.concatenate(E); ts = np.concatenate(T)
    o = np.argsort(eids, kind='stable'); eids, ts = eids[o], ts[o]
    u, i = np.unique(eids, return_index=True)       # an event split over two chunks: keep its first timestamp
    eids, ts = u, ts[i]
    mult = pd.concat(M).groupby(level=0).sum().reindex(eids, fill_value=0).to_numpy()
    nchip = pd.concat(C).groupby(level=0).sum().reindex(eids, fill_value=0).to_numpy()
    return eids, ts, mult, nchip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True)
    ap.add_argument('--registry', default=os.path.join(REPO, 'rerun', 'pass_registry.csv'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pass-root', default=PASS, help='reco tables root (default Analysis_pass1; two-track: Analysis_pass1_twotrack)')
    ap.add_argument('--eff-root', default=EFF, help='budget dir (default paper1_eff/budget; two-track: paper1_eff_twotrack/budget)')
    a = ap.parse_args()
    PASS_, EFF_ = a.pass_root, a.eff_root
    r = {x['key']: x for x in csv.DictReader(open(a.registry))}[a.key]
    rel = f"{r['run']}/{r['subrun']}/{r['det']}"
    meta = json.load(open(f'{PASS_}/{rel}/wft/events.meta.json'))
    feus = [int(meta['run']['feu_x']), int(meta['run']['feu_y'])]
    eids, ts, mult, nchip = triggers(f"{BENCH}/{r['run']}/{r['subrun']}/combined_hits_root_rr1_cm0", feus)
    veto = mult > SPARK_VETO_HITS
    spark = veto & (nchip >= 1)                          # the physical discharges
    burst = veto & ~spark
    # epochs: a discharge more than EPOCH_MS after the previous one opens a new epoch
    st = ts[spark]
    new = np.r_[True, np.diff(st) > EPOCH_MS * 1e-3] if len(st) else np.zeros(0, bool)
    ep_t = st[new]
    dur = float(ts.max() - ts.min()) if len(ts) > 1 else float('nan')
    # DAQ: time to the next trigger after a discharge vs after a normal trigger
    gap = np.diff(ts)
    gs, gn = gap[spark[:-1]], gap[~spark[:-1]]
    R = pd.read_parquet(f"{EFF_}/{rel}/rays_{SHORT[r['det']]}.parquet")
    R = R[(R.region == 'interior') & R.good_x.astype(bool)].copy()
    tr = np.interp(R.event_id.to_numpy(float), eids.astype(float), ts)   # rays with no hit at all: interpolated
    R['t'] = tr
    R['in_spark'] = R.cat_xy_5 == 'spark'
    k = np.clip(np.searchsorted(eids, R.event_id.to_numpy()), 0, len(eids) - 1)
    hit = eids[k] == R.event_id.to_numpy()
    R['in_discharge'] = R.in_spark.to_numpy() & hit & spark[k]
    R['in_burst'] = R.in_spark & ~R.in_discharge
    R['det'] = R.cat_xy_5 == 'detected'
    j = np.searchsorted(ep_t, tr, side='left') - 1       # previous epoch strictly before the ray
    R['dt'] = np.where(j >= 0, tr - ep_t[np.clip(j, 0, None)], np.inf) if len(ep_t) else np.inf
    T = pd.read_parquet(f'{PASS_}/{rel}/wft/events.parquet', columns=['event_id', 'x_q_sum', 'y_q_sum',
                                                                   'x_n_sat_strips', 'y_n_sat_strips'])
    T = T.drop_duplicates('event_id', keep=False).set_index('event_id')
    q = (T.x_q_sum + T.y_q_sum).where((T.x_n_sat_strips == 0) & (T.y_n_sat_strips == 0))
    R['q_unsat'] = q.reindex(R.event_id).to_numpy()
    N = R[~R.in_spark]
    bins = []
    for lo, hi in zip(DT_EDGES_S[:-1], DT_EDGES_S[1:]):
        b = N[(N.dt >= lo) & (N.dt < hi)]
        n, k = len(b), int(b.det.sum())
        qq = b.q_unsat[b.det].dropna()
        bins.append(dict(lo=float(lo), hi=float(hi), n=n, k=k, eff=k / n if n else None,
                         err=float(np.sqrt(max(k, 1) * max(n - k, 1) / n ** 3)) if n else None,
                         q_med=float(qq.median()) if len(qq) >= 10 else None, n_q=int(len(qq))))
    out = dict(key=a.key, det=r['det'], run=r['run'], subrun=r['subrun'], cls=r['cls'],
               resist_V=float(r['resist_V']), drift_V=float(r['drift_V']), feus=feus,
               n_trig=int(len(eids)), duration_s=dur, trig_rate_hz=len(eids) / dur if dur else None,
               n_veto=int(veto.sum()), veto_per_trig=float(veto.mean()),
               n_spark=int(spark.sum()), n_epoch=int(len(ep_t)), spark_per_trig=float(spark.mean()),
               n_burst=int(burst.sum()), burst_per_trig=float(burst.mean()),
               epoch_rate_hz=len(ep_t) / dur if dur else None,
               daq_gap_after_spark_med_s=float(np.median(gs)) if len(gs) else None,
               daq_gap_after_normal_med_s=float(np.median(gn)) if len(gn) else None,
               n_rays=int(len(R)), f_in_spark=float(R.in_spark.mean()) if len(R) else None,
               f_in_discharge=float(R.in_discharge.mean()) if len(R) else None,
               f_in_burst=float(R.in_burst.mean()) if len(R) else None,
               eff_all=float(R.det.mean()) if len(R) else None,
               eff_non_spark=float(N.det.mean()) if len(N) else None, dt_bins=bins,
               spark_veto_hits=SPARK_VETO_HITS, sig_rel_floor=SIG_REL_FLOOR, epoch_ms=EPOCH_MS,
               dis_adc=DIS_ADC, chip_min=CHIP_MIN)
    os.makedirs(f'{a.out_dir}/rows', exist_ok=True); os.makedirs(f'{a.out_dir}/rays', exist_ok=True)
    json.dump(out, open(f'{a.out_dir}/rows/{a.key}.json', 'w'), indent=1)
    np.savez_compressed(f'{a.out_dir}/rays/{a.key}.npz', event_id=R.event_id.to_numpy(), t=tr, dt=R.dt.to_numpy(),
                        det=R.det.to_numpy(), in_spark=R.in_spark.to_numpy(),
                        in_discharge=R.in_discharge.to_numpy(), in_burst=R.in_burst.to_numpy(), q_unsat=R.q_unsat.to_numpy(),
                        mesh_x=R.mesh_x.to_numpy(), mesh_y=R.mesh_y.to_numpy(), spark_t=st, epoch_t=ep_t,
                        mult_spark=mult[spark], mult_burst=mult[burst], nchip_veto=nchip[veto])
    print(json.dumps({k: v for k, v in out.items() if k != 'dt_bins'}))
    for b in bins:
        print(b)


if __name__ == '__main__':
    main()
