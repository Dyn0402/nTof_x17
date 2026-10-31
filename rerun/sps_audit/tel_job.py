#!/usr/bin/env python3
"""tel_job.py --run R --subrun S --group G --out O.parquet [--key REGISTRY_KEY | --nsamp 32 --raw 0] [--feu3-dir decoded_root]
              [--scan SCAN.npz] [--jobs 8] [--max-tel N]

One FEU file group of an SPS H4 run: the uRWELL telescope track and det4 reconstructed through the pass-1 SPS chain on
the SAME events, for the position-resolution audit (rerun/sps_audit/README.md).

Telescope:
  ZS runs  banco's combined_hits (feu 1; both uRWELL planes, sps_beam_test_26/analysis/urw_mapping/mapping_urwell.csv):
           per plane-view, hits clustered in position (gap 3 mm), the leading cluster by charge and the number of clusters
           -- exactly sps_beam_test_26/det4_sps_assessment/mapping_check/extract2.py, the input of the run-53 176 um.
  RAW run71  the followup's FEU 1 scan (rerun/sps_followup/urwell_scan.py: in-time 3-strip peak, 5-strip centroid, peak
           count), because banco's run71 combined_hits merge events (pre-fix RAW decoder).
det4 on the telescope events:
  hits     the run-53 estimator: feu-3 hits of combined_hits (amp > 60), clustered in position, lead cluster (ZS only).
  pass     sps_build_cache.event_record (the pass's cache code, WITH the 2026-10-30 seed fix) -> sps_reco._init/one with
           the pass bundle, the H4 channel layer, the row's v/K/samples/zeros.  The own charge centroid ref_* comes along.
  old      the same event with the pre-fix seed (seed_bad = none), fitted only where the seed differs.
Plateau (with --key): the registry row's [t_lo + 20 s, t_hi - 20 s] on the wall clock of sps_build_cache.build_cache.
"""
import argparse, csv, glob, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc                                    # noqa: E402
from sps_build_cache import POSITION_MM, VIEW                    # noqa: E402

URW_MAP = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'urw_mapping', 'mapping_urwell.csv')
GAP_MM = 3.0
DET4_AMP_MIN = 60.0
CHUNK = 5000


def urwell_map():
    view = np.full(512, '', dtype='<U1'); pos = np.full(512, np.nan); det = np.full(512, '', dtype='<U1')
    for r in csv.DictReader(open(URW_MAP)):
        c = int(r['channel']); view[c] = r['view']; pos[c] = float(r['position_mm'])
        det[c] = 'f' if r['detector'] == 'EIC_uRWELL_front' else 'b'
    return view, pos, det


def clusters(ev, pos, amp, n_ev, gap=GAP_MM):
    """Per event: leading-cluster position, its charge, its strip count and the number of clusters (extract2.py)."""
    lead = np.full(n_ev, np.nan); qlead = np.zeros(n_ev); ncl = np.zeros(n_ev, np.int16); size = np.zeros(n_ev, np.int16)
    if len(ev) == 0:
        return lead, qlead, ncl, size
    o = np.lexsort((pos, ev)); ev, pos, amp = ev[o], pos[o], amp[o]
    new = np.empty(len(ev), bool); new[0] = True
    new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > gap)
    cid = np.cumsum(new) - 1; nc = cid[-1] + 1
    cq = np.bincount(cid, weights=amp, minlength=nc)
    cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
    cn = np.bincount(cid, minlength=nc)
    cev = np.zeros(nc, np.int64); cev[cid] = ev
    np.add.at(ncl, cev, 1)
    order = np.argsort(cq, kind='stable')
    lead[cev[order]] = cp[order]; qlead[cev[order]] = cq[order]; size[cev[order]] = cn[order]
    return lead, qlead, ncl, size


def telescope_zs(ch_file):
    """{eid: row} for the events of one combined_hits file with a cluster in all four uRWELL plane-views, plus the
    det4 hits-cluster estimator (amp > 60, no time gate here; the hit times are kept for an offline gate)."""
    import uproot
    a = uproot.open(ch_file)['hits'].arrays(['feu', 'eventId', 'channel', 'amplitude', 'time'], library='np')
    feu, eid, ch = a['feu'], a['eventId'].astype(np.int64), a['channel'].astype(int)
    amp, tm = np.abs(a['amplitude']).astype(np.float64), a['time'].astype(np.float64)
    ev_u, ev_i = np.unique(eid, return_inverse=True); n = len(ev_u)
    view, pos, det = urwell_map()
    out = dict(eid=ev_u)
    m1 = feu == 1
    for d in 'fb':
        for v in 'xy':
            k = m1 & (det[ch] == d) & (view[ch] == v)
            p, q, nc, sz = clusters(ev_i[k], pos[ch[k]], amp[k], n)
            out[f'{d}{v}_p'], out[f'{d}{v}_q'], out[f'{d}{v}_n'], out[f'{d}{v}_size'] = p, q, nc, sz
    keep = np.all([out[f'{d}{v}_n'] >= 1 for d in 'fb' for v in 'xy'], axis=0)
    # det4 hits of the kept events (for the hits estimator; the gate is applied offline)
    m3 = (feu == 3) & keep[ev_i] & (amp > DET4_AMP_MIN)
    hits = dict(ev=ev_u[ev_i[m3]], ch=ch[m3].astype(np.int16), amp=amp[m3].astype(np.float32), time=tm[m3].astype(np.float32))
    return {k: v[keep] for k, v in out.items()}, hits


def telescope_raw(scan):
    """The followup's FEU 1 scan of one run71 group: events with front and back lit in at least one coordinate."""
    S = np.load(scan)
    if 'fx_size' in S.files:          # urw_raw_scan.py: ZS-like clusters; keep the four-view events as telescope_zs
        keep = np.all([S[f'{p}_n'] >= 1 for p in ('fx', 'fy', 'bx', 'by')], axis=0)
        return {k: S[k][keep] for k in S.files if k != 'n_bad'}, None
    lit = {p: S[p + '_s3'] > 100.0 for p in ('fx', 'fy', 'bx', 'by')}
    keep = (lit['fx'] & lit['bx']) | (lit['fy'] & lit['by'])
    out = dict(eid=S['eid'][keep].astype(np.int64))
    for p in ('fx', 'fy', 'bx', 'by'):
        out[p + '_p'] = S[p + '_p'][keep]; out[p + '_q'] = S[p + '_s3'][keep]; out[p + '_n'] = S[p + '_npk'][keep]
    return out, None


def wall_clock(run, subrun, files):
    """{file: (t_lo, ts0, scale)}: sps_build_cache.build_cache's per-acquisition timestamp calibration."""
    import uproot
    out = {}
    for (t_lo, t_hi), afiles in sbc.acquisitions(files, sbc.subrun_spans(run, subrun)):
        ts0 = uproot.open(afiles[0])['nt'].arrays(['timestamp'], entry_stop=1, library='np')['timestamp'][0]
        tl = uproot.open(afiles[-1])['nt']
        ts1 = tl.arrays(['timestamp'], entry_start=tl.num_entries - 1, library='np')['timestamp'][0]
        for f in afiles:
            out[f] = (t_lo, ts0, (t_hi - t_lo) / float(ts1 - ts0))
    return out


def hot_probe(f3, raw, nsamp, n=None):
    """The pass's hot-seed rule (sps_build_cache.hot_channels) on the first HOT_SCAN events of this file that pass the
    PASS selection (peak >= 300 ADC, >= 4 strips), seeded with SEED_BAD: the same probe sps_build_cache runs per plateau."""
    import uproot
    n = n or 2000                       # its own per-file probe (sbc.HOT_SCAN is None = the row's events since 10-31)
    t = uproot.open(f3)['nt']; cnt = np.zeros(512); k = 0
    for lo in range(0, t.num_entries, CHUNK):
        arr = t.arrays(['sample', 'channel', 'amplitude'], entry_start=lo, entry_stop=min(lo + CHUNK, t.num_entries), library='np')
        for j in range(len(arr['channel'])):
            r = sbc.event_record(arr['channel'][j].astype(int), arr['sample'][j].astype(int), arr['amplitude'][j].astype(np.float32),
                                 raw, nsamp)
            if r is not None:
                cnt[r['seed_ch_x']] += 1; cnt[r['seed_ch_y']] += 1; k += 1
            if k >= n:
                return sbc.hot_channels(cnt, k), k
    return sbc.hot_channels(cnt, max(k, 1)), k


def main():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    import sps_reco
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--subrun', required=True); ap.add_argument('--group', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--key', default='')
    ap.add_argument('--nsamp', type=int, default=64); ap.add_argument('--raw', type=int, default=0)
    ap.add_argument('--feu3-dir', default='decoded_root_fa80c06', help='under the sub-run, or an absolute directory')
    ap.add_argument('--scan', default='')
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--max-tel', type=int, default=0)
    ap.add_argument('--qmin', type=float, default=50.0, help='seed peak threshold; the pass: 300 ADC; here 5 x noise')
    ap.add_argument('--min-strips', type=int, default=1, help='strips with samples in the window; the pass: 4')
    ap.add_argument('--strict', type=int, default=1, help='keep only single-cluster four-view tracks for the reco')
    ap.add_argument('--work', default=os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    ap.add_argument('--bundle', default=sps_reco.BUNDLE)
    a = ap.parse_args()
    sub_dir = os.path.join(sbc.RUNS, a.run, a.subrun)
    f3dir = a.feu3_dir if os.path.isabs(a.feu3_dir) else os.path.join(sub_dir, a.feu3_dir)
    f3 = sorted(glob.glob(os.path.join(f3dir, f'*_datrun_*_{a.group}_03.root')))[0]
    if a.scan:
        tel, hits = telescope_raw(a.scan)
    else:
        chf = sorted(glob.glob(os.path.join(sub_dir, 'combined_hits_root', f'*_{a.group}_feu-combined_hits.root')))[0]
        tel, hits = telescope_zs(chf)
    print(f'{a.run} {a.subrun} {a.group}: {len(tel["eid"])} telescope events', flush=True)
    rows_k = {}
    if a.key:
        reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
        rows_k = {k_: reg[k_] for k_ in a.key.split(',')}
        r0 = next(iter(rows_k.values()))
        a.nsamp, a.raw = int(r0['n_samples']), int(r0['raw'])
    plateau = bool(rows_k)

    def key_of(tw):
        for k_, r_ in rows_k.items():
            if float(r_['t_lo_s']) + 20 <= tw < float(r_['t_hi_s']) - 20:
                return k_
        return None
    hot, nprobe = hot_probe(f3, bool(a.raw), a.nsamp)
    bad = sbc.SEED_BAD | hot
    print(f'  hot seed channels {list(np.flatnonzero(hot))} (probe of {nprobe} events)', flush=True)
    reco_sel = np.ones(len(tel['eid']), bool)
    if a.strict and 'fx_size' in tel:
        reco_sel = np.all([tel[f'{d}{v}_n'] == 1 for d in 'fb' for v in 'xy'], axis=0)
    want = set(tel['eid'][reco_sel].astype(int).tolist())
    if a.max_tel and len(want) > a.max_tel:
        want = set(sorted(want)[:a.max_tel])
    clock = None
    if plateau:
        files = sorted(glob.glob(os.path.join(f3dir, '*_datrun_*_03.root')))
        clock = wall_clock(a.run, a.subrun, files)[f3]
    import uproot
    t = uproot.open(f3)['nt']; recs, keyof = {}, {}
    for lo in range(0, t.num_entries, CHUNK):
        arr = t.arrays(['eventId', 'timestamp', 'ftst', 'sample', 'channel', 'amplitude'], entry_start=lo,
                       entry_stop=min(lo + CHUNK, t.num_entries), library='np')
        for j, eid in enumerate(arr['eventId']):
            eid = int(eid)
            if eid not in want:
                continue
            tw = clock[0] + (float(arr['timestamp'][j]) - clock[1]) * clock[2] if clock else np.nan
            kk = key_of(tw) if plateau else ''
            if kk is None:
                continue
            ch = arr['channel'][j].astype(int); sm = arr['sample'][j].astype(int); am = arr['amplitude'][j].astype(np.float32)
            r = sbc.event_record(ch, sm, am, bool(a.raw), a.nsamp, qmin=a.qmin, seed_bad=bad, min_strips=a.min_strips)
            if r is not None:
                r.update(eid=eid, t_wall=tw, file=os.path.basename(f3), timestamp=int(arr['timestamp'][j]),
                         ftst_x=int(arr['ftst'][j]), ftst_y=int(arr['ftst'][j]))
                recs[eid] = r; keyof[eid] = kk
    print(f'  det4 records: {len(recs)} of {len(want)} telescope tracks', flush=True)
    fits = {}
    extra = {p: [int(c) for c in np.flatnonzero(hot) if sbc.VIEW[c] == p] for p in ('x', 'y')}
    for kk in sorted(set(keyof.values())):
        row = rows_k.get(kk)
        v = float(row['v_span_um_ns']) if row and row['v_span_um_ns'] else None
        k = int(row['K']) if row and row['K'] else int(np.ceil(28800.0 / (v or 12.0) / 60.0)) + 15
        evs = {e: r_ for e, r_ in recs.items() if keyof[e] == kk}
        cache = os.path.join(a.work, f'tel_{a.run}_{a.group}.pkl'); pickle.dump(evs, open(cache, 'wb'), protocol=4)
        with ProcessPoolExecutor(a.jobs, initializer=sps_reco._init,
                                 initargs=(cache, a.bundle, k, v, a.nsamp, 'h4', bool(a.raw), extra)) as pool:
            fits.update({r_['event_id']: r_ for r_ in pool.map(sps_reco.one, sorted(evs), chunksize=4)})
        os.remove(cache)
        print(f'  {kk or "(no plateau)"}: v {v}, K {k}, {len(evs)} events fitted', flush=True)
    rows = []
    ti = {int(e): i for i, e in enumerate(tel['eid'])}
    for eid in sorted(recs):
        o = dict(run=a.run, subrun=a.subrun, group=a.group, key=keyof.get(eid, ''), eid=eid,
                 hot=' '.join(str(c) for c in np.flatnonzero(hot)))
        for kk, vv in tel.items():
            if kk != 'eid':
                o[kk] = vv[ti[eid]]
        r = recs[eid]
        for v_ in 'xy':
            o[f'seed_{v_}'] = r[f'seed_ch_{v_}']
            o[f'seed_amp_{v_}'] = float(np.asarray(r[v_]['W']).max())
            o[f'win_strips_{v_}'] = int(len(r[v_]['ch']))
        for c, val in fits.get(eid, {}).items():
            if c != 'event_id':
                o[f'd4_{c}'] = val
        rows.append(o)
    df = pd.DataFrame(rows) if rows else pd.DataFrame({'eid': pd.Series(dtype='int64'), 'group': pd.Series(dtype=str), 'key': pd.Series(dtype=str)})
    df.to_parquet(a.out, index=False)
    if hits is not None:
        sel = np.isin(hits['ev'], df.eid.values)
        np.savez_compressed(a.out.replace('.parquet', '_hits.npz'), **{kk: vv[sel] for kk, vv in hits.items()})
    print(f'wrote {a.out}: {len(df)} rows', flush=True)


if __name__ == '__main__':
    main()
