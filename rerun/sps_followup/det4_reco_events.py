#!/usr/bin/env python3
"""det4_reco_events.py --scan scan_<g>.npz --group <g> --out O.parquet [--jobs 4]

det4 reconstruction of the run_71 events the uRWELL telescope sees, for the position-resolution measurement
(README §i).  The pass-1 run71 rows hold the first 3000 det4 events of each plateau, and the telescope has a track
(front and back plane lit in one view) in only ~0.5 % of triggers, so those rows overlap it in a few tens of events.
Here the selection is telescope-first: the events of FEU 3 file <g> whose FEU 1 scan (urwell_scan.py) has front AND
back lit in at least one view are reconstructed exactly as the pass's run71 rows were (code a72b889):
  - the event record: sps_build_cache's RAW baseline (_baseline: pre-sample median, masked block common mode, pre median),
    brightest strip per view, peak >= 300 ADC, +-16 strips, both views required (same code path as _fill, restricted
    to an event list);
  - the fit: sps_reco._init / one (the pass's det4 bundle, H4 channel layer, dropped RAW samples censored, bench
    bow / v map / w0 / t0 dropped) with the row's v and K from rerun/sps_registry.csv for the event's plateau.
Plateau by wall clock as in build_cache (one acquisition: first event of file 000 / last of file 035 -> the sub-run
span in dream_daq.log), 20 s plateau edges dropped.  Output: one row per event with the wft columns, the plateau and
the four uRWELL clusters.
"""
import argparse, csv, glob, os, pickle, sys
import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis')]
import sps_build_cache as sbc                                    # noqa: E402
from sps_build_cache import POSITION_MM, VIEW, PITCH_MM, NOISE_ADC  # noqa: E402

RUN, SUB, DATASET = 'run_71', 'cfg_gain4.5_peaktime50', 'run71_raw'
LIT = 100.0          # 3-strip in-time sum [ADC] for a uRWELL plane-view to count as lit (urwell_scan.py PEAK_THR)


def wall_clock():
    files = sorted(glob.glob(os.path.join(sbc.RUNS, RUN, SUB, 'decoded_root_fa80c06', '*_datrun_*_03.root')))
    import uproot
    (t_lo, t_hi), _ = sbc.acquisitions(files, sbc.subrun_spans(RUN, SUB))[0]
    ts0 = uproot.open(files[0])['nt'].arrays(['timestamp'], entry_stop=1, library='np')['timestamp'][0]
    tl = uproot.open(files[-1])['nt']; ts1 = tl.arrays(['timestamp'], entry_start=tl.num_entries - 1, library='np')['timestamp'][0]
    return files, t_lo, ts0, (t_hi - t_lo) / float(ts1 - ts0)


def records(f, want, t_lo, ts0, scale, plateaus, half_win=16, qmin=300.0, nsmp=64):
    """{eid: (plateau, record)} for the events of FEU 3 file f in want: the body of sps_build_cache._fill."""
    import uproot
    t = uproot.open(f)['nt']; out = {}
    for lo in range(0, t.num_entries, 20000):
        arr = t.arrays(['eventId', 'timestamp', 'ftst', 'sample', 'channel', 'amplitude'], entry_start=lo,
                       entry_stop=min(lo + 20000, t.num_entries), library='np')
        tw = t_lo + (arr['timestamp'].astype(float) - ts0) * scale
        for j in range(len(tw)):
            eid = int(arr['eventId'][j])
            L = next((n for n, l0, l1 in plateaus if l0 <= tw[j] < l1), None)
            if eid not in want or L is None:
                continue
            ch = arr['channel'][j].astype(int); sm = arr['sample'][j].astype(int); am = arr['amplitude'][j].astype(np.float32)
            am = sbc._baseline(ch, sm, am, True)
            rec = dict(eid=eid, tan_x=0.0, tan_y=0.0, t_wall=float(tw[j]), file=os.path.basename(f), timestamp=int(arr['timestamp'][j]))
            ok = True
            for view in ('x', 'y'):
                k = VIEW[ch] == view
                if not k.any():
                    ok = False; break
                pk = np.zeros(512); np.maximum.at(pk, ch[k], am[k])
                c0 = int(np.argmax(pk))
                if pk[c0] < qmin:
                    ok = False; break
                sel = k & (np.abs(POSITION_MM[ch] - POSITION_MM[c0]) <= half_win * PITCH_MM)
                cc, ss, aa = ch[sel], sm[sel], am[sel]
                chs = np.unique(cc)
                if len(chs) < 4:
                    ok = False; break
                W = np.zeros((len(chs), nsmp), np.float32); ci = np.searchsorted(chs, cc); m = (ss >= 0) & (ss < nsmp)
                W[ci[m], ss[m]] = aa[m]
                o = np.argsort(POSITION_MM[chs]); q = np.clip(W, 0, None).sum(1)
                rec[view] = dict(ch=chs[o].astype(np.int16), pos=POSITION_MM[chs[o]].astype(np.float32), W=W[o],
                                 noise=np.full(len(chs), NOISE_ADC, np.float32))
                rec[f'ref_mesh_{view}'] = float((q * POSITION_MM[chs]).sum() / max(q.sum(), 1e-9))
                rec[f'ftst_{view}'] = int(arr['ftst'][j])
            if ok:
                out[eid] = (L, rec)
    return out


def main():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    import datasets, sps_reco
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', required=True); ap.add_argument('--group', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--work', default=os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    ap.add_argument('--bundle', default=sps_reco.BUNDLE)
    a = ap.parse_args()
    S = np.load(a.scan)
    lit = {p: S[p + '_s3'] > LIT for p in ('fx', 'fy', 'bx', 'by')}
    keep = (lit['fx'] & lit['bx']) | (lit['fy'] & lit['by'])
    want = set(S['eid'][keep].astype(int))
    files, t_lo, ts0, scale = wall_clock()
    f = [x for x in files if f'_{a.group}_03.root' in x][0]
    D = datasets.get(DATASET)
    plateaus = [(n, lo + 20.0, hi - 20.0) for n, lo, hi, dv, rv in D['plateaus']]
    R = records(f, want, t_lo, ts0, scale, plateaus)
    reg = {r['plateau']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv'))) if r['dataset'] == DATASET}
    tel = {k: S[k][keep] for k in S.files}
    ti = {int(e): i for i, e in enumerate(tel['eid'])}
    rows = []
    for L in sorted({L for L, _ in R.values()}):
        evs = {eid: rec for eid, (LL, rec) in R.items() if LL == L}
        row = reg[L]
        v = float(row['v_span_um_ns']) if row['v_span_um_ns'] else None
        k = int(row['K']) if row['K'] else int(np.ceil(28800.0 / (v or 12.0) / 60.0)) + 15
        cache = os.path.join(a.work, f'tel_{a.group}_{L}.pkl'); pickle.dump(evs, open(cache, 'wb'), protocol=4)
        with ProcessPoolExecutor(a.jobs, initializer=sps_reco._init, initargs=(cache, a.bundle, k, v, 64, 'h4', True)) as pool:
            out = list(pool.map(sps_reco.one, sorted(evs), chunksize=4))
        os.remove(cache)
        for r in out:
            r['plateau'] = L; r['group'] = a.group; r['v_used'] = v; r['K'] = k
            i = ti[int(r['event_id'])]
            for p in ('fx', 'fy', 'bx', 'by'):
                for s in ('_p', '_s3', '_npk'):
                    r[p + s] = tel[p + s][i]
            rows.append(r)
        print(f'group {a.group} {L}: {len(evs)} telescope events reconstructed (v {v}, K {k})', flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out, index=False)
    print(f'wrote {a.out}: {len(df)} rows ({len(want)} telescope events in the scan)')


if __name__ == '__main__':
    main()
