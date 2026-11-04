#!/usr/bin/env python3
"""ringing.py -- the trigger-uncorrelated "clusters" of chamber D are a ~2.8 MHz oscillation, not particles.

Per FEU file (pedestal + CM as wft.io.FeuReader reads them), per event: the pulsed channels (peak > 5 sigma), and among
them the RINGING channels: >= 3 local maxima above 3 sigma, consecutive maxima 4-8 samples apart (240-480 ns).  A
drift signal on one strip is a single hump (column <= ~1 us, the RC copies of the neighbours are delayed by ~50 ns, not
hundreds), so three evenly spaced humps on one strip are not ionisation.  An event is RINGING in a plane when >= 4
channels ring.  Also kept: the median spacing of the maxima (the period), the odd-channel fraction of the ringing
channels, and the 64-channel blocks they sit in.

    ringing.py --out OUT.json [--max-events 2000]
    ringing.py --vs-time OUT.csv      # D, A, C (x and y) on run_145 stat090_0000 _000: ringing vs time since the flash
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
W = '/media/ucla/work/ntof_pass1/runs'
FEUS = dict(A=(3, 4), B=(5, 6), C=(7, 8), D=(1, 2))
SAMPLES = [('run_145', 'stat090_0000', '_000'), ('run_149', 'cosbounce_cos_0000', '_000')]


def humps(w, thr):
    """indices of local maxima of w above thr"""
    m = (w[1:-1] > w[:-2]) & (w[1:-1] >= w[2:]) & (w[1:-1] > thr)
    return np.flatnonzero(m) + 1


def scan(path, max_events):
    from wft import io as wio
    r = wio.FeuReader(path)
    n = nring = npuls = 0
    periods, odd, blocks, nch = [], [], np.zeros(8), []
    for eid, ftst, Wf in r.iter_events(None):
        n += 1
        noise = np.maximum(r.noise, 3.0)
        pk = Wf.max(1)
        pul = np.flatnonzero(pk > 5 * noise)
        if len(pul) >= 3:
            npuls += 1
        ring = []
        for c in pul:
            h = humps(Wf[c], 3 * noise[c])
            if len(h) >= 3:
                d = np.diff(h)
                if np.all((d >= 4) & (d <= 8)):
                    ring.append(c)
                    periods.append(float(np.median(d)))
        if len(ring) >= 4:
            nring += 1
            ring = np.array(ring)
            odd.append(float(np.mean(ring % 2 == 1)))
            blocks += np.bincount(ring // 64, minlength=8)
            nch.append(len(ring))
        if n >= max_events:
            break
    return dict(file=os.path.basename(path), n_events=n, frac_ge3_pulsed=npuls / max(n, 1),
                frac_ringing=nring / max(n, 1),
                period_samples_med=float(np.median(periods)) if periods else None,
                period_ns_med=float(np.median(periods)) * 60 if periods else None,
                ringing_odd_channel_frac=float(np.mean(odd)) if odd else None,
                ringing_channels_per_event_med=float(np.median(nch)) if nch else None,
                ringing_blocks=blocks.astype(int).tolist())


def n_ringing(Wf, noise):
    noise = np.maximum(noise, 3.0)
    pul = np.flatnonzero(Wf.max(1) > 5 * noise)
    nr = 0
    for c in pul:
        h = humps(Wf[c], 3 * noise[c])
        if len(h) >= 3 and np.all((np.diff(h) >= 4) & (np.diff(h) <= 8)):
            nr += 1
    return nr, len(pul)


def vs_time(out):
    """per event ringing flag joined to the scint stack's time since the flash and own-trigger flag (events with a
    track in some arm in the old production; the stack is the only per-event n_TOF clock for beam triggers)."""
    import pandas as pd
    from wft import io as wio
    st = pd.read_parquet('/media/dylan/data/x17/scint_stack/tracks/stack_run_145.parquet',
                         columns=['subrun', 'event_id', 'hw_A', 'hw_B', 'hw_C', 'hw_D', 't_since_flash_ns'])
    st = st[st.subrun.astype(str) == 'stat090_0000'].drop_duplicates('event_id').set_index('event_id')
    rows = []
    for arm in 'ADC':
        for plane, feu in zip('xy', FEUS[arm]):
            f = glob.glob(f'{W}/run_145/stat090_0000/decoded_root/*_000_0{feu}.root')[0]
            r = wio.FeuReader(f)
            ev = []
            for eid, ftst, Wf in r.iter_events(None):
                ev.append((eid,) + n_ringing(Wf, r.noise))
            d = pd.DataFrame(ev, columns=['event_id', 'nring', 'npul']).set_index('event_id').join(st, how='inner')
            ms = d.t_since_flash_ns / 1e6
            for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 45), (45, 80)):
                m = (ms >= lo) & (ms < hi)
                own = m & d[f'hw_{arm}'].astype(bool)
                rows.append(dict(arm=arm, plane=plane, t_lo_ms=lo, t_hi_ms=hi, n=int(m.sum()),
                                 frac_ringing=float((d.nring[m] >= 4).mean()),
                                 frac_ringing_own=float((d.nring[own] >= 4).mean()) if own.any() else np.nan,
                                 n_own=int(own.sum()), med_ringing_channels=float(d.nring[m].median())))
            print(arm, plane, 'done', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def example(out):
    """strip x time images for the report: D x beam with ringing (> 10 ms), D x beam on D's own trigger (> 10 ms),
    A x beam on A's own trigger (> 30 ms, quiet), D x cosmic on a B-D line muon."""
    import pandas as pd
    from wft import io as wio
    st = pd.read_parquet('/media/dylan/data/x17/scint_stack/tracks/stack_run_145.parquet',
                         columns=['subrun', 'event_id', 'hw_A', 'hw_D', 't_since_flash_ns'])
    st = st[st.subrun.astype(str) == 'stat090_0000'].drop_duplicates('event_id').set_index('event_id')
    L = pd.read_parquet('/media/ucla/work/ntof_pass1/d_b/line_kw/lines_D_v1.parquet')
    bd = set(L[(L.partner == 'B') & (L.miss_mm < 30) & (L.subrun == 'cosbounce_cos_0000')].event_id)
    want = [('D_beam_ring', 'run_145', 'stat090_0000', 1,
             lambda e, nr: e in st.index and st.t_since_flash_ns[e] > 1e7 and nr >= 10 and not st.hw_D[e]),
            ('D_beam_own', 'run_145', 'stat090_0000', 1,
             lambda e, nr: e in st.index and st.t_since_flash_ns[e] > 1e7 and bool(st.hw_D[e])),
            ('A_beam_own', 'run_145', 'stat090_0000', 3,
             lambda e, nr: e in st.index and st.t_since_flash_ns[e] > 3e7 and bool(st.hw_A[e]) and nr == 0),
            ('D_cosmic_BDline', 'run_149', 'cosbounce_cos_0000', 1, lambda e, nr: e in bd)]
    rec = {}
    for lab, run, sub, feu, ok in want:
        f = glob.glob(f'{W}/{run}/{sub}/decoded_root/*_000_0{feu}.root')[0]
        r = wio.FeuReader(f)
        for eid, ftst, Wf in r.iter_events(None):
            nr, npul = n_ringing(Wf, r.noise)
            if ok(eid, nr):
                rec[lab] = Wf.astype(np.float32)
                rec[f'{lab}_eid'] = np.array(eid)
                rec[f'{lab}_nring'] = np.array(nr)
                break
    np.savez_compressed(out, **rec)
    print('wrote', out, [k for k in rec if not k.endswith(('_eid', '_nring'))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None)
    ap.add_argument('--vs-time', default=None)
    ap.add_argument('--example', default=None)
    ap.add_argument('--max-events', type=int, default=2000)
    a = ap.parse_args()
    if a.vs_time:
        return vs_time(a.vs_time)
    if a.example:
        return example(a.example)
    res = []
    for run, sub, tag in SAMPLES:
        for arm, feus in FEUS.items():
            for plane, feu in zip('xy', feus):
                f = glob.glob(f'{W}/{run}/{sub}/decoded_root/*{tag}_0{feu}.root')[0]
                r = scan(f, a.max_events) | dict(run=run, subrun=sub, arm=arm, plane=plane, feu=feu)
                print(json.dumps({k: r[k] for k in ('run', 'arm', 'plane', 'n_events', 'frac_ge3_pulsed',
                                                     'frac_ringing', 'period_ns_med', 'ringing_odd_channel_frac',
                                                     'ringing_channels_per_event_med')}), flush=True)
                res.append(r)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
