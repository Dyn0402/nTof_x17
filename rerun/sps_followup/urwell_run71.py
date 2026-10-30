#!/usr/bin/env python3
"""urwell_run71.py --pass-dir D --feu1-dir F --out O.npz <key> [<key> ...]

The uRWELL telescope for the pass-1 run_71 RAW rows: for every det4 event of a pass table
(Analysis_pass1/sps/<key>/events.parquet), the leading cluster of each uRWELL plane-view (front / back, x / y), read
from FEU 1 of the same acquisition file and DREAM eventId.  FEU 1 carries both uRWELL planes (channels 0-255 front,
256-511 back, sps_beam_test_26/analysis/urw_mapping/mapping_urwell.csv).  The FEU 1 files are decoded with the pinned
decoder (fa80c06, the one the pass's FEU 3 files went through); banco's decoded_root/combined_hits for run_71 predate
the RAW decoder fix and merge events, so they are not used.

RAW baseline, per event, the same recipe as sps_build_cache._baseline without the det4 lead mask: per channel the
median of the first RAW_PRE samples, then per sample the median of each 64-channel block (the uRWELL clusters are a
few strips of a block), then the pre-sample median again.  A strip's amplitude is its maximum after RAW_PRE; a strip
is hit above N_SIG x its own noise (robust sigma of its pre samples over the file).  Clusters: hit strips of one
plane-view sorted in position, split at gaps > 3 mm (as pair_dataset.py); the lead cluster is the largest-charge one,
its position the charge-weighted centroid.

Writes per event: key, acq group, event_id, det4 x/y p0 + flags + own centroid ref_x/ref_y (for comparison only),
and {f,b}{x,y}_p / _n / _q.
"""
import argparse, csv, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
URW_MAP = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'urw_mapping', 'mapping_urwell.csv')
if not os.path.exists(URW_MAP):                  # a copy deployed next to the script
    URW_MAP = os.path.join(HERE, 'mapping_urwell.csv')
RAW_PRE, NS, N_SIG, GAP_MM = 8, 64, 5.0, 3.0
DET4_COLS = ['event_id', 'x_ok', 'x_quality_ok', 'x_p0', 'x_tan_theta', 'x_n_strips', 'x_q_sum',
             'y_ok', 'y_quality_ok', 'y_p0', 'y_tan_theta', 'y_n_strips', 'y_q_sum', 'ref_x', 'ref_y', 'acq_tag', 't_wall']


def urwell_map():
    view = np.full(512, '', dtype='<U1'); pos = np.full(512, np.nan); det = np.full(512, '', dtype='<U1')
    for r in csv.DictReader(open(URW_MAP)):
        c = int(r['channel']); view[c] = r['view']; pos[c] = float(r['position_mm'])
        det[c] = 'f' if r['detector'] == 'EIC_uRWELL_front' else 'b'
    return view, pos, det


def grid_of(ch, sm, am):
    g = np.full((512, NS), np.nan, np.float32)
    k = (sm >= 0) & (sm < NS)
    g[ch[k], sm[k]] = am[k]
    return g


def baseline(g):
    with np.errstate(all='ignore'):
        import warnings
        warnings.simplefilter('ignore', RuntimeWarning)
        g = g - np.nanmedian(g[:, :RAW_PRE], axis=1, keepdims=True)
        cm = np.nanmedian(g.reshape(8, 64, NS), axis=1)                  # (block, sample)
        g = g - np.repeat(cm, 64, axis=0)
        g = g - np.nanmedian(g[:, :RAW_PRE], axis=1, keepdims=True)
    return g


def lead_cluster(pos, amp):
    if not len(pos):
        return np.nan, 0, 0.0
    o = np.argsort(pos); pos, amp = pos[o], amp[o]
    cid = np.r_[0, np.cumsum(np.diff(pos) > GAP_MM)]
    q = np.bincount(cid, weights=amp); p = np.bincount(cid, weights=pos * amp) / q
    i = int(np.argmax(q))
    return float(p[i]), int(len(q)), float(q[i])


def main():
    import pandas as pd, uproot
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+'); ap.add_argument('--pass-dir', required=True); ap.add_argument('--feu1-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    view, pos, det = urwell_map()
    planes = [(d, v, (det == d) & (view == v)) for d in 'fb' for v in 'xy']
    rows, amps, tmaxs, noises, negs, tmins = [], [], [], [], [], []
    for key in a.keys:
        t = pd.read_parquet(os.path.join(a.pass_dir, key, 'events.parquet'), columns=DET4_COLS)
        for tag, sub in t.groupby('acq_tag'):
            f1 = os.path.join(a.feu1_dir, tag.replace('_03.root', '_01.root'))
            # the pass keys an event as eventId + 1e7 x (index of its file in the sub-run; sps_build_cache._fill)
            want = set(sub.event_id.astype(int) % 10_000_000)
            tr = uproot.open(f1)['nt']
            # per-channel noise from the pre samples of the first 400 events of this file
            pre = []
            got = {}
            for lo in range(0, tr.num_entries, 4000):
                arr = tr.arrays(['eventId', 'sample', 'channel', 'amplitude'], entry_start=lo,
                                entry_stop=min(lo + 4000, tr.num_entries), library='np')
                for j, eid in enumerate(arr['eventId']):
                    eid = int(eid)
                    if len(pre) < 400:
                        g = baseline(grid_of(arr['channel'][j].astype(int), arr['sample'][j].astype(int),
                                             arr['amplitude'][j].astype(np.float32)))
                        pre.append(g[:, :RAW_PRE])
                    if eid in want:
                        got[eid] = (arr['channel'][j].astype(int), arr['sample'][j].astype(int), arr['amplitude'][j].astype(np.float32))
                if len(got) == len(want):
                    break
            P = np.concatenate(pre, axis=1)
            med = np.nanmedian(P, axis=1, keepdims=True)
            noise = 1.4826 * np.nanmedian(np.abs(P - med), axis=1)
            noise[~np.isfinite(noise) | (noise <= 0)] = np.inf
            print(f'{key} {tag}: {len(want)} det4 events, {len(got)} found in FEU1; median noise {np.median(noise[np.isfinite(noise)]):.1f} ADC', flush=True)
            for _, r in sub.iterrows():
                eid = int(r.event_id) % 10_000_000
                o = dict(key=key, acq_tag=tag, **{c: r[c] for c in DET4_COLS if c not in ('acq_tag',)})
                o['urw_found'] = eid in got
                if eid in got:
                    g = baseline(grid_of(*got[eid]))
                    with np.errstate(all='ignore'):
                        amp = np.nanmax(g[:, RAW_PRE:], axis=1)
                    with np.errstate(all='ignore'):
                        tmax = np.nanargmax(np.nan_to_num(g[:, RAW_PRE:], nan=-1e9), axis=1) + RAW_PRE
                        neg = np.nanmax(-g[:, RAW_PRE:], axis=1)
                        tmin = np.nanargmax(np.nan_to_num(-g[:, RAW_PRE:], nan=-1e9), axis=1) + RAW_PRE
                    amps.append(amp.astype(np.float32)); tmaxs.append(tmax.astype(np.int8)); noises.append(noise.astype(np.float32))
                    negs.append(neg.astype(np.float32)); tmins.append(tmin.astype(np.int8))
                    hit = np.isfinite(amp) & (amp > N_SIG * noise)
                    for d, v, m in planes:
                        k = m & hit
                        p, n, q = lead_cluster(pos[k], amp[k])
                        o[f'{d}{v}_p'], o[f'{d}{v}_n'], o[f'{d}{v}_q'] = p, n, q
                rows.append(o)
    df = pd.DataFrame(rows)
    df.to_parquet(a.out, index=False)
    # per strip, for re-clustering offline: amplitude, sample of the maximum, the strip noise (rows with urw_found)
    np.savez_compressed(a.out.replace('.parquet', '_strips.npz'), amp=np.array(amps), tmax=np.array(tmaxs), neg=np.array(negs), tmin=np.array(tmins),
                        noise=np.array(noises), view=view, pos=pos, det=det)
    print(f'wrote {a.out}: {len(df)} rows, uRWELL found {df.urw_found.mean():.3f}')


if __name__ == '__main__':
    main()
