#!/usr/bin/env python3
"""v4: drift-column timing per detector, de-noised.

Improvements over v3:
 - common mode per 64-ch DREAM-chip group (noise is per-connector), iterated with
   per-channel baseline; track strips excluded from the CM estimate on 2nd pass
 - hot-channel mask (channels hit in >20% of events are noise, not track)
 - robust per-channel MAD threshold + amplitude floor + width>=2
 - spatial cluster (adjacent strips, gap<=1, size>=3); largest cluster = track
 - outputs: earliest/latest/span percentiles, ceiling occupancy, latest-sample
   histogram, mean cluster-strip amplitude vs peak sample (attachment probe)

Usage: python drift_time_v4.py <decoded_dir_glob_base> <maxev_per_feu> 3=A_x_600 4=A_y_600 ...
  <decoded_dir_glob_base> may contain '*' (quoted) to span subruns; all file parts used.
"""
import sys, glob, json, numpy as np, uproot

SP = 60.0
FLOOR = 400.0
NSIG = 8
MINCL = 3
HOTFRAC = 0.20
NHITMAX = 100   # events with more hit strips are flash/shower/spark: fixed-channel
                # ringing after saturation fakes late drift; excluded from timing

def clusters(chs):
    chs = np.sort(chs); groups = []; cur = [chs[0]]
    for c in chs[1:]:
        if c - cur[-1] <= 1: cur.append(c)
        else: groups.append(cur); cur = [c]
    groups.append(cur)
    return groups

def denoise(M):
    """M[ch,32] raw. Returns residual after baseline + per-64ch-group CM, 2 passes."""
    nch = M.shape[0]
    R = M.astype(np.float32).copy()
    for it in range(2):
        # per-channel baseline
        R -= np.nanmedian(R, axis=1)[:, None]
        # per-group common mode; on 2nd pass mask strips with big peaks out of the median
        W = R.copy()
        if it == 1:
            mad = 1.4826 * np.nanmedian(np.abs(R), axis=1)
            big = np.nanmax(R, axis=1) > np.maximum(FLOOR / 2, 5 * mad)
            W[big, :] = np.nan
        for g0 in range(0, nch, 64):
            g1 = min(g0 + 64, nch)
            cm = np.nanmedian(W[g0:g1], axis=0)
            cm = np.where(np.isfinite(cm), cm, 0.0)
            R[g0:g1] -= cm[None, :]
    return R

def analyze(files, label, maxev):
    ev_earliest = []; ev_latest = []; ev_span = []; ev_nstr = []
    ev_first = []; ev_lastsig = []
    amp_by_sample = np.zeros(32); n_by_sample = np.zeros(32)
    latest_hist = np.zeros(32); earliest_hist = np.zeros(32); lastsig_hist = np.zeros(32)
    n_seen = 0; n_trk = 0; n_monster = 0
    late_ch = np.zeros(1024)
    for path in files:
        if n_seen >= maxev: break
        try:
            t = uproot.open(path)['nt']
        except Exception as e:
            print(f'  ! skip {path}: {e}'); continue
        n = min(t.num_entries, maxev - n_seen)
        if n <= 0: break
        S = t['sample'].array(library='np', entry_stop=n)
        C = t['channel'].array(library='np', entry_stop=n)
        A = t['amplitude'].array(library='np', entry_stop=n)
        # hot-channel pass: fraction of events where channel has a >FLOOR excursion (raw)
        nch_all = max(int(c.max()) for c in C if len(c)) + 1
        hitcnt = np.zeros(nch_all); evcnt = 0
        for i in range(min(n, 400)):
            a = A[i].astype(np.float32); c = C[i]; s = S[i]
            M = np.full((nch_all, 32), np.nan, np.float32); M[c, s] = a
            R = denoise(M)
            mad = 1.4826 * np.nanmedian(np.abs(R), axis=1)
            pk = np.nanmax(R, axis=1)
            hitcnt[:len(pk)] += (pk > np.maximum(FLOOR, NSIG * mad))
            evcnt += 1
        hot = hitcnt / max(evcnt, 1) > HOTFRAC
        for i in range(n):
            a = A[i].astype(np.float32); c = C[i]; s = S[i]
            M = np.full((nch_all, 32), np.nan, np.float32); M[c, s] = a
            R = denoise(M)
            mad = 1.4826 * np.nanmedian(np.abs(R), axis=1)
            peak = np.nanmax(R, axis=1)
            ps = np.nanargmax(np.nan_to_num(R, nan=-1e9), axis=1)
            rows = np.arange(nch_all)
            nb = np.maximum(R[rows, np.clip(ps - 1, 0, 31)], R[rows, np.clip(ps + 1, 0, 31)])
            thr = np.maximum(FLOOR, NSIG * mad)
            hit = (peak > thr) & (nb > thr / 2) & (~hot)
            hch = np.where(hit)[0]
            if hch.size > NHITMAX:
                n_monster += 1
                continue
            if hch.size >= MINCL:
                gs = [g for g in clusters(hch) if len(g) >= MINCL]
                if gs:
                    g = np.array(max(gs, key=len))
                    p = ps[g]; amp = peak[g]
                    # pulse rise/end: contiguous region around peak above thr/2
                    rises = []; ends = []
                    for ch in g:
                        pk = ps[ch]; half = max(thr[ch] / 2, 150.0)
                        r = pk
                        while r > 0 and np.nan_to_num(R[ch, r - 1], nan=-1e9) > half: r -= 1
                        e = pk
                        while e < 31 and np.nan_to_num(R[ch, e + 1], nan=-1e9) > half: e += 1
                        rises.append(r); ends.append(e)
                    ev_earliest.append(p.min()); ev_latest.append(p.max())
                    ev_first.append(min(rises)); ev_lastsig.append(max(ends))
                    ev_span.append(p.max() - p.min()); ev_nstr.append(len(g))
                    latest_hist[p.max()] += 1; earliest_hist[p.min()] += 1
                    lastsig_hist[max(ends)] += 1
                    for pp, aa in zip(p, amp):
                        amp_by_sample[pp] += aa; n_by_sample[pp] += 1
                    for ch, pp in zip(g, p):
                        if pp >= 28: late_ch[ch] += 1
                    n_trk += 1
        n_seen += n
    E = np.array(ev_earliest); L = np.array(ev_latest); SPn = np.array(ev_span); NS = np.array(ev_nstr)
    F = np.array(ev_first); LS = np.array(ev_lastsig)
    out = {'label': label, 'n_events': int(n_seen), 'n_track': int(n_trk), 'n_monster': int(n_monster)}
    print(f'\n== {label} ==  n={n_seen} track-evts={n_trk} monster={n_monster} med-clstrips={np.median(NS) if NS.size else 0:.0f}')
    if L.size > 30:
        def pc(x, q): return {str(p): float(np.percentile(x, p)) for p in q}
        out['earliest'] = pc(E, [5, 25, 50, 75, 95])
        out['first_sig'] = pc(F, [1, 5, 25, 50])
        out['latest'] = pc(L, [50, 75, 90, 95, 99])
        out['last_sig'] = pc(LS, [50, 75, 90, 95, 99])
        out['span'] = pc(SPn, [50, 75, 90, 95, 99])
        big = NS >= 6
        if big.sum() > 30:
            out['span_big'] = pc(SPn[big], [50, 75, 90, 95, 99])
            out['latest_big'] = pc(L[big], [50, 90, 95, 99])
            out['n_big'] = int(big.sum())
        out['ceil31'] = float(np.mean(L >= 31)); out['ceil30'] = float(np.mean(L >= 30)); out['ceil28'] = float(np.mean(L >= 28))
        out['ceil_sig31'] = float(np.mean(LS >= 31))
        out['latest_hist'] = latest_hist.tolist(); out['earliest_hist'] = earliest_hist.tolist()
        out['lastsig_hist'] = lastsig_hist.tolist()
        out['late_ch_top'] = {str(c): int(late_ch[c]) for c in np.argsort(late_ch)[::-1][:20] if late_ch[c] > 0}
        out['ev'] = {'earliest': [int(x) for x in E], 'latest': [int(x) for x in L],
                     'first': [int(x) for x in F], 'lastsig': [int(x) for x in LS],
                     'span': [int(x) for x in SPn], 'nstr': [int(x) for x in NS]}
        out['amp_vs_sample'] = (amp_by_sample / np.maximum(n_by_sample, 1)).tolist()
        out['n_vs_sample'] = n_by_sample.tolist()
        print(f"  first-sig p1/5/25/50: {[out['first_sig'][k] for k in ['1','5','25','50']]}   earliest-peak p5/50: {out['earliest']['5']}/{out['earliest']['50']}")
        print(f"  LATEST-peak  p50/90/95/99: {[out['latest'][k] for k in ['50','90','95','99']]}")
        print(f"  LAST-SIGNAL  p50/90/95/99: {[out['last_sig'][k] for k in ['50','90','95','99']]}")
        print(f"  SPAN    p50/90/95/99: {[out['span'][k] for k in ['50','90','95','99']]}")
        if 'span_big' in out:
            print(f"  SPAN(nstr>=6, n={out['n_big']}) p50/90/95/99: {[out['span_big'][k] for k in ['50','90','95','99']]}"
                  f"  -> v(p95,30mm)={30000/max(out['span_big']['95']*SP,1):.0f} um/ns")
        print(f"  ceiling peak>=31: {100*out['ceil31']:.1f}%  last-sig>=31: {100*out['ceil_sig31']:.1f}%")
    return out

if __name__ == '__main__':
    base = sys.argv[1]; maxev = int(sys.argv[2])
    results = {}
    for arg in sys.argv[3:]:
        feu, label = arg.split('=')
        fs = sorted(glob.glob(f'{base}/decoded_root/*_{int(feu):02d}.root'))
        if not fs:
            print(f'! no files for FEU {feu} under {base}'); continue
        results[label] = analyze(fs, label, maxev)
    outp = sys.argv[0].replace('.py', '_out.json')
    with open(outp, 'w') as f: json.dump(results, f)
    print(f'\nwrote {outp}')
