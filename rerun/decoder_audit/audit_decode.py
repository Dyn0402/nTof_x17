#!/usr/bin/env python3
"""audit_decode.py <file.fdf> <decoded.root> <out.json> [<out.npz>] -- check the C++ DREAM decoder against an independent one.

The rerun re-decodes every June row with mm_strip_reconstruction fa80c06 (RERUN_PLAN §8.1).  Before trusting it, this
decodes the same .fdf a second, independent way -- vectorised numpy on the 16-bit word stream, written from the word
format (decoder/include/dreamdataline.h) and not from DreamDecoder.cpp's control flow -- and compares the two event by
event.  It also measures what the format itself says about each event (is every sample there, every channel, any
duplicate), which no comparison between two decoders can show.

Word classes (big-endian words; bits 14..12):
    0,1  data           12-bit amplitude (RAW)
    2,3  data header    4 words per DREAM block; the 4th carries the DREAM id (bits 11..9)
    4,5  data trailer
    6    FEU header     8 words: FeuID, eventID lo, ts, sampleID+ftst, eventID hi, ts, ts, ts
    7    final trailer  bit 11 = end of event; the word after it is a VEP word and is skipped

Independent event definition: an event is a run of consecutive FEU frames with the same eventID.  Data words count
only after a frame's header and before that frame's first final trailer, and only inside a <data header>(<data>)
block.  RAW mode only (every June bench row is RAW, run_config zero_suppress = False).

Content comparison is by two position-weighted hashes per event over (sample, channel, amplitude) in word order
(uint64, wrapping), plus the word count, eventId, timestamp and ftst -- exact on every field but the content, where a
difference goes unseen only if two independent 64-bit sums both cancel.  The file is processed in chunks cut at FEU
frame headers, so memory stays ~1-2 GB for a 500 MB file.
"""
import json
import sys
from collections import Counter

import numpy as np
import uproot

CHUNK = 16_000_000          # words per chunk (cut at the next frame header)
M = np.uint64
FLIP = int(__import__('os').environ.get('AUDIT_FLIP', 0))


def vals(s, ch, amp):
    s, ch, amp = (np.asarray(x).astype(np.uint64) for x in (s, ch, amp))
    v1 = amp * M(1000003) + ch * M(1009) + s * M(7919) + M(1)
    v2 = (amp + M(1)) * (ch + M(3)) * (s + M(11)) * M(2654435761) + amp * M(97)
    return v1, v2


def classes(w):
    c3 = ((w >> 12) & 7).astype(np.int8)
    return np.choose(c3, [0, 0, 1, 1, 2, 2, 3, 4]).astype(np.int8)   # 0 data 1 dhdr 2 dtrl 3 feu 4 ft


def runs(cls):
    b = np.flatnonzero(np.r_[True, cls[1:] != cls[:-1]])
    return b, np.diff(np.r_[b, len(cls)])


def frame_starts(w):
    """Global pass: VEP words (after each final trailer) and the start of every FEU header run."""
    n = len(w)
    ft, hs = [], []
    prev_last = -1
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        g = classes(w[a:b])
        ft.append(np.flatnonzero(g == 4) + a)
        st = np.flatnonzero((g == 3) & np.r_[prev_last != 3, g[:-1] != 3]) + a
        hs.append(st)
        prev_last = g[-1]
    ft = np.concatenate(ft)
    vep = []
    skip = -1
    for i in ft.tolist():
        if i == skip:
            continue
        skip = i + 1
        if skip < n:
            vep.append(skip)
    vep = np.array(vep, np.int64)
    hs = np.concatenate(hs)
    hs = hs[~np.isin(hs, vep)]
    return vep, hs


def decode(path):
    w = np.memmap(path, dtype='>u2', mode='r')
    n = len(w)
    vep, hs_all = frame_starts(w)
    rep = Counter()
    vep_cls = Counter(classes(np.asarray(w[vep])).tolist()) if len(vep) else Counter()
    rep_d = dict(n_words=int(n), vep_word_class={str(k): int(v) for k, v in vep_cls.items()})
    hl_c, dh_c, dl_c, orphan_len, orphan_nonzero = Counter(), Counter(), Counter(), Counter(), 0
    F = {k: [] for k in ('evt', 'samp', 'ts', 'ftst', 'feu', 'zs', 'nblk', 'ndream', 'nw', 'S1', 'T1', 'S2', 'T2',
                         'words_after_ft')}
    cuts = [0]
    for target in range(CHUNK, n, CHUNK):
        j = np.searchsorted(hs_all, target)
        if j < len(hs_all):
            cuts.append(int(hs_all[j]))
    cuts.append(n)
    cuts = sorted(set(cuts))
    for a, b in zip(cuts[:-1], cuts[1:]):
        wa = np.asarray(w[a:b]).astype(np.uint16)
        g = classes(wa)
        vl = vep[(vep >= a) & (vep < b)] - a
        g[vl] = 5
        hs = hs_all[(hs_all >= a) & (hs_all < b)] - a
        rs, rl = runs(g)
        rg = g[rs]
        is_h = np.isin(rs, hs)
        hl = rl[is_h]
        hl_c.update(hl.tolist())
        nf = len(hs)
        H = np.zeros((nf, 8), np.int64)
        for k in range(8):
            ok = hl > k
            H[ok, k] = wa[hs[ok] + k]
        F['feu'].append(H[:, 0] & 0xFF)
        F['zs'].append((H[:, 0] & 0x400) >> 10)
        F['samp'].append(((H[:, 0] & 0x800) >> 3) + ((H[:, 3] & 0xFF8) >> 3))
        F['ftst'].append(H[:, 3] & 7)
        F['evt'].append((H[:, 1] & 0xFFF) | ((H[:, 4] & 0xFFF) << 12))
        F['ts'].append((H[:, 2] & 0xFFF) + ((H[:, 5] & 0xFFF) << 12) + ((H[:, 6] & 0xFFF) << 24)
                       + ((H[:, 7] & 0xFF) << 36))
        # frame of each word; words after the frame's first final trailer are outside it
        fst = np.zeros(len(wa), np.int32)
        fst[hs] = 1
        fidx = np.cumsum(fst, dtype=np.int32) - 1
        del fst
        ftc = np.cumsum(g == 4, dtype=np.int32)
        ft0 = np.r_[0, ftc][hs]
        inside = fidx >= 0
        in_frame = np.zeros(len(wa), bool)
        in_frame[inside] = (ftc[inside] - ft0[fidx[inside]]) == 0
        # data/header/trailer words that sit outside every frame (after a trailer, before the next header)
        stray = (~in_frame) & (g <= 2)
        di = np.flatnonzero(rg == 0)
        di = di[di > 0]
        valid = (rg[di - 1] == 1) & in_frame[rs[di]]
        for s0, L in zip(rs[di[~valid]], rl[di[~valid]]):
            orphan_len[int(L)] += 1
            orphan_nonzero += int(np.any(wa[s0:s0 + L] != 0))
        stray_other = int(stray.sum()) - int(rl[di[~valid]].sum())
        rep['stray_non_data_words'] += max(0, stray_other)
        di = di[valid]
        dstart, dlen = rs[di], rl[di]
        hstart, hlen = rs[di - 1], rl[di - 1]
        dh_c.update(hlen.tolist())
        dl_c.update(dlen.tolist())
        dream = np.where(hlen >= 4, (wa[np.minimum(hstart + 3, len(wa) - 1)].astype(np.int64) & 0xE00) >> 9, 0)
        tot = int(dlen.sum())
        blk = np.repeat(np.arange(len(dstart)), dlen)
        off = np.arange(tot) - np.repeat(np.cumsum(dlen) - dlen, dlen)
        widx = dstart[blk] + off
        amp = wa[widx] & 0xFFF
        ch = dream[blk] * 64 + off
        fr = fidx[widx].astype(np.int64)
        del fidx, ftc, in_frame
        # position of each word inside its frame
        fw = np.bincount(fr, minlength=nf)
        kin = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(fw) - fw, fw)
        samp_w = F['samp'][-1][fr]
        v1, v2 = vals(samp_w, ch, amp)
        kk = (kin + 1).astype(np.uint64)
        S1 = np.zeros(nf, np.uint64); T1 = np.zeros(nf, np.uint64)
        S2 = np.zeros(nf, np.uint64); T2 = np.zeros(nf, np.uint64)
        np.add.at(S1, fr, v1); np.add.at(T1, fr, v1 * kk)
        np.add.at(S2, fr, v2); np.add.at(T2, fr, v2 * kk)
        F['S1'].append(S1); F['T1'].append(T1); F['S2'].append(S2); F['T2'].append(T2)
        F['nw'].append(fw)
        bf = fr[np.r_[True, np.diff(blk) != 0]] if tot else np.zeros(0, np.int64)       # frame of each block
        bd = dream[np.unique(blk)] if tot else np.zeros(0, np.int64)
        F['nblk'].append(np.bincount(bf, minlength=nf))
        F['ndream'].append(np.bincount(np.unique(bf * 8 + bd) // 8, minlength=nf))
        F['words_after_ft'].append(np.zeros(nf, np.int64))
    F = {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in F.items()}
    rep_d.update(feu_header_len={str(k): v for k, v in hl_c.items()}, dhdr_len={str(k): v for k, v in dh_c.items()},
                 data_run_len={str(k): v for k, v in dl_c.items()},
                 orphan_data_runs={str(k): v for k, v in orphan_len.items()}, orphan_runs_nonzero=orphan_nonzero,
                 stray_non_data_words=int(rep['stray_non_data_words']),
                 zs_frames=int(F['zs'].sum()), feu_ids=sorted(set(F['feu'].tolist())))
    # events = runs of frames with one eventID
    nf = len(F['evt'])
    first = np.flatnonzero(np.r_[True, F['evt'][1:] != F['evt'][:-1]])
    last = np.r_[first[1:], nf] - 1
    ev = np.repeat(np.arange(len(first)), np.diff(np.r_[first, nf]))
    # words before each frame inside its event
    cw = np.cumsum(F['nw']) - F['nw']
    p0 = (cw - cw[first][ev]).astype(np.uint64)
    h1 = np.zeros(len(first), np.uint64); h2 = np.zeros(len(first), np.uint64)
    np.add.at(h1, ev, p0 * F['S1'] + F['T1'])
    np.add.at(h2, ev, p0 * F['S2'] + F['T2'])
    nwe = np.bincount(ev, weights=F['nw'], minlength=len(first)).astype(np.int64)
    # sample coverage
    key = ev.astype(np.int64) * 1024 + F['samp']
    fr_with_data = F['nw'] > 0
    uk, ck = np.unique(key[fr_with_data], return_counts=True)
    nsamp = np.bincount((uk // 1024).astype(np.int64), minlength=len(first))
    smax = np.full(len(first), -1, np.int64)
    np.maximum.at(smax, (uk // 1024).astype(np.int64), uk % 1024)
    E = dict(eventId=F['evt'][first], timestamp=F['ts'][last], ftst=F['ftst'][last], ts_first=F['ts'][first],
             nframes=np.diff(np.r_[first, nf]), nwords=nwe, nsamp=nsamp, smax=smax, h1=h1, h2=h2,
             frames_full=np.bincount(ev, weights=(F['ndream'] == 8) & (F['nblk'] == 8) & (F['nw'] == 512),
                                     minlength=len(first)).astype(np.int64))
    rep_d['frames_repeating_sample_in_event'] = int((ck > 1).sum())
    rep_d['frames_without_data'] = int((~fr_with_data).sum())
    rep_d['frames_not_8x64'] = int((fr_with_data & ~((F['ndream'] == 8) & (F['nblk'] == 8) & (F['nw'] == 512))).sum())
    return rep_d, E


def cpp_hashes(root):
    f = uproot.open(root)
    t = f['nt']
    out = {k: [] for k in ('eventId', 'timestamp', 'ftst', 'nwords', 'h1', 'h2')}
    import awkward as ak
    for A in t.iterate(['eventId', 'timestamp', 'ftst', 'sample', 'channel', 'amplitude'], step_size=2000):
        cnt = ak.to_numpy(ak.num(A['amplitude']))
        fs, fc, fa = (ak.to_numpy(ak.flatten(A[b])) for b in ('sample', 'channel', 'amplitude'))
        if FLIP and not out['h1']:                 # negative control: one amplitude off by one must be caught
            fa = fa.astype(np.int64)
            fa[FLIP] += 1
        k = ak.to_numpy(ak.flatten(ak.local_index(A['amplitude']))).astype(np.uint64) + M(1)
        v1, v2 = vals(fs, fc, fa)
        e = np.repeat(np.arange(len(cnt)), cnt)
        h1 = np.zeros(len(cnt), np.uint64); h2 = np.zeros(len(cnt), np.uint64)
        np.add.at(h1, e, v1 * k); np.add.at(h2, e, v2 * k)
        out['h1'].append(h1); out['h2'].append(h2); out['nwords'].append(cnt)
        for b in ('eventId', 'timestamp', 'ftst'):
            out[b].append(ak.to_numpy(A[b]).astype(np.int64))
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    stats = None
    if 'decode_stats' in f:
        stats = {k: (v.tolist()[0] if len(v) else None) for k, v in f['decode_stats'].arrays(library='np').items()}
    return out, stats


def main():
    fdf, root, out = sys.argv[1:4]
    npz = sys.argv[4] if len(sys.argv) > 4 else None
    rep, E = decode(fdf)
    nev = len(E['eventId'])
    r = dict(fdf=fdf, root=root, indep=rep, n_indep=nev)
    mode_ns = int(np.bincount(E['nsamp']).argmax()) if nev else 0
    complete = (E['nsamp'] == mode_ns) & (E['smax'] == mode_ns - 1) & (E['nwords'] == mode_ns * 512) \
        & (E['frames_full'] == mode_ns)
    r.update(mode_samples=mode_ns, complete_events=int(complete.sum()),
             samples_per_event={str(k): int(v) for k, v in Counter(E['nsamp'].tolist()).most_common(12)},
             eventid_repeats=int(nev - len(np.unique(E['eventId']))),
             ts_differs_within_event=int((E['ts_first'] != E['timestamp']).sum()))
    de = np.diff(E['eventId'])
    r['eventid_steps'] = {str(k): int(v) for k, v in Counter(de.tolist()).most_common(10)}
    r['eventid_missing_in_span'] = int((de[de > 0] - 1).sum())
    C, stats = cpp_hashes(root)
    r['decode_stats'] = stats
    r['n_cpp'] = int(len(C['eventId']))
    nz = C['nwords'] > 0
    r['cpp_empty_entries'] = int((~nz).sum())
    cmp = {}
    if int(nz.sum()) == nev:
        for b, mine in (('eventId', E['eventId']), ('timestamp', E['timestamp']), ('ftst', E['ftst']),
                        ('nwords', E['nwords']), ('h1', E['h1']), ('h2', E['h2'])):
            cmp[b] = int((C[b][nz].astype(mine.dtype) != mine).sum())
        r['identical'] = all(v == 0 for v in cmp.values())
    else:
        r['identical'] = False
        common = np.intersect1d(C['eventId'][nz], E['eventId'])
        cmp.update(entries_cpp_nonempty=int(nz.sum()), eventids_common=int(len(common)),
                   only_cpp=int(nz.sum() - len(common)), only_indep=int(nev - len(common)))
    r['compare'] = cmp
    json.dump(r, open(out, 'w'), indent=1, default=int)
    if npz:
        np.savez_compressed(npz, eventId=E['eventId'], timestamp=E['timestamp'], ftst=E['ftst'], nsamp=E['nsamp'],
                            smax=E['smax'], nwords=E['nwords'], complete=complete)
    print(json.dumps(dict(identical=r['identical'], n_cpp=r['n_cpp'], n_indep=nev, compare=cmp,
                          complete=r['complete_events'], samples=r['samples_per_event'])))


if __name__ == '__main__':
    main()
