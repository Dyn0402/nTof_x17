#!/usr/bin/env python3
"""fdfcheck.py -- item (a): are the noise bands in the raw DREAM words, and does anything in the raw frame mark them?

    python fdfcheck.py [N_BAND]     # golden row sat_det3, file 003, FEU 7 (x) and 8 (y) -> $BANDS_OUT/fdfcheck.json

For N_BAND band events (and as many non-band controls) of the golden-row file 003, the raw .fdf frames are walked word
by word with an independent parser (big-endian 16-bit words; word classes as rerun/decoder_audit/audit_decode.py:
FEU header 8 words, then per DREAM chip a data header + 64 data words + data trailer, then a final trailer + VEP word).
For every (sample, channel) the 12-bit raw amplitude is compared with the decoded_root entry the band scan read.

It also keeps every non-data word of each chip block (data header and data trailer words, by position) and asks
whether any of them differs between band chips and quiet chips (a corrupted / flagged / differently-indexed block would
show here; the DREAM header carries the trigger id and the SCA cell index of the sample).
"""
import json, os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', 'satstudy'), os.path.join(HERE, '..', 'decoder_audit')]
import bands as B          # noqa: E402
import audit_decode as A   # noqa: E402

RUN = os.path.expanduser('~/x17/cosmic_bench/det3/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V')
TAG = 'MX17_long_run_resist_490V_drift_1000V_datrun_260627_22H49_003'


def walk(fdf, want):
    """{eventId: dict(amp=(ns,512) int, hdr=(ns,8,nh), trl=(ns,8,nt))} for the wanted eventIds."""
    w = np.memmap(fdf, dtype='>u2', mode='r')
    vep, hs = A.frame_starts(w)
    H = np.stack([np.asarray(w[np.minimum(hs + k, len(w) - 1)]).astype(np.int64) for k in range(8)], 1)
    evt = (H[:, 1] & 0xFFF) | ((H[:, 4] & 0xFFF) << 12)
    samp = ((H[:, 0] & 0x800) >> 3) + ((H[:, 3] & 0xFFF) >> 3)
    vset = set(vep.tolist())
    out, shapes = {}, Counter()
    nxt = np.r_[hs[1:], len(w)]
    for i in np.flatnonzero(np.isin(evt, list(want))):
        e = int(evt[i]); fr = np.asarray(w[hs[i]:nxt[i]]).astype(np.int64)
        cls = A.classes(fr.astype(np.uint16))
        for v in range(len(fr)):
            if hs[i] + v in vset:
                cls[v] = 5
        rs, rl = A.runs(cls)
        d = out.setdefault(e, dict(amp={}, hdr={}, trl={}))
        s = int(samp[i]); k = 0
        sh = []
        while k < len(rs):
            if cls[rs[k]] == 1 and k + 1 < len(rs) and cls[rs[k + 1]] == 0:
                hw = fr[rs[k]:rs[k] + rl[k]]; dw = fr[rs[k + 1]:rs[k + 1] + rl[k + 1]]
                tw = fr[rs[k + 2]:rs[k + 2] + rl[k + 2]] if k + 2 < len(rs) and cls[rs[k + 2]] == 2 else np.zeros(0, np.int64)
                dream = int((hw[-1] & 0xE00) >> 9)
                d['amp'][(s, dream)] = dw & 0xFFF
                d['hdr'][(s, dream)] = hw.copy(); d['trl'][(s, dream)] = tw.copy()
                sh.append((len(hw), len(dw), len(tw)))
                k += 3 if len(tw) else 2
            else:
                k += 1
        shapes[tuple(sh)] += 1
    return out, shapes


def main(nband=60):
    import uproot
    nband = int(nband)
    Z = np.load(B.SCAN)
    res = dict(file=TAG)
    for v, feu in (('x', 7), ('y', 8)):
        root = os.path.join(RUN, 'decoded_root', f'{TAG}_{feu:02d}.root')
        fdf = os.path.join(RUN, 'raw_daq_data', f'{TAG}_{feu:02d}.fdf')
        t = uproot.open(root)['nt']
        eids = t['eventId'].array(library='np')
        bv = B.band_events(Z, v); off = B.flags(Z, v)
        zi = {int(e): i for i, e in enumerate(Z[f'eid_{v}'])}
        inf = [e for e in eids.tolist() if e in zi]
        band = [e for e in inf if bv[zi[e]].any()]
        quiet = [e for e in inf if not bv[zi[e]].any()]
        rng = np.random.default_rng(1)
        pick_b = sorted(rng.choice(band, min(nband, len(band)), replace=False).tolist())
        pick_q = sorted(rng.choice(quiet, len(pick_b), replace=False).tolist())
        R, shapes = walk(fdf, set(pick_b + pick_q))
        nwords = nmis = 0; nev = 0
        hdr_band, hdr_q = Counter(), Counter()
        rows = []                                   # (chip, band, sample, chip raw mean, words...)
        for e in pick_b + pick_q:
            j = int(np.flatnonzero(eids == e)[0])
            a = t.arrays(['amplitude', 'sample', 'channel'], entry_start=j, entry_stop=j + 1, library='np')
            amp, sm, ch = a['amplitude'][0], a['sample'][0], a['channel'][0]
            mine = np.full((int(sm.max()) + 1, 512), -1, np.int64)
            for (s, dr), dw in R[e]['amp'].items():
                mine[s, dr * 64:dr * 64 + len(dw)] = dw
            nwords += len(amp); nmis += int((mine[sm, ch] != amp).sum()); nev += 1
            chips_b = off[zi[e]].reshape(8, 64).sum(1) >= 8
            for (s, dr), hw in R[e]['hdr'].items():
                tw = R[e]['trl'][(s, dr)]
                (hdr_band if chips_b[dr] else hdr_q)[(len(hw), len(tw))] += 1
                if len(hw) == 4 and len(tw) == 6:
                    rows.append([dr, int(chips_b[dr]), s, float(R[e]['amp'][(s, dr)].mean())] + [int(x) & 0xFFF for x in hw] + [int(x) & 0xFFF for x in tw])
        rows = np.array(rows, float)
        # per word position, conditioned on the chip: KS distance band-vs-quiet blocks of the SAME chip (chips that band),
        # and the correlation of the word with the chip's raw mean amplitude in that sample (a firmware baseline / CM word
        # would track it)
        words = []
        bch = np.unique(rows[rows[:, 1] == 1, 0]).astype(int)
        for p in range(4, rows.shape[1]):
            ks = []
            for c in bch:
                vb = rows[(rows[:, 0] == c) & (rows[:, 1] == 1), p]; vq = rows[(rows[:, 0] == c) & (rows[:, 1] == 0), p]
                if len(vb) < 50 or len(vq) < 50:
                    continue
                grid = np.unique(np.r_[vb, vq])
                ks.append(float(np.abs(np.searchsorted(np.sort(vb), grid, 'right') / len(vb)
                                       - np.searchsorted(np.sort(vq), grid, 'right') / len(vq)).max()))
            sd = rows[:, p].std()
            words.append(dict(pos=int(p - 4), part='hdr' if p < 8 else 'trl', n_distinct=int(len(np.unique(rows[:, p]))),
                              ks_same_chip=ks, corr_with_chip_mean=float(np.corrcoef(rows[:, p], rows[:, 3])[0, 1]) if sd > 0 else None,
                              corr_with_sample=float(np.corrcoef(rows[:, p], rows[:, 2])[0, 1]) if sd > 0 else None))
        cb = rows[rows[:, 1] == 1]; cq = rows[rows[:, 1] == 0]
        res[v] = dict(feu=feu, n_events=nev, n_band=len(pick_b), n_quiet=len(pick_q), words_compared=int(nwords),
                      mismatches=int(nmis), frame_shapes={str(k[:1]) + f'x{len(k)}': c for k, c in shapes.most_common(3)},
                      block_layout_band={str(k): c for k, c in hdr_band.items()},
                      block_layout_quiet={str(k): c for k, c in hdr_q.items()}, header_trailer_words=words,
                      n_band_blocks=int(len(cb)), n_quiet_blocks=int(len(cq)))
        print(v, json.dumps({k: res[v][k] for k in ('n_events', 'words_compared', 'mismatches', 'block_layout_band', 'block_layout_quiet')}), flush=True)
        for wd in words:
            print('  ', wd, flush=True)
    OUTD = os.environ.get('BANDS_OUT', os.path.expanduser('~/x17/cosmic_bench/satstudy/bands')); os.makedirs(OUTD, exist_ok=True)
    json.dump(res, open(os.path.join(OUTD, 'fdfcheck.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
