#!/usr/bin/env python3
"""band_census.py <key> --out DIR [--max-per-file N] -- the noise-band census of one bench row (REPROCESSING_TODO
§C item 13, PASS_PLAN §4), read straight from the pass's decoded files on EOS (rerun/inputs.plan, generation rr1:
decoded_root_fa80c06), no staging.

Band definition = rerun/satstudy/bands.py (the 10-26 characterisation, unchanged):
  per trigger and FEU, each channel's window mean after the 64-channel block common mode (pedestal = the file's own
  first N_PED_EVENTS, as wft.io.FeuReader) is "offset" when |mean| > OFF_SIG x its noise while its RMS about that mean
  stays < FLAT_RMS x noise (shifted, not pulsing); a chip (64 channels) is in a band when >= 8 of its channels are.
Writes <out>/<rel>/bands.npz: per view eid, ts, chip bitmask (uint8) and the number of offset channels, plus per-view
noise, the number of discharge chips (>= 8 strips of a chip above 3000 ADC, bands.py nulls item 6), and census.json (rates per view / chip, x-y coincidence, events scanned).  Spark-like triggers (> 150 strips
above 6 sigma in x + y, bands.py nulls) are counted apart so a band riding on a discharge is not confused with one on
a quiet trigger.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun')]
import inputs  # noqa: E402
from wft import io as wio  # noqa: E402

OFF_SIG, FLAT_RMS, BLK, CHIP_MIN = 5.0, 2.0, 64, 8
SPARK_STRIPS = 150


def scan_file(path, nmax):
    import uproot
    r = wio.FeuReader(path); t = uproot.open(path)['nt']
    stop = min(r.n_entries, nmax) if nmax else r.n_entries
    E, TS, MASK, NOFF, NSIG, NDIS = [], [], [], [], [], []
    for lo in range(0, stop, 1000):
        a = t.arrays(['eventId', 'timestamp', 'amplitude'], entry_start=lo, entry_stop=min(lo + 1000, stop),
                     library='np')
        for e, ts, amp in zip(a['eventId'], a['timestamp'], a['amplitude']):
            if len(amp) != r.n_sample * 512:
                continue
            w = amp.reshape(-1, 512).astype(np.float32) - r.ped
            cm = np.median(w.reshape(w.shape[0], 512 // BLK, BLK), axis=2)
            wc = w - np.repeat(cm, BLK, axis=1)
            mc, sd = wc.mean(0), wc.std(0)
            off = (np.abs(mc) > OFF_SIG * r.noise) & (sd < FLAT_RMS * r.noise)
            chips = off.reshape(8, 64).sum(1) >= CHIP_MIN
            E.append(int(e)); TS.append(int(ts)); NOFF.append(int(off.sum()))
            MASK.append(int(np.packbits(chips[::-1].astype(np.uint8))[0]))
            mx = wc.max(0)
            NSIG.append(int((mx > 6 * r.noise).sum()))
            NDIS.append(int(((mx > 3000).reshape(8, 64).sum(1) >= CHIP_MIN).sum()))
    return dict(eid=np.array(E, np.int64), ts=np.array(TS, np.int64), mask=np.array(MASK, np.uint8),
                noff=np.array(NOFF, np.int16), nsig=np.array(NSIG, np.int16), ndis=np.array(NDIS, np.int8)), r.noise, r.n_entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--out', required=True)
    ap.add_argument('--max-per-file', type=int, default=4000)
    a = ap.parse_args()
    row = inputs.row_for(a.key)
    files, _, _, feus = inputs.plan(row, 'rr1')
    fx, fy = int(row['feu_x']), int(row['feu_y'])
    res, info = {}, dict(key=a.key, max_per_file=a.max_per_file, files={})
    for v, feu in (('x', fx), ('y', fy)):
        parts, noise = [], []
        for src, name in files['decoded_root']:
            if not name.endswith(f'_{feu:02d}.root'):
                continue
            d, nz, ntot = scan_file(src, a.max_per_file)
            parts.append(d); noise.append(nz); info['files'][name] = dict(n_entries=int(ntot), scanned=int(len(d['eid'])))
            print(v, name, len(d['eid']), 'of', ntot, flush=True)
        for k in ('eid', 'ts', 'mask', 'noff', 'nsig', 'ndis'):
            res[f'{k}_{v}'] = np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)
        res[f'noise_{v}'] = np.median(np.array(noise), axis=0) if noise else np.zeros(512)
    rel = os.path.join(row['run'], row['subrun'], row['det'])
    od = os.path.join(a.out, rel); os.makedirs(od, exist_ok=True)
    np.savez_compressed(os.path.join(od, 'bands.npz'), **res)
    # census numbers on the triggers both views read
    ex, ey = res['eid_x'], res['eid_y']
    common, ix, iy = np.intersect1d(ex, ey, return_indices=True)
    bx, by = res['mask_x'][ix] > 0, res['mask_y'][iy] > 0
    spark = (res['nsig_x'][ix].astype(int) + res['nsig_y'][iy].astype(int)) > SPARK_STRIPS
    dis = (res['ndis_x'][ix] > 0) | (res['ndis_y'][iy] > 0)
    chips = lambda m: np.unpackbits(m[:, None], axis=1)[:, ::-1]
    S = dict(info, rel=rel, det=row['det'], resist_V=row.get('resist_V'), drift_V=row.get('drift_V'),
             n_triggers=int(len(common)), n_spark=int(spark.sum()), n_discharge=int(dis.sum()),
             p_band_no_discharge=float((bx | by)[~dis].mean()) if (~dis).any() else None,
             p_band_any=float((bx | by).mean()) if len(common) else None,
             p_band_x=float(bx.mean()) if len(common) else None, p_band_y=float(by.mean()) if len(common) else None,
             p_band_both=float((bx & by).mean()) if len(common) else None,
             p_band_quiet=float((bx | by)[~spark].mean()) if (~spark).any() else None,
             p_band_on_spark=float((bx | by)[spark].mean()) if spark.any() else None,
             chip_rate_x=chips(res['mask_x'][ix]).mean(0).round(5).tolist() if len(common) else None,
             chip_rate_y=chips(res['mask_y'][iy]).mean(0).round(5).tolist() if len(common) else None,
             chip_rate_x_quiet=chips(res['mask_x'][ix][~spark]).mean(0).round(5).tolist() if (~spark).any() else None,
             chip_rate_y_quiet=chips(res['mask_y'][iy][~spark]).mean(0).round(5).tolist() if (~spark).any() else None)
    json.dump(S, open(os.path.join(od, 'census.json'), 'w'), indent=1)
    print(json.dumps({k: S[k] for k in ('n_triggers', 'p_band_any', 'p_band_quiet', 'p_band_on_spark')}))


if __name__ == '__main__':
    main()
