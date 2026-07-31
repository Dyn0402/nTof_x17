"""Full-run amp/satuflag census on v12/224572, signed-decoding-correct bands.

Produces the table in ntof_handoff/README.md §8b(a). One row per tree over all
16 partials (3018 bunches), classifying hits against the REAL ceiling of
~63 800 counts rather than the ~31 000 baseline an unsigned decode suggests:

  band          31 000 < amp <= 63 800   ordinary half-scale pulses. The
                                         retracted "wrap" cut removed these.
  over          amp > 63 800             fit extrapolations through clipped
                                         samples; not measurements.
  over_unflag   over & ~satuflag         what a satuflag-only cut misses
                                         (9-15 % on LIQ, up to 100 % on PSS).
  satu & ~over  flagged, amp in range    what an amp-only cut misses.

Cut both: satuflag OR amp > 63 800. See FINDINGS_2026-07-29_signed_decoding.md.
Companion output: amp_ceiling_census_v12_224572.json (2026-07-30 run).

    python amp_ceiling_census.py out.json
"""
import uproot, numpy as np, glob, json, sys
files = sorted(glob.glob('/media/dylan/data/x17/ntof_reproc/v12_liqpileup/run224572_*.root'))
TREES = [f'{g}{q}' for g in ('LIQ','WAL','PSS') for q in 'ABCD']
acc = {t: dict(hits=0, gt31k=0, band=0, over=0, satu=0, satu_lt31k=0,
               over_unflag=0, phys=0, phys_band=0, phys_satu=0, amax=0.0,
               satu_amin=np.inf, satu_amax=0.0) for t in TREES}
bunches = set()
for i, f in enumerate(files):
    with uproot.open(f) as fh:
        for t in TREES:
            if t not in [k.split(';')[0] for k in fh.keys()]:
                continue
            a = fh[t].arrays(['amp','satuflag','tof','BunchNumber'], library='np')
            amp = a['amp'].astype(float); sf = a['satuflag'].astype(bool)
            hi = amp > 31_000; over = amp > 63_800
            band = hi & ~over; phys = a['tof'] > 1e6
            d = acc[t]
            d['hits'] += amp.size; d['gt31k'] += int(hi.sum())
            d['band'] += int(band.sum()); d['over'] += int(over.sum())
            d['satu'] += int(sf.sum()); d['satu_lt31k'] += int((sf & ~hi).sum())
            d['over_unflag'] += int((over & ~sf).sum())
            d['phys'] += int(phys.sum()); d['phys_band'] += int((band & phys).sum())
            d['phys_satu'] += int((sf & phys).sum())
            d['amax'] = max(d['amax'], float(amp.max()) if amp.size else 0.0)
            if sf.any():
                d['satu_amin'] = min(d['satu_amin'], float(amp[sf].min()))
                d['satu_amax'] = max(d['satu_amax'], float(amp[sf].max()))
            if t == 'LIQA':
                bunches.update(np.unique(a['BunchNumber']).tolist())
    print(f'  {i+1}/{len(files)} {f.split("/")[-1]}', flush=True)
for t in TREES:
    acc[t]['satu_amin'] = None if not np.isfinite(acc[t]['satu_amin']) else acc[t]['satu_amin']
acc['_bunches'] = len(bunches)
json.dump(acc, open(sys.argv[1], 'w'), indent=1)
print('bunches (LIQA):', len(bunches))
for t in TREES:
    d = acc[t]
    print(f"{t} hits {d['hits']:>10,} >31k {d['gt31k']:>6,} band {d['band']:>6,} "
          f">63.8k {d['over']:>5,} satu {d['satu']:>6,} over_unflag {d['over_unflag']:>5,} "
          f"satu_amp {d['satu_amin']}-{d['satu_amax']:.0f} amax {d['amax']:.0f}")
