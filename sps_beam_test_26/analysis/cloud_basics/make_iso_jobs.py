#!/usr/bin/env python3
"""make_iso_jobs.py -- Magboltz grid in the bench gas's isobutane fraction x water (x_width.py, FINDINGS §29).

The bench X width growth sits at ~dry-95/5 D_T while v(E) needs ~1 % water.  A water-only grid cannot
give a slow AND hot gas; less quencher can (cooler electrons sit in argon's Ramsauer minimum, so iso
raises v and lowers D_T).  Ar/iso (100-i)/i, i = 2-8 %, water 0-1 %, at the bench fields (V / 2.88 cm).
Tags benchi<iso>_w<water>_a0 (magboltz_drift.air_mix); 1.5e8 collisions (D_T and v only, no eta).

    make_iso_jobs.py > iso_jobs.txt     (lxplus ~/cloud_basics_condor/att; condor_mb_hs.sub with this list)
"""
NCOLL = 15
ISO = (2, 3, 4, 5, 6, 8)
WATER = (0, 0.25, 0.5, 0.75, 1.0)
FIELDS = (104, 174, 208, 243, 312, 347, 382)

# second pass (--ext): the first grid's joint det3 fit landed on its 2 % iso edge; extend to near-pure argon
# and wetter, so the grid edge does not decide whether a composition exists
ISO_EXT = (0.5, 1, 1.5)
WATER_EXT = (1.25, 1.5)

if __name__ == '__main__':
    import sys
    f = lambda x: f'{x:g}'.replace('.', 'p')
    if '--ext' in sys.argv:
        pts = [(i, w) for i in ISO_EXT for w in WATER + WATER_EXT] + [(i, w) for i in (2, 3) for w in WATER_EXT]
    else:
        pts = [(i, w) for i in ISO for w in WATER]
    for i, w in pts:
        for e in FIELDS:
            print(f'benchi{f(i)}_w{f(w)}_a0', e, NCOLL)
