#!/usr/bin/env python3
"""Task 2: decomposition of the truth-level pointing scale with each electron's true creation vertex.

  e_c  = expected tan from the assumed centre (the sample's band crossings) to the gas line's mesh point (as the data)
  e_o  = tan of the straight line from the dominant gas track's creation vertex to the same mesh point
  s    = measured: the truth gas line (edep-weighted fit of every DriftGas step)

  K_full   = scale(e_c, s)      level (a)
  K_point  = scale(e_o, s)      source collapsed to each electron's own vertex: scattering (+ selection) only
  K_nosc   = scale(e_c, e_o)    no scattering (each electron flies straight from its vertex): source size (+ selection)
Selection variants: the data cut |s| < 0.5 (truncation by the measured angle) against |e_o| < 0.5 and no tan cut.
"""
import glob
import sys

import numpy as np
import pandas as pd

import w3lib as L

libs = sys.argv[1:] or ['nose']
rows, prof = [], []
for lib in libs:
    R = L.add_global(L.load_reduced(lib))
    O = pd.concat([pd.read_parquet(f, columns=['eventID', 'trackID', 'armID', 'ox', 'oy', 'oz', 'job'])
                   for f in sorted(glob.glob(str(L.HERE / f'origin_{lib}' / 'o*.parquet')))], ignore_index=True)
    O = O.drop_duplicates(['job', 'eventID', 'armID', 'trackID'])
    jobs = set(O.job.unique())
    R = R[R.job.isin(jobs)]
    n0 = len(R)
    R = R.merge(O.rename(columns={'armID': 'arm', 'trackID': 'dom_track'}), on=['job', 'eventID', 'arm', 'dom_track'],
                how='inner')
    print(lib, 'jobs', len(jobs), 'reduced rows', n0, 'matched', len(R), flush=True)
    base = R.full & (R.v_mesh.abs() <= 125) & ((R.A != 'D') | R.wall)
    cut5 = (R.sh.abs() < 0.5) & (R.tan_gap_v.abs() < 0.5)
    xs, zs, cr = L.source_from_crossings(R[base & cut5])
    ho = np.where(R.A == 'D', R.oz, R.ox)
    no = np.where(R.A == 'D', R.ox, R.oz)
    R['ho'], R['no'] = ho, no
    R['eo'] = (R.h - R.ho) / (R.n - R.no)
    # vertical line from the vertex, for a no-scattering version of the vertical cut
    R['evo'] = (R.v_mesh - R.oy) / (R.n - R.no)
    R['r_o'] = np.hypot(R.ox, R.oz)
    print(lib, 'source from crossings xs %.2f zs %.2f' % (xs, zs), flush=True)
    for arm in 'ACD':
        hs, ns = (xs, zs) if arm in 'AC' else (zs, xs)
        for sel_lab, sel in (('data cut |s|<0.5 (both views)', cut5),
                             ('cut on vertex line |e_o|,|e_vo|<0.5', (R.eo.abs() < 0.5) & (R.evo.abs() < 0.5)),
                             ('no tan cut', R.sh.abs() < 50)):
            g = R[base & sel & (R.A == arm)]
            ec = L.expected(g, arm, (xs, zs))
            out = dict(lib=lib, arm=arm, selection=sel_lab, n=len(g))
            for lab, (x, y) in (('K_full', (ec, g.sh.to_numpy())), ('K_point', (g.eo.to_numpy(), g.sh.to_numpy())),
                                ('K_nosc', (ec, g.eo.to_numpy()))):
                k, ke, a = L.boot_scale(x, y, nboot=150)
                out[lab], out[lab + '_err'] = k, ke
            # paired bootstrap of K_full - K_nosc - K_point + 1 (non-additivity)
            # vertex spread seen by this arm (band tracks)
            band = (np.abs(ec) >= 0.18) & (np.abs(ec) <= 0.44)
            dh = g.ho.to_numpy()[band] - hs
            dn = g.no.to_numpy()[band] - ns
            out.update(h_med=float(np.median(dh)), h_mad=float(1.4826 * np.median(np.abs(dh - np.median(dh)))),
                       h_rms_clip=float(np.std(dh[np.abs(dh) < 50])), n_med=float(np.median(dn)),
                       n_mad=float(1.4826 * np.median(np.abs(dn - np.median(dn)))),
                       f_far=float(np.mean(np.hypot(g.ox, g.oz).to_numpy()[band] > 30)),
                       L=float(np.median(np.abs(g.n - ns))),
                       res_sigma_full=float(1.4826 * np.median(np.abs((g.sh - ec)[band] - np.median((g.sh - ec)[band])))),
                       res_sigma_scat=float(1.4826 * np.median(np.abs((g.sh - g.eo)[band] - np.median((g.sh - g.eo)[band])))),
                       res_sigma_src=float(1.4826 * np.median(np.abs((g.eo - ec)[band] - np.median((g.eo - ec)[band])))))
            rows.append(out)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
            if sel_lab.startswith('data'):
                for lab, (x, y) in (('full', (ec, g.sh.to_numpy())), ('point', (g.eo.to_numpy(), g.sh.to_numpy())),
                                    ('nosc', (ec, g.eo.to_numpy()))):
                    k, a, B = L.scale(x, y, return_bins=True)
                    if B is not None:
                        B['lib'], B['arm'], B['case'] = lib, arm, lab
                        prof.append(B)
        # restrict to vertices near the capsule (r < 30 mm from the axis, |y| < 80): is the deficit from far vertices?
        for lab, m in (('vertex upstream of the chamber (|n_o| < 195 mm)', R['no'].abs() < 195),
                       ('vertex in the chamber (|n_o| >= 195 mm)', R['no'].abs() >= 195),
                       ('vertex r<30 mm, |oy|<80', (R.r_o < 30) & (R.oy.abs() < 80)),
                       ('vertex r>=30 mm or |oy|>=80', ~((R.r_o < 30) & (R.oy.abs() < 80)))):
            g = R[base & cut5 & (R.A == arm) & m]
            ec = L.expected(g, arm, (xs, zs))
            out = dict(lib=lib, arm=arm, selection='data cut; ' + lab, n=len(g))
            for kl, (x, y) in (('K_full', (ec, g.sh.to_numpy())), ('K_point', (g.eo.to_numpy(), g.sh.to_numpy())),
                               ('K_nosc', (ec, g.eo.to_numpy()))):
                k, ke, a = L.boot_scale(x, y, nboot=150, nmin=25)
                out[kl], out[kl + '_err'] = k, ke
            rows.append(out)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
pd.DataFrame(rows).to_csv(L.HERE / f'decompose_{"_".join(libs)}.csv', index=False)
if prof:
    pd.concat(prof).to_csv(L.HERE / f'decompose_profiles_{"_".join(libs)}.csv', index=False)
