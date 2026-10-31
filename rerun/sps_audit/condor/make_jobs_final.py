#!/usr/bin/env python3
"""jobs_final.txt for tel.sub (final telescope tables: final seed rule + pass hot probe, no pass cut, all estimators) and
keys_flat.txt for footprint.sub (every flat SPS row)."""
import csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = '/eos/user/d/dneff/x17rerun/sps_audit/scan_v2/scan_{g}.npz'
D53 = '/eos/user/d/dneff/x17rerun/sps_audit/run53_fa80c06'
K66 = '+'.join(f'run66_flat_resist__r{r}' for r in (580, 605, 630, 655, 680, 705, 730, 755, 780))
K71 = 'run71_raw__raw700+run71_raw__raw450+run71_raw__raw275'
L = []
for g in range(4):
    L.append(f'F53 run_53 cfg_gain4.5_peaktime200_deflt {g:03d} {D53} 32 - 50')
for sub, keys in (('meshscan_m00V', 'run55_flatdrift__d700+run55_flatdrift__d600'), ('meshscan_m10V', 'run55_flatdrift__d500'),
                  ('meshscan_m20V', 'run55_flatdrift__d400')):
    for g in range(2):
        L.append(f'F55{sub[-3:-1]} run_55 {sub} {g:03d} decoded_root_fa80c06 64 {keys} 50')
for sub, n in (('operating_02', 4), ('operating_03', 5)):
    for g in range(n):
        L.append(f'F63{sub[-2:]} run_63 {sub} {g:03d} decoded_root_fa80c06 64 run63_flat__flat700 50')
for g in range(4):
    L.append(f'F66 run_66 cfg_gain4.5_peaktime200_opt {g:03d} decoded_root_fa80c06 32 {K66} 50')
for g in range(36):
    L.append(f'F71 run_71 cfg_gain4.5_peaktime50 {g:03d} decoded_root_fa80c06 64 {K71} 50 ' + S2.format(g=f'{g:03d}'))
open(os.path.join(HERE, 'jobs_final.txt'), 'w').write('\n'.join(L) + '\n')
flat = [r['key'] for r in csv.DictReader(open(os.path.join(HERE, '..', '..', 'sps_registry.csv'))) if r['mount'] == 'flat']
open(os.path.join(HERE, 'keys_flat.txt'), 'w').write('\n'.join(flat) + '\n')
print(len(L), 'tel jobs;', len(flat), 'flat rows')
