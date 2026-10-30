#!/usr/bin/env python3
"""Write jobs.txt for tel.sub: one line per (dataset tag, FEU file group)."""
SCAN = '/eos/user/d/dneff/x17rerun/sps_followup/scan/scan_{g}.npz'
K66 = '+'.join(f'run66_flat_resist__r{r}' for r in (580, 605, 630, 655, 680, 705, 730, 755, 780))
K71 = 'run71_raw__raw700+run71_raw__raw450+run71_raw__raw275'
lines = []
for g in range(4):
    lines.append(f'r53 run_53 cfg_gain4.5_peaktime200_deflt {g:03d} decoded_root 32 - 300')
    lines.append(f'r53q100 run_53 cfg_gain4.5_peaktime200_deflt {g:03d} decoded_root 32 - 100')
for g in range(2):
    lines.append(f'r55 run_55 meshscan_m00V {g:03d} decoded_root_fa80c06 64 run55_flatdrift__d700+run55_flatdrift__d600 300')
for g in range(4):
    lines.append(f'r63op02 run_63 operating_02 {g:03d} decoded_root_fa80c06 64 run63_flat__flat700 300')
for g in range(5):
    lines.append(f'r63op03 run_63 operating_03 {g:03d} decoded_root_fa80c06 64 run63_flat__flat700 300')
for g in range(4):
    lines.append(f'r66 run_66 cfg_gain4.5_peaktime200_opt {g:03d} decoded_root_fa80c06 32 {K66} 300')
for g in range(36):
    lines.append(f'r71 run_71 cfg_gain4.5_peaktime50 {g:03d} decoded_root_fa80c06 64 {K71} 300 ' + SCAN.format(g=f'{g:03d}'))
open('jobs.txt', 'w').write('\n'.join(lines) + '\n')
print(len(lines), 'jobs')
