"""run first_epoch last_epoch n_bunches, from the index tree of one partial.

The index tree is replicated IN FULL in every partial (verified on 224632:
partials 1, 32 and 63 all carry bunches 1..4966 with identical Date/Time), so a
single open per run is enough. Date is 1YYMMDD, Time is HHMMSS, both LOCAL
(UTC+2) -- emitted here as a naive epoch exactly like ntof_index_times.py, so
coverage_map's INDEX_LOCAL_SHIFT_S still applies.
"""
import glob, sys, calendar, uproot, numpy as np

C = '/eos/experiment/ntof/processing/official/completed'
def epoch(d, t):
    d = int(d) % 1000000
    yy, mm, dd = 2000 + d // 10000, (d // 100) % 100, d % 100
    t = int(t); h, mi, s = t // 10000, (t // 100) % 100, t % 100
    return calendar.timegm((yy, mm, dd, h, mi, s, 0, 0, 0))

for run in sys.argv[1:]:
    fs = sorted(glob.glob(f'{C}/{run}/run{run}_[0-9]*.root'))
    if not fs:
        print(f'{run} ERR ERR 0'); continue
    try:
        a = uproot.open(fs[0])['index'].arrays(['BunchNumber', 'Date', 'Time'],
                                               library='np')
        ok = a['Date'] > 0
        if not ok.any():
            print(f'{run} ERR ERR 0'); continue
        e = np.array([epoch(d, t) for d, t in zip(a['Date'][ok], a['Time'][ok])])
        print(f'{run} {e.min()} {e.max()} {int(ok.sum())}')
    except Exception:
        print(f'{run} ERR ERR 0')
