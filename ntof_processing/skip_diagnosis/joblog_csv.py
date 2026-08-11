import re, sys, glob, datetime, os
print("run,job,cluster,attempt_s,outcome,mem_mb,nfiles")
for run in sys.argv[1:]:
    base = f"/afs/cern.ch/work/d/dneff/x17_reproc/aux_prod_{run}/{run}"
    for f in sorted(glob.glob(f"{base}/run{run}_0*_process.sub.log.*")):
        t = open(f).read()
        m = re.match(rf"run{run}_(\d+)_process\.sub\.log\.(\d+)\.0", os.path.basename(f))
        if not m: continue
        job, cluster = m.group(1), m.group(2)
        ev = re.findall(r'^(\d{3}) \(\S+\) (\d\d/\d\d \d\d:\d\d:\d\d)', t, re.M)
        ts = lambda s: datetime.datetime.strptime('2026/' + s, '%Y/%m/%d %H:%M:%S')
        st = [ts(x) for c, x in ev if c == '001']
        en = [ts(x) for c, x in ev if c == '005']
        ab = [ts(x) for c, x in ev if c == '009']
        mem = [int(x) for x in re.findall(r'^\t(\d+)  -  MemoryUsage of job \(MB\)', t, re.M)]
        wall = 'SYSTEM_PERIODIC_REMOVE' in t and 'wall time' in t
        if st and en:
            dur, out = (en[-1] - st[-1]).total_seconds(), 'completed'
        elif st and ab:
            dur = (ab[-1] - st[-1]).total_seconds()
            out = 'killed_walltime' if wall else 'aborted'
        else:
            dur, out = -1, 'unknown'
        nf = 0
        try: nf = sum(1 for _ in open(f"{base}/run{run}_{job}.files"))
        except OSError: pass
        print(f"{run},{job},{cluster},{int(dur)},{out},{max(mem) if mem else 0},{nf}")
