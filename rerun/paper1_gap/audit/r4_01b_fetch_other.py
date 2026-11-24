"""r4_01_fetch: list and fetch events tables + bundles of the det3 sub-runs (EOS, xrootd, adler32)."""
import os, sys, csv
from XRootD import client
URL = 'root://eospublic.cern.ch'
CB = '/eos/experiment/ntof/data/x17/cosmic_bench'
OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/r4/eos'
fs = client.FileSystem(URL)
def ls(p):
    st, l = fs.dirlist(p, client.flags.DirListFlags.STAT)
    return [(e.name, e.statinfo.size, bool(e.statinfo.flags & client.flags.StatInfoFlags.IS_DIR)) for e in l] if st.ok else []
reg = list(csv.DictReader(open('/home/dylan/PycharmProjects/nTof_x17/rerun/pass_registry.csv')))
rows = [(x['run'], x['subrun']) for x in reg if x['det'] == 'mx17_3' and (
    x['run'] in ('mx17_det2_det3_overnight_6-22-26', 'mx17_det3_quick_6-26-26', 'mx17_det3_test_6-22-26',
                 'mx17_det3_det4_overnight_6-23-26x'))]
cp = client.CopyProcess(); n = 0
for run, sub in rows:
    w = f'{CB}/Analysis_pass1/{run}/{sub}/mx17_3/wft'
    files = [(f'{w}/{a}', s) for a, s, d in ls(w) if not d and a in ('events.parquet', 'bow_gate_table.parquet')]
    files += [(f'{w}/calib_bundle/bundle.json', None)]
    for src, s in files:
        dst = f'{OUT}/{run}/{sub}/{os.path.basename(src)}'
        if os.path.exists(dst) and (s is None or os.path.getsize(dst) == s): continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cp.add_job(f'{URL}/{src}', dst, force=True, checksummode='end', checksumtype='adler32'); n += 1
print(n, 'files', flush=True)
cp.prepare(); st, res = cp.run()
bad = [r for r in res if not r['status'].ok]
print('done', st, 'failed', len(bad))
for r in bad: print(r['status'].message)
