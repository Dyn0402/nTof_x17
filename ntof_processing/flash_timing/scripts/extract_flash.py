"""Extract every hit near the gamma flash, per channel, for one n_TOF run.

Output: <outdir>/flash_run<N>.npz with one struct-array per tree.
Selection: |tof - anchor| < 3000 ns, where anchor = that bunch's WALB tflash
(WALB is the tree whose flash finder is reliable in both gate states).
"""
import sys, numpy as np, uproot

run, outdir = sys.argv[1], sys.argv[2]
path = f"/eos/experiment/ntof/processing/official/done/run{run}.root"
f = uproot.open(path)
keys = [k.split(';')[0] for k in f.keys(recursive=False)]
print(f"run{run} trees={[k for k in keys if len(k)==4]}", flush=True)

BR = ['BunchNumber', 'detn', 'tof', 'peak_tof', 'amp', 'area', 'fwhm',
      'risetime', 'satuflag', 'tflash', 'PulseIntensity']

# --- anchor: per-bunch WALB tflash
a = f['WALB'].arrays(['BunchNumber', 'tflash'], library='np')
b, tf = a['BunchNumber'], a['tflash']
o = np.argsort(b, kind='stable'); b, tf = b[o], tf[o]
fi = np.ones(len(b), bool); fi[1:] = b[1:] != b[:-1]
anchor_b, anchor_t = b[fi], tf[fi]
amap = dict(zip(anchor_b.tolist(), anchor_t.tolist()))
print(f"  anchor bunches: {len(amap)}", flush=True)

out = {}
for t in ('WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD', 'PKUP', 'SILI'):
    if t not in keys or f[t].num_entries == 0:
        continue
    br = [x for x in BR if x in f[t].keys()]
    chunks = []
    ntot = 0
    for arr in f[t].iterate(br, step_size='300 MB', library='np'):
        ntot += len(arr['tof'])
        anc = np.array([amap.get(int(x), np.nan) for x in arr['BunchNumber']])
        sel = np.abs(arr['tof'] - anc) < 3000
        if sel.sum() == 0:
            continue
        d = {k: arr[k][sel] for k in br}
        d['anchor'] = anc[sel]
        chunks.append(d)
    if not chunks:
        print(f"  {t}: {ntot} hits, 0 near flash", flush=True); continue
    merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    rec = np.zeros(len(merged['tof']), dtype=[(k, 'f8' if merged[k].dtype.kind == 'f' else 'i8')
                                              for k in merged])
    for k in merged:
        rec[k] = merged[k]
    out[t] = rec
    print(f"  {t}: {ntot} hits -> {len(rec)} near flash", flush=True)

np.savez_compressed(f"{outdir}/flash_run{run}.npz", **out)
print(f"  wrote {outdir}/flash_run{run}.npz", flush=True)
