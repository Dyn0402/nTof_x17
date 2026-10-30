import json, numpy as np, pyarrow.parquet as pq
R='/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps/'
def q(a,p): a=a[np.isfinite(a)]; return np.round(np.percentile(a,p),3) if len(a) else None
for k in ['run70_flat_drift__d700','run70_flat_drift__d500','run71_raw__raw700','run71_raw__raw450','run71_raw__raw275','run66_flat_resist__r780']:
    t=pq.read_table(R+k+'/events.parquet').to_pandas(); m=json.load(open(R+k+'/events.meta.json'))
    rr=m['registry_row']
    print(f"== {k} gas={rr['gas']} resist={rr['resist_V']} n={len(t)} ref finite x/y {np.isfinite(t.ref_x).mean():.2f}/{np.isfinite(t.ref_y).mean():.2f}")
    for v in 'xy':
        ok=t[v+'_ok'].astype(bool); qo=ok&t[v+'_quality_ok'].astype(bool)
        tan=t[v+'_tan_theta']; sane=qo&(tan.abs()<=3)&t[v+'_p0'].between(-20,420)
        res=(t[v+'_p0']-t['ref_'+v])
        print(f"  {v}: ok {ok.mean():.3f} q {qo.mean():.3f} sane {sane.mean():.3f} | nstrips p50 {np.median(t[v+'_n_strips'][ok]):.0f} | res(sane) p16/50/84 {q(res[sane].values,[16,50,84])} | |res|<1mm {np.mean(np.abs(res[sane])<1):.2f} | nsat p50 {np.median(t[v+'_n_sat_strips'][ok]):.0f} | p0 p1/p99 {q(t[v+'_p0'][ok].values,[1,99])} ref p1/99 {q(t['ref_'+v].values,[1,99])}")
    # x-y wild correlation
    wx=(t.x_tan_theta.abs()>3)&t.x_ok; wy=(t.y_tan_theta.abs()>3)&t.y_ok
    print(f"  wild x {wx.mean():.3f} y {wy.mean():.3f} both {(wx&wy).mean():.3f}; wild x n_strips p50 {np.median(t.x_n_strips[wx]) if wx.any() else 0}; isochronous among wild {t.x_isochronous[wx].mean() if wx.any() else 0:.2f}; isochronous all-ok {t.x_isochronous[t.x_ok.astype(bool)].mean():.2f}")
