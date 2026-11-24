import glob, numpy as np, pandas as pd, w3lib as L
R = L.add_global(L.load_reduced('nose'))
O = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('origin_nose/o*.parquet'))]).drop_duplicates(['job','eventID','armID','trackID'])
R = R.merge(O.rename(columns={'armID':'arm','trackID':'dom_track'}), on=['job','eventID','arm','dom_track'])
R=R[R.full&(R.v_mesh.abs()<=125)&(R.sh.abs()<0.5)&(R.tan_gap_v.abs()<0.5)&((R.A!='D')|R.wall)]
for arm in 'AC':
    g=R[R.A==arm]; ho=g.ox; no=g.oz
    eo=(g.h-ho)/(g.n-no); ec=(g.h-0.82)/(g.n+0.01)
    b=(ec.abs()>0.18)&(ec.abs()<0.44)
    print(arm, 'median (eo/ec)', np.median((eo/ec)[b]), 'mean', np.mean((eo/ec)[b]), 'median no', np.median(no[b]), 'mean no', no[b].mean(), 'median(n)',np.median(g.n))
    print('  quantiles no', np.quantile(no[b],[.05,.25,.5,.75,.95]))
    print('  quantiles ho', np.quantile(ho[b],[.05,.25,.5,.75,.95]))
    print('  oy', np.quantile(g.oy[b],[.05,.25,.5,.75,.95]))
    # regression of (eo-ec) on ec
    print('  OLS slope of eo on ec', np.polyfit(ec[b], eo[b], 1))
