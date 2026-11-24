import sys; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import r4_lib as L, json
out={}
for t,rel in (('6-25',L.R625),('golden1',L.GOLD1),('500V',L.SC+'/drift_scan_resist_490V_drift_500V'),('700V',L.SC+'/drift_scan_resist_490V_drift_700V'),('6-22','mx17_det2_det3_overnight_6-22-26/long_run')):
    d=L.load_events(rel); out[t]={}
    for cov in ((),('walk',),('walk','tan')):
        f=L.fit_shape(d,cov=cov,nboot=60); k='+'.join(cov) or 'none'
        out[t][k]=[f['c'][3:6].round(4).tolist(),f['err'][3:6].round(4).tolist(),(f['c'][6:].round(3).tolist())]
        print(t,k,out[t][k],flush=True)
d=L.load_events(L.R625); d=d[(d.x_n_sat_strips==0)&(d.y_n_sat_strips==0)]; f=L.fit_shape(d,nboot=60); print('6-25 unsat only',len(d),f['c'][3:6].round(4),f['err'][3:6].round(4))
d=L.load_events(L.GOLD2); d=d[(d.x_n_sat_strips==0)&(d.y_n_sat_strips==0)]; f=L.fit_shape(d,nboot=60); print('run2 unsat only',len(d),f['c'][3:6].round(4),f['err'][3:6].round(4))
