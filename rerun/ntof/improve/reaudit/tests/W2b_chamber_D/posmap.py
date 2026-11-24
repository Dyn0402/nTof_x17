import sys, json, numpy as np
sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17'); sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/ntof/channel_layer')
import cabling as C
out={}
for arm in 'BD':
    m=C.bench_maps(arm); out[arm]={v:[None if np.isnan(p) else float(p) for p in m[v]] for v in 'xy'}; out[arm]['feu']=m['feu']
json.dump(out,open('posmap.json','w'))
for arm in 'BD':
    for v in 'xy':
        p=np.array(out[arm][v],float)
        print(arm,v,out[arm]['feu'][v],'conn ranges:',[ (round(np.nanmin(p[i*64:(i+1)*64]),1),round(np.nanmax(p[i*64:(i+1)*64]),1)) for i in range(8)])
