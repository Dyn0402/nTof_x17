import json, numpy as np
D='/eos/user/d/dneff/vladder/'
for det in ('det2','det3','det4','det6','det7'):
    for kern in ('', 'r06_'):
        if kern: continue
        a=json.load(open(f'{D}{det}_prod.json'))['events']; b=json.load(open(f'{D}{det}_seeded.json'))['events']
        for p in 'xy':
            k=[e for e in a if p in a[e] and p in b.get(e,{})]
            tr=np.array([a[e][p]['tr'] for e in k]); wa=np.array([a[e][p]['w'] for e in k]); wb=np.array([b[e][p]['w'] for e in k])
            ca=np.array([a[e][p]['chi2'] for e in k]); cb=np.array([b[e][p]['chi2'] for e in k])
            diff=np.abs(wa-wb)>0.5
            s=f'{det} {p}: n {len(k)}  differ(|dw|>0.5) {diff.mean()*100:4.1f}%  of those seeded lower chi2 {np.mean(cb[diff]<ca[diff])*100:4.0f}%'
            for lo,hi in ((0,.15),(.15,.3),(.3,.5),(.5,2)):
                m=(abs(tr)>=lo)&(abs(tr)<hi)
                s+=f' | |t|{lo}-{hi}: {diff[m].mean()*100:4.1f}%'
            print(s)
