import sys, os, json, pickle
import numpy as np
REPO='/home/dylan/PycharmProjects/nTof_x17'
sys.path[:0]=[REPO+'/papers/paper1/figs', REPO+'/papers']
import make_displays as md, evd
tree=sys.argv[1]
D=md.Data(*md.TREES[tree], md.DEF_EOS)
EV=[(7884,'tracks','A cosmic muon'),(47235,'tracks','Two cosmic muons'),(1063,'tracks','Two particles, separate in both views'),
    (17907,'all','Two particles, merged in Y'),(28537,'tracks','A δ-ray along the track'),(41373,'tracks','A saturated δ blob'),
    (41538,'none','A shower at the crossing'),(963,'tracks','A second deposit, not particle-like'),(5423,'tracks','A curling electron'),
    (33974,'none','A muon and a wide deposit')]
out=[]
for e,c,title in EV:
    try:
        ev=D.event(e,c,split=md.NOBRK)
    except SystemExit as ex:
        print('skip',e,ex); continue
    except Exception as ex:
        print('skip',e,repr(ex)); continue
    rec=dict(id=e,title=title,v=ev.v_mm,gap=evd.GAP,pitch=evd.PITCH,ts=evd.TS,t0=ev.t0,ref={k:list(map(float,v)) for k,v in ev.ref.items()},
             fits=[{k:(float(x) if not isinstance(x,str) else x) for k,x in f.items()} for f in ev.fits],views={})
    for v in 'xy':
        W,P=evd._sorted(ev.W[v],ev.P[v])
        segs=evd.window(ev,v,split=1e4)
        cen,half=segs[0]
        half=max(half+12,22)
        sel=np.abs(P-cen)<half
        rec['views'][v]=dict(P=[round(float(p),3) for p in P[sel]],W=np.round(W[sel]).astype(int).tolist(),cen=cen,half=half)
        print(e,v,sel.sum(),W.shape, W[sel].max())
    out.append(rec)
r=pickle.load(open(md.LOCAL+'/principle/principle_7884.pkl','rb'))
print(r.keys(), r['views']['x'].keys())
m={}
for v in 'xy':
    d=r['views'][v]
    m[v]=dict(P=[round(float(p),3) for p in np.asarray(d['pos'],float)],M=np.round(np.asarray(d['M'],float)).astype(int).tolist(),
              W=np.round(np.asarray(d['W'],float)).astype(int).tolist(),t0=float(d['t0']),p0=float(d['p0']),w=float(d['w']))
json.dump(dict(events=out,model7884=m,tree=tree),open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'events.json'),'w'),separators=(',',':'))
