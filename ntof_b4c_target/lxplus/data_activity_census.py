# Coarse multi-chamber activity census from combined_hits (candidate finding only, no geometry).
import uproot,numpy as np,glob,sys,os,json
R="/eos/experiment/ntof/data/x17/july_beam/runs"
DET={"A":(3,4),"B":(5,6),"C":(7,8),"D":(1,2)}
def census(run,subs,smin,smax,thr=500,nmin=2,nmax=40):
    out=[]
    for s in subs:
        files=sorted(glob.glob(f"{R}/{run}/{s}/combined_hits_root/*.root"))
        tot=dict(ev=0,ge1=0,ge2=0,ge3=0,pairs={},t0=None,t1=None)
        for f in files:
            a=uproot.open(f)["hits"].arrays(["eventId","feu","amplitude","max_sample","trigger_timestamp_ns"],library="np")
            ev=a["eventId"].astype(np.int64)
            ue,inv=np.unique(ev,return_inverse=True)
            ts=a["trigger_timestamp_ns"]
            tot["t0"]=int(ts.min()) if tot["t0"] is None else min(tot["t0"],int(ts.min()))
            tot["t1"]=int(ts.max()) if tot["t1"] is None else max(tot["t1"],int(ts.max()))
            sel=(a["amplitude"]>thr)&(a["max_sample"]>=smin)&(a["max_sample"]<=smax)
            n=np.zeros((len(ue),9),int)
            np.add.at(n,(inv[sel],a["feu"][sel]),1)
            act={}
            for d,(fx,fy) in DET.items():
                act[d]=(n[:,fx]>=nmin)&(n[:,fx]<=nmax)&(n[:,fy]>=nmin)&(n[:,fy]<=nmax)
            k=sum(act[d].astype(int) for d in DET)
            tot["ev"]+=len(ue); tot["ge1"]+=int((k>=1).sum()); tot["ge2"]+=int((k>=2).sum()); tot["ge3"]+=int((k>=3).sum())
            for i,d1 in enumerate("ABCD"):
                for d2 in "ABCD"[i+1:]:
                    tot["pairs"][d1+d2]=tot["pairs"].get(d1+d2,0)+int((act[d1]&act[d2]).sum())
        tot["hours"]=(tot["t1"]-tot["t0"])*1e-9/3600 if tot["t0"] else 0
        print(run,s,json.dumps(tot),flush=True)
        out.append(tot)
    return out
run=sys.argv[1]; smin,smax=int(sys.argv[2]),int(sys.argv[3]); subs=sys.argv[4:]
if subs==["ALL"]: subs=sorted(x for x in os.listdir(f"{R}/{run}") if os.path.isdir(f"{R}/{run}/{x}"))
census(run,subs,smin,smax)
