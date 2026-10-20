import os
base="/eos/experiment/ntof/data/x17/full_sim/b4c_july/he3_ref"; os.makedirs(base,exist_ok=True)
jd=os.path.expanduser("~/condor/mx17_b4c")
L=[f"executable = {jd}/he3_job.sh","log = "+jd+"/logs/he3_$(ClusterId).log",
   "output = "+jd+"/logs/he3_$(Process).out","error = "+jd+"/logs/he3_$(Process).err",
   '+JobFlavour = "longlunch"',"request_cpus = 1","queue arguments from ("]
seed=9000
for ipc,k,tag in ((1.0,"M1","ipcM1"),(1.0,"E0","ipcE0"),(0.0,"M1","x17")):
    for j in range(20):
        seed+=1; L.append(f"  {ipc} {k} 10000 {seed} {base}/he3_{tag}_job{j:02d}")
L.append(")"); open(f"{jd}/he3.sub","w").write("\n".join(L)+"\n")
