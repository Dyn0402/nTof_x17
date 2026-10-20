import os
base="/eos/experiment/ntof/data/x17/full_sim/b4c_july"
jd=os.path.expanduser("~/condor/mx17_b4c")
lines_=[(7.007,"E1"),(4.711,"E1"),(2.533,"E1"),(11.448,"E1"),(4.444,"M1"),(8.917,"M1"),(6.740,"M1")]
L=[f"executable = {jd}/ipc_job.sh","log = "+jd+"/logs/$(ClusterId).log",
   "output = "+jd+"/logs/ipc_$(Process).out","error = "+jd+"/logs/ipc_$(Process).err",
   '+JobFlavour = "longlunch"',"request_cpus = 1","queue arguments from ("]
seed=5000
for r in (50,25):
    for e,k in lines_:
        name=f"ipc_r{r}_{e:.3f}_{k}"
        os.makedirs(f"{base}/ipc_r{r}",exist_ok=True)
        for j in range(20):
            seed+=1
            L.append(f"  {r} {e} {k} 50000 {seed} {base}/ipc_r{r}/{name}_job{j:02d}")
L.append(")")
open(f"{jd}/ipc.sub","w").write("\n".join(L)+"\n")
