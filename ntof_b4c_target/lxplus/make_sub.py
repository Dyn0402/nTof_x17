import os
base="/eos/experiment/ntof/data/x17/full_sim/b4c_july"
jd=os.path.expanduser("~/condor/mx17_b4c")
sets=[("neutron",50,40,500000),("neutron",25,40,500000),("casc",50,50,200000),("casc",25,50,200000)]
lines=[f"executable = {jd}/b4c_job.sh","log = "+jd+"/logs/$(ClusterId).log",
       "output = "+jd+"/logs/$(name)_$(Process).out","error = "+jd+"/logs/$(name)_$(Process).err",
       '+JobFlavour = "longlunch"',"request_cpus = 1",""]
seed=1000
for mode,r,nj,n in sets:
    name=f"{'neutrons' if mode=='neutron' else 'casc'}_r{r}"
    os.makedirs(f"{base}/{name}",exist_ok=True)
    lines.append(f"name = {name}")
    lines.append("queue arguments from (")
    for j in range(nj):
        seed+=1
        lines.append(f"  {mode} {r} {n} {seed} {base}/{name}/{name}_job{j:03d}")
    lines.append(")"); lines.append("")
open(f"{jd}/b4c.sub","w").write("\n".join(lines))
print("written",sum(s[2] for s in sets),"jobs")
