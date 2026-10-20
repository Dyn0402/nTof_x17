import os,glob
base="/eos/experiment/ntof/data/x17/full_sim/b4c_july"
parts="/eos/user/d/dneff/scratch/b4c_sim/parts"; os.makedirs(parts,exist_ok=True)
jd=os.path.expanduser("~/condor/mx17_b4c")
open(f"{jd}/red_job.sh","w").write("""#!/usr/bin/env bash
set -e
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
out=$1; shift
python3 /eos/user/d/dneff/scratch/b4c_sim/b4c_reduce.py "$out" "$@"
""")
os.chmod(f"{jd}/red_job.sh",0o755)
L=[f"executable = {jd}/red_job.sh","log = "+jd+"/logs/red_$(ClusterId).log",
   "output = "+jd+"/logs/red_$(Process).out","error = "+jd+"/logs/red_$(Process).err",
   '+JobFlavour = "longlunch"',"queue arguments from ("]
n=0
for d in sorted(os.listdir(base)):
    files=sorted(glob.glob(f"{base}/{d}/*.root"))
    groups={}
    for f in files:
        key=os.path.basename(f).rsplit("_job",1)[0]   # ipc: per line; others: per dataset
        groups.setdefault(key,[]).append(f)
    for key,fs in groups.items():
        for i in range(0,len(fs),5):
            L.append(f"  {parts}/{d}__{key}__{i//5:02d}.npz "+" ".join(fs[i:i+5])); n+=1
L.append(")")
open(f"{jd}/red.sub","w").write("\n".join(L)+"\n"); print(n,"jobs")
