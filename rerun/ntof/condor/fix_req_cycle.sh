#!/bin/bash
# Restore Requirements / PreVerRequirements on 4636418 idle jobs whose chain was rewritten into a cycle.
# Values copied verbatim from the cluster's factory ad (and every completed job).
TARGET="$1"   # a single job id (e.g. 4636418.9029) or "all"
PREVER='((Machine != "b9g42p2031.cern.ch")) && (TARGET.Arch == "X86_64") && (TARGET.OpSys == "LINUX") && (TARGET.Disk >= RequestDisk) && (TARGET.Memory >= RequestMemory) && (TARGET.Cpus >= RequestCpus) && (TARGET.FileSystemDomain == MY.FileSystemDomain)'
REQ='(TARGET.Hostgroup =?= "bi/condor/gridworker/share/mixed" || TARGET.Hostgroup =?= "bi/condor/gridworker/shareshort" || TARGET.Hostgroup =?= "bi/condor/gridworker/sharelong") && PreLocalRequirements'
CONS='ClusterId == 4636418 && JobStatus == 1 && regexp("PreVerRequirements", unparse(Requirements))'
if [ "$TARGET" = "all" ]; then SEL=(-constraint "$CONS"); else SEL=("$TARGET"); fi
condor_qedit "${SEL[@]}" PreVerRequirements "$PREVER" || exit 1
condor_qedit "${SEL[@]}" Requirements "$REQ"
