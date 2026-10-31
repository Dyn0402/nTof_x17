#!/bin/bash
# wait_cluster.sh <cluster> -- block until the lxplus condor cluster has left the queue (polled every 5 min), then count
# the paper1_gas stack outputs and the logs with a traceback.
C=$1
while true; do
  n=$(timeout 120 ssh lxplus "condor_q $C -af ClusterId 2>/dev/null | wc -l; echo rc\$?" 2>/dev/null | tr '\n' ' ')
  [[ "$n" =~ ^0\ rc0 ]] && break
  sleep 300
done
timeout 200 ssh lxplus 'find /eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_gas/Analysis -name s4_stacks.json | wc -l; grep -l -i traceback ~/x17rerun/paper1_gas/log/*.err 2>/dev/null | wc -l'
