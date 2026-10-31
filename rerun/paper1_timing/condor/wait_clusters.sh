#!/bin/bash
# wait_clusters.sh <cluster> ... -- block until none of the given condor clusters has a job left in the queue (lxplus).
while true; do
  n=$(timeout 120 ssh lxplus "condor_q $* -af ClusterId 2>/dev/null | wc -l; echo rc\$?" 2>/dev/null | tr '\n' ' ')
  case "$n" in "0 rc0 ") echo "clusters $* done"; exit 0;; esac
  sleep 240
done
