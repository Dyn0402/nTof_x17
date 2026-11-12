#!/bin/bash
# One-shot check of the n_TOF pass-1 condor clusters; repairs the requirements cycle on the main cluster.
# Run ON LXPLUS:  bash ~/x17rerun/ntof/pass1/watch_pass1.sh          (status + auto-fix)
#                 bash ~/x17rerun/ntof/pass1/watch_pass1.sh --hist   (also count finished jobs; slow, ~3 min)
# Guide: rerun/ntof/condor/PASS1_WATCH.md in the nTof_x17 repo.
MAIN=4636418
SIDES="4636419 4636420"
cd "$(dirname "$0")" || exit 1
echo "== $(date '+%F %T')"
for c in $MAIN $SIDES; do
  n=$(condor_q $c -af JobStatus 2>/dev/null | sort | uniq -c | awk '{s=s" "$2"="$1} END{print s}')
  echo "cluster $c in queue:${n:- (none)}   (1=idle 2=running 5=held)"
done
for c in $MAIN $SIDES; do
  bad=$(condor_q $c -af:r Requirements PreVerRequirements 2>/dev/null | grep -c 'PreVerRequirements\|PreLocalRequirements.*PreLocalRequirements')
  echo "cluster $c jobs with the cyclic requirement: $bad"
  if [ "$bad" -gt 0 ]; then
    if [ "$c" = "$MAIN" ]; then
      echo "  -> repairing with fix_req_cycle.sh all"
      bash fix_req_cycle.sh all
      echo "  -> after repair: $(condor_q $c -af:r Requirements PreVerRequirements | grep -c 'PreVerRequirements\|PreLocalRequirements.*PreLocalRequirements') still bad"
    else
      echo "  -> SIDE CLUSTER AFFECTED: not auto-fixed (its values may differ). STOP and report."
    fi
  fi
done
held=$(condor_q $MAIN $SIDES -hold -af ClusterId ProcId HoldReason 2>/dev/null)
if [ -n "$held" ]; then echo "HELD JOBS (do not release; report):"; echo "$held" | head -20; else echo "held: 0"; fi
last=$(condor_q $MAIN $SIDES -run -af JobCurrentStartDate 2>/dev/null | sort -n | tail -1)
[ -n "$last" ] && echo "latest job start: $(date -d @$last '+%F %T')"
if [ "$1" = "--hist" ]; then
  for c in $MAIN $SIDES 4638252; do
    echo "cluster $c finished (count JobStatus ExitCode; 4=completed, exit 0 = ok):"
    timeout 170 condor_history $c -limit 20000 -af JobStatus ExitCode 2>/dev/null | sort | uniq -c || echo "  (timed out)"
  done
fi
