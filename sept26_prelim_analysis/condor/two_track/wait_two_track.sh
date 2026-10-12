#!/bin/bash
# Poll a condor cluster until it has no running/idle jobs left (or any is held).
#   wait_two_track.sh <cluster> [poll-seconds]
C=${1:-4334051}; S=${2:-600}
while true; do
  L=$(timeout 120 ssh -o BatchMode=yes lxplus "condor_q $C -totals 2>/dev/null | grep 'for query'" 2>/dev/null)
  if [ -n "$L" ]; then
    idle=$(echo "$L" | sed -E 's/.* ([0-9]+) idle.*/\1/'); run=$(echo "$L" | sed -E 's/.* ([0-9]+) running.*/\1/')
    held=$(echo "$L" | sed -E 's/.* ([0-9]+) held.*/\1/')
    echo "$(date '+%F %T') idle=$idle running=$run held=$held"
    if [ "$held" != "0" ]; then echo "HELD jobs present"; exit 2; fi
    if [ "$idle" = "0" ] && [ "$run" = "0" ]; then echo "cluster $C finished"; exit 0; fi
  else
    echo "$(date '+%F %T') ssh/condor_q failed"
  fi
  sleep $S
done
