#!/bin/bash
# fix_req_cycle.sh <job id | all> [cluster]
# Restore Requirements / PreVerRequirements on idle jobs whose chain was rewritten into a cycle (bigbird101, seen
# 2026-11-09 -> 11-13 on newly materialised jobs of pass-1 cluster 4636418; no slot matches, nothing starts).
# The good values are read from the cluster's factory ad (condor_q -factory), which was never rewritten; the
# 4636418 values were copied verbatim from it before, and every completed job carries the same.
# cluster: default 4636418, or the cluster of the job id given.
TARGET="$1"
CL=${2:-4636418}
[ "$TARGET" != all ] && CL=${TARGET%%.*}
PREVER=$(condor_q "$CL" -factory -af:r PreVerRequirements | head -1)
REQ=$(condor_q "$CL" -factory -af:r Requirements | head -1)
# a sane factory ad: PreVer is the plain Arch/OpSys/Disk expression, Requirements chains to PreLocalRequirements
if [ -z "$PREVER" ] || [ "$PREVER" = undefined ] || [[ $PREVER = *PreLocalRequirements* ]] || [[ $REQ != *PreLocalRequirements* ]]; then
  echo "fix_req_cycle: factory ad of $CL does not look sane, not editing:"; echo "  PreVer: $PREVER"; echo "  Req: $REQ"
  exit 1
fi
# Two variants: (a) Requirements references PreVerRequirements; (b, seen 11-12) PreVerRequirements itself holds the
# Requirements text (contains PreLocalRequirements), so the chain loops back on itself.
CONS="ClusterId == $CL && JobStatus == 1 && (regexp(\"PreVerRequirements\", unparse(Requirements)) || regexp(\"PreLocalRequirements\", unparse(PreVerRequirements)))"
if [ "$TARGET" = "all" ]; then SEL=(-constraint "$CONS"); else SEL=("$TARGET"); fi
condor_qedit "${SEL[@]}" PreVerRequirements "$PREVER" || exit 1
condor_qedit "${SEL[@]}" Requirements "$REQ"
