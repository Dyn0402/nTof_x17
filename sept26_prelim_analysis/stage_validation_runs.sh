#!/usr/bin/env bash
# Stage one sub-run each of three runs spanning the campaign, so stage 1 can be
# validated somewhere other than run_145 before the full census is committed to.
#
# Chosen for what they exercise, not at random:
#   run_79   the first production run, and the only one carrying the dead
#            chamber-A x-connector -- exercises the hot/dead mask path
#   run_116  the largest run in the sample (29 sub-runs, 2.87 M triggers)
#   run_162  the last run of the campaign
#
# Only ONE combined_hits tag per run: the census fractions are what is being
# checked, and one tag of ~90 MB measures them well enough to say whether the
# class partition is stable across six weeks.
set -u

# Where the runs go comes from paths.py, so this and stage 1 cannot disagree;
# $X17_ROOT / $X17_RUNS move both. Resolved in a subshell so a staging script
# does not change the caller's working directory.
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST=$(cd "$REPO" && .venv/bin/python -m sept26_prelim_analysis.paths --path runs)
[ -n "$DEST" ] || exit 1
EOS=/eos/experiment/ntof/data/x17/july_beam/runs
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25"
SUB=stat090_0000

for RUN in run_79 run_116 run_162; do
  echo "=== $RUN"
  mkdir -p "$DEST/$RUN/$SUB"
  # the run config -- detector_transforms needs it
  rsync -a -e "$SSH" "lxplus:$EOS/$RUN/run_config.json" \
        "$DEST/$RUN/" 2>&1 | tail -1
  # the first combined_hits tag only
  FIRST=$($SSH lxplus "ls $EOS/$RUN/$SUB/combined_hits_root/ | head -1")
  echo "  tag file: $FIRST"
  mkdir -p "$DEST/$RUN/$SUB/combined_hits_root"
  rsync -a -e "$SSH" \
        "lxplus:$EOS/$RUN/$SUB/combined_hits_root/$FIRST" \
        "$DEST/$RUN/$SUB/combined_hits_root/" 2>&1 | tail -1
  # the slim n_TOF file -- the IMPLIED class needs the arm coincidences
  rsync -a -e "$SSH" "lxplus:$EOS/$RUN/$SUB/ntof_hits/" \
        "$DEST/$RUN/$SUB/ntof_hits/" 2>&1 | tail -1
  du -sh "$DEST/$RUN/$SUB" 2>/dev/null
done
echo "=== staged $(date -Is)"
