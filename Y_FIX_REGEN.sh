#!/usr/bin/env bash
# Y_FIX_REGEN.sh -- regenerate everything the stage-3 y mirror touched, on the
# machine that holds the data (the desktop), after the 2026-10-13 code fix.
# HANDOFF_Y_FLIP_FIX.md §3-4; results go in Y_FLIP_AUDIT.md.
#
#   bash Y_FIX_REGEN.sh [step ...]      # no args = every step, in order
#
# Each step logs to $LOG/<step>.log and the chain stops at the first failure.
# Every stage-3 rebuild is checked row by row against the old table as an exact
# y mirror before it replaces it (regen_stage3.py); old tables are kept under
# <dir>/pre_yfix/ until you delete them.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONPATH=.:ntof_athens_26
OUT=$($PY -m sept26_prelim_analysis.paths --path out)
LOG=${LOG:-$OUT/y_fix_logs}
mkdir -p "$LOG"
JOBS=${JOBS:-8}

step() {                                   # step <name> <cmd...>
  local n=$1; shift
  if [ ${#WANT[@]} -gt 0 ] && [[ ! " ${WANT[*]} " =~ " $n " ]]; then return 0; fi
  echo "=== [$n] $*" | tee -a "$LOG/chain.log"
  if ! "$@" > "$LOG/$n.log" 2>&1; then
    echo "!!! [$n] FAILED -- see $LOG/$n.log" | tee -a "$LOG/chain.log"; tail -20 "$LOG/$n.log"; exit 1
  fi
  tail -4 "$LOG/$n.log"
}
WANT=("$@")

df -h / /media/dylan/data | tee -a "$LOG/chain.log"

# step() exits the chain on failure; s3() below adds a free-space check on /.

# --- 0. the code: unit tests (convention, guard, loader flag, regen check)
step tests      $PY -m pytest -q sept26_prelim_analysis/tests/test_y_convention.py

# --- 1. stage 3, every live set (each rebuilt from its own sidecar).
#        The data disk is full: staging and the pre_yfix backups go on / under
#        $WORK (<dir>/pre_yfix is a symlink there).  Largest sets first, while
#        / has the room; each set needs its sub-run tables' size free on / plus
#        a $SPARE_GB GB margin, or the chain stops before touching it.
WORK=${WORK:-$HOME/x17_local/yfix_work}
SPARE_GB=${SPARE_GB:-6}
room() {                                   # room <stage3 dir>: enough space on / ?
  local need free
  need=$(du -s --block-size=1G --exclude=tracks_campaign.parquet --exclude=pre_yfix "$1" | cut -f1)
  free=$(df --output=avail --block-size=1G "$HOME" | tail -1 | tr -d ' ')
  echo "$1: sub-run tables ${need} GB, / free ${free} GB (margin ${SPARE_GB})"
  [ "$free" -ge $((need + SPARE_GB)) ]
}
s3() {                                     # s3 <step> <dir> [--combine]
  local n=$1 d=$2; shift 2
  if [ ${#WANT[@]} -gt 0 ] && [[ ! " ${WANT[*]} " =~ " $n " ]]; then return 0; fi
  if ! room "$d" | tee -a "$LOG/chain.log"; then
    echo "!!! [$n] not enough room on / for staging + backup -- stopping" | tee -a "$LOG/chain.log"; exit 1
  fi
  step "$n" $PY -m sept26_prelim_analysis.regen_stage3 "$d" --jobs $JOBS --work "$WORK" "$@"
}
s3 s3_full    "$OUT/stage3_fullpass"     --combine
s3 s3_is2v2   "$OUT/stage3_is2_v2"       --combine
s3 s3_cf      "$HOME/x17_local/stage3_is2_v2_cf" --combine
s3 s3_camp    "$OUT/stage3_campaign"     --combine
s3 s3_pilot   "$OUT/stage3_is2_v1_pilot" --combine
s3 s3_cos147  ntof_cosmics/results/tracking/k_run_147
s3 s3_cos150  ntof_cosmics/results/tracking/k_run_150

# --- 2. the guards on the regenerated tables
step guard_cf   $PY -m sept26_prelim_analysis.y_guard "$HOME/x17_local/stage3_is2_v2_cf"
step guard_full $PY -m sept26_prelim_analysis.y_guard "$OUT/stage3_fullpass"
step cosmic     $PY ntof_cosmics/y_orientation_check.py cosmic      # rebuilds ac149.parquet
step verdict    $PY ntof_cosmics/y_orientation_check.py verdict     # exits 1 unless y is up
step tests_data $PY -m pytest -q sept26_prelim_analysis/tests/test_y_convention.py

# --- 3. Geant4 digitised samples on the fixed digitiser (~1 h per arm), then the
#        check that the digitised chain still finds the truth y crossing (-28 mm)
step g4digi     bash ntof_cosmics/g4_digi/run_full.sh
step capsule    $PY -m sept26_prelim_analysis.capsule_tof all       # data tof/tilt + g4: y crossings
step pose       $PY -m sept26_prelim_analysis.capsule_pose all
step pose_rep   $PY -m sept26_prelim_analysis.make_capsule_pose_report

# --- 4. products that depend on absolute y (audit items 1-7)
step capsule_y  $PY -m sept26_prelim_analysis.capsule_y --campaign --jobs 6
step capy_rep   $PY -m sept26_prelim_analysis.make_capsule_y_report
step focus      $PY -m ntof_cosmics.focus_split --src "$OUT/stage3_is2_v2" --label is2_v2
for ch in prod cf is2; do
  case $ch in
    prod) SRC="$OUT/stage3_fullpass"; X="" ;;
    cf)   SRC="$HOME/x17_local/stage3_is2_v2_cf"; X="--tan-max-true 0.6" ;;
    is2)  SRC="$OUT/stage3_is2_v2"; X="--tan-max-true 0.6" ;;
  esac
  for arm in A C; do
    step sc_${ch}_$arm $PY -m sept26_prelim_analysis.det_a_intra --arm $arm --t0-max 300 --axis measured \
         --wall --src "$SRC" $X --out "$HOME/x17_local/same_chamber/${ch}_$arm" --jobs 6
  done
done
step sc_phys    $PY -m sept26_prelim_analysis.same_chamber_physics
step sc_cmp     $PY -m sept26_prelim_analysis.same_chamber_compare
step sc_rep     $PY -m sept26_prelim_analysis.make_same_chamber_report
step scint_x    $PY -m ntof_scint_stack.extract --jobs $JOBS
step scint_ana  $PY -m ntof_scint_stack.ana
step scint_rep  $PY -m ntof_scint_stack.make_report
step pmt        $PY ntof_cosmics/y_orientation_check.py pmt          # plastics: stored-high end now EARLIER
step y_image    $PY -m pair_vertex_imaging.y_image --jobs $JOBS
step overhead_y $PY ntof_athens_26/make_overhead_figure.py --projection y
step trackqa    $PY -m sept26_prelim_analysis.trackqa_shards --force

echo "=== done. Fill Y_FLIP_AUDIT.md 'after' columns from $LOG and the reports; then republish:" | tee -a "$LOG/chain.log"
echo "    make_y_orientation_deck / make_same_chamber_deck / make_capsule_pose_deck -> add-note.py --force --deploy"
