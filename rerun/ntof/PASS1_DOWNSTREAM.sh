#!/usr/bin/env bash
# PASS1_DOWNSTREAM.sh -- rebuild the downstream analysis on n_TOF pass 1 (NTOF_PASS_PLAN P7), on the desktop.
#
#   bash rerun/ntof/PASS1_DOWNSTREAM.sh [step ...]      # no args = every step, in order
#
# Inputs: $WORK_DATA/ntof_pass1/{stage2,rerun_dbeam_L1,run104,stage3} (copies of EOS Analysis_pass1, checksummed).
# Derived here (regenerable, not archived): stage3_vt/ (the analysis view: v(t) angles under the plain names),
# reco_tree/ (canonical stage 2 merged per sub-run and arm, old layout), ana/<study>/ (every downstream product).
# Status of each study and its numbers against the retired chain: rerun/ntof/PASS1_DOWNSTREAM.md.
# Each step logs to $ANA/logs/<step>.log; the chain stops at the first failure.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export PYTHONPATH=.:ntof_athens_26
P=${WORK_DATA:-/media/ucla/work}/ntof_pass1
ANA=$P/ana
LOG=$ANA/logs
mkdir -p "$LOG"
JOBS=${JOBS:-8}
PROD=$($PY -c "
import csv
r=[x for x in csv.DictReader(open('rerun/ntof/ntof_registry.csv')) if x['excluded']!='True' and x['rclass']=='production']
print(','.join(f'run_{i}' for i in sorted({int(x['run']) for x in r})))")

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

# --- 0. the derived inputs
step vt_view    $PY rerun/ntof/stage3/vt_view.py --jobs $JOBS
step reco_tree  $PY rerun/ntof/stage3/reco_tree.py --jobs $JOBS

# --- 1. capsule pointing per run (scale-free crossing; feeds the measured axis of the pair studies)
step imaging    $PY -m sept26_prelim_analysis.campaign_imaging --reco $P/reco_tree --runs $PROD --no-y \
                    --jobs 10 --out $ANA/imaging_campaign

# --- 1b. capsule focus (x view, nominal axis; and split by population) and the capsule-pose data side
step focus      $PY -m ntof_cosmics.capsule_focus --src $P/stage3_vt --label pass1 --tan-max 0.5 --jobs 4
step focus_split $PY -m ntof_cosmics.focus_split --src pass1 --label pass1 --tan-max 0.5
for s in data data_z data_depth; do
  step pose_$s  $PY -m sept26_prelim_analysis.capsule_pose $s --out $ANA/capsule_pose
done
step capsule_tof $PY -m sept26_prelim_analysis.capsule_tof tof --chains pass1,is2_v2_cf
step capsule_tilt $PY -m sept26_prelim_analysis.capsule_tof tilt --chains pass1

# --- 2. same-chamber pairs (A, C): all ToF and late (>= 12 ms), then data vs Geant4
for arm in A C; do
  step sc_$arm      $PY -m sept26_prelim_analysis.det_a_intra --arm $arm --runs $PROD --t0-max 300 --axis measured \
                        --wall --src $P/stage3_vt --reco $P/reco_tree --imaging $ANA/imaging_campaign/per_run.csv \
                        --tan-max-true 0.5 --out $ANA/same_chamber/pass1_$arm --jobs 6
  step sc_late_$arm $PY -m sept26_prelim_analysis.det_a_intra --arm $arm --runs $PROD --t0-max 300 --axis measured \
                        --wall --src $P/stage3_vt --reco $P/reco_tree --imaging $ANA/imaging_campaign/per_run.csv \
                        --tan-max-true 0.5 --tof-min-ms 12 --out $ANA/same_chamber_late/pass1_$arm --jobs 6
done
step sc_events  $PY -m sept26_prelim_analysis.same_chamber_compare events --chain pass1
step sc_phys    $PY -m sept26_prelim_analysis.same_chamber_physics --chain pass1 --compare cf
step sc_cmp     $PY -m sept26_prelim_analysis.same_chamber_compare --chain pass1 --compare cf

# --- 3. scintillator stack mapped by the tracks (k = 1: m = tan, the v(t) angle)
export X17_SCINT_OUT=$ANA/scint_stack
step scint_x    $PY -m ntof_scint_stack.extract --src $P/stage3_vt --runs $PROD --jobs 4
step scint_ana  $PY -m ntof_scint_stack.ana --jobs 4

echo "=== done: $ANA" | tee -a "$LOG/chain.log"
