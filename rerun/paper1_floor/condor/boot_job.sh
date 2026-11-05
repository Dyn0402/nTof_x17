#!/bin/bash
# residual bootstrap (rerun/paper1_floor/BOOTSTRAP_2026-11-01.md): one boot_run.py job from the frozen code archive.
#   boot_job.sh <code_dir> <view> <out name> <boot_run.py args...>
CODE=$1; VIEW=$2; NAME=$3; shift 3
E=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/floor_boot
export FD_AUDIT=$E/bundles FD_BASIS=pass FD_PLACE=ig FD_INPUTS=$E/bundles/toy_inputs_pass1.json FD_VIEW=$VIEW
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER/boot_$$}; mkdir -p $W
/afs/cern.ch/user/d/dneff/bin/lockpy $CODE/rerun/paper1_floor/boot_run.py "$@" --out $W/$NAME.npz
rc=$?
if [ $rc -eq 0 ]; then xrdcp -f $W/$NAME.npz root://eosexperiment.cern.ch/$E/raw/$NAME.npz || rc=1; fi
rm -f $W/$NAME.npz
exit $rc
