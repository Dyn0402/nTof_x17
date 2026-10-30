#!/bin/bash
# tel_job.sh <tag> <run> <subrun> <group> <feu3dir> <nsamp> <keys|-> <qmin> [scan]: one tel_job.py group on condor.
# Outputs go to EOS from inside the job; nothing large is left in the scratch dir.
TAG=$1; RUN=$2; SUB=$3; G=$4; F3=$5; NS=$6; KEYS=${7//+/,}; QMIN=$8; SCAN=$9   # keys joined by + (condor splits on commas)
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=/eos/user/d/dneff/x17rerun/sps_audit/code MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
W=${_CONDOR_SCRATCH_DIR:-/tmp/dneff_tel_${TAG}_$G}; mkdir -p $W
O=/eos/user/d/dneff/x17rerun/sps_audit/tel; mkdir -p $O
A="--run $RUN --subrun $SUB --group $G --feu3-dir $F3 --nsamp $NS --qmin $QMIN --out $W/${TAG}_$G.parquet --work $W --jobs 8"
[ "$KEYS" != "-" ] && A="$A --key $KEYS"
[ -n "$SCAN" ] && A="$A --scan $SCAN"
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $WFT_REPO/rerun/sps_audit/tel_job.py $A || exit 1
cp $W/${TAG}_$G.parquet $O/ || exit 1
[ -f $W/${TAG}_${G}_hits.npz ] && cp $W/${TAG}_${G}_hits.npz $O/
rm -f $W/*.parquet $W/*.npz $W/*.pkl
