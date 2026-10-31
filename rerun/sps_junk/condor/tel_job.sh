#!/bin/bash
# tel_job.sh <code dir> <tag> <run> <subrun> <group> <feu3dir> <nsamp> <keys|-> <qmin>: rerun/sps_audit/tel_job.py on the
# final oscillator rules (rerun/sps_junk), the run53 + run63 telescope tables -> /eos/user/d/dneff/x17rerun/sps_junk/tel.
CODE=$1; TAG=$2; RUN=$3; SUB=$4; G=$5; F3=$6; NS=$7; KEYS=${8//+/,}; QMIN=$9
set +u; source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=$CODE MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
W=${_CONDOR_SCRATCH_DIR:-/tmp/dneff_tel_${TAG}_$G}; mkdir -p $W
O=/eos/user/d/dneff/x17rerun/sps_junk/tel; mkdir -p $O
A="--run $RUN --subrun $SUB --group $G --feu3-dir $F3 --nsamp $NS --qmin $QMIN --out $W/${TAG}_$G.parquet --work $W --jobs 8"
[ "$KEYS" != "-" ] && A="$A --key $KEYS"
env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python $CODE/rerun/sps_audit/tel_job.py $A || exit 1
cp $W/${TAG}_$G.parquet $O/ || exit 1
[ -f $W/${TAG}_${G}_hits.npz ] && cp $W/${TAG}_${G}_hits.npz $O/
rm -f $W/*.parquet $W/*.npz $W/*.pkl
