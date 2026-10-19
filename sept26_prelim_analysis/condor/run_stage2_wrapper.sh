#!/bin/bash
# Condor executable for one CAMPAIGN stage-2 reco job.
#   run_stage2_wrapper.sh <arm> <tag> <outname> [extra run_beam_job.py args...]
# where extra carries --run/--subrun/--allow, so one package covers all 293
# sub-runs instead of one package per sub-run.
#
# Differs from ntof_tracking/condor/run_beam_wrapper.sh in exactly one way: the
# allowlists arrive as a TARBALL of 293 JSONs rather than a single file, because
# condor's transfer_input_files would otherwise need a per-job file list. The
# job picks its own out of the unpacked directory.
set -e
ARM=$1; TAG=$2; OUTNAME=$3; shift 3 || true

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

tar xzf code.tar.gz          # -> code/
tar xzf bundles.tar.gz       # -> bundles/
tar xzf allowlists.tar.gz    # -> allow/

python3 run_beam_job.py "$ARM" "$TAG" --jobs "${RECO_JOBS:-8}" "$@"

# OUTNAME is passed in rather than reconstructed here, so it cannot drift from
# what the submit file declares. It carries run and sub-run because across the
# campaign a bare (arm, tag) is NOT unique -- two sub-runs can share a file tag,
# and the collision would silently overwrite one sub-run's reco with another's.
tar czf "${OUTNAME}.tar.gz" out

# Push to EOS from the JOB and hand condor back only a marker: the access point
# has no EOS mount, so transfer_output_remaps to EOS completes the job and then
# holds it. Measured, cluster 4141483 -- see EOS_WRITE_TEST.md.
# Overridable: the calibration pass writes to its own directory so its
# products (a DIFFERENT event selection under the same run/subrun/arm/tag
# name) cannot overwrite the main pass's.
EOSOUT=${EOS_STAGE2_OUT:-/eos/user/d/dneff/x17/sept26_stage2}
XH=root://eosuser.cern.ch/
xrdcp -f "${OUTNAME}.tar.gz" "$XH$EOSOUT/${OUTNAME}.tar.gz" \
  || { echo "[wrapper] FATAL: xrdcp of ${OUTNAME}.tar.gz failed"; exit 5; }
{ echo "outname=$OUTNAME"
  echo "bytes=$(stat -c %s "${OUTNAME}.tar.gz")"
  echo "sha256=$(sha256sum "${OUTNAME}.tar.gz" | cut -d" " -f1)"
  echo "host=$(hostname) date=$(date -Is)"
} > "done_${OUTNAME}.txt"
echo "[wrapper] pushed ${OUTNAME}.tar.gz to EOS"
