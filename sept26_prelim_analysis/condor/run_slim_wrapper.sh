#!/bin/bash
# Condor executable for one n_TOF slim -> parquet conversion.
#   run_slim_wrapper.sh <run> <subrun>
# Output: slim_<run>_<subrun>.tar.gz containing out/slim/
#
# Cheap by design: the slim is 50-90 MB (against combined_hits' 1.2 GB), so this
# is ~1 min of work and needs neither the waveforms nor the hits.
set -e
RUN=$1; SUB=$2

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh
tar xzf code.tar.gz          # -> code/

EOSDIR=/eos/experiment/ntof/data/x17/july_beam/runs
HOST=root://eosexperiment.cern.ch/     # trailing slash: see run_stage1_wrapper.sh
mkdir -p "stage/runs/$RUN/$SUB/ntof_hits" out

n=0
for f in $(xrdfs eosexperiment.cern.ch ls "$EOSDIR/$RUN/$SUB/ntof_hits" 2>/dev/null || true); do
  xrdcp -f -s "$HOST$f" "stage/runs/$RUN/$SUB/ntof_hits/" && n=$((n+1)) || true
done
ls "stage/runs/$RUN/$SUB/ntof_hits/"*.root >/dev/null 2>&1 || {
  echo "[wrapper] no slim ROOT for $RUN/$SUB -- nothing to export"; exit 3; }

export X17_BEAM_JULY="$PWD/stage"
export X17_RUNS="$PWD/stage/runs"
export X17_SEPT26_OUT="$PWD/out"

cd code
python3 -W ignore -m sept26_prelim_analysis.slim_export \
        --run "$RUN" --subrun "$SUB" --out ../out/slim
cd ..

# Push the product to EOS FROM THE JOB, and hand condor back only a marker.
# transfer_output_remaps to EOS does NOT work: the transfer is performed by the
# access point (the schedd), which has no EOS mount, so the job runs to
# completion and is then held. Measured -- see EOS_WRITE_TEST.md.
EOSOUT=/eos/user/d/dneff/x17/sept26_slim
XH=root://eosuser.cern.ch/
P="out/slim/ntof_hits_${RUN}_${SUB}.parquet"
M="out/slim/ntof_hits_${RUN}_${SUB}.meta.json"
[ -s "$P" ] || { echo "[wrapper] FATAL: no parquet at $P"; exit 4; }

xrdcp -f "$P" "$XH$EOSOUT/$(basename "$P")" || { echo "[wrapper] FATAL: xrdcp parquet failed"; exit 5; }
xrdcp -f "$M" "$XH$EOSOUT/$(basename "$M")" || echo "[wrapper] WARNING: xrdcp meta failed"

# The marker is the ONLY thing condor transfers back. transfer_output_files must
# name it explicitly: left unset, HTCondor returns every file in the scratch
# directory, which is the whole product and defeats the point.
{ echo "run=$RUN subrun=$SUB"
  echo "bytes=$(stat -c %s "$P")"
  echo "sha256=$(sha256sum "$P" | cut -d" " -f1)"
  echo "host=$(hostname) date=$(date -Is)"
} > "done_${RUN}_${SUB}.txt"
echo "[wrapper] pushed $(basename "$P") to EOS ($(stat -c %s "$P") bytes)"
