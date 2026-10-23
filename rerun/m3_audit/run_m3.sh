#!/bin/bash
# run_m3.sh <bindir> <config.json> <subrun dir> <date_time stamp prefix e.g. MX17_long_run_datrun_260626_06H03_> <pedbase> <NNN> <out.root>
# The production M3 v2 job (cosmic_bench_m3_tracking scripts/condor/m3_job.sh) with the binaries and config as arguments,
# so variants of the tracking can be run on the same inputs.
BIN=$1; CFG=$2; SUB=$3; COSBASE=$4; PEDBASE=$5; NUM=$6; OUT=$7
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/m3_$(basename $OUT .root); rm -rf $W; mkdir -p $W/root_files; cd $W
cp $BIN/DataReader $BIN/tracking . ; cp $CFG cfg_base.json
cp $SUB/raw_daq_data/${PEDBASE}000_01.fdf ped_000_01.fdf || exit 1
cp $SUB/raw_daq_data/${COSBASE}${NUM}_01.fdf dat_${NUM}_01.fdf || exit 1
strip(){ python3 -c "
import sys
for line in sys.stdin:
    o=[];q=False;i=0
    while i<len(line):
        c=line[i]
        if c=='\"': q=not q
        if not q and line[i:i+2]=='//': break
        o.append(c); i+=1
    print(''.join(o).rstrip())
"; }
mk(){ { cat cfg_base.json; echo "\"data_file_first\" : $1,"; echo "\"data_file_last\" : $2,"; echo "\"data_file_basename\" : \"$3\"}"; } | strip > $4; }
mk 0 0 "$W/ped_" cfg_ped.json
./DataReader cfg_ped.json read > read.log 2>&1 && ./DataReader cfg_ped.json ped > ped.log 2>&1 || { echo ped failed; exit 1; }
mk $((10#$NUM)) $((10#$NUM)) "$W/dat_" cfg_cos.json
./DataReader cfg_cos.json analyse > analyse.log 2>&1 || { echo analyse failed; tail analyse.log; exit 1; }
./tracking cfg_cos.json rays $W/out.root > track.log 2>&1; tail -1 track.log
cp $W/out.root $OUT && cp analyse.log ${OUT%.root}_analyse.log && cd / && rm -rf $W
