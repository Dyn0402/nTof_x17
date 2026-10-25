#!/bin/bash
# job.sh <subrun dir> <cosbase> <pedbase> <NNN> -- the production M3 v2 chain (rerun/m3_audit/run_m3.sh) on one file,
# keeping the DataReader cluster tree (all clusters on all 8 M3 planes) next to the rays.
# Output /eos/user/d/dneff/x17rerun/m3clus/<run>__<subrun>__<NNN>_{analyse,rays}.root
SUB=$1; COSBASE=$2; PEDBASE=$3; NUM=$4
D=/afs/cern.ch/work/d/dneff/m3_var; CFG=$D/config_ref_june2026.json
RUN=$(basename $(dirname $SUB)); S=$(basename $SUB); O=/eos/user/d/dneff/x17rerun/m3clus; mkdir -p $O
TAG=${RUN}__${S}__${NUM}
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
W=${_CONDOR_SCRATCH_DIR:-/tmp/$USER}/m3c_$TAG; rm -rf $W; mkdir -p $W/root_files; cd $W
cp $D/DataReader $D/tracking . ; cp $CFG cfg_base.json
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
cp root_files/test_analyse.root $O/${TAG}_analyse.root && cp out.root $O/${TAG}_rays.root
cd / && rm -rf $W
