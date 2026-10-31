#!/bin/bash
# dec53_job.sh <group>: decode run_53 FEU 3 (det4) with the pinned decoder fa80c06, the decoder of every pass-1 SPS row,
# so run 53 goes through exactly the pass chain.  Output to EOS; scratch cleaned.
G=$1
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
R=/eos/experiment/ntof/data/x17/p2_sps_july/runs/run_53/cfg_gain4.5_peaktime200_deflt/raw_daq_data
O=/eos/user/d/dneff/x17rerun/sps_audit/run53_fa80c06; mkdir -p $O
FB=EicP2Bt_cfg_gain4.5_peaktime200_deflt_datrun_260801_12H59_${G}_03
W=${_CONDOR_SCRATCH_DIR:-/tmp/dneff_dec53_$G}; mkdir -p $W; cd $W
cp $R/$FB.fdf in.fdf || exit 1
$HOME/x17rerun/mm_strip_reconstruction/build/decoder/decode in.fdf $FB.root > $O/$FB.log 2>&1 || exit 1
cp $FB.root $O/$FB.root.part && mv $O/$FB.root.part $O/$FB.root
rm -f in.fdf $FB.root
