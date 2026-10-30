#!/bin/bash
# r71v2_job.sh <group>: run_71 with the ZS-like uRWELL telescope (urw_raw_scan.py): decode FEU 1 (pinned decoder
# fa80c06) in scratch, scan it, then tel_job.py on the four-view events.  Outputs to EOS; scratch cleaned.
G=$1
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc14-opt/setup.sh >/dev/null 2>&1
export WFT_REPO=/eos/user/d/dneff/x17rerun/sps_audit/code MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
PY="env -u PYTHONPATH -u PYTHONHOME /eos/user/d/dneff/venvs/ntof-lock-cvmfs/bin/python"
R=/eos/experiment/ntof/data/x17/p2_sps_july/runs/run_71/cfg_gain4.5_peaktime50/raw_daq_data
O=/eos/user/d/dneff/x17rerun/sps_audit; mkdir -p $O/scan_v2 $O/tel
FB=EicP2Bt_cfg_gain4.5_peaktime50_datrun_260803_05H22_${G}_01
W=${_CONDOR_SCRATCH_DIR:-/tmp/dneff_r71v2_$G}; mkdir -p $W; cd $W
if [ ! -s $O/scan_v2/scan_$G.npz ]; then
  if [ -s /eos/user/d/dneff/x17rerun/sps_followup/feu1/$FB.root ]; then IN=/eos/user/d/dneff/x17rerun/sps_followup/feu1/$FB.root
  else cp $R/$FB.fdf in.fdf || exit 1; $HOME/x17rerun/mm_strip_reconstruction/build/decoder/decode in.fdf $FB.root > dec.log 2>&1 || exit 1; rm -f in.fdf; IN=$W/$FB.root; fi
  $PY $WFT_REPO/rerun/sps_audit/urw_raw_scan.py --out $W/scan_$G.npz $IN || exit 1
  cp $W/scan_$G.npz $O/scan_v2/scan_$G.npz || exit 1
  rm -f $W/*.root
fi
$PY $WFT_REPO/rerun/sps_audit/tel_job.py --run run_71 --subrun cfg_gain4.5_peaktime50 --group $G \
  --key run71_raw__raw700,run71_raw__raw450,run71_raw__raw275 --scan $O/scan_v2/scan_$G.npz \
  --out $W/r71v2_$G.parquet --work $W --jobs 8 || exit 1
cp $W/r71v2_$G.parquet $O/tel/ || exit 1
rm -f $W/*.parquet $W/*.npz $W/*.pkl $W/*.root
