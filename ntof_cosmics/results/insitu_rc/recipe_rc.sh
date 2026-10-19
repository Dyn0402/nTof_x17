#!/bin/bash
# Physical (RC) kernel from cloud_basics (FINDINGS §12, arms rcm) on the n_TOF A-C cosmic truth.
# Same settings as is2ts_s3 (recipe_ts.sh): seeder 3, TAN_MAX 1.0 raw, two-sided search.
# Bundles start from is2_<arm> (same template, v) and add the head-on-measured constants, no refit.
set -e
cd ~/PycharmProjects/nTof_x17/.claude/worktrees/capsule-scale-kernel
W=~/scratch/ntof_insitu; PY="/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python -W ignore ntof_cosmics/insitu_calib.py"
export WFT_BEAM_MIN_STRIPS=3 WFT_TAN_MAX=1.0 WFT_TWO_SIDED=1
J=${J:-14}
RC="rc_D_x=0,rc_tmpl=1"
# A = det3
B=$($PY mkbundle --work $W --arm A --bundle $W/bundles/is2_A --label rcm3 --v 38.0 \
     --hyper "sigma_p0=0.4103844485267755,Dp=0.028031762650303486,rc_D_y=0.0002778295432456987,$RC" | tail -1)
$PY reco --work $W --arm A --bundle $B --label rcm3ts_s3 --jobs $J
# C = det6 physically; production in-situ C runs det7's r06 kernel. Both constant sets.
B=$($PY mkbundle --work $W --arm C --bundle $W/bundles/is2_C --label rcm7 --v 28.7 \
     --hyper "sigma_p0=0.5591589074698122,Dp=0.030645769588839238,rc_D_y=0.0005742205686006513,$RC" | tail -1)
$PY reco --work $W --arm C --bundle $B --label rcm7ts_s3 --jobs $J
B=$($PY mkbundle --work $W --arm C --bundle $W/bundles/is2_C --label rcm6 --v 28.7 \
     --hyper "sigma_p0=0.34554369779118366,Dp=0.018391525692165577,rc_D_y=0.0007927485597013558,$RC" | tail -1)
$PY reco --work $W --arm C --bundle $B --label rcm6ts_s3 --jobs $J
$PY cmp --work $W --arm A --labels is2ts_s3 rcm3ts_s3 > $W/cmp_A_rc.txt
$PY cmp --work $W --arm C --labels is2ts_s3 rcm7ts_s3 rcm6ts_s3 > $W/cmp_C_rc.txt
echo RECIPE_RC_DONE
