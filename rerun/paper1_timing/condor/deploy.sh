#!/bin/bash
# deploy.sh -- git archive of HEAD -> lxplus EOS (/eos/user/d/dneff/x17rerun-code is not used: the archive goes next to the
# outputs on the n_TOF area), the condor dir ~/x17rerun/paper1_timing with chunks.txt and the submit file.
# Run from the repo root on the desktop/laptop.  Prints the code dir.
set -e
H=$(git rev-parse --short=8 HEAD)
DST=/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_timing/code_$H
git archive --format=tar.gz -o ${TMPDIR:-/tmp}/p1t_$H.tgz HEAD rerun/paper1_timing rerun/model_audit/followups/fa_gas_v_tilt/colsum.py
scp -q ${TMPDIR:-/tmp}/p1t_$H.tgz lxplus:/tmp/p1t_$H.tgz
ssh lxplus "mkdir -p $DST && tar xzf ${TMPDIR:-/tmp}/p1t_$H.tgz -C $DST && rm /tmp/p1t_$H.tgz && mkdir -p ~/x17rerun/paper1_timing/log \
  && cp $DST/rerun/paper1_timing/condor/timing.sub ~/x17rerun/paper1_timing/ \
  && n=\$(wc -l < $DST/rerun/paper1_timing/jobs.txt) && (for i in \$(seq 1 4 \$n); do echo \"\$i 4\"; done) > ~/x17rerun/paper1_timing/chunks.txt"
rm -f ${TMPDIR:-/tmp}/p1t_$H.tgz
echo "$DST"
