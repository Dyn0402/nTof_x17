#!/bin/bash
# run_streams.sh A|B|C -- the 2026-10-31 user-EOS -> n_TOF-EOS move (rerun/EOS_MOVE_2026-10-31.md), three streams.
# Each item: eosmove.py copy, then finish (verify size + adler32; only then remove + symlink).
cd ~/x17rerun/eos_move
U=/eos/user/d/dneff; X=/eos/experiment/ntof/data/x17
item() { python3 eosmove.py copy "$1" "$2" ${3:+--dedupe-root $3} -j 16 && python3 eosmove.py finish "$1" "$2"; echo "EXIT $? $1"; }
streamA() {
  item $U/x17/reproc/prod_v12_recover $X/reproc/from_user_2026-10-31/prod_v12_recover $X/reproc/prod_v12_recover
  item $U/x17/reproc/prod_v12 $X/reproc/from_user_2026-10-31/prod_v12 $X/reproc/prod_v12
}
streamB() {
  item $U/x17/sps_run53_det4_check $X/p2_sps_july/sps_run53_det4_check
  for d in $(ls $U/x17); do
    case $d in reproc|sps_run53_det4_check|*.MOVED.txt) continue;; esac
    [ -L $U/x17/$d ] && continue
    item $U/x17/$d $X/analysis/$d
  done
  item $U/tmp/gx $X/response_sim/scratch_gx_2026-10
  item $U/tmp/d0ar $X/response_sim/scratch_d0ar_2026-10
}
streamC() {
  for d in asym_beam calorimetry cloud_basics mx17_hdr_audit_20261016 pairing_sim plane_ratio satblob seedfit sim vladder x17dig_out scratch; do
    [ -L $U/$d ] || item $U/$d $X/analysis/user_misc/$d
  done
  for d in $(ls $U/x17rerun); do
    [ -d $U/x17rerun/$d ] && [ ! -L $U/x17rerun/$d ] || continue
    case $d in sps_gasfit|sps_junk|sps_final_stage|sps_final2_stage) continue;; esac
    item $U/x17rerun/$d $X/cosmic_bench/x17rerun/$d
  done
}
case $1 in A) streamA;; B) streamB;; C) streamC;; L) shift; for d in "$@"; do item $U/x17rerun/$d $X/cosmic_bench/x17rerun/$d; done;; esac
