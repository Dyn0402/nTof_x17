#!/bin/bash
# pack_lock_venv.sh -- (re)build the tarball that lockpy stages into condor jobs. Run on lxplus after any change to the
# lock venv /eos/user/d/dneff/venvs/ntof-lock-cvmfs (e.g. after mk_cvmfs2.sh). Takes ~8 min (reads 14k files off FUSE).
set -euo pipefail
V=/eos/user/d/dneff/venvs
T=/tmp/$USER-lockvenv-$$.tgz
tar -C $V -cf - ntof-lock-cvmfs | gzip -1 > $T
S=$(sha256sum $T | cut -d' ' -f1)
xrdcp -f $T root://eosuser.cern.ch/$V/ntof-lock-cvmfs.tgz
[ "$(xrdcp -s root://eosuser.cern.ch/$V/ntof-lock-cvmfs.tgz - | sha256sum | cut -d' ' -f1)" = "$S" ]
echo "$S  ntof-lock-cvmfs.tgz" > $V/ntof-lock-cvmfs.tgz.sha256
rm -f $T
echo "packed $V/ntof-lock-cvmfs.tgz  sha256 $S"
