# Getting condor output to EOS — measured, 2026-09-09

Tested with a one-job cluster (4141483) that tried three routes in one go.
Do not re-derive this; it cost a 155-job cluster that had to be held.

| route | result |
|---|---|
| `transfer_output_remaps` → `/eos/user/...` | **FAILS.** Job runs fine, then is **held** on output transfer. |
| `xrdcp` to `root://eosuser.cern.ch/` from inside the job | **works**, rc=0 |
| plain `cp` to `/eos/user/...` from inside the job | **works**, rc=0 |

The error names the mechanism exactly:

```
Transfer output files failure at access point bigbird101 while receiving files
from execution point slot1_16@... Details: writing to file
/eos/user/d/dneff/x17/eostest/remap_test_out.txt: (errno 2) No such file or directory
Code 12 Subcode 2
```

**The transfer is done by the ACCESS POINT (the schedd, `bigbird101`), not by
the worker, and the access point has no EOS mount.** The worker does — which is
why the job's own `xrdcp` and `cp` both succeed. `errno 2` is EOS not existing
as a path on the schedd, not a permission or quota problem, so no amount of
`SendCredential` or directory pre-creation fixes it.

`MY.SendCredential = true` does give the worker a valid Kerberos ticket
(`krbtgt/CERN.CH`, ~24 h), which is what the job's own `xrdcp` uses.

## The pattern to use

The job pushes its own output to EOS and transfers back only a small marker:

```bash
xrdcp -f "$PRODUCT" "root://eosuser.cern.ch//eos/user/d/dneff/.../$PRODUCT"
echo "..." > done_$TAG.txt          # a few bytes, this is what condor returns
```

```
transfer_output_files = done_$(tag).txt
```

`transfer_output_files` must be set **explicitly**: left unset, HTCondor
returns every file the job created in its scratch directory, which is the whole
product and defeats the point.

## When AFS is fine after all

Small per-job output (stage 1: ~1 MB) is fine on AFS **provided something
drains it**. AFS home is 10 GB; the slim pass at ~30 MB/job would make 8.8 GB
and did fill it to 100 %, holding 3 stage-1 jobs. With a drain every ~4 minutes
only 10-20 jobs' worth is ever resident. Both routes are legitimate; the
failure mode to avoid is "large output, no drain".
