**STATUS: DONE — completed 2026-07-30 by Claude (Dylan's session).** See "Completion
record" at the bottom.

# Handoff: push the 2026-07-24 reprocessed hits to EOS

**Task:** the bench data on EOS still carries the *pre*-matched-filter hits. The
local mirror was fully reprocessed on 2026-07-24 and never uploaded (the upload
was parked that day). Replace `hits_root/` and `combined_hits_root/` on EOS with
the local versions. **Nothing else changes** — `decoded_root/`,
`m3_tracking_root*/` and `raw_daq_data/` on EOS are already correct and must be
left alone.

## Why it matters

`wft` uses the hits tree for *seeding* (which events, which strips get fitted)
and for the efficiency numerator/denominator. Running any analysis against the
stale EOS hits silently produces a different event selection from everything
computed locally since 7-24 — the results look fine and are not comparable.
That is the trap this upload closes.

Verified 2026-07-30 on the det3 Saturday long run:

| file | EOS (stale) | local (correct) |
|---|---|---|
| `..._000_feu-combined_hits.root` | 8,285,614 B, dated 6-27 | 19,113,003 B, dated 7-24 |
| `..._000_07_hits.root` | 3,662,858 B, dated 6-27 | (reprocessed 7-24) |
| `..._000_07.root` (decoded) | 263,483,073 B | **263,483,073 B — identical** |

The size jump is the expected one: the matched-filter analyzer recovers ~40 %
more hits (biggest on the low-gain det6/det7).

## Paths

- **EOS root:** `/eos/experiment/ntof/data/x17/cosmic_bench/`
  (note *experiment*, singular. `/afs/cern.ch/user/d/dneff/x17/cosmic_bench` is
  a symlink to the same place — do not treat it as a second copy.)
- **local root:** `/home/dylan/x17/cosmic_bench/` (mirror of
  `/media/dylan/data/x17/cosmic_bench`)

Layout differs by one level. Locally, runs are grouped into *bench-area*
directories; on EOS the June runs sit directly under `june_tests/`:

| local | EOS |
|---|---|
| `det3/<run>/<subrun>/` | `june_tests/<run>/<subrun>/` |
| `det2_det3/<run>/<subrun>/` | `june_tests/<run>/<subrun>/` |
| `det1_det2/`, `det3_det4/`, `det4_day/`, `det6_det7/` | `june_tests/<run>/<subrun>/` |
| `det_3/<run>/...` (May and earlier) | `det_3_old/<run>/...` |
| `det_1/`, `det_4/` (Jan–Apr) | `det_1_old/`, `det_4_old/` |

So the area directory is **dropped** for June runs and **renamed with `_old`**
for the older ones. Confirm each run name exists on the EOS side before copying
— do not create new run directories.

## Scope

Local dirs with mtime ≥ 2026-07-24 (i.e. reprocessed):

```
278 directories, 1,070 root files, 22.4 GB total
  det3       56    det6_det7  50    det2_det3  38    det_1  36
  det3_det4  32    det_3      24    det1_det2  20    det_4  20
  det4_day    2
```

Priority order if you split the work: `det3`, `det2_det3`, `det6_det7`,
`det4_day`, `det3_det4` (these back the active June analysis), then
`det1_det2`, then the `det_*` older areas.

## Procedure

1. **Ticket:** `kinit dneff@CERN.CH` (transfers die silently when it expires;
   for a multi-hour upload renew it or use a keytab).
2. **Enumerate** the local reprocessed dirs:
   ```bash
   cd /home/dylan/x17/cosmic_bench
   find . -maxdepth 4 -type d \( -name hits_root -o -name combined_hits_root \) \
        -newermt 2026-07-23 | sort > /tmp/reproc_dirs.txt
   ```
3. **Preserve the old copy before overwriting** — rename on EOS rather than
   deleting, so a bad upload is recoverable:
   ```bash
   ssh lxplus 'mv <subrun>/combined_hits_root <subrun>/combined_hits_root_prewfa'
   ```
   (or, if space is tight, verify the new upload first and remove afterwards.)
4. **Copy.** `xrdcp` is the reliable route for EOS; `rsync` over ssh to the EOS
   fuse mount is slower and has bitten this project before. Per file:
   ```bash
   xrdcp -f <local file> \
     root://eosuser.cern.ch//eos/experiment/ntof/data/x17/cosmic_bench/june_tests/<run>/<subrun>/combined_hits_root/
   ```
   Uplink from this laptop is capped ~200 Mbit/s (see the download-methods
   note), so 22.4 GB is roughly 20-40 min at best, realistically an hour or two.
   Run it in the background and log every file.
5. **Verify** — this is the part that must not be skipped:
   - file count per directory matches local,
   - byte size matches local for every file (`xrdfs stat` or `ls -l` on the
     mount),
   - spot-check one file per bench area by opening the `hits` tree and
     comparing the entry count with the local copy,
   - confirm no `decoded_root/` or `m3_tracking_root*/` timestamps changed.
6. **Record** the completion in this file (date, which areas, who ran it) and
   update the `june-cosmics-reprocessing-2026-07-24` memory, which currently
   says the EOS upload is PARKED.

## Watch out for

- **Do not** upload the local `Analysis/` tree. It contains per-run calibration
  bundles and reco tables that are conditions-specific and much more volatile;
  the EOS `june_tests/Analysis` is a separate concern.
- Some subruns have `_backup_*` directories locally (e.g.
  `_backup_feu7only_003`). They are local artefacts — do not upload them.
- `det_3/mx17_det3_ArIso_Test_6-16-26` exists in BOTH `det_3_old/` and
  `june_tests/` on EOS. Check which one the local file corresponds to (the
  local copy lives under `det_3/`, so it maps to `det_3_old/`) before copying.
- If a run exists locally but not on EOS, stop and ask — it may be a local-only
  test rather than a missing upload.

## Completion record — 2026-07-30

All 278 reprocessed directories / 1,070 files / 22.38 GB uploaded and verified.
Zero failures, zero size mismatches, zero exceptions.

- **Path resolution:** built an explicit local-run → EOS-run map for the 18
  unique runs in scope (not a blanket area-name rule — old-style local areas
  don't map 1:1 to a single `_old` EOS area). Confirmed against live EOS
  listings:
  - `det3`, `det2_det3`, `det1_det2`, `det3_det4`, `det4_day`, `det6_det7` →
    `june_tests/<run>` (area dropped), as the handoff table says.
  - `det_1/mx17_det0_He_HV_Scan_4-1-26` and `det_1/mx17_det1_daytime_run_1-28-26`
    → **`det_0_old/`**, not `det_1_old/` — `det_1_old` on EOS only holds
    `mx17_det1_Ar_CF4_HV_Scan_4-25-26`; the Jan/Apr runs are filed by the
    detector-in-use name at the time, which was det0 for both.
  - `det_4/mx17_det4_ArIso_HV_Scan_5-7-26` → `det_4_old/` as expected.
  - `det_3/mx17_det3_HV_Scan_5-5-26`, `det_3/mx17_det3_long_run_5-6-26` →
    `det_3_old/` (unambiguous, only exist there).
  - `det_3/mx17_det3_test_6-22-26` → **`june_tests/`**, not `det_3_old` — this
    run was never uploaded under `det_3_old` at all, only `june_tests` has it.
  - `det_3/mx17_det3_ArIso_Test_6-16-26` and `det_3/zs_compression_scan_4_6-6-26`
    exist **identically in both** `det_3_old/` and `june_tests/` on EOS.
    Per this doc's original note, targeted `det_3_old/` for both (matches
    where the local `det_3/` copy corresponds); the `june_tests/` duplicates
    were left untouched as out of scope.
- **Procedure used:** upload each file to `<name>.new_upload` alongside the
  target, verify byte size on EOS against local before touching anything, then
  `xrdfs rm` the old file and `xrdfs mv` the staged file into place. This is
  the "verify new upload first, then delete old" variant (no permanent
  `_prewfa` backup kept) — chosen over the rename-first default to save EOS
  space, on explicit instruction.
- **Verification:** (1) per-file size check during upload before delete
  (1,070/1,070 OK), (2) a fully independent second pass re-diffing all 1,070
  files fresh from EOS after the fact (0 remaining diffs), (3) ROOT tree
  entry-count spot-check, one file per bench area, local vs. downloaded EOS
  copy — all 9 areas matched exactly, (4) confirmed no leftover
  `*.new_upload` staging files anywhere in `june_tests` or any `det_*_old`
  area, (5) confirmed `decoded_root/`, `m3_tracking_root*/`, `raw_daq_data/`
  sit untouched alongside the updated `hits_root/combined_hits_root` (the
  upload only ever wrote inside `hits_root/`/`combined_hits_root/` paths by
  construction).
- No runs were found locally-only-not-on-EOS; no manual stop-and-ask cases
  beyond the two dual-location ones above, which the doc had already resolved.
