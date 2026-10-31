# eos_move: user EOS → n_TOF experiment EOS, verified

Used for the 2026-10-31 move (`rerun/EOS_MOVE_2026-10-31.md`). It runs on lxplus from `~/x17rerun/eos_move/`, with
manifests in `state/` and logs in `log/`.

- `eosmove.py copy|verify|finish SRC DST [--dedupe-root R]`
  - `copy` uses xrdcp to `root://eosexperiment.cern.ch`: per file for few or large files, one recursive xrdcp for
    trees of more than 300 files.
  - `verify` compares size and adler32 from one server-side `eos find --checksum` per side. It never reads the
    destination through FUSE, which can be stale.
  - `finish` removes the source only when every file verifies, then leaves a symlink and `<name>.MOVED.txt`.
  - `--dedupe-root R`: a file already in R with the same name, size and adler32 is a duplicate. It is kept in place
    and listed, and the other files of the tree become per-file symlinks.
- `run_streams.sh A|B|C|L <x17rerun dirs>`: the 2026-10-31 move in three streams. `L` moves named `x17rerun`
  subdirectories.
- `ledger.py`: writes `/eos/user/d/dneff/MOVED_TO_NTOF_2026-10-31.md` from the manifests.
- `repoint.py --moved moved_paths.txt [--apply]`: points code, submit files, `docs/PARAMETERS.md` and READMEs at the
  new paths, and switches `root://eosuser` to `root://eosexperiment` on those lines.
- `wait_streams.sh`: a local helper that blocks until no stream is left running.
