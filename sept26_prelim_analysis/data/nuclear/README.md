# Nuclear data staged for the IPC modules

Small, quoted, machine-readable inputs so `ipc_aluminium.py` runs offline and
so the provenance of every branching ratio is one `grep` away.  Nothing here is
ours; nothing here has been edited.

| file | what | where it came from |
|---|---|---|
| `28AL_EGAF.ens` | the `27Al(n,γ)28Al` thermal capture level scheme in ENSDF format: 182 levels with Jπ, 578 placed transitions, and the capture-state block with 47 primary branchings summing to 100 % | IAEA Evaluated Gamma-ray Activation File, `https://www-nds.iaea.org/pgaa/EGAF/28AL_EGAF.ens`, evaluated by R. B. Firestone (LBNL), Dec 2003 |
| `13C_EGAF.ens` | the same for `12C(n,γ)13C` — the carbon-fibre half of the capsule | `https://www-nds.iaea.org/pgaa/EGAF/13C_EGAF.ens` |
| `pgaa_lines_subset.tsv` | the `28-Al`, `13-C`, `2-H` and `4-He` rows of the IAEA PGAA gamma catalogue: **partial cross sections σ(Eγ) in barns**, which is what makes the per-capture intensities absolute rather than relative | `GAMJAVA.DAT` inside `https://www-nds.iaea.org/pgaa/pgaa7/pgaadata.zip` |

## The one thing to know before using them

The two files disagree on normalisation and only one of them is absolute.

* `pgaa_lines_subset.tsv` column 7 is σ(Eγ) in **barns**.  Divide by
  σ₀(27Al) = 0.231(6) b for the intensity per capture.  The delayed 1778.9 keV
  line (28Al β⁻ → 28Si) carries σ = 0.232 b, i.e. exactly σ₀, which is the
  check that the column means what it says.
* The capture-state block of `28AL_EGAF.ens` lists primaries as relative
  intensities **renormalised to sum to 100**.  The same lines in the PGAA
  catalogue sum to 0.187 b, i.e. 81 % of σ₀, so the EGAF numbers are ~1.24×
  too large if read as per-capture intensities.  `ipc_aluminium.py` therefore
  takes intensities from PGAA and takes only the **level assignments** (which
  γ leaves which level, and each level's Jπ) from EGAF.

Two completeness numbers fall out of that and are quoted on the report page:
the placed prompt lines carry 86 % of σ₀ × S_n in γ energy, and the identified
primaries carry 81 % of the captures.
