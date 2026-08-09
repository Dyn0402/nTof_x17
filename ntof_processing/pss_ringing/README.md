# pss_ringing — the plastic after-pulses, and how to cut them

**The n_TOF plastic scintillators emit a train of real secondary pulses after
every large hit, from ~20 ns out to about 1 µs. The PSA reports them, and they
are the long late tail in the DREAM/PSS match.** A per-hit flag built only from
"this hit is under 5 % of a bigger hit on the same channel in the previous
microsecond" removes 99.5 % of that tail.

Two reports, both generated — re-run the analysis and rebuild and the numbers,
tables and verdicts move together:

| | |
|---|---|
| [`report.html`](report.html) | the measurement: is it real, and what is it? |
| [`report_veto.html`](report_veto.html) | what to do about it in the slim |

## The headline numbers

| | plastics | SiPM walls |
|---|---|---|
| excess PSA hits per large pulse, 18–1000 ns | **4.4** | 0.007 |
| the sharp 81–82 ns echo alone | 0.30 | 0.000 |

On the DREAM side (reference pair, ±3 µs slim): the plastic excess at
150–1000 ns is **122,133 hits against a core of 47,292** — the tail was 2.6× the
signal.

## What is established, and how

Four independent checks, because the obvious explanation — the PSA mis-fitting a
long pulse tail — had to be excluded:

1. **An event-mixed accidental control.** Each leader's time transplanted into a
   different bunch of the same channel: same rate profile, same dead time, no
   correlation. The control is ~0 below 500 ns.
2. **Time reversal.** Forward +4.13 excess hits per leader against +0.90
   backward; the 81 ns echo is 25× asymmetric.
3. **The walls, in the same beam and the same PSA**, with a pulse 3× wider and a
   tail an order of magnitude fatter — and no excess at all.
4. **The raw stream1 traces.** The secondary pulses are visible one event at a
   time (`figures/event_display.png`), and the traces behind the 81 ns hits carry
   a +1.5 % bump that the traces without them do not.

The 81–82 ns component is a *fixed-delay echo*, identical on all four plastics,
2 ns wide — a reflection (~8 m of cable one way at 0.66 c, which is arithmetic
and not a measurement). The broad component peaks at 32–40 ns, which is too
early for ion-feedback afterpulsing; PMT late pulses or multiple reflections both
fit and this does not separate them.

## The cut

```python
from afterpulse_flag import flag_afterpulses     # or prev_max_amp for a scan

flag = flag_afterpulses(bunch * 100 + detn, tof, amp_0,
                        t_hold=1000.0, ratio=0.05)
```

Removes 99.5 % of the 150–1000 ns excess and 94.8 % of the 25–150 ns excess for
10.4 % of the core, all of it small-amplitude.

**Compute it on the full n_TOF hit stream, with a full `t_hold` of lookback.** An
after-pulse whose parent falls just outside the slim window is exactly the case a
slim-only recomputation gets wrong. **Store `shadow` and `dt_prev` as floats
rather than the boolean** — ~8 B/hit, ~18 MB on a 74 MB segment — so an analysis
can re-tune `ratio` and `t_hold` without re-slimming 21 TB.

*Cheap fallback needing no new branch, computable on today's slims:*
`amp_0 > 250` removes 95.7 % of the late tail for the same core cost, losing only
in the 25–150 ns band. It works because the core is a MIP peak (72 % above
5000 ADC) while the tail piles up near threshold.

*Per-trigger metric:* per (DREAM trigger, arm) take the **largest-amplitude**
plastic hit — not the earliest — and cut on its residual. On the trigger's own
arm **89.5 % land within ±25 ns**, median −5.6 ns. "Earliest" gives −589 ns,
because in a microsecond-wide window at 720 kHz singles the earliest hit is
almost always an unrelated single.

## The scripts

| | |
|---|---|
| `afterpulse_spectrum.py` | the Δt spectrum behind isolated leaders, with the event-mixed control and the `--reverse` time-reversal mode |
| `raw_pss_blocks.py` | amplitude-normalised median trace after a pulse, from raw stream1 |
| `echo_probe.py` | keeps the individual traces, so a minority-of-pulses feature is not medianed away |
| `echo_conditional.py` | traces split by whether the PSA gave the 81 ns hit |
| `same_block.py` | where the followers sit relative to the zero-suppressed record |
| `event_display.py` | single blocks with the PSA hits drawn on them |
| `afterpulse_flag.py` | **the flag itself** — `flag_afterpulses` and `prev_max_amp` |
| `veto_on_dream.py` | the flag against a real slim: what it removes, what it costs |
| `make_figures.py`, `veto_figures.py`, `make_report.py`, `make_veto_report.py` | the two reports |

## Reproducing

Everything runs from local data
(`/media/dylan/data/x17/ntof_reproc/v12_liqpileup` and
`/media/dylan/data/x17/ntof_raw_224572`) in a few minutes. `*.png` and `*.npz`
are gitignored repo-wide, so the figures and caches regenerate rather than being
carried:

```bash
V=../../.venv/bin/python
RAW=/media/dylan/data/x17/ntof_raw_224572

$V afterpulse_spectrum.py --parts 1 -o afterpulse.json
$V afterpulse_spectrum.py --parts 1 --dets PSSB --quiet 0 --max-dt 2000 -o fwd.json
$V afterpulse_spectrum.py --parts 1 --dets PSSB --quiet 0 --max-dt 2000 --reverse -o rev.json
$V raw_pss_blocks.py $RAW/head_8.bin --dets PSSA PSSB PSSC PSSD WALA WALB LIQA \
     --stack stack_head8.npz
$V echo_conditional.py $RAW/head_{8,20,40}.bin --det PSSB -o echo_cond_PSSB.npz
$V same_block.py $RAW/head_{8,20,40}.bin -o same_block.json
$V event_display.py $RAW/head_8.bin --det PSSB -n 4
$V make_figures.py && $V make_report.py

# the veto, against a locally slimmed reference pair
$V ../slim_pipeline/run_segment.py run_79 stat090_0000 224572 \
     --ntof-source /media/dylan/data/x17/ntof_reproc/v12_liqpileup \
     --out /tmp/slim_ap --slim-ns 3000 --nb 400
$V veto_on_dream.py /tmp/slim_ap/runs/run_79/stat090_0000/ntof_hits/ntof_hits_*.root \
     --scan -o veto_scan.json
$V veto_figures.py && $V make_veto_report.py
```

## Scope, and what is not settled

Run 224572 only — ten segments of v12 hits, three raw chunks, and one DREAM
sub-run over 400 bunches for the veto. All four plastics agree to within 30 % on
every number, which argues the effect is structural rather than a channel fault,
but:

- **the flag is tuned on one segment.** Its cost scales with the singles rate,
  which varies across the campaign; `ratio` and `t_hold` want re-checking on a
  high-rate and a low-rate segment before a campaign-wide number is quoted;
- **the 10.4 % core loss is not separated.** Some of the small-amplitude core
  excess is genuinely correlated with the trigger, and some may be after-pulsing
  landing inside ±25 ns — the effect turns on at 18 ns;
- **the amplitude floor is in ADC, not MIPs.** Per-channel gains differ, so
  250 ADC is not the same physical threshold on all four arms;
- **no wall-side requirement is folded in.** The metric is plastic-only;
- **the mechanism of the broad component** would take a bench pulse-injection
  test to pin down, and the ~8 m cable is inferred from the delay, not measured.
