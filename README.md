# nTof_x17
DAQ and analysis for x17 experiment at nTof

**Reconstruction basis:** positions, angles and drift depths come from the
waveforms (`decoded_root` → forward-model fit), **not** from `combined_hits`
times — a per-strip hit time is an aggregate of resistively shared charge and
compresses the drift ladder 20–30 %. Hits are for cluster finding, efficiency
and QA. Read [`RECONSTRUCTION_BASIS.md`](RECONSTRUCTION_BASIS.md) before writing
any analysis that produces a position, an angle or a depth.

Entry points: [`CLAUDE.md`](CLAUDE.md) (layout and conventions),
`mx_june_cosmic_qa/MICROTPC_RUNBOOK.md` (June cosmic bench),
`mx_june_cosmic_qa/waveform_first_threading/WAVEFORM_FIRST_THREADING.md`
(the waveform-first reconstruction study).
