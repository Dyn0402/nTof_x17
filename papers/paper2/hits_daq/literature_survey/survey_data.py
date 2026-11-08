"""survey_data.py -- the literature survey as data: one record per experiment or group of experiments.

Every number and quote comes from the notes in this directory (pad_tpcs.md, strip_microtpcs.md, lartpc.md,
short_track_tpcs.md), which give the source and how it was checked.  make_deck.py builds the slide note from this file.
geo = dict(D=um/sqrt(cm) or Dcm2=(lo, hi) with t_ms, L=drift mm, pitch=mm, assumed=True if D_T is assumed, not measured).
"""
import math

STRIP, PAD, LAR, SHORT = 'strip', 'pad', 'lar', 'short'
TAN_REF = 0.3


def sigma(g):
    """(lo, hi) sigma_T at full drift [mm]"""
    if 'Dcm2' in g:
        return tuple(math.sqrt(2 * d * g['t_ms'] * 1e-3) * 10 for d in g['Dcm2'])
    return tuple(d * math.sqrt(g['L'] / 10.0) / 1000.0 for d in (g['D'] if isinstance(g['D'], tuple) else (g['D'], g['D'])))


def lam(g, tan=TAN_REF):
    """Lambda: sigma_T(full drift) / (L tan) (derived)"""
    return tuple(s / (g['L'] * tan) for s in sigma(g))


MX17 = dict(D=379, L=28.0, pitch=0.78)

#: rows of the map: label, family, geo, note
MAP = [
    ('MX17 det3', 'mx', MX17, 'D_T from the fitted Dp'),
    ('ATLAS NSW', STRIP, dict(D=365, L=5.0, pitch=0.45), 'D_T: Scholer 2021'),
    ('BESIII CGEM', STRIP, dict(D=350, L=5.0, pitch=0.65), 'D_T: Farinelli'),
    ('LMU floating strip', STRIP, dict(D=365, L=6.0, pitch=0.5, assumed=True), 'D_T assumed = NSW'),
    ('LMU GEM (Flierl)', STRIP, dict(D=365, L=8.0, pitch=0.4, assumed=True), 'D_T assumed = NSW'),
    ('MIGDAL strips', SHORT, dict(D=260, L=30.0, pitch=0.833), 'D_T minimum at 200 V/cm'),
    ('GridPix', SHORT, dict(D=306, L=20.0, pitch=0.055), 'single electrons: no aggregation'),
    ('T2K HA-TPC', PAD, dict(D=310, L=972.5, pitch=11.0), 'pads 10 x 11 mm'),
    ('ArgoNeuT', LAR, dict(Dcm2=(5, 12), t_ms=0.295, L=469.0, pitch=4.0), 'D_T 5-12 cm2/s (ICARUS estimate)'),
    ('STAR', PAD, dict(D=230, L=2093.0, pitch=6.7), 'outer pads'),
    ('ALICE GEM', PAD, dict(D=209, L=2500.0, pitch=None), 'pad width not read'),
    ('ALICE MWPC', PAD, dict(D=220, L=2500.0, pitch=5.0), 'pads 4-6 mm across'),
    ('ICARUS', LAR, dict(Dcm2=(5, 12), t_ms=1.0, L=1500.0, pitch=3.0), 'D_T 5-12 cm2/s'),
    ('MicroBooNE', LAR, dict(Dcm2=(5, 12), t_ms=2.3, L=2560.0, pitch=3.0), 'D_T 5-12 cm2/s'),
    ('ProtoDUNE-SP', LAR, dict(Dcm2=(5, 12), t_ms=2.25, L=3572.0, pitch=4.79), 'D_T 5-12 cm2/s'),
    ('ILD (3.5 T / 1 T)', PAD, dict(D=(30, 91), L=2300.0, pitch=1.3), 'D_T 30 at 3.5 T, 91 at 1 T'),
]

#: published strip uTPC bias points: label, set angle [deg], reconstructed [deg], Lambda-geometry, statistic
LIT_POINTS = [
    ('Bortfeldt 10 deg (FSM)', 10.0, 15.0, dict(D=365, L=6.0), 'most probable angle; D_T assumed'),
    ('Flierl 15 deg (GEM)', 15.0, 19.0, dict(D=365, L=8.0), 'most probable value; D_T assumed'),
    ('Flierl 30 deg (GEM)', 30.0, 32.0, dict(D=365, L=8.0), 'mean; D_T assumed'),
]

E = []


def add(**k):
    E.append(k)


add(id='nsw', short='ATLAS NSW', fam=STRIP, name='ATLAS New Small Wheel resistive Micromegas',
    title='ATLAS NSW: tracks read steep, blamed on strip coupling; production avoids the µTPC fit',
    use='Muon tracking in the ATLAS forward region; high rate, large area. µTPC is a test-beam and inclined-track option.',
    specs=[('Drift gap', '5 mm, about 106 ns'), ('Strip pitch', '0.425-0.45 mm, resistive strips over copper'), ('σ<sub>T</sub> / pitch', '0.57-0.61 (derived)'),
           ('Gas', 'Ar:CO<sub>2</sub> 93:7, 47 µm/ns'), ('Electronics', 'VMM3 peak time; APV25 in earlier tests'), ('Reference', 'set chamber angle, SPS H4')],
    geo=dict(D=365, L=5.0, pitch=0.45),
    did=['Fit a line through one time per strip (APV25 rise time, VMM peak)',
         'Measured neighbour sharing: 10 % (simulated) to 67 % (longest strips)',
         'Simulated reconstruction with charge sharing on and off, with diffusion on',
         'Fixes: drop edge strips, position correction in earliest time, angle filters'],
    concl=['Bias to larger angles from capacitive coupling and edge strips, shrinking with angle',
           '“Adding CS pulls the time measurements more towards the mean time”',
           'With sharing off, simulation still peaks +7° high (tracks above 25°); cause left open',
           'Test beam at 28° peaks at the true angle, in tension with that simulation'],
    us='Closest published relative. The +7° without sharing is the nearest analogue of our “a metallic readout keeps most of it”, but they did not separate diffusion from geometry.',
    quotes=[('For the case of CS disabled, the distribution is symmetric around a peak value of 7° while in the case of a correct angle reconstruction a peak position at 0° would be expected. This offset most likely stems from the positions of the first and last ionizations above the corresponding strips.', 'Scholer 2021, Sec. 11.3.3, p. 141', 'https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b'),
            ('The peak of the distribution corresponds to the actual inclination of the chamber of 28°. A resolution of 7.1° was obtained around the peak, but a long tail towards higher angles', 'Scholer 2021, Fig. 8.8, p. 90', 'https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b'),
            ('due to capacitive coupling between the strips, the first and last strips in a cluster may have a signal which is only coming from the coupling with their neighbours. It was determined in simulation that charge-sharing between neighbouring strips is at the 10% level. This effect may lead to a bias in µTPC reconstruction.', 'Vanadia, ATL-MUON-PROC-2015-008', 'https://inspirehep.net/files/cf2a35f9dc497f132c073ce83c5ef49c'),
            ('A small angular bias is observed due to capacitative induction', 'Sessa 2014', 'https://inspirehep.net/files/09876cb48a49208d97e7312de1a60d94')],
    gaps='Numbers in Vanadia Fig. 8 and ATLAS public Figs. 12-13 exist only as images and were not read. Iodice and Lösel were not retrieved.')

add(id='fsm', short='LMU floating strip', fam=STRIP, name='LMU floating-strip Micromegas (Bortfeldt, Klitzner)',
    title='LMU floating strips: +5° at 10°, and the drift velocity comes out too large, as ours does',
    use='High-rate tracking R&D; low-energy proton and ion tracking in Ne:CF<sub>4</sub> (Klitzner).',
    specs=[('Drift gap', '6 mm'), ('Strip pitch', '0.5 mm, metallic, capacitive readout'), ('σ<sub>T</sub> / pitch', '0.57 (derived, D<sub>T</sub> assumed)'),
           ('Gas', 'Ar:CO<sub>2</sub>; Ne:CF<sub>4</sub> 80:20 (Klitzner)'), ('Electronics', 'APV25, Fermi rise-time fit'), ('Reference', 'set angle (10 ± 1)°; doublet 25.8 ± 0.4°')],
    geo=dict(D=365, L=6.0, pitch=0.5, assumed=True),
    did=['Fitted a Fermi function per strip and a line through the times',
         'Simulated capacitive chains in LTspice forward',
         'Scanned reconstructed angle against drift field at four angles',
         'Derived an angle-dependent rescale, per sample rather than per event'],
    concl=['Reconstructed angle always larger than true, approaching it with angle',
           '+5° at 10° measured; +9° in the LTspice model',
           'Slopes too small, so velocities too large: our hit-ladder symptom',
           'Klitzner: fewer strips means more bias; but most Ne:CF<sub>4</sub> layers read slightly small'],
    us='Same sign and same velocity symptom on metallic strips, so the sign does not need a resistive layer. Not a controlled comparison.',
    quotes=[('The most probable reconstructed angle is always larger than the true track inclination and approaches the correct value for increasing track angles', 'Bortfeldt PhD, Sec. 7.4, p. 182', 'https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf'),
            ('signals couple capacitively onto neighboring strips. Thus measured signal timing is shifted towards earlier values for the strips, hit by electrons, created close to the cathode.', 'Bortfeldt et al., Sec. 3.2', 'https://arxiv.org/pdf/1508.00802'),
            ('The decrease in cluster length for smaller inclination angles leads to a false reconstruction of the earliest and latest timing, as they are more biased by neighboring strips', 'Klitzner MSc, Sec. 5, p. 53', 'https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_fklitzner.pdf'),
            ('All other layers reconstruct a slightly smaller angle than expected', 'Klitzner MSc, p. 115', 'https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_fklitzner.pdf')],
    gaps='Klitzner PhD not retrieved. D_T for these gases was not found as a number in text, so 365 µm/√cm is assumed.')

add(id='flierl', short='LMU GEM (Flierl)', fam=STRIP, name='LMU triple-GEM µTPC (Flierl)',
    title='LMU GEM: the closest published correction, a 20 % neighbour subtraction per time bin',
    use='GEM µTPC development, including ions from <sup>10</sup>B neutron converters.',
    specs=[('Drift gap', '8 mm'), ('Strip pitch', '0.4 mm'), ('σ<sub>T</sub> / pitch', '0.82 (derived, D<sub>T</sub> assumed)'), ('Electronics', 'APV25'), ('Reference', 'set chamber angle')],
    geo=dict(D=365, L=8.0, pitch=0.4, assumed=True),
    did=['Line fit through per-strip times',
         'Subtracted 15 % (simulated) or 20 % (empirical) of each neighbour, per time bin',
         'Compared angle resolution against drift field'],
    concl=['19° at a set 15° (most probable); 32° at 30° (mean)',
           'After correction: 16.99 ± 0.04° at 15°',
           'First electrons occupy the full cluster width, so signals appear simultaneous',
           'Best resolution at the drift voltage with minimal diffusion'],
    us='States our mechanism in words, in one passage about short ion tracks. The per-bin subtraction touches the waveform but is a fixed-fraction correction, not a fit.',
    quotes=[('For clusters with a smaller inclination angle the timing difference of the incoming electrons is suppressed by the first arriving electrons, as they already occupy the width of the full cluster and the signal on all strips appears simultaneous', 'Flierl PhD, Ch. 3, p. 56', 'https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf'),
            ('a significant deviation even at a set angle of 30° was observed, where the mean reconstructed angle was 32°', 'Flierl PhD, Sec. 5.3, p. 80', 'https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf')],
    gaps='D_T is shown in figures only, so 365 µm/√cm is assumed.')

add(id='besiii', short='BESIII CGEM', fam=STRIP, name='BESIII cylindrical GEM inner tracker',
    title='BESIII GEM: the one strip µTPC paper that names transverse diffusion as a cause',
    use='Inner tracker upgrade for BESIII: charged tracks in a 1 T field, inclined crossings.',
    specs=[('Drift gap', '3 or 5 mm'), ('Strip pitch', '0.65 mm'), ('σ<sub>T</sub> / pitch', '0.38 (derived)'), ('Gas', 'Ar:iC<sub>4</sub>H<sub>10</sub> 90:10'), ('Electronics', 'APV25; TIGER later'), ('Reference', 'rotated test chambers')],
    geo=dict(D=350, L=5.0, pitch=0.65),
    did=['µTPC line fit plus a charge-centroid combination',
         'Weighted or suppressed first and last strips',
         'Three-step data-driven position shift (Farinelli App. B)'],
    concl=['Angular bias from capacitive induction on neighbouring strips',
           'Transverse diffusion “plays another important role”',
           'Correction depends on the gas mixture; “still in a preliminary stage”'],
    us='Our mechanism as a second cause, in one sentence and without a measurement. We can quantify it.',
    quotes=[('In addition the transverse diffusion plays another important role since the signal of a single primary electron can be spread on more than one strip', 'Alexeev et al. 2019, Sec. 4, p. 8', 'https://arxiv.org/pdf/1908.06253')],
    gaps='Alexopoulos NIM A 937 (2019) was not retrieved. No angle-bias number was published in the passages read.')

add(id='alice', short='ALICE', fam=PAD, name='ALICE TPC (MWPC, then GEM)',
    title='ALICE: same σ<sub>T</sub>/pad as us, charge-weighted point per row, no bias reported',
    use='Heavy-ion tracking and particle identification at the LHC.',
    specs=[('Drift', '2.5 m, 2.6 cm/µs'), ('Pads', '4×7.5, 6×10, 6×15 mm; 152 rows (GEM)'), ('σ<sub>T</sub> / pad', '0.58-0.87 (derived)'), ('D<sub>T</sub>', '220 (MWPC), 209 µm/√cm (GEM)'), ('Timing', '5-10 MHz; 160-190 ns shaping'), ('Reference', 'lasers, ITS, vertices, cosmics')],
    geo=dict(D=220, L=2500.0, pitch=5.0),
    did=['One centre-of-gravity point per pad row, in pad and time',
         'Parametrised resolution in drift length and tan²α',
         'Tail-cancellation filter in the readout',
         'Drift scale from lasers and ITS matching'],
    concl=['σ²<sub>COG</sub> = p₀² + p²<sub>L</sub>L<sub>drift</sub> + p²<sub>A</sub>tan²α: a variance term, not a bias',
           'Below-threshold charge grows with drift and inclination; handled in the error'],
    us='Proves σ<sub>T</sub>/pitch is not what protects a TPC. A track-matched velocity calibration would also absorb a compression, but the expected effect is about 0.1 %.',
    quotes=[('The fraction of the signal below the threshold is proportional to the width of the response function and increases with the drift length and the track inclination angle', 'Alme et al. 2010, Sec. 11.1', 'https://arxiv.org/abs/1001.1950'),
            ('matching laser tracks with the surveyed mirror positions; matching with tracks from the Inner Tracking System (ITS); matching of the TPC primary vertices from the two halves of the TPC; matching tracks from two halves of the TPC using cosmic tracks.', 'Alme et al. 2010, drift-velocity calibration', 'https://arxiv.org/abs/1001.1950')],
    gaps='ALICE GEM pad dimensions and TDR-016 not read. No column-resolved tan-lambda check found.')

add(id='star', short='STAR', fam=PAD, name='STAR TPC',
    title='STAR: time is an explicit charge-weighted mean, and dip-angle spread is a resolution term',
    use='Heavy-ion tracking at RHIC.',
    specs=[('Drift', '2.09 m, 5.45 cm/µs'), ('Pads', 'inner 2.85×11.5, outer 6.2×19.5 mm; 45 rows'), ('σ<sub>T</sub> / pad', '0.49 outer, 0.99 inner (derived)'), ('D<sub>T</sub>', '230 µm/√cm at 0.5 T'), ('Timing', '9.4 MHz, 180 ns FWHM'), ('Reference', 'lasers, vertex matching')],
    geo=dict(D=230, L=2093.0, pitch=6.7),
    did=['z from charge-weighted time buckets',
         'x from a three-pad Gaussian, or weighted mean at large crossing angle',
         'Velocity known to 0.1 % from lasers every few hours'],
    concl=['Resolution worsens with dip angle: pad path longer than shaping × velocity',
           'Landau clusters at the row edge pull the centroid: a crossing-angle error',
           'Diffusion and shaping set the z cluster width'],
    us='The textbook counter-design: a charge-weighted time. Analytically that leaves the interior slope unchanged. We have not tried one.',
    quotes=[('calculated by measuring the time of arrival of the electrons in ‘time buckets’ and weighting the average by the amount of charge collected in each bucket.', 'Anderson et al. 2003', 'https://arxiv.org/abs/nucl-ex/0301015'),
            ('The ionization electrons are spread over a distance d along the beam axis, with d = L/ tan(θ) and L is the length of the pad.', 'Anderson et al. 2003', 'https://arxiv.org/abs/nucl-ex/0301015')],
    gaps='No published tan-lambda versus external tracker per column found.')

add(id='t2k', short='T2K ERAM', fam=PAD, name='T2K HA-TPC with resistive ERAM modules',
    title='T2K ERAM: closest resistive relative, but the time is never used as a slope ladder',
    use='Near-detector upgrade (ND280) for long-baseline neutrino oscillations.',
    specs=[('Drift', '97 cm, 7.68 cm/µs'), ('Pads', '10.09 × 11.18 mm resistive'), ('σ<sub>T</sub> / pad', '0.27-0.31 (derived)'), ('D<sub>T</sub>', '286 MC, 310 µm/√cm fitted'), ('Timing', 'AFTER, 40 ns; 200 / 412 ns peaking'), ('Reference', 'scintillator trigger only')],
    geo=dict(D=310, L=972.5, pitch=11.0),
    did=['Time = bin of waveform maximum per pad; position from the pad response',
         'Separated undelayed “sharing” from delayed RC “spreading”',
         'Fitted all pads’ waveforms jointly, for X-ray RC and gain maps only',
         'Angle-adapted clustering'],
    concl=['Resolution 200-800 µm at all angles and drifts',
           'Neighbour signals delayed, smaller, longer, growing with distance',
           'In-plane biases tied to E×B and field non-uniformity',
           'Global waveform fit for tracks: “studies are underway”'],
    us='Our RC physics and a waveform model already exist here. What does not: an RC effect on the drift coordinate, and a per-event track fit. They are working towards the second.',
    quotes=[('Contrary to the ‘charge sharing’ effect, the signals in neighbouring pads are delayed compared to the one of the leading pad in case of the ‘charge spreading’.', 'DESY test beam, arXiv:2212.06541', 'https://arxiv.org/abs/2212.06541'),
            ('Studies are underway aiming at using simultaneously the full information of the waveforms of all the pads around the tracks in a global fit, to improve the track reconstruction accuracy.', 'arXiv:2212.06541', 'https://arxiv.org/abs/2212.06541'),
            ('All the waveforms generated in one X-ray event are fitted simultaneously', 'arXiv:2303.04481, Sec. 7', 'https://arxiv.org/abs/2303.04481')],
    gaps='The sources are silent on whether RC delay shifts the track z. Silent, not negative.')

add(id='lctpc', short='LCTPC / ILD', fam=PAD, name='LCTPC prototypes and ILD (GEM, resistive Micromegas)',
    title='LCTPC: resistive-anode bias is real but in-plane, and “can easily be corrected”',
    use='R&D for a TPC at a future linear e⁺e⁻ collider.',
    specs=[('Drift', '2.3 m (ILD); prototypes shorter'), ('Pads', '1.3 mm; about 1×6 mm typical; 2×6 mm resistive'), ('σ<sub>T</sub> / pad', '0.35 at 3.5 T (derived)'), ('D<sub>T</sub>', '30 (3.5 T), 91 µm/√cm (1 T)')],
    geo=dict(D=(30, 91), L=2300.0, pitch=1.3),
    did=['Studied resistive-anode charge dispersion for sub-pad position',
         'Measured x<sub>row</sub> − x<sub>track</sub> against x<sub>track</sub>',
         'Compared MPGD to MWPC angular effects'],
    concl=['Bias up to 100-150 µm from RC and gap non-uniformity, corrected to about ±20 µm',
           'MPGDs free of E×B and angular wire effects',
           'Angular pad effect adds a near-constant resolution offset'],
    us='A strong field squeezes D<sub>T</sub>, the extreme of the lever arm. Their resistive bias is a map in position, ours a depth effect in time.',
    quotes=[('A bias of up to 150 µm is observed which may be attributed to a non-uniform RC constant due to inhomogeneities in the gap size and the resistivity of the foil. However, this bias is due to geometry only and can easily be corrected', 'Boudjemline, Dixit et al., physics/0510085', 'https://arxiv.org/abs/physics/0510085'),
            ('free from the E × B and the angular wire effects inherent in conventional TPCs with MWPC readout', 'Kobayashi 2014, arXiv:1404.0982', 'https://arxiv.org/abs/1404.0982')],
    gaps='Only |phi| < 5 degrees analysed in the resistive-anode work.')

add(id='uboone', short='MicroBooNE', fam=LAR, name='MicroBooNE LArTPC',
    title='MicroBooNE: the convolution in channel and time is already published. Credit it',
    use='Short-baseline neutrino physics at Fermilab.',
    specs=[('Drift', '2.56 m, 2.3 ms, 1.1 mm/µs'), ('Wires', '3 mm pitch, 3 planes'), ('σ<sub>T</sub> / pitch', '0.51-0.78 (derived)'), ('Sampling', '0.5 µs; 2 µs peaking'), ('Response', 'field computed over ±10 wires'), ('Reference', 'PMT flash t₀, cosmic tagger')],
    geo=dict(Dcm2=(5, 12), t_ms=2.3, L=2560.0, pitch=3.0),
    did=['2D deconvolution in wire and time (Adams 2018)',
         'Per-event L1 regularised fit, 1D, for shorted-wire regions only',
         'Longitudinal diffusion from pulse widths (2021)',
         'Data/simulation charge and width ratios against angle (WireMod)'],
    concl=['Signal = charge ⊗ field ⊗ electronics, solved jointly over wires',
           'At large θ<sub>xz</sub> neighbour-wire charge is “stretched in time”',
           'Transverse diffusion adds time smearing, growing with θ<sub>xz</sub>',
           'No hit-time or slope bias reported (negative search)'],
    us='Same equation class as ours; the prior art a referee will name. Their kernel excludes diffusion, which stays in the image, and they fit no track or depth profile.',
    quotes=[('the raw digitized TPC signal is a convolution of the arriving ionization electron distribution, the field response describing the induced current on wires due to moving charge, and the overall electronics response', 'Adams et al. 2018, Sec. 3.1, p. 17', 'https://arxiv.org/abs/1802.08709'),
            ('the contributions from the charge in neighboring wires are stretched in time and produce a triangular-like shape', 'Adams et al. 2018, Sec. 4.3.5, p. 49', 'https://arxiv.org/abs/1802.08709'),
            ('A measured pulse width may have contributions to the pulse width from the effects of D_T. Adjacent electron clouds begin to overlap as they spread in the yz-plane under the influence of D_T, causing additional σ_t smearing', 'MicroBooNE 2021, Sec. 5.1, p. 18', 'https://arxiv.org/abs/2104.06551'),
            ('limited to a 1D deconvolution instead of the 2D deconvolution that is used elsewhere', 'Adams et al. Part II, Sec. 3.4', 'https://arxiv.org/abs/1804.02583')],
    gaps='The D_T null in Table 3 of 2104.06551 was taken at ±6 degrees, where the effect is expected to be small from kinematics alone (derived), so it does not transfer. Theses and internal notes not covered.')

add(id='protodune', short='ProtoDUNE-SP', fam=LAR, name='ProtoDUNE single-phase',
    title='ProtoDUNE-SP: 2D deconvolution, then one Gaussian time per wire, with no bias seen',
    use='Prototype of the DUNE far detector, in a CERN charged-particle beam.',
    specs=[('Drift', '3.57 m, 2.25 ms, 1.59 mm/µs'), ('Wires', '4.67-4.79 mm'), ('σ<sub>T</sub> / pitch', '0.31-0.48 (derived)'), ('Sampling', '2 MHz, 2 µs peaking'), ('Response', '21 wires, 1/10 pitch grid'), ('Reference', 'CRT, beam instrumentation')],
    geo=dict(Dcm2=(5, 12), t_ms=2.25, L=3572.0, pitch=4.79),
    did=['Same 2D deconvolution equation as MicroBooNE',
         'Per-wire Gaussian hit fit on the deconvolved waveform',
         'Noted non-Gaussian shapes for tracks near the wire-field plane'],
    concl=['A hit is one charge deposit on one wire at one time, same sense as ours',
           'A single Gaussian is “clearly inadequate” for tracks along the field (Baller)'],
    us='Shows one time per channel alone gives no visible bias when columns are metres long. Short segments in LAr are not protected and are simply untested.',
    quotes=[('Because of the long-range induction effect, instead of a 1D deconvolution involving only the time dimension, a two-dimensional (2D) deconvolution involving both the time and wire dimensions is performed', 'ProtoDUNE-SP, Sec. 4.4, p. 42', 'https://arxiv.org/abs/2007.06722'),
            ('Ionization electrons produced on tracks that travel roughly in the electric field direction, φ ≈ 0, will arrive over many microseconds. In this case a single Gaussian fit is clearly inadequate', 'Baller 2017, Sec. 3.5, p. 10', 'https://arxiv.org/abs/1703.04024')],
    gaps='A beam or CRT slope comparison for hit times was not located.')

add(id='icarus', short='ICARUS, ArgoNeuT, SBND', fam=LAR, name='ICARUS, ArgoNeuT, SBND',
    title='ICARUS: tunes the waveform response in angle bins, and confirms diffusion in data',
    use='ICARUS: SBN far detector. ArgoNeuT: small test LArTPC. SBND: SBN near detector.',
    specs=[('ICARUS', '1.5 m drift, 3 mm wires, 2.5 MHz'), ('ArgoNeuT', '0.47 m (derived), 4 mm, 5.05 MHz'), ('SBND', '2 m drift, 3 mm'), ('σ<sub>T</sub> / pitch', '0.32-0.50 ICARUS; 0.14-0.21 ArgoNeuT (derived)')],
    geo=dict(Dcm2=(5, 12), t_ms=1.0, L=1500.0, pitch=3.0),
    did=['ICARUS: average waveforms in 2° angle bins from 20° to 76°, near the anode',
         'Fitted the field response to data',
         'Tested wire-by-wire against coarse-grained dQ/dx'],
    concl=['Simulation disagrees “especially at high track angle”',
           'First data confirmation of transverse diffusion on the muon charge scale',
           'A single angle controls the signal shape'],
    us='Data-driven tuning of the response against angle resembles our calibration step. Their expected slope comes from the TPC track, so a scale bias would not show.',
    quotes=[('the first confirmation in data of the impact of transverse diffusion on the muon charge scale', 'ICARUS 2024, p. 8', 'https://arxiv.org/abs/2407.11925'),
            ('This single angle controls the shape of the ionization signal response observed within the TPC waveform', 'ICARUS 2024, p. 28', 'https://arxiv.org/abs/2407.11925')],
    gaps='ICARUS and SBND 2D deconvolution status comes from talks only. SBND sampling not in any primary source read.')

add(id='mimac', short='MIMAC', fam=SHORT, name='MIMAC directional dark matter',
    title='MIMAC: the only quantified external-truth test here, and its z error has the opposite sign',
    use='Directional dark-matter search; recoils of a few keV.',
    specs=[('Drift', '25 cm'), ('Strips', '424 µm, 256 per axis, binary per 20 ns'), ('Gas', 'iC<sub>4</sub>H<sub>10</sub>+CHF<sub>3</sub> 50/50, 30 mbar'), ('Track', 'a few mm'), ('Reference', '0° ion beam; neutron kinematics')],
    geo=None,
    did=['Barycentre per time slice, then a line (BCW)',
         'η rescale of z for the space-charge stretch',
         'At keV: abandoned the slice fit for a width ratio with D<sub>L</sub>/D<sub>T</sub>'],
    concl=['First and last slices deviate: the anode samples the diffused cloud ends',
           'z is stretched, attributed to ion backflow, so tan reads too small',
           '0° beam: mean under 1.6°; neutrons: 10-16° resolution'],
    us='Same end-of-column effect in a time-slice estimator. A warning that the sign is not universal in high-gain, large-gap Micromegas.',
    quotes=[('initial and final time slices of the track usually have a larger than average deviation from the track direction. This is because the anode samples the endpoints of the transversely diffused primary electron cloud. Removing the first and last time slice does not produce better results either.', 'Tao 2020, p. 7', 'https://arxiv.org/pdf/2003.11812'),
            ('leading to a decrease of effective drift velocity, and thus we measure more time slices than predicted by simulations', 'Tao 2019, p. 8-9', 'https://arxiv.org/pdf/1903.02159')],
    gaps='D_T not read, so MIMAC cannot be placed on the Λ map. A 3-4° bias would be hard to exclude at 10-16° resolution.')

add(id='directional', short='NEWAGE, DRIFT, CYGNO', fam=SHORT, name='NEWAGE, DRIFT, CYGNO',
    title='Directional searches: resolution of tens of degrees would hide a bias of a few degrees',
    use='Directional dark-matter searches at low pressure; CYGNO reads light with a camera and PMTs.',
    specs=[('NEWAGE', '400 µm µ-PIC, CF<sub>4</sub> 76 Torr, 10 ns TOT clock'), ('DRIFT', '50 cm, negative ion CS<sub>2</sub>, 2 mm wires'), ('CYGNO', '50 cm, He:CF<sub>4</sub>, sCMOS + PMTs'), ('Reference', 'neutron simulation; scintillator bars (CYGNO)')],
    geo=None,
    did=['NEWAGE: relative arrival time per strip, line fit; TOT for charge and head-tail',
         'DRIFT: head-tail from time-profile asymmetry',
         'CYGNO: track extent from PMT time over threshold, 20 % level'],
    concl=['NEWAGE angle resolution 58° at 50-100 keVee; head-tail about 52-54 %',
           'NEWAGE: TOT stops helping once longitudinal diffusion matters',
           'CYGNO: noise-level TOT biased “towards larger values”'],
    us='No constraint on our effect: per-channel time, but neither the resolution nor a slope reference to see it.',
    quotes=[('Angles of a track in the X-Z and Y-Z planes are known by fitting the hit-positions with straight lines.', 'NEWAGE 2023, p. 7', 'https://arxiv.org/pdf/2301.04779')],
    gaps='D_T not found for these detectors. DMTPC not read.')

add(id='pixtpc', short='D3, BEAST', fam=SHORT, name='D3-Micro and BEAST pixel TPCs',
    title='BEAST’s “canoe shape”: one threshold time per pixel bends tracks even with fine pixels',
    use='D3: directional recoil imaging. BEAST: beam-background neutrons at SuperKEKB.',
    specs=[('D3-Micro', '9.2-45 mm, 50×400 µm pixels, 25 ns'), ('BEAST', '10-15 cm, 50×250 µm, 25 ns, He:CO<sub>2</sub>'), ('Velocity', 'BEAST chose 10 µm/ns so 25 ns = 250 µm'), ('Reference', 'source pointing, near-horizontal tracks')],
    geo=None,
    did=['Threshold-crossing time and ToT per pixel, 3D track',
         'Angle checks with alpha sources',
         'D3: fit-uncertainty and collimation methods'],
    concl=['BEAST: only one entry per position, “leading to the canoe shape”',
           'BEAST φ width 0.9° on 2.1 cm tracks',
           'D3: no absolute z, so cannot collimate in θ'],
    us='Fine pixels alone do not remove the mixing. The canoe shape is our family, unquantified, in a figure caption.',
    quotes=[('Because the pixel integration time is longer than the typical charge cloud size, only one entry per horizontal position is measured, leading to the canoe shape of the reconstructed tracks.', 'Jaegle 2019, p. 3', 'https://arxiv.org/pdf/1901.06657'),
            ('Since, however, a TPC cannot measure absolute position in the drift direction (z), we cannot know the true dz, and hence cannot select on it to collimate the source in theta.', 'Vahsen 2014, p. 7', 'https://arxiv.org/pdf/1407.7013')],
    gaps='Later D3 papers not located. BEAST checks are near horizontal, so blind to a z scale.')

add(id='active', short='Active targets', fam=SHORT, name='MAIKo, AT-TPC, ACTAR, TexAT',
    title='Active targets: peak or midpoint times and simulated angles; none tests a z-scale bias',
    use='Low-energy nuclear reactions with the gas as target, often with radioactive beams.',
    specs=[('MAIKo', '140 mm, 400 µm strips, 100 MHz'), ('AT-TPC', 'about 1 m, GET, 6.25 MHz'), ('ACTAR', '2×2 mm pads, 128×128, GET'), ('TexAT', '1.67×3.42 mm pads, GET')],
    geo=None,
    did=['MAIKo: time = midpoint of time over threshold',
         'AT-TPC: deconvolve, peak-find, z linear between window and mesh',
         'ACTAR: response deconvolution to an effective 3D charge',
         'TexAT: waveform fit for maximum and peak time'],
    concl=['MAIKo 1.3° from up/down halves of one track: blind to a common scale error',
           'ACTAR simulated FWHM 2.09° at 11 pads, 0.26° at 56',
           'TexAT 3° FWHM from simulation'],
    us='MAIKo’s layout is the nearest to ours in this group. Its TOT midpoint is a cheap estimator we could emulate from our own waveforms.',
    quotes=[('the time over threshold was recorded for each strip, but only the middle clock between the leading edge and the trailing edge was used in the analysis.', 'Furuno 2018, p. 6', 'https://arxiv.org/pdf/1809.02287'),
            ('The angle reconstruction depends mostly in the granularity of the detector as well as the length of the tracks.', 'ACTAR tracking, arXiv:2011.07138, p. 9-10', 'https://arxiv.org/pdf/2011.07138')],
    gaps='Bradt, Mauss and Giovinazzo 2020 full texts not retrieved. It is not established whether ACTAR angles use the deconvolved charge. SpecMAT: search summaries only.')

add(id='migdal', short='MIGDAL', fam=SHORT, name='MIGDAL optical TPC',
    title='MIGDAL: deconvolves electronics and strip coupling, then takes one peak time per strip',
    use='First observation of the Migdal effect in nuclear scattering.',
    specs=[('Drift', '3 cm: our gap'), ('Strips', '833 µm ITO, 120; camera for x-y'), ('Gas', 'CF<sub>4</sub>, 50 Torr'), ('D<sub>T</sub>', '260 µm/√cm minimum'), ('σ<sub>T</sub> / pitch', '0.54 (derived)'), ('Sampling', '2 ns')],
    geo=dict(D=260, L=30.0, pitch=0.833),
    did=['Two planned 2D deconvolutions: electronics and cross-talk, then single-electron response',
         'Simulated ITO series resistance and strip coupling (Cadence Spectre)',
         'Time of peak amplitude per strip gives z'],
    concl=['Peak time “gives an approximation for the center of the charge distribution”',
           'Strip coupling is non-negligible; no angle bias study read'],
    us='Same gap and similar Λ. They remove coupling but not mixing across depth. A forward model is directly relevant to them.',
    quotes=[('The time of the peak amplitude of the deconvolved ITO signal in a given strip is converted to a relative z position using the drift velocity.', 'Tilly and Handley 2023, p. 4', 'https://arxiv.org/pdf/2307.10477')],
    gaps='No angle bias study read.')

add(id='gridpix', short='GridPix', fam=SHORT, name='GridPix and GasPix Timepix TPCs',
    title='GridPix removes the aggregation by reading single electrons; the slope test is still unpublished',
    use='Pixel TPC R&D for future colliders; 55 µm Timepix readout.',
    specs=[('Drift', 'up to 20 mm (GridPix), 13.5 mm (GasPix)'), ('Pixels', '55 µm'), ('Gas', 'Ar/CF<sub>4</sub>/iC<sub>4</sub>H<sub>10</sub> 95/3/2; CO<sub>2</sub>/DME'), ('σ<sub>T</sub> / pitch', '7.9 at 20 mm'), ('Time', '1.56 ns ToA'), ('Reference', 'silicon telescope')],
    geo=dict(D=306, L=20.0, pitch=0.055),
    did=['Time and position of each electron, with time-walk correction by ToT',
         'Telescope tests of residuals',
         'GasPix: z(t) calibrated against telescope tracks, on a tilted chamber'],
    concl=['Diffusion dominates the position error in both directions',
           'Mean residuals: 8 µm in plane, 31 µm in drift',
           'Only explicit time bias is time walk'],
    us='The hardware answer: no per-channel sum, no mechanism. Their telescope calibration of z would absorb a compression, so it does not report one.',
    quotes=[('Diffusion is found to be the dominating error on the track position measurement both in the pixel plane and in the drift direction', 'Ligtenberg 2018, p. 6', 'https://arxiv.org/pdf/1808.04565'),
            ('the assumption of a constant drift velocity for the z-position reconstruction is not useful', 'Tsigaridas 2019, p. 5', 'https://arxiv.org/pdf/1902.00480')],
    gaps='Telescope tests have beam across the chip, so they test offsets at fixed z, not a slope along z.')

add(id='ixpe', short='IXPE', fam=SHORT, name='IXPE gas pixel detector',
    title='IXPE: different physics, but the same lesson: a null source found a 1 % effect',
    use='X-ray polarimetry in space from photoelectron track directions.',
    specs=[('Gap', '1 cm'), ('Pixels', '50 µm hexagonal'), ('Gas', 'DME, 800 mbar'), ('Track', '0.1 mm (2 keV) to 1 mm (8 keV)'), ('σ<sub>T</sub> / pitch', '1.4 (derived)'), ('Reference', 'polarised and unpolarised sources')],
    geo=None,
    did=['Reconstructed track direction in the plane, without a drift ladder',
         'Calibrated spurious modulation with unpolarised sources at two rotations'],
    concl=['Spurious modulation traced to the GEM hole structure',
           'Empirical correction, “hard to model from first principles”',
           'Absorption-point error along the track biases the direction (leakage)'],
    us='Not our mechanism. A model for how a small systematic is established: a null plus a known source.',
    quotes=[('a stretch of the effective pitch of the GEM holes in one of the two orthogonal directions as small as ∼ 1% could explain the observed modulation', 'Baldini 2021, p. 13', 'https://arxiv.org/pdf/2107.05496')],
    gaps='PolarLight-specific results not found.')

NEIGHBOURS = [
    ('CLAS12 BMT', 'Barrel Micromegas at JLab; 3 mm gap, 330-860 µm, DREAM', 'Centroid; time selects the cluster', 'No per-strip drift fit'),
    ('COMPASS', 'Micromegas, 360 µm centre; 85 ns peaking', 'Centroid; tracks to 70 mrad', 'No per-strip drift fit'),
    ('KLOE-2 CGEM', 'Cylindrical GEM, 650 µm, digital readout', 'No per-strip time', 'Cannot show it'),
    ('sTGC', '2.8 mm gap, 3.2 mm strips', 'Strip charge', 'Cannot show it'),
    ('µ-RWELL', '0.4 mm, Ar:CO2:CF4, APV25', 'µTPC + centroid', 'No angle-bias statement found'),
]
OTHER_PAD = [
    ('ALEPH', 'LEP; 6.2×30 mm, 21 rows', 'Resolution “dominated by the E×B and angular effects”', 'Slides and abstract only'),
    ('PEP-4', 'PEP; 8.5 atm Ar/CH4', 'Studied resolution vs drift and crossing angle', 'Abstract only'),
    ('sPHENIX', 'RHIC; 1 m, zigzag 2×12.5 mm, 80 ns', 'Weighted centroid; notes time-constant mismatch', 'TDR; no bias statement'),
    ('CEPC', 'Future e+e−; 500 µm pixels or 1×6 mm pads', 'Pad resolution depends on angle, pixels do not', 'Design level'),
]
CORRECTIONS = [
    ('Drop or down-weight edge strips', 'Sessa, Rinnagel, Alexeev, Vanadia', 'Fixes the symptom at the ends; loses lever arm'),
    ('Shift edge-strip positions', 'Farinelli, Vanadia', 'Data-driven; depends on the gas'),
    ('Subtract 15-20 % of each neighbour, per time bin', 'Flierl', 'Fixed fraction, removes coupling not diffusion'),
    ('Polynomial in earliest time', 'Scholer', 'Empirical, per setup'),
    ('Angle-dependent rescale', 'Bortfeldt', 'Calibration; needs a known angle'),
    ('Combine with centroid', 'Bencivenni, Alexeev', 'Avoids the problem at small angle'),
    ('Filter angles below 5° and above 85°', 'Scholer', 'Selection only'),
    ('2D deconvolution', 'Adams, ProtoDUNE-SP', 'Removes field and electronics, leaves diffusion'),
    ('Forward fit of depth profile', 'MX17 (not found elsewhere)', 'Includes diffusion, footprint and RC in the kernel'),
]
HYPOTHESES = [
    ('Reconstruction axis', 'Partly right', 'Large TPCs use a charge-weighted time per row, not time slices. LAr has both; MIMAC slices still show end deviations. Charge-weighted time is untested on MX17.', 'gold'),
    ('Lever arm Λ', 'Main separator', 'Orders the families by 1-2 decades. Necessary, not sufficient: GridPix has no aggregation; LAr short segments are untested.', 'green'),
    ('Same effect, other name', 'True for strip µTPCs', 'Capacitive coupling, charge sharing, edge strips; BEAST canoe shape. LAr: width and charge only. Pad TPCs: variance terms, weak.', 'green'),
    ('Few check an external angle', 'Supported', 'Drift scale is checked to 0.1 % but track calibration absorbs a compression. Strip µTPCs use a set angle, ±1°. Per-track telescope slope: not found.', 'green'),
]
KNOWN_NEW = [
    ('Tracks read too steep, shrinking with angle', 'Bortfeldt, Flierl, Sessa, Vanadia, Scholer, Alexeev', 'Per-track slope ratio against an independent telescope, vs |tan θ|', 'MEASURED'),
    ('Cause: capacitive coupling and edge strips', 'Same', 'Decomposition: diffusion, footprint, strip ends, RC, by switching each off in a validated model', 'MEASURED + MODEL'),
    ('Diffusion mentioned as a second cause', 'Alexeev, Flierl (a sentence each)', 'Size of the diffusion term, and its depth dependence', 'MEASURED + MODEL'),
    ('Sharing-off simulation still biased', 'Scholer (+7°, cause open)', 'Metallic arm keeps most of it; split between diffusion and strip ends in this geometry', 'MODEL'),
    ('Signal = charge ⊗ response in channel and time', 'Adams 2018 (LAr), T2K for RC maps', 'Per-event fit of a track and depth-binned charge, diffusion and RC in the kernel', 'METHOD'),
    ('Corrections by filter or calibration', 'Several groups', 'Fit reads 0.99-1.00 of the true slope on re-simulation up to 10× RC, with no edge filter', 'METHOD'),
    ('One time per channel suffices for long tracks', 'LAr, pad TPCs (implicit)', 'Λ as a design rule for when it fails', 'INFERENCE'),
    ('Strip pitch and shaping', 'No controlled study found', 'Scans by re-simulation and by summing strips on data', 'RUNNING'),
]
