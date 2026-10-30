"""Head-on det4 cluster of one SPS fit window: the charge-centroid position estimator and the footprint (sps_audit 2026-10-30).

At normal incidence there is no drift ladder, so a charge centroid is a legitimate position (RECONSTRUCTION_BASIS.md
forbids hit TIMES for geometry, not charge centroids of a head-on column).  It is stored next to the wft fit's p0 so
the two can be compared on the same events.

Definition, per view:
  - strip amplitude = its maximum within +-T_HALF samples of the lead strip's peak sample (pile-up at other times in
    the 1.9-3.8 us window does not enter);
  - lead strip = the largest amplitude among the strips that are not censored (``bad``: the fit's dead list, i.e. the
    H4 census-dead, the oscillating ch 372/510 and the row's hot channels);
  - the cluster grows from the lead to each side while the next strip (in position) is above THR_ADC; a censored strip
    is bridged (its charge is unknown, it is not counted) when the strip after it is above threshold;
  - position = amplitude-weighted mean of the cluster strips; n_strips = their number; lead_frac = lead / sum;
    nb_lo / nb_hi = the lower / upper neighbour's amplitude over the lead's (charge sharing; NaN at an edge or bridge).
"""
import numpy as np

PITCH_MM = 0.78
THR_ADC = 50.0          # 5 x the 10 ADC per-strip noise of the SPS caches (NOISE_ADC)
T_HALF = 4              # samples (240 ns) around the lead strip's peak
SAMPLE_NS = 60.0


def cluster(P, bad=(), thr=THR_ADC, t_half=T_HALF):
    """dict(pos, n_strips, q_sum, q_lead, lead_frac, nb_lo, nb_hi, t_peak_ns) or None when no strip passes thr."""
    W = np.asarray(P['W'], float); pos = np.asarray(P['pos'], float); ch = np.asarray(P['ch'], int)
    good = ~np.isin(ch, np.asarray(list(bad), int))
    afull = np.where(good, W.max(1), -np.inf)
    i0 = int(np.argmax(afull))
    if not np.isfinite(afull[i0]) or afull[i0] < thr:
        return None
    tpk = int(np.argmax(W[i0]))
    a = W[:, max(0, tpk - t_half):tpk + t_half + 1].max(1)
    members = [i0]
    for step in (-1, 1):
        i = i0
        while True:
            j = i + step
            if j < 0 or j >= len(a) or abs(pos[j] - pos[i]) > 1.5 * PITCH_MM:
                break
            if good[j] and a[j] > thr:
                members.append(j); i = j; continue
            k = j + step                                          # bridge one censored strip
            if (not good[j]) and 0 <= k < len(a) and abs(pos[k] - pos[j]) <= 1.5 * PITCH_MM and good[k] and a[k] > thr:
                members.append(k); i = k; continue
            break
    m = np.array(sorted(members))
    q = np.clip(a[m], 0, None)
    out = dict(pos=float((q * pos[m]).sum() / q.sum()), n_strips=int(len(m)), q_sum=float(q.sum()), q_lead=float(a[i0]),
               lead_frac=float(a[i0] / q.sum()), t_peak_ns=float(tpk * SAMPLE_NS), nb_lo=np.nan, nb_hi=np.nan)
    for key, j in (('nb_lo', i0 - 1), ('nb_hi', i0 + 1)):
        if 0 <= j < len(a) and good[j] and abs(pos[j] - pos[i0]) <= 1.5 * PITCH_MM:
            out[key] = float(a[j] / a[i0])
    return out
