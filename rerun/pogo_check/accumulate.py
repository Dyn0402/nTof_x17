"""Per-pair event statistics for the pogo check, from hits (QA only: which channels fire together and how big;
no position, angle or depth is reconstructed from hit times).

For one readout plane (512 channels of one FEU) and a stream of events, each event a list of (channel, amplitude):

  F[a]      events in which channel a fires (amplitude > FIRE_ADC)
  J[a, b]   events in which a and b both fire **in different clusters**, a cluster being a run of hit strips in
            strip order broken by >= GAP_MIN quiet strips.  Charge sharing and the track's own extent only ever
            make contiguous runs; a and b firing with quiet strips between them is a jump that the detector
            cannot make by itself (apart from accidentals, a second track, or a delta ray).
  K[a, b]   events in which a and b both fire (any cluster)
  R1, R2    sums of l = ln(A_a / A_b) and l^2 over the K events: the spread of the amplitude ratio.  A shorted
            pair reads one shared charge, so l ~ 0 in every event; charge sharing between neighbours moves the
            ratio from event to event with where the track crossed.
  X, XX, XY sums of ln A_a, (ln A_a)^2 and ln A_a ln A_b over the J events: the log-amplitude regression of the
            jumps.  A pad short makes A_b follow A_a over the whole dynamic range (correlation ~ 1, slope ~ 1);
            pickup, crosstalk and accidentals fire together without following each other, and hits just above
            threshold all look alike, which is why a plain |ln ratio| < tol identity test is fooled by them.
  S, Nq     the census correlation (cosine of amplitude products), kept for comparison only.

Events with a saturated hit are dropped (C20: crosstalk in saturated events), as are events with more than
MAX_HITS hits.
"""
import numpy as np

FIRE_ADC = 50.0
GAP_MIN = 2
MAX_HITS = 40
SIG_MIN = 5.0


class PairStats:
    def __init__(self, strip_index):
        """strip_index: (512,) int strip number of each channel in position order (-1 = unmapped)."""
        self.si = np.asarray(strip_index, int)
        n = 512
        self.F = np.zeros(n); self.H = np.zeros(n)
        self.J = np.zeros((n, n), np.float32); self.K = np.zeros((n, n), np.float32)
        self.R1, self.R2, self.X, self.XX, self.XY = (np.zeros((n, n)) for _ in range(5)); self.S = np.zeros((n, n)); self.Nq = np.zeros(n)
        self.nev = 0

    def add(self, ch, amp):
        ch = np.asarray(ch, int); amp = np.asarray(amp, float)
        m = self.si[ch] >= 0
        ch, amp = ch[m], amp[m]
        if len(ch) < 2 or len(ch) > MAX_HITS:
            return
        self.nev += 1
        self.H[ch] += 1
        self.S[np.ix_(ch, ch)] += np.outer(amp, amp); self.Nq[ch] += amp ** 2
        o = np.argsort(self.si[ch]); ch, amp = ch[o], amp[o]
        s = self.si[ch]
        lab = np.r_[0, np.cumsum(np.diff(s) > GAP_MIN)]            # > GAP_MIN index step = >= GAP_MIN quiet strips
        f = amp > FIRE_ADC
        if f.sum() == 0:
            return
        fc, fa, fl = ch[f], amp[f], lab[f]
        self.F[fc] += 1
        if len(fc) < 2:
            return
        ix = np.ix_(fc, fc)
        self.K[ix] += 1
        sep = fl[:, None] != fl[None, :]
        la = np.log(fa); lr = la[:, None] - la[None, :]
        self.J[ix] += sep
        self.R1[ix] += lr; self.R2[ix] += lr ** 2
        self.X[ix] += sep * la[:, None]; self.XX[ix] += sep * la[:, None] ** 2
        self.XY[ix] += sep * np.outer(la, la)

    def save(self, path, **meta):
        np.savez_compressed(path, F=self.F, H=self.H, J=self.J, K=self.K, R1=self.R1, R2=self.R2, X=self.X, XX=self.XX, XY=self.XY, S=self.S, Nq=self.Nq,
                            nev=self.nev, strip_index=self.si,
                            cuts=dict(FIRE_ADC=FIRE_ADC, GAP_MIN=GAP_MIN, MAX_HITS=MAX_HITS,
                                      SIG_MIN=SIG_MIN), **meta)


def strip_index_from_positions(pos, pitch=0.78):
    pos = np.asarray(pos, float)
    si = np.full(len(pos), -1)
    ok = np.isfinite(pos)
    si[ok] = np.rint((pos[ok] - pos[ok].min()) / pitch).astype(int)
    return si
