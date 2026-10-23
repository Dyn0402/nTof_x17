"""Signal-masked common mode (wft.io.masked_block_median, opt-in)."""
import numpy as np

from wft import io as wio


def test_plain_when_off():
    rng = np.random.default_rng(1)
    w = rng.normal(0, 10, (32, 512)).astype(np.float32)
    plain = w - np.repeat(np.median(w.reshape(32, 8, 64), axis=2), 64, axis=1)
    assert np.array_equal(wio.masked_block_median(w, np.full(512, 10.0), 0), plain)


def test_wide_cluster_does_not_bias_the_baseline():
    rng = np.random.default_rng(2)
    ns = 32
    cm = rng.normal(0, 100, ns)[:, None]                       # coherent pickup, per sample
    w = rng.normal(0, 10, (ns, 512)) + cm
    t = np.arange(ns)
    pulse = 800 * np.exp(-0.5 * ((t - 12) / 3.0) ** 2)
    w[:, 70:100] += pulse[:, None]                              # a 30-strip track inside block 1
    noise = np.full(512, 10.0)
    plain = wio.masked_block_median(w, noise, 0)
    masked = wio.masked_block_median(w, noise, 3)
    quiet = np.r_[64:68, 104:128]                               # block-1 channels without signal
    bias_plain = abs(np.median(plain[12, quiet]))
    assert bias_plain > 10                                      # the plain median is pulled up by the track
    assert abs(np.median(masked[12, quiet])) < bias_plain / 4   # the masked one is not
