"""
Initial conditions for model simulations.
"""

import numpy as np
import pyfftw.interfaces.numpy_fft as fftw
import diagnostic_tools as tools


def JMcW(m):
    """
    Initial condition from J. McWilliams' 1984 JFM paper.
    """
    fk = m.wv != 0
    ckappa = np.zeros_like(m.wv2)
    ckappa[fk] = np.sqrt(m.wv2[fk] * (1. + (m.wv2[fk] / 36.) ** 2)) ** -1
    nhx, nhy = m.wv2.shape
    m.rng_init = np.random.default_rng(seed=1234)
    Pi_hat = np.reshape(m.rng_init.normal(size=nhx * nhy)
                        + 1j * m.rng_init.normal(size=nhx * nhy),
                        m.wv.shape) * ckappa
    Pi = fftw.irfft2(Pi_hat[:, :])
    Pi = Pi - Pi.mean()
    Pi_hat = fftw.rfft2(Pi)
    KEaux = tools.spec_var(m, m.wv * Pi_hat)
    pik = (Pi_hat / np.sqrt(KEaux))
    zik = -m.wv2 * pik
    return zik


def zero(m):
    """
    Zero initial condition.
    """
    return 0 * m.wv
