"""
Module for parameter inference for the Brownian model.
"""

import numpy as np
from scipy.optimize import minimize


def logLikeB1(t, k, dx):
    """
    Likelihood for single 1D position increment.
    """
    return -0.5 * np.log(4 * np.pi * k * t) - dx ** 2 / (4 * k * t)


def dk_logLikeB1(t, k, dx):
    return -0.5 * k ** (-1) + dx ** 2 / (4 * t * k ** 2)


def dkk_logLikeB1(t, k, dx):
    return 0.5 * k ** (-2) - dx ** 2 / (2 * t * k ** 3)


def logLikeB(DX, kappa, tau):
    D, Ntau, Np = DX.shape
    lLike = 0
    for j in range(D):
        for p in range(Np):
            for n in range(Ntau):
                lLike += logLikeB1(tau, kappa, DX[j, n, p])
    return lLike


def log_exp_prior(kappa, kappa_mean):
    imean = kappa_mean ** -1.
    return np.log(imean) - imean * kappa


def logPostB(DX, kappa, tau, kappa_mean):
    """
    Natural log of unnormalised posterior density of kappa.
    """
    return logLikeB(DX, kappa, tau) + log_exp_prior(kappa, kappa_mean)


def dk_logLikeB(DX, kappa, tau):
    D, Ntau, Np = DX.shape
    lLike = 0
    for j in range(D):
        for p in range(Np):
            for n in range(Ntau):
                lLike += dk_logLikeB1(tau, kappa, DX[j, n, p])
    return lLike


def dk_log_exp_prior(kappa, kappa_mean):
    return -kappa_mean ** -1.


def dk_logPostB(DX, kappa, tau, kappa_mean):
    return dk_logLikeB(DX, kappa, tau) + dk_log_exp_prior(kappa, kappa_mean)


def dkk_logLikeB(DX, kappa, tau):
    D, Ntau, Np = DX.shape
    lLike = 0
    for j in range(D):
        for p in range(Np):
            for n in range(Ntau):
                lLike += dkk_logLikeB1(tau, kappa, DX[j, n, p])
    return lLike


def max_lPostB(Xobs, tau, kappa_mean):
    DX = (Xobs[:, 1:, :] - Xobs[:, :-1, :])

    def nlPost(kappa):
        return -logPostB(DX, kappa, tau, kappa_mean)

    def dk_nlPost(kappa):
        return -dk_logPostB(DX, kappa, tau, kappa_mean)
    resB = minimize(nlPost, x0=0.5, jac=dk_nlPost,
                    method='L-BFGS-B', bounds=((0.000001, np.inf),))
    kappa_star = resB.x[0]
    maxlB = -resB.fun[0]

    if not resB.success:
        print('Optimisation unsuccessful')

    return kappa_star, maxlB


def lEviB(Xobs, tau, kappa_mean):
    """
    Log-evidence
    """
    DX = (Xobs[:, 1:, :] - Xobs[:, :-1, :])

    kappa_post_star, maxlPB = max_lPostB(Xobs, tau, kappa_mean)
    maxlLB = logLikeB(DX, kappa_post_star, tau)
    log_prior = log_exp_prior(kappa_post_star, kappa_mean)

    obsInfo = -dkk_logLikeB(DX, kappa_post_star, tau)
    lnOccamB = log_prior - 0.5 * np.log(obsInfo / (2 * np.pi))
    lnLapEviB = maxlLB + lnOccamB
    return lnLapEviB
