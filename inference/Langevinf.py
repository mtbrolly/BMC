"""
Module for parameter inference for the Langevin model.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.linalg import toeplitz


def phif(x):
    return -np.expm1(-x) / x


def CXXf(g, k, tau):
    return 2 * k * tau * (1 - 2 * phif(g * tau) + phif(2 * g * tau))


def CXUf(g, k, tau):
    return k * (phif(g * tau) * g * tau) ** 2


def CUUf(g, k, tau):
    return 2 * k * g ** 2 * tau * phif(2 * g * tau)


"""
Independent exponential priors for gamma and k.
"""


def log_exp_prior(g, k, g_mean, k_mean):
    imean_g = g_mean ** -1.
    imean_k = k_mean ** -1.
    lp_g = np.log(imean_g) - imean_g * g
    lp_k = np.log(imean_k) - imean_k * k
    return lp_g + lp_k


def grad_log_exp_prior(g, k, g_mean, k_mean):
    return np.array([-g_mean ** -1., -k_mean ** -1])


def hess_log_exp_prior(g, k, g_mean, k_mean):
    return 0


"""
Lkelihood and posterior for position observations.
"""


def logLikeLmarg(X, g, k, tau):
    DX = (X[:, 1:, :] - X[:, :-1, :])
    D, Ntau, Np = DX.shape

    phi = phif(g * tau)
    mean = np.zeros((Ntau,))
    if Ntau == 1:
        cov = 2 * k * tau * (1 - phi)
    if Ntau > 1:
        col = ([2 * k * tau * (1 - phi), ]
               + [k * g * tau ** 2 * phi ** 2 * np.exp(-(h - 1) * g * tau)
                  for h in range(1, Ntau)])
        cov = toeplitz(np.array(col))

    logLike = 0
    for p in range(Np):
        for d in range(D):
            logLike += multivariate_normal.logpdf(DX[d, :, p], mean, cov)
    return logLike


def logPostLmarg(X, g, k, tau, g_mean, k_mean):
    return logLikeLmarg(X, g, k, tau) + log_exp_prior(g, k, g_mean, k_mean)


def grad_logLikeLmarg(X, g, k, tau, h=1e-2):
    h1 = g * h
    h2 = k * h

    def lLike(theta):
        return logLikeLmarg(X, theta[0], theta[1], tau)
    dg = (lLike((g - 2 * h1, k)) - 8 * lLike((g - h1, k))
          + 8 * lLike((g + h1, k)) - lLike((g + 2 * h1, k))) / (12 * h1)
    dk = (lLike((g, k - 2 * h2)) - 8 * lLike((g, k - h2))
          + 8 * lLike((g, k + h2)) - lLike((g, k + 2 * h2))) / (12 * h2)
    return np.array([dg, dk])


def dg_logLikeLmarg(X, g, k, tau, h=1e-2):
    def lLike(theta):
        return logLikeLmarg(X, theta[0], theta[1], tau)
    h *= g
    dg = (lLike((g - 2 * h, k)) - 8 * lLike((g - h, k))
          + 8 * lLike((g + h, k)) - lLike((g + 2 * h, k))) / (12 * h)
    return dg


def hess_logLikeLmarg(X, g, k, tau, h=1e-2):
    def lLike(theta):
        return logLikeLmarg(X, theta[0], theta[1], tau)

    def dg_lLike(theta):
        return dg_logLikeLmarg(X, theta[0], theta[1], tau, h=h)

    h1 = g * h
    h2 = k * h
    dgg = (-lLike((g - 2 * h1, k)) + 16 * lLike((g - h1, k))
           - 30 * lLike((g, k)) + 16 * lLike((g + h1, k))
           - lLike((g + 2 * h1, k))) / (12 * (h1 ** 2))
    dgk = (dg_lLike((g, k - 2 * h2)) - 8 * dg_lLike((g, k - h2))
           + 8 * dg_lLike((g, k + h2)) - dg_lLike((g, k + 2 * h2))) / (12 * h2)
    dkk = (-lLike((g, k - 2 * h2)) + 16 * lLike((g, k - h2))
           - 30 * lLike((g, k)) + 16 * lLike((g, k + h2))
           - lLike((g, k + 2 * h2))) / (12 * (h2 ** 2))
    return np.array([[dgg, dgk], [dgk, dkk]])


def hess_logPostLmarg(X, g, k, tau, g_mean, k_mean, h=1e-2):
    hess_lL = hess_logLikeLmarg(X, g, k, tau, h=1e-2)
    hess_log_prior = hess_log_exp_prior(g, k, g_mean, k_mean)
    return hess_lL + hess_log_prior


def max_lPostLmarg(X, tau, g_mean, k_mean, x0=(0.5, 0.5), h=1e-2):
    def nlPost(theta):
        return -logPostLmarg(X, theta[0], theta[1], tau, g_mean, k_mean)

    resL = minimize(nlPost, x0=x0, method='Nelder-Mead',
                    bounds=((0.0000001, None), (0.0000001, None)))
    if not resL.success:
        print('Optimisation failed. Reason:', resL.message)
    theta_star = resL.x
    maxlL = -resL.fun

    return theta_star, maxlL
