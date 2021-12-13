import numpy as np
import time
import matplotlib.pyplot as plt
from inference.Brownianf import logLikeB, dkk_logLikeB, max_lPostB, logPostB
plt.style.use('paper.mplstyle')

root_directory = "/home/s1511699/github/BMC/"

# Data
Xdata = np.load(root_directory + 'experiments/data/NS2D_experiment/X.npy')
dt = 2.5e-3


# Prior
RMSV = 0.6805062486538265
t_zeta = 0.07347211220002751
kappa_mean = RMSV ** 2 * t_zeta


# Subsampling data
Np = 100
Ntau = 10
tau = 1. * t_zeta
obsInt = int(tau / dt)
Xobs = Xdata[:, : 1 + Ntau * obsInt: obsInt, :Np]
DX = (Xobs[:, 1:, :] - Xobs[:, :-1, :])


def lLikeB(kappa):
    return logLikeB(DX, kappa, tau)


def dkk_nlLikeB(kappa):
    return -dkk_logLikeB(DX, kappa, tau)


def lPostB(kappa):
    return logPostB(DX, kappa, tau, kappa_mean)


# Optimise to find maximum posterior value and parameter
t0o = time.time()
kappa_star, maxlPB = max_lPostB(Xobs, tau, kappa_mean)
t1o = time.time()
print('Posterior optimisation took: ', t1o - t0o, ' seconds')

maxlB = lLikeB(kappa_star)

# Evidence calculation by Laplace's Method
uni_prior = 1 / 9.9
obsInfo = dkk_nlLikeB(kappa_star)
occamB = uni_prior * (obsInfo / (2 * np.pi)) ** (-0.5)
lnLapEviB = maxlB + np.log(occamB)

# Evidence approximation given by Bayesian information criterion (BIC)
BIC = 1. * np.log(Np * Ntau) - 2. * maxlB
BIC_lnEviB = -0.5 * BIC

# Print results
print('kappa* =', kappa_star)
print('maxlL =', maxlB)
print('lnLapEviB =', lnLapEviB)
print('Var(kappa) =', obsInfo ** (-1))

do_plot = 0

# Plot log posterior
if do_plot:
    plt.figure()
    ks = np.logspace(np.log10(kappa_star) - 1, np.log10(kappa_star) + 1, 1000)
    lLs = np.array([lLikeB(ks[i]) for i in range(len(ks))])
    plt.plot(ks, lLs, 'k', label=r'$\ln p(\kappa|\Delta \mathcal{X}_{\tau})$')
    plt.title(rf'$\tau={tau:.1f}$, $N_p={Np:.0f}$, $N_{{\tau}}$={Ntau}')
    plt.xlabel(r'$\kappa$')
    plt.vlines(kappa_star, lLs.min(), lLs.max(), 'grey', label=r'$\kappa^*$',
               linestyle='dashdot')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.tight_layout()
