import time
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from Langevinf import (logPostLmarg, max_lPostLmarg, hess_logPostLmarg,
                       log_exp_prior, logLikeLmarg)
plt.style.use('paper.mplstyle')


experiment = 'NS2D'
save_figs = 0

if experiment == 'Langevin':
    fig_folder = "/home/s1511699/github/transport/figures/Langevin_experiment/"
elif experiment == 'NS2D':
    fig_folder = "/home/s1511699/github/transport/figures/NS2D_experiment/"

# Data
if experiment == 'Langevin':
    from LangevinSim import LSim  # noqa: E402
    d = 3
    g, k = 1., 1.
elif experiment == 'NS2D':
    Xdata = np.load('/home/s1511699/github/transport/'
                    + 'data/NS2D_fd/kf64_nx1024/X500_sub.npy')
    dt = 1e-3 / 4 * 10

# Prior
if experiment == 'Langevin':
    k_mean = 1.
    g_mean = 1.
elif experiment == 'NS2D':
    RMSV = 0.666
    t_zeta = 162.91 ** -0.5
    k_mean = RMSV ** 2 * t_zeta
    g_mean = t_zeta ** -1

# Experiment parameters
if experiment == 'Langevin':
    Np = 20
    Ntau = 10
    tau = 1
elif experiment == 'NS2D':
    Np = 100
    Ntau = 10
    tau = 50 * t_zeta
    obsInt = int(tau / dt)

# Sampled data
if experiment == 'Langevin':
    Xobs, _ = LSim(Np, Ntau, tau, d=d, g=g, k=k)
elif experiment == 'NS2D':
    Xobs = Xdata[:, : 1 + Ntau * obsInt: obsInt, :Np]


def lPost(theta):
    return logPostLmarg(Xobs, theta[0], theta[1], tau, g_mean, k_mean)


def lLike(theta):
    return logLikeLmarg(Xobs, theta[0], theta[1], tau)


def hess_nlPost(theta):
    return -hess_logPostLmarg(Xobs, theta[0], theta[1], tau, g_mean, k_mean,
                              h=1e-5)


t_opt0 = time.time()
theta_star, maxlPL = max_lPostLmarg(Xobs, tau, g_mean, k_mean)
t_opt1 = time.time()
print(f"Optimisation took {t_opt1 - t_opt0:.2f} seconds.")

lLMAP = logLikeLmarg(Xobs, theta_star[0], theta_star[1], tau)
obsInfo = hess_nlPost(theta_star)
postVar = np.linalg.inv(obsInfo)


log_prior = log_exp_prior(theta_star[0], theta_star[1], g_mean, k_mean)
lnOccamL = log_prior - 0.5 * np.log(np.linalg.det(obsInfo
                                                  / (2 * np.pi)))
lnLapEviL = lLMAP + lnOccamL
lnEviL = lnLapEviL


do_plot = 0

if do_plot:
    ng = 40
    nk = 40
    grange = np.logspace(np.log10(theta_star[0]) - 2.,
                         np.log10(theta_star[0]) + 2., ng)
    krange = np.logspace(np.log10(theta_star[1]) - 0.7,
                         np.log10(theta_star[1]) + 0.7, nk)

    t0c = time.time()
    lLM = np.zeros((ng, nk))
    for gni in range(ng):
        for kni in range(nk):
            lLM[gni, kni] = lPost(np.array((grange[gni], krange[kni])))
    t1c = time.time()
    print('Plotting calculations time: {}'.format(t1c-t0c)+'seconds')

    lmax = np.max(lLM)
    lmin = np.min(lLM)
    if lmax < 0:
        levels = (-np.logspace(np.log10(-lmax), np.log10(-(lmax-500)), 12))[::-1]  # noqa: E501
    else:
        levels = np.linspace(lmax-5000, lmax, 12)

    plt.figure()
    plt.contourf(grange, krange, np.transpose(lLM), levels,
                 cmap=cmocean.cm.thermal)
    plt.colorbar()
    plt.contour(grange, krange, np.transpose(lLM), levels,
                colors='k', linestyles='solid', linewidths=0.5)
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$k$')
    plt.title(r'$\ln p(\gamma,k | \mathcal{X}_{\tau},\mathcal{M}_L)$, '
              + rf'$\tau={tau:.3g}$, $N_p={Np:.0f}$, $N_{{\tau}}={Ntau}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(theta_star[0], theta_star[1],
             'kx', markersize=10, label=r'$\theta^*$')
    plt.legend()
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_folder + "lnPost_tau1.pdf", format='pdf')
