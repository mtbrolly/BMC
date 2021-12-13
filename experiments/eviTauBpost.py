import time
import pandas as pd
import numpy as np
from evidenceBpost import evidenceBpost

experiment = 'NS2D'  # Either 'NS2D' or 'Langevin'
save_results = 0
hpc = 0


# DataFrame for results
B = {'Np': [], 'Ntau': [], 'tau': [], 'lBMAP': [], 'kappaMAP': [],
     'varkappa': [], 'CVkappa': [], 'lnEviB': [], 'lnOccamB': []}
B = pd.DataFrame(B)


# Data
if experiment == 'Langevin':
    from LangevinSim import LSim  # noqa: E402
    d = 3
    g, k = 1., 1.
elif experiment == 'NS2D':
    if hpc:
        Xdata = np.load('/data/s1511699/NS2D_fd/kf64_nx1024/X500_sub.npy')
    else:
        Xdata = np.load('data/NS2D_fd/kf64_nx1024/X500_2000spinup_sub.npy')
    dt = 1e-3 / 4 * 10

# Prior
if experiment == 'Langevin':
    kappa_mean = 1.
elif experiment == 'NS2D':
    RMSV = 0.666
    t_zeta = 162.91 ** -0.5
    kappa_mean = RMSV ** 2 * t_zeta


# Experiment parameters
if experiment == 'Langevin':
    Np = 100
    Ntau = 10
    nTau = 100
    taus = np.logspace(-2, 2, nTau)
elif experiment == 'NS2D':
    Np = 100
    Ntau = 10
    nTau = 30
    taus = np.linspace(0.01, 30., nTau)
    # taus = np.logspace(np.log10(0.01), np.log10(20.), nTau)


for i in range(nTau):
    t0e = time.time()
    tau = taus[i]

    if experiment == 'Langevin':
        Xdata, _ = LSim(Np, Ntau, tau, d=d, g=g, k=k, seed=i)
        dt = tau

    lBMAP, kappa, lnEviB, postVarKappa = evidenceBpost(tau, dt, Xdata,
                                                       kappa_mean, Np,
                                                       Ntau, evi=True)
    B = B.append({'Np': Np, 'Ntau': Ntau, 'tau': tau, 'lBMAP': lBMAP,
                  'kappaMAP': kappa, 'varkappa': postVarKappa,
                  'CVkappa': (np.sqrt(postVarKappa) / kappa), 'lnEviB': lnEviB,
                  'lnOccamB': lnEviB - lBMAP}, ignore_index=True)
    t1e = time.time()
    print(f'Iteration number {i + 1} took: {t1e - t0e:.2f} seconds')

if save_results:
    if experiment == 'Langevin':
        pd.to_pickle(B, './data/Langevin/inference/B.pkl')
    elif experiment == 'NS2D':
        if hpc:
            pd.to_pickle(B,'/data/s1511699/NS2D_fd/kf64_nx1024/inference/basic_prior/B_X500_sub_nTau100.pkl')  # noqa: E501
        else:
            pd.to_pickle(B,'./data/NS2D_fd/kf64_nx1024/inference/basic_prior/B_X500_sub_nTau100.pkl')  # noqa: E501
