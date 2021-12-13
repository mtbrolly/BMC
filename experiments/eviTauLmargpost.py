import time
import pandas as pd
import numpy as np
from evidenceLmargpost import evidenceLmargpost

experiment = 'NS2D'  # Either 'NS2D' or 'Langevin'
save_results = 0

# DataFrame for results
L = {'Np': [], 'Ntau': [], 'tau': [], 'lLMAP': [], 'gMAP': [], 'kMAP': [],
     'varg': [], 'vark': [], 'CVg': [], 'CVk': [], 'lnEviL': [],
     'lnOccamL': []}
L = pd.DataFrame(L)


# Data
if experiment == 'Langevin':
    from LangevinSim import LSim  # noqa: E402
    d = 3
    g, k = 1., 1.
elif experiment == 'NS2D':
    Xdata = np.load('data/NS2D_fd/kf64_nx1024/X500_sub.npy')
    dt = 2.5e-3


# Prior
if experiment == 'Langevin':
    k_mean = 1.
    g_mean = 1.
elif experiment == 'NS2D':
    RMSV = 0.6805062486538265
    t_zeta = 0.07347211220002751
    k_mean = RMSV ** 2 * t_zeta
    g_mean = t_zeta ** -1


# Experiment parameters
if experiment == 'Langevin':
    Np = 100
    Ntau = 10
    nTau = 100
    taus = np.logspace(-2, 2, nTau)
elif experiment == 'NS2D':
    Np = 1000
    Ntau = 25
    nTau = 100
    taus = np.linspace(0.01, 20., nTau)

for i in range(nTau):

    t0e = time.time()
    tau = taus[i]

    if experiment == 'Langevin':
        Xdata, _ = LSim(Np, Ntau, tau, d=d, g=g, k=k, seed=i)
        dt = tau

    lLMAP, theta, lnEviL, postVar = evidenceLmargpost(tau, dt, Xdata,
                                                      g_mean, k_mean, Np, Ntau,
                                                      evi_method='Lap',
                                                      do_ML=False)
    L = L.append({'Np': Np, 'Ntau': Ntau, 'tau': tau, 'lLMAP': lLMAP,
                  'gMAP': theta[0], 'kMAP': theta[1], 'varg': postVar[0, 0],
                  'vark': postVar[1, 1],
                  'CVg': (np.sqrt(postVar[0, 0]) / theta[0]),
                  'CVk': (np.sqrt(postVar[1, 1]) / theta[1]),
                  'lnEviL': lnEviL, 'lnOccamL': lnEviL - lLMAP},
                 ignore_index=True)
    t1e = time.time()
    print(f'Iteration number {i + 1} took: {t1e - t0e:.2f} seconds')


if save_results:
    if experiment == 'Langevin':
        pd.to_pickle(L, './data/Langevin/inference/L.pkl')
    elif experiment == 'NS2D':
        pd.to_pickle(L, './data/NS2D_fd/kf64_nx1024/inference/basic_prior/'
                     + 'L_X500_sub_nTau100.pkl')
