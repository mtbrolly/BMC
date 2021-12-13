import time
import pandas as pd
import numpy as np
from inference.evidenceBpost import evidenceBpost

experiment = 'NS2D'  # Either 'NS2D' or 'Langevin'
save_results = 0


# DataFrame for results
B = {'Np': [], 'Ntau': [], 'tau': [], 'lBMAP': [], 'kappaMAP': [],
     'varkappa': [], 'CVkappa': [], 'lnEviB': [], 'lnOccamB': []}
B = pd.DataFrame(B)


# Data
if experiment == 'Langevin':
    from experiments.LangevinSim import LSim  # noqa: E402
    d = 3
    g, k = 1., 1.
elif experiment == 'NS2D':
    Xdata = np.load('data/NS2D_experiment/X.npy')
    dt = 2.5e-3


# Prior
if experiment == 'Langevin':
    kappa_mean = 1.
elif experiment == 'NS2D':
    RMSV = 0.6805062486538265
    t_zeta = 0.07347211220002751
    kappa_mean = RMSV ** 2 * t_zeta


# Experiment parameters
if experiment == 'Langevin':
    Np = 10  # 0
    Ntau = 10
    nTau = 10  # 0
    taus = np.logspace(-2, 2, nTau)
elif experiment == 'NS2D':
    Np = 10  # 00
    Ntau = 25
    nTau = 10  # 0
    taus = np.linspace(0.01, 20., nTau)


for i in range(nTau):
    t0e = time.time()
    tau = taus[i]

    if experiment == 'Langevin':
        Xdata, _ = LSim(Np, Ntau, tau, d=d, g=g, k=k, seed=i)
        dt = tau

    lBMAP, kappa, lnEviB, postVarKappa = evidenceBpost(tau, dt, Xdata,
                                                       kappa_mean, Np, Ntau)
    B = B.append({'Np': Np, 'Ntau': Ntau, 'tau': tau, 'lBMAP': lBMAP,
                  'kappaMAP': kappa, 'varkappa': postVarKappa,
                  'CVkappa': (np.sqrt(postVarKappa) / kappa), 'lnEviB': lnEviB,
                  'lnOccamB': lnEviB - lBMAP}, ignore_index=True)
    t1e = time.time()
    print(f'Iteration number {i + 1} took: {t1e - t0e:.2f} seconds')

if save_results:
    if experiment == 'Langevin':
        pd.to_pickle(B, 'experiments/data/Langevin_experiment/B.pkl')
    elif experiment == 'NS2D':
        pd.to_pickle(B, 'experiments/data/NS2D_experiment/B.pkl')
