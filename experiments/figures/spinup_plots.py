import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('paper.mplstyle')
save_figs = 1
fig_folder = "/home/s1511699/github/experiments/figures/flow/"

data = pd.read_pickle('/home/s1511699/github/experiments/'
                      + 'data/NS2D_experiment/data_T1000.pkl')

t_zeta = data['Ens'][data['t'] > 500].mean() ** -0.5
rmsv = np.sqrt(2 * data['ke'][data['t'] > 500].mean())
mean_kurt_u = data['kurt_u'][data['t'] > 500].mean()
mean_kurt_z = data['kurt_z'][data['t'] > 500].mean()

data['t'] /= t_zeta

diagnostics = ['ke', 'Ens', 'kurt_u', 'kurt_z']
labels = ['Energy', 'Enstrophy', 'Velocity kurtosis', 'Vorticity kurtosis']

for i in range(len(diagnostics)):
    plt.figure()
    plt.plot(data['t'], data[diagnostics[i]], 'k')
    plt.xlabel(r'$t/\tau_{\zeta}$')
    plt.ylabel(labels[i])
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_folder + diagnostics[i] + "_time_series.pdf",
                    format='pdf')
