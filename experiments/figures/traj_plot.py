import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('paper.mplstyle')

save_figs = 0
fig_folder = "/home/s1511699/github/experiments/figures/flow/"

X = np.load('/home/s1511699/github/experiments/data/NS2D_experiment/X.npy')

data = pd.read_pickle('/home/s1511699/github/experiments/'
                      + 'data/NS2D_experiment/data_T1000.pkl')
t_zeta = data['Ens'][data['t'] > 500].mean() ** -0.5
dt = 1e-2 / 4
max_tn = int(100 * t_zeta / dt)

fig, ax = plt.subplots(1, 1, figsize=[3, 3])
for p in range(100):
    ax.plot(X[0, :max_tn, p], X[1, :max_tn, p], linewidth=0.5)
ax.set_aspect('equal')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, 2 * np.pi)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
fig.tight_layout()
if save_figs:
    fig.savefig(fig_folder + "traj.pdf", format='pdf')
