import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
plt.style.use('paper.mplstyle')

save_figs = 1
experiments_dir = str(Path().absolute().parent)
NS2D_dir = str(Path().absolute().parent.parent) + "/NS2D"
fig_folder = NS2D_dir + "/figures/flow/"

X = np.load(experiments_dir + '/data/NS2D_experiment/X.npy')

data = pd.read_pickle(experiments_dir + '/data/NS2D_experiment/data_T1000.pkl')
t_zeta = data['Ens'][data['t'] > 500].mean() ** -0.5
dt = 2.5e-3
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
    fig.savefig(fig_folder + "Fig5.pdf", format='pdf')
