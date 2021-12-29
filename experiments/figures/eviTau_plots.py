import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use('paper.mplstyle')
plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsfonts}'})

experiment = 'NS2D'  # Either 'NS2D' or 'Langevin'
save_figs = 1
experiments_dir = str(Path().absolute().parent)

if experiment == 'Langevin':
    fig_folder = "Langevin_experiment/"
    fig = "Fig1"
elif experiment == 'NS2D':
    fig_folder = "NS2D_experiment/"
    fig = "Fig6"

if experiment == 'Langevin':
    B = pd.read_pickle(experiments_dir + '/data/Langevin_experiment/B.pkl')
    L = pd.read_pickle(experiments_dir + '/data/Langevin_experiment/L.pkl')
elif experiment == 'NS2D':
    B = pd.read_pickle(experiments_dir + '/data/NS2D_experiment/B.pkl')
    L = pd.read_pickle(experiments_dir + '/data/NS2D_experiment/L.pkl')

taus = np.array(B['tau'])
if experiment == 'NS2D':
    data = pd.read_pickle(experiments_dir
                          + '/data/NS2D_experiment/data_T1000.pkl')
    t_zeta = data['Ens'][data['t'] > 500].mean() ** -0.5
    RMSV = 0.696
    k_mean = RMSV ** 2 * t_zeta
    g_mean = t_zeta ** -1
    taus /= t_zeta
    B['kappaMAP'] /= k_mean
    L['gMAP'] /= g_mean
    L['kMAP'] /= k_mean
    B['varkappa'] /= k_mean ** 2
    L['varg'] /= g_mean ** 2
    L['vark'] /= k_mean ** 2

# kappa lines colour
ka_c = tuple(np.array((117, 112, 179)) / 256)

# gamma lines colour
g_c = tuple(np.array((27, 158, 119)) / 256)

# k lines colour
k_c = tuple(np.array((217, 95, 2)) / 256)


fig0, ax0 = plt.subplots(figsize=(4.0, 2.0))
ax0.plot(taus, B['kappaMAP'], '--', color='k', linewidth='0.5')
ax0.fill_between(taus,
                 B['kappaMAP'] - B['varkappa'] ** 0.5,
                 B['kappaMAP'] + B['varkappa'] ** 0.5,
                 color=ka_c)
if experiment == 'Langevin':
    ax0.set_xlabel(r'$\tau$')
    ax0.set_xscale('log')
    ax0.set_xlim(taus[0], taus[-1])
    ax0.set_ylabel(r'Inference of $\kappa$')
elif experiment == 'NS2D':
    ax0.set_xlabel(r'$\tau / \tau_{\zeta}$')
    ax0.set_xlim(0, taus[-1])
    ax0.set_ylabel(r'Inference of $\kappa/\mathbb{E}[\kappa]$')
ax0.set_ylim(0, None)
fig0.tight_layout()
if save_figs:
    plt.savefig(fig_folder + fig + "a.pdf", format='pdf')


fig1, ax1 = plt.subplots(figsize=(4.0, 2.0))
ax1.plot(taus, L['gMAP'], '--', color='k', linewidth='0.5')
ax1.fill_between(taus,
                 L['gMAP'] - L['varg'] ** 0.5,
                 L['gMAP'] + L['varg'] ** 0.5,
                 color=g_c)
if experiment == 'Langevin':
    ax1.set_xlabel(r'$\tau$')
    ax1.set_xscale('log')
    ax1.set_xlim(taus[0], taus[-1])
    ax1.set_ylabel(r'Inference of $\gamma$')
elif experiment == 'NS2D':
    ax1.set_xlabel(r'$\tau / \tau_{\zeta}$')
    ax1.set_xlim(0, taus[-1])
    ax1.set_ylabel(r'Inference of $\gamma/\mathbb{E}[\gamma]$')
ax1.set_ylim(0, None)
fig1.tight_layout()
if save_figs:
    plt.savefig(fig_folder + fig + "b.pdf", format='pdf')

fig2, ax2 = plt.subplots(figsize=(4.0, 2.0))
ax2.plot(taus, L['kMAP'], '--', color='k', linewidth='0.5')
ax2.fill_between(taus,
                 L['kMAP'] - L['vark'] ** 0.5,
                 L['kMAP'] + L['vark'] ** 0.5,
                 color=k_c)
if experiment == 'Langevin':
    ax2.set_xlabel(r'$\tau$')
    ax2.set_xscale('log')
    ax2.set_xlim(taus[0], taus[-1])
    ax2.set_ylabel(r'Inference of $k$')
elif experiment == 'NS2D':
    ax2.set_xlabel(r'$\tau / \tau_{\zeta}$')
    ax2.set_ylabel(r'Inference of $k/\mathbb{E}[k]$')
    ax2.set_xlim(0, taus[-1])
ax2.set_ylim(0, None)
fig2.tight_layout()
if save_figs:
    plt.savefig(fig_folder + fig + "c.pdf", format='pdf')
