import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

experiment = 'NS2D'  # Either 'NS2D' or 'Langevin'
save_figs = 1

if experiment == 'Langevin':
    fig_folder = "Langevin_experiment/"
elif experiment == 'NS2D':
    fig_folder = "NS2D_experiment/"
    import matplotlib.ticker as mtick  # noqa

if experiment == 'Langevin':
    B = pd.read_pickle('data/Langevin_experiment/B.pkl')
    L = pd.read_pickle('data/Langevin_experiment/L.pkl')
elif experiment == 'NS2D':
    B = pd.read_pickle('/home/s1511699/github/experiments/'
                       + 'data/NS2D_experiment/B.pkl')
    L = pd.read_pickle('/home/s1511699/github/experiments/'
                       + 'data/NS2D_experiment/L.pkl')
taus = np.array(B['tau'])
if experiment == 'NS2D':
    data = pd.read_pickle('/home/s1511699/github/experiments/'
                          + 'data/NS2D_experiment/data_T1000.pkl')
    t_zeta = data['Ens'][data['t'] > 500].mean() ** -0.5
    taus /= t_zeta

# ln Bayes factors plot
fig0, ax0 = plt.subplots(figsize=(4.0, 2.5))
lnKs = L['lnEviL'] - B['lnEviB']
if experiment == 'Langevin':
    ax0.plot(taus, lnKs, 'k', linewidth='1.0')
    ax0.set_xlabel(r'$\tau$')
elif experiment == 'NS2D':
    ax0.semilogy(taus, lnKs, 'k', base=np.e, linewidth='1.0')

    def ticks(y, pos):
        return r'$e^{{{:.0f}}}$'.format(np.log(y))

    ax0.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
    ax0.set_xlabel(r'$\tau / \tau_{\zeta}$')
ax0.set_ylabel('Log Bayes factor')
ax0.set_xlim(0, None)
plt.tight_layout()
if save_figs:
    plt.savefig(fig_folder + "lnK_semilog.pdf", format='pdf')


# ln Occam factors plot
fig1, ax1 = plt.subplots(figsize=(4.0, 2.5))
ax1.plot(taus, B['lnOccamB'], color='k',
         label=r'$\ln$'+r'Occam'+r'$_B$')
ax1.plot(taus, L['lnOccamL'], '--', color='dimgrey',
         label=r'$\ln$'+r'Occam'+r'$_L$')
if experiment == 'Langevin':
    ax1.set_xlabel(r'$\tau$')
elif experiment == 'NS2D':
    ax1.set_xlabel(r'$\tau / \tau_{\zeta}$')
ax1.set_ylabel('Log Occam factors')
ax1.legend()
plt.tight_layout()
if save_figs:
    plt.savefig(fig_folder + "lnOccam.pdf", format='pdf')
