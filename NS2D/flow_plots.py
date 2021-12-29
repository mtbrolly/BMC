import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cmocean
from NS2D.model import Model
from NS2D.diagnostic_tools import KEspec, calc_ispec
plt.style.use('figures/paper.mplstyle')

save_figs = 1
fig_folder = "figures/flow/"
package_dir = str(Path().absolute().parent)

m = Model(Tend=500, nx=1024, dt=0.001/4, twrite=200,
          k_f=64, f_a=0.1/(2.8e-14), lsf_a=1)

m.zk = np.load(package_dir + '/experiments/data/NS2D_experiment/zk_T1000.npy')
m._calc_derived_fields()


fig0, ax0 = plt.subplots(1, 2, figsize=(5, 2.75))
lims = [2 * np.pi, 0.5 * np.pi]
for a in range(ax0.size):
    ax0[a].set(aspect='equal')
    pc = ax0[a].pcolormesh(m.x, m.y, m.z, cmap=cmocean.cm.curl,
                           shading='gouraud')
    cmax = 150.
    pc.set_clim(-cmax, cmax)
    if a == 0:
        ax0[a].plot([0, lims[a + 1]], [lims[a + 1], lims[a + 1]], 'k--')
        ax0[a].plot([lims[a + 1], lims[a + 1]], [0, lims[a + 1]], 'k--')
    ax0[a].set_xlim([0, lims[a]])
    ax0[a].set_ylim([0, lims[a]])
    ax0[a].set_xticks([])
    ax0[a].set_yticks([])
fig0.tight_layout()
if save_figs:
    # fig0.savefig(fig_folder + "Fig3.png", format='png', dpi=576)
    fig0.savefig(fig_folder + "Fig3.eps", format='eps', dpi=576)


spec2D = KEspec(m)
kr, iso_E_spec = calc_ispec(m, spec2D)

fig1, ax1 = plt.subplots(figsize=(3.0, 2.5))
ax1.set_ylabel(r'$E(k)$', rotation=0)
ax1.loglog(kr, iso_E_spec, 'k')
range1 = np.array([10, m.k_f - 2])
range2 = np.array([m.k_f + 2, .35 * m.nx])
ax1.loglog(range1, .5 * range1 ** -(2.), color='grey',
           linestyle='--')
ax1.loglog(range2, 1e2 * range2 ** -3.5, color='grey',
           linestyle='--')
ax1.text(15, .00005, r'$k^{-2}$')
ax1.text(83, .00000009, r'$k^{-3.5}$')
ax1.vlines(m.k_f, 1e-10, 1e0, color='grey', linestyles='-.')
ax1.set_xlim(1, m.nx * 0.6)
ax1.set_ylim(1e-10, 0.5e-1)
ax1.set_xlabel(r'$k$')
fig1.tight_layout()
if save_figs:
    fig1.savefig(fig_folder + "Fig4.pdf", format='pdf')
