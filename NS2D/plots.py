"""
Plotting functions for simulations.
"""

import numpy as np
import diagnostic_tools as tools
import matplotlib.pyplot as plt
from drawnow import drawnow
import cmocean
plt.rcParams.update({'font.size': 14})
plt.rc('text', usetex=True)


def run_and_plot(m):
    """
    Run model with animated plot.
    """
    m._calc_derived_fields()
    plt.ion()
    plt.figure(figsize=(13, 6))

    m.plotint = .1
    m.plotints = np.ceil(m.plotint / m.dt)

    # cmax = 20.

    def plot_z():
        plt.subplot(1, 2, 1)
        plt.title(r'$\zeta(t = {:.2f})$'.format(m.t))
        plt.pcolormesh(m.x, m.y, m.z, cmap=cmocean.cm.curl,
                       shading='gouraud')
        # plt.clim([-cmax, cmax])
        plt.xlim([0, 2 * np.pi])
        plt.ylim([0, 2 * np.pi])
        plt.colorbar()

        plt.subplot(1, 2, 2)
        m.spec2D = tools.KEspec(m)
        m.kr, m.iso_E_spec = tools.calc_ispec(m, m.spec2D)
        plt.loglog(m.kr, m.iso_E_spec, 'k')
        m.ke = tools.calc_ke(m)
        m.energy_cent = tools.calc_centroid_k(m.kr, m.iso_E_spec, m.ke)
        plt.title(r'Energy spectrum: $\bar{k}$ '+'$={:.1f}$'.format(m.energy_cent))  # noqa: E501

        if m.k_f:
            range1 = np.array([4, m.k_f - 2])
            range2 = np.array([m.k_f + 2, .35 * m.nx])
            plt.loglog(range1, .3 * range1 ** -(5/3), color='grey',
                       linestyle='--')
            plt.loglog(range2, 1e1 * range2 ** -3., color='grey',
                       linestyle='--')
            plt.text(10, .0006, r'$k^{-5/3}$')
            plt.text(100, .00001, r'$k^{-3}$')

            plt.vlines(m.k_f, 1e-8, 1e-2, linestyles='--')

        plt.xlim(1, m.nx * 0.6)
        plt.ylim(1e-8, 1e0)
        plt.xlabel(r'$k$')
        plt.tight_layout()

    drawnow(plot_z)

    for _ in m.run_with_snapshots(tsnapint=m.dt):
        if (m.tn % m.plotints) == 0:
            m._calc_derived_fields()
            drawnow(plot_z)
