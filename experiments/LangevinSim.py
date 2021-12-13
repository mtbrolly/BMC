import numpy as np
from scipy import linalg as la
from inference.Langevinf import CXXf, CXUf, CUUf


def LSim(Np, Ntau, tau, d=1, g=1, k=1, seed=1, dt=None):
    """
    Exact simulations of trajectories of the Langevin model.
    """
    if not dt:
        dt = tau
    rng = np.random.default_rng(seed=seed)
    Tn = int(np.ceil(Ntau * tau / dt - 1e-15))

    XU = np.zeros((2 * d, Tn + 1, Np))
    XU[:d, 0, :] = rng.uniform(size=d * Np).reshape((d, Np)) * 2 * np.pi
    XU[d:, 0, :] = (rng.standard_normal(d * Np).reshape((d, Np))
                    * ((k*g) ** 0.5))

    xi = rng.standard_normal(2 * d * Tn * Np).reshape(2 * d, Tn, Np)

    CXX = CXXf(g, k, dt)
    CXU = CXUf(g, k, dt)
    CUU = CUUf(g, k, dt)

    m = np.concatenate((np.tile(np.array([[(1 - np.exp(-g*dt))/g], ]), (d, 1)),
                       np.tile(np.array([[np.exp(-g*dt)], ]), (d, 1))), axis=0)

    cxxicxui = np.concatenate([CXX * np.eye(d), CXU * np.eye(d)], axis=1)
    cxuicuui = np.concatenate([CXU * np.eye(d), CUU * np.eye(d)], axis=1)
    C = np.concatenate([cxxicxui, cxuicuui], axis=0)

    Sigma = la.cholesky(C).T

    for tn in range(Tn):
        mean = (np.concatenate([XU[:d, tn, :], np.zeros((d, Np))])
                + np.concatenate([XU[d:, tn, :], XU[d:, tn, :]]) * m)
        XU[:, tn + 1] = mean + Sigma @ xi[:, tn, :]

    return XU[:d, :, :], XU[d:, :, :]
