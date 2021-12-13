"""
A model class for simulating homogeneous two-dimensional turbulence.

Several dissipation mechanisms are implemented, namely:
    - linear friction;
    - hypoviscosity (or "large-scale friction");
    - Newtonian viscosity;
    - hyperviscosity;
    - a low-pass, exponential, spectral filter for small-scale dissipation.

A white-noise forcing is also implemented.

The beta-effect is implemented for optional use.


Martin Brolly, 2021.
"""

import numpy as np
from numpy import pi
import diagnostic_tools
import pyfftw.interfaces.numpy_fft as fftw
import pandas as pd
import logging

try:
    import mkl  # noqa: F401
    np.use_fastnumpy = True
except ImportError:
    pass


class Model():
    def __init__(
                 self,
                 L=2 * pi,
                 W=None,
                 nx=512,
                 ny=None,
                 dt=5e-4,
                 Tend=40,
                 Tout=.5,
                 twrite=1000.,

                 dissipation='filter',
                 filterfac=23.6,
                 nu=None,
                 nu_2=None,

                 beta=0.,

                 k_f=None,
                 f_a=None,
                 E_input_rate=None,

                 d_a=None,
                 lsf_a=None,

                 dealias=False,

                 log_level=1,
                 logfile=None  # logfile; None prints to screen
                 ):

        if ny is None:
            ny = nx
        if W is None:
            W = L

        self.nx = nx
        self.ny = ny
        self.L = L
        self.W = W

        self.beta = beta

        self.k_f = k_f
        if E_input_rate:
            self.f_a = E_input_rate * nx ** -4. * k_f ** 2. * 6e20
        else:
            self.f_a = f_a

        self.d_a = d_a
        self.lsf_a = lsf_a

        self.dt = dt
        self.Tend = Tend
        self.Tout = Tout
        self.Tendn = int(self.Tend / self.dt)
        self.twrite = twrite
        self.logfile = logfile
        self.log_level = log_level

        self.dissipation = dissipation
        self.filterfac = filterfac
        self.nu = nu
        self.nu_2 = nu_2

        self.dealias = dealias

        self._initialise_grid()
        self._initialise_filter()
        self._initialise_time()
        self._initialise_logger()
        self._initialise_dataset()

        if self.k_f:
            self._initialise_forcing()

    def _initialise_time(self):
        """
        Initialise time and timestep at zero.
        """
        self.t = 0
        self.tn = 0

    def _initialise_grid(self):
        """
        Define spatial and spectral grids and related constants, as well as
        padding tools if dealiasing is in use.
        """
        self.x, self.y = np.meshgrid(
            np.arange(0.5, self.nx, 1.) / self.nx * self.L,
            np.arange(0.5, self.ny, 1.) / self.ny * self.W)

        self.dk = 2.*pi / self.L
        self.dl = 2.*pi / self.W

        self.nl = self.ny
        self.nk = int(self.nx / 2 + 1)
        self.ll = self.dl * np.append(np.arange(0., self.nx / 2),
                                      np.arange(-self.nx / 2, 0.))
        self.kk = self.dk * np.arange(0., self.nk)

        self.k, self.l = np.meshgrid(self.kk, self.ll)  # noqa: E741
        self.ik = 1j * self.k
        self.il = 1j * self.l

        self.dx = self.L / self.nx
        self.dy = self.W / self.ny

        # Constant for spectral normalizations
        self.nxny = self.nx * self.ny

        self.wv2 = self.k**2 + self.l**2
        self.wv = np.sqrt(self.wv2)

        iwv2 = self.wv2 != 0.
        self.wv2i = np.zeros_like(self.wv2)
        self.wv2i[iwv2] = self.wv2[iwv2] ** -1

        # Spectral dealiasing
        if self.dealias:
            self.pad = 3. / 2.
            self.mx = int(self.pad * self.nx)
            self.mk = int(self.pad * self.nk)
            self.padder = np.ones(self.mx, dtype=bool)
            self.padder[int(self.nx / 2):
                        int(self.nx * (self.pad - 0.5)):] = False

    def _initialise_filter(self):
        """
        Define low-pass, exponential, spectral filter for small scale
        dissipation.
        """
        cphi = 0.65 * pi
        wvx = np.sqrt((self.k * self.dx) ** 2. + (self.l * self.dy) ** 2.)
        exp_filter = np.exp(-self.filterfac * (wvx - cphi) ** 4.)
        exp_filter[wvx <= cphi] = 1.
        self.exp_filter = exp_filter

    def _initialise_forcing(self):
        """
        Set up random forcing.
        """
        self.f_rng = np.random.default_rng(seed=1234)
        F = ((self.wv > self.k_f - 2.) & (self.wv < self.k_f + 2.)) * self.f_a
        self.fk_vars = F / ((self.wv + (self.wv == 0)) * pi) / 2

    def _generate_forcing(self):
        """
        Generate a (new) realisation of random forcing.
        """
        self.fk = np.reshape(self.f_rng.normal(size=self.wv.size)
                             + 1j * self.f_rng.normal(size=self.wv.size),
                             self.wv.shape) * np.sqrt(self.fk_vars)

    def _initialise_logger(self):
        self.logger = logging.getLogger(__name__)
        # if not (self.logfile is None):
        if self.logfile is not None:
            fhandler = logging.FileHandler(filename=self.logfile, mode='w')
        else:
            fhandler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        fhandler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(fhandler)
        self.logger.setLevel(self.log_level*10)

        self.logger.info('Logger initialized')

    def _initialise_dataset(self):
        """
        Create a pandas DataFrame for recording diagnostics.
        """
        data = {'tn': [], 't': [], 'ke': [], 'Ens': [], 'cfl': [],
                'kurt_u': [], 'kurt_z': []}
        self.data = pd.DataFrame(data)

    def _print_status(self):
        """
        Calculate, log and record diagnostics in DataFrame.
        """
        if (self.log_level) and ((self.tn % self.twrite) == 0):
            self._calc_derived_fields()
            self.ke = diagnostic_tools.calc_ke(self)
            self.Ens = diagnostic_tools.calc_Ens(self)
            self.cfl = diagnostic_tools.calc_cfl(self)
            self.kurt_u = diagnostic_tools.calc_kurt_u(self)
            self.kurt_z = diagnostic_tools.calc_kurt_z(self)

            self.logger.info('Step: %i, Time: %3.2e, ke: %3.2e, Ens: %3.2e, '
                             + 'CFL: %4.3f, kurt_u: %3.2e, kurt_z: %3.2e',
                             self.tn, self.t, self.ke, self.Ens,
                             self.cfl, self.kurt_u, self.kurt_z)

            self.data = self.data.append({'tn': int(self.tn), 't': self.t,
                                          'ke': self.ke, 'Ens': self.Ens,
                                          'cfl': self.cfl,
                                          'kurt_u': self.kurt_u,
                                          'kurt_z': self.kurt_z},
                                         ignore_index=True)

            assert self.cfl < 1., self.logger.error('CFL condition violated')

    def _calc_z(self):
        """
        Compute z from zk.
        """
        self.z = fftw.irfft2(self.zk)

    def _calc_zk(self):
        """
        Compute zk from z.
        """
        self.zk = fftw.rfft2(self.z)

    def _calc_psi(self):
        """
        Compute psi from zk.
        """
        self.psik = -self.wv2i * self.zk
        self.psi = fftw.irfft2(self.psik)

    def _calc_dealiased_advection(self):
        """
        Calculate dealiased advection term from zk.
        """
        self.nlk = np.zeros(self.zk.shape, dtype='complex128')
        self.uk = -self.il * self.psik
        self.vk = self.ik * self.psik

        # Create padded arrays
        self.uk_padded = np.zeros((self.mx, self.mk), dtype='complex128')
        self.vk_padded = np.zeros((self.mx, self.mk), dtype='complex128')
        self.z_xk_padded = np.zeros((self.mx, self.mk), dtype='complex128')
        self.z_yk_padded = np.zeros((self.mx, self.mk), dtype='complex128')

        # Enter known coefficients, leaving padded entries equal to zero
        self.uk_padded[self.padder, :self.nk] = self.uk[:, :]
        self.vk_padded[self.padder, :self.nk] = self.vk[:, :]
        self.z_xk_padded[self.padder, :self.nk] = (self.ik * self.zk)[:, :]
        self.z_yk_padded[self.padder, :self.nk] = (self.il * self.zk)[:, :]

        # Inverse transform padded arrays
        self.u_padded = fftw.irfft2(self.uk_padded)
        self.v_padded = fftw.irfft2(self.vk_padded)
        self.z_x_padded = fftw.irfft2(self.z_xk_padded)
        self.z_y_padded = fftw.irfft2(self.z_yk_padded)

        # Calculate Jacobian term
        self.nlk[:, :] = fftw.rfft2((self.u_padded * self.z_x_padded
                                     + self.v_padded
                                     * self.z_y_padded)
                                    )[self.padder, :self.nk] * self.pad ** 2
        return self.nlk

    def _calc_tendency(self):
        """
        Calculates "RHS" of barotropic vorticity equation. exp_filter should be
        used in conjunction with this to apply small scale dissipation.
        """
        self.psik = -self.wv2i * self.zk

        if self.dealias:
            self.RHS = -self._calc_dealiased_advection()
        else:
            self.uk = -self.il * self.psik
            self.vk = self.ik * self.psik

            self.u = fftw.irfft2(self.uk)
            self.v = fftw.irfft2(self.vk)

            self.z_x = fftw.irfft2(self.ik * self.zk)
            self.z_y = fftw.irfft2(self.il * self.zk)

            # Advection
            self.nlk = (fftw.rfft2(self.u * self.z_x)
                        + fftw.rfft2(self.v * self.z_y))
            self.RHS = -self.nlk

        # Beta effect
        if self.beta:
            self.RHS -= self.beta * (self.ik * self.psik)

        # Random forcing
        if self.f_a:
            self._generate_forcing()
            self.RHS += self.fk

        # Linear drag
        if self.d_a:
            self.dragk = -self.d_a * self.zk
            self.RHS += self.dragk

        # Large scale friction
        if self.lsf_a:
            self.lsf_k = self.lsf_a * self.psik
            self.RHS += self.lsf_k

        # Viscosity
        if self.dissipation == 'viscosity':
            self.RHS += -self.nu * self.wv2 * self.zk

        # Hyperviscosity
        if self.dissipation == 'hyperviscosity':
            if not self.nu_2:
                print('nu_2 must defined for hyperviscosity.')
            self.RHS += -self.nu_2 * self.wv2 ** 2 * self.zk

    def _step_forward(self):
        """
        Evolve zk one step according to barotropic vorticity equation.
        """
        self._calc_tendency()

        # Timestepping: Adams--Bashforth 3rd order.
        if self.tn == 0:
            self.zk = (self.zk + self.dt * self.RHS)
            if self.dissipation == 'filter':
                self.zk *= self.exp_filter

        elif self.tn == 1:
            self.zk = self.zk + (self.dt / 2) * (3 * self.RHS - self.RHS_m1)
            if self.dissipation == 'filter':
                self.zk *= self.exp_filter

        else:
            self.zk = self.zk + (self.dt / 12) * (23 * self.RHS
                                                  - 16 * self.RHS_m1
                                                  + 5 * self.RHS_m2)
            if self.dissipation == 'filter':
                self.zk *= self.exp_filter

        # Record preceding tendencies
        if self.tn > 0:
            self.RHS_m2 = self.RHS_m1.copy()
        self.RHS_m1 = self.RHS.copy()

        self._print_status()

        self.tn += 1
        self.t += self.dt

    def run(self):
        """
        Run model uninterrupted until final time.
        """
        while(self.t < self.Tend):
            self._step_forward()
            if self.tn == self.Tendn:
                break
        self._calc_derived_fields()

    def run_with_snapshots(self, tsnapint=1.):
        """
        Run model with interruptions at set intervals to allow for other
        calculations to be done, e.g. plotting, evolving particles.
        """
        tsnapints = np.ceil(tsnapint / self.dt)

        while(self.t < self.Tend):
            self._step_forward()
            if (self.tn % tsnapints) == 0:
                self._calc_z()
                yield self.t
            if self.tn > self.Tendn:
                break
        self._calc_derived_fields()

    def _calc_derived_fields(self):
        """
        Typically only zk is explicitly updated during timestepping; this
        updates other fields based on the current zk.
        """
        self.psik = -self.wv2i * self.zk
        self.uk = -self.il * self.psik
        self.vk = self.ik * self.psik

        self.z = fftw.irfft2(self.zk)
        self.psi = fftw.irfft2(self.psik)
        self.u = fftw.irfft2(self.uk)
        self.v = fftw.irfft2(self.vk)
