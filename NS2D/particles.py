import warnings
import numpy as np

try:
    import scipy.ndimage
except ImportError:
    warnings.warn('Failed to import scipy.ndimage. '
                  'Gridded interpolation will not work',
                  ImportWarning)


class LagrangianParticleArray2D(object):
    """A class for keeping track of a set of lagrangian particles
    in two-dimensional space. Tries to be fast.
    """

    def __init__(self, x0, y0,
                 periodic_in_x=False,
                 periodic_in_y=False,
                 xmin=-np.inf, xmax=np.inf,
                 ymin=-np.inf, ymax=np.inf,
                 particle_dtype='f8'):
        """
        Parameters
        ----------

        x0, y0 : array-like
            Two arrays (same size) representing the particle initial
            positions.
        periodic_in_x : bool
            Whether the domain wraps in the x direction.
        periodic_in_y : bool
            Whether the domain 'wraps' in the y direction.
        xmin, xmax : numbers
            Maximum and minimum values of x coordinate
        ymin, ymax : numbers
            Maximum and minimum values of y coordinate
        particle_dtype : dtype
            Data type to use for particles
        """

        self.x = np.array(x0, dtype=np.dtype(particle_dtype)).ravel()
        self.y = np.array(y0, dtype=np.dtype(particle_dtype)).ravel()

        assert self.x.shape == self.y.shape
        self.N = len(self.x)

        # check that the particles are within the specified boundaries
        assert np.all(self.x >= xmin) and np.all(self.x <= xmax)
        assert np.all(self.y >= ymin) and np.all(self.y <= ymax)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.pix = periodic_in_x
        self.piy = periodic_in_y

        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin

    def step_forward_with_function(self, uv0fun, uv1fun, dt):
        """Advance particles using a function to determine u and v.

        Parameters
        ----------
        uv0fun : function
            Called like ``uv0fun(x,y)``. Should return the velocity field
            u, v at time t.
        uv1fun(x,y) : function
            Called like ``uv1fun(x,y)``. Should return the velocity field
            u, v at time t + dt.
        dt : number
            Timestep."""

        dx, dy = self._rk4_integrate(self.x, self.y, uv0fun, uv1fun, dt)
        self.x = self._wrap_x(self.x + dx)
        self.y = self._wrap_y(self.y + dy)

    def _rk4_integrate(self, x, y, uv0fun, uv1fun, dt):
        """Integrates positions x, y using velocity functions
           uv0fun, uv1fun. Returns dx and dy, the displacements."""
        u0, v0 = uv0fun(x, y)
        k1u = dt * u0
        k1v = dt * v0
        x11 = self._wrap_x(x + 0.5 * k1u)
        y11 = self._wrap_y(y + 0.5 * k1v)
        u11, v11 = uv1fun(x11, y11)
        k2u = dt * u11
        k2v = dt * v11
        x12 = self._wrap_x(x + 0.5 * k2u)
        y12 = self._wrap_y(y + 0.5 * k2v)
        u12, v12 = uv1fun(x12, y12)
        k3u = dt * u12
        k3v = dt * v12
        x13 = self._wrap_x(x + k3u)
        y13 = self._wrap_y(y + k3v)
        u13, v13 = uv1fun(x13, y13)
        k4u = dt * u13
        k4v = dt * v13

        # update
        dx = 6 ** -1 * (k1u + 2 * k2u + 2 * k3u + k4u)
        dy = 6 ** -1 * (k1v + 2 * k2v + 2 * k3v + k4v)
        return dx, dy

    def _AB3_integrate(self, x, y, xm1, ym1, xm2, ym2,
                       uvfun, uvm1fun, uvm2fun, dt):
        f1x, f1y = uvfun(x, y)
        f2x, f2y = uvm1fun(xm1, ym1)
        f3x, f3y = uvm2fun(xm2, ym2)
        dx = (dt / 12) * (23 * f1x - 16 * f2x + 5 * f3x)
        dy = (dt / 12) * (23 * f1y - 16 * f2y + 5 * f3y)
        return dx, dy

    def _wrap_x(self, x):
        if self.pix:
            return np.mod(x - self.xmin, self.Lx) + self.xmin
        else:
            return x

    def _wrap_y(self, y):
        if self.piy:
            return np.mod(y-self.ymin, self.Ly) + self.ymin
        else:
            return y

    def _distance(self, x0, y0, x1, y1):
        """Utitlity function to compute distance between points."""
        dx = x1-x0
        dy = y1-y0
        # roll displacements across the borders
        if self.pix:
            dx[dx > self.Lx/2] -= self.Lx
            dx[dx < -self.Lx/2] += self.Lx
        if self.piy:
            dy[dy > self.Ly/2] -= self.Ly
            dy[dy < -self.Ly/2] += self.Ly
        return dx, dy

    def _step_forward_with_model(self, model, method='RK4'):
        if method == 'RK4':
            [model.um1p, model.vm1p,
             model.up, model.vp] = [self._pad_field(c, pad=1)
                                    for c in [model.um1, model.vm1,
                                              model.u, model.v]]
        elif method == 'AB3':
            [model.um2p, model.vm2p,
             model.um1p, model.vm1p,
             model.up, model.vp] = [self._pad_field(c, pad=1)
                                    for c in [model.um2, model.vm2,
                                              model.um1, model.vm1,
                                              model.u, model.v]]
        else:
            print('Particle numerics method not recognised.')

        if method == 'AB3':
            uvm2fun = (lambda x, y:
                       (self.interpolate_gridded_scalar(self._wrap_x(x),
                                                        self._wrap_y(y),
                                                        model.um2p, pad=0,
                                                        order=1, offset=1),
                        self.interpolate_gridded_scalar(self._wrap_x(x),
                                                        self._wrap_y(y),
                                                        model.vm2p, pad=0,
                                                        order=1, offset=1)))
        uvm1fun = (lambda x, y:
                   (self.interpolate_gridded_scalar(self._wrap_x(x),
                                                    self._wrap_y(y),
                                                    model.um1p, pad=0,
                                                    order=1, offset=1),
                    self.interpolate_gridded_scalar(self._wrap_x(x),
                                                    self._wrap_y(y),
                                                    model.vm1p, pad=0,
                                                    order=1, offset=1)))
        uvfun = (lambda x, y:
                 (self.interpolate_gridded_scalar(self._wrap_x(x),
                                                  self._wrap_y(y),
                                                  model.up, pad=0,
                                                  order=1, offset=1),
                  self.interpolate_gridded_scalar(self._wrap_x(x),
                                                  self._wrap_y(y),
                                                  model.vp, pad=0,
                                                  order=1, offset=1)))

        if method == 'RK4':
            dx, dy = self._rk4_integrate(self.x, self.y, uvm1fun, uvfun,
                                         model.dt)
        else:
            dx, dy = self._AB3_integrate(self.x, self.y, self.xm1, self.ym1,
                                         self.xm2, self.ym2,
                                         uvfun, uvm1fun, uvm2fun, model.dt)
        self.x = self.x + dx
        self.y = self.y + dy
        self.uv = uvfun(self.x, self.y)


class GriddedLagrangianParticleArray2D(LagrangianParticleArray2D):
    """Lagrangian particles with velocities given on a regular cartesian grid.
    """

    def __init__(self, x0, y0, Nx, Ny, grid_type='A', **kwargs):
        """
        Parameters
        ----------

        x0, y0 : array-like
            Two arrays (same size) representing the particle initial
            positions.
        Nx, Ny: int
            Number of grid points in the x and y directions
        grid_type: {'A'}
            Arakawa grid type specifying velocity positions.
        """

        super(GriddedLagrangianParticleArray2D, self).__init__(x0, y0,
                                                               **kwargs)
        self.Nx = Nx
        self.Ny = Ny

        if grid_type != 'A':
            raise ValueError('Only A grid velocities supported at this time.')

        if not (self.pix and self.piy):
            raise ValueError('Interpolation only works with doubly'
                             + 'periodic grids at this time.')

        # figure out grid geometry, assuming velocities are at cell centers

    def interpolate_gridded_scalar(self, x, y, c, order=1, pad=1, offset=0):
        """Interpolate gridded scalar C to points x,y.

        Parameters
        ----------
        x, y : array-like
            Points at which to interpolate
        c : array-like
            The scalar, assumed to be defined on the grid.
        order : int
            Order of interpolation
        pad : int
            Number of pad cells added
        offset : int
            ???

        Returns
        -------
        ci : array-like
            The interpolated scalar
        """

        # first pad the array to deal with the boundaries
        # (map_coordinates can't seem to deal with this by itself)
        # pad twice so cubic interpolation can be used
        if pad > 0:
            cp = self._pad_field(c, pad=pad)
        else:
            cp = c
        # cp has shape (Nx + 2 * pad, Nx + 2 * pad)
        i = (x - self.xmin) / self.Lx * self.Nx + pad + offset - 0.5
        j = (y - self.ymin) / self.Ly * self.Ny + pad + offset - 0.5

        # for some reason this still does not work with high precision near
        # the boundaries
        return scipy.ndimage.map_coordinates(cp, [j, i], mode='constant',
                                             order=order, cval=np.nan)

    def _pad_field(self, c, pad=5):
        return np.pad(c, ((pad, pad), (pad, pad)), mode='wrap')
