"""
    Integrate module
    ================

    Module with the function to integrate the ordinary differential equations

    .. math:: \dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    of the model and its linearized version.

    Description of the module functions
    -----------------------------------

    Two main functions:

    * :obj:`integrate_runge_kutta`
    * :obj:`integrate_runge_kutta_tgls`

"""

from numba import njit
import numpy as np

@njit
def reverse(a):
    """Numba-jitted function to reverse a 1D array.

    Parameters
    ----------
    a: ~numpy.ndarray
        The 1D array to reverse.

    Returns
    -------
    ~numpy.ndarray
        The reversed array.
    """
    out = np.zeros_like(a)
    ii = 0
    for i in range(len(a)-1,-1,-1):
        out[ii] = a[i]
        ii +=1
    return out

def integrate_runge_kutta(f, t0, t, dt, ic=None, forward=True, write_steps=1, b=None, c=None, a=None):
    """
    Integrate the ordinary differential equations (ODEs)

    .. math:: \dot{\\boldsymbol{x}} = \\boldsymbol{f}(t, \\boldsymbol{x})

    with a specified `Runge-Kutta method`_. The function :math:`\\boldsymbol{f}` should
    be a `Numba`_ jitted function. This function must have a signature ``f(t, x)`` where ``x`` is
    the state value and ``t`` is the time.

    .. _Runge-Kutta method: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    .. _Numba: https://numba.pydata.org/

    Parameters
    ----------
    f: callable
        The `Numba`_-jitted function :math:`\\boldsymbol{f}`.
        Should have the signature``f(t, x)`` where ``x`` is the state value and ``t`` is the time.
    t0: float
        Initial time of the time integration. Corresponds to the initial condition's `ic` time.
        Important if the ODEs are non-autonomous.
    t: float
        Final time of the time integration. Corresponds to the final condition.
        Important if the ODEs are non-autonomous.
    dt: float
        Timestep of the integration.
    ic: None or ~numpy.ndarray(float), optional
        Initial condition of the system. Can be a 1D or a 2D array:

        * 1D: Provide a single initial condition.
          Should be of shape (`n_dim`,) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`.
        * 2D: Provide an ensemble of initial condition.
          Should be of shape (`n_traj`, `n_dim`) where `n_dim` = :math:`\mathrm{dim}(\\boldsymbol{x})`,
          and where `n_traj` is the number of initial conditions.

        If `None`, use a zero initial condition. Default to `None`.

    forward: bool, optional
        Whether to integrate the ODEs forward or backward in time. In case of backward integration, the
        initial condition `ic` becomes a final condition. Default to forward integration.
    write_steps: int, optional
        Save the state of the integration in memory every `write_steps` steps. The other intermediary
        steps are lost. It determine the size of the returned objects. Default is 1.
        Set to 0 to return only the final state.
    b: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`b_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    c: None or ~numpy.ndarray, optional
        Matrix of coefficients :math:`c_{i,j}` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.
    a: None or ~numpy.ndarray, optional
        Vector of coefficients :math:`a_i` of the `Runge-Kutta method`_ .
        If `None`, use the classic RK4 method coefficients. Default to `None`.

    Returns
    -------
    time, traj: ~numpy.ndarray
        The result of the integration:

        * `time` is the time at which the state of the system was saved. Array of shape (`n_step`,) where
          `n_step` is the number of saved states of the integration.
        * `traj` are the saved states. 3D array of shape (`n_traj`, `n_dim`, `n_steps`). If `n_traj` = 1,
          a 2D array of shape (`n_dim`, `n_steps`) is returned instead.

    Examples
    --------

    >>> from numba import njit
    >>> import numpy as np
    >>> from integrators.integrate import integrate_runge_kutta
    >>> a = 0.25
    >>> F = 16.
    >>> G = 3.
    >>> b = 6.
    >>> # Lorenz 84 example
    >>> @njit
    ... def fL84(t, x):
    ...     xx = -x[1] ** 2 - x[2] ** 2 - a * x[0] + a * F
    ...     yy = x[0] * x[1] - b * x[0] * x[2] - x[1] + G
    ...     zz = b * x[0] * x[1] + x[0] * x[2] - x[2]
    ...     return np.array([xx, yy, zz])
    >>> # no ic
    >>> # write_steps is 1 by default
    >>> tt, traj = integrate_runge_kutta(fL84, t0=0., t=10., dt=0.1)  # 101 steps
    >>> print(traj.shape)
    (3, 101)
    >>> # 1 ic
    >>> ic = 0.1 * np.random.randn(3)
    >>> tt, traj = integrate_runge_kutta(fL84, t0=0., t=10., dt=0.1, ic=ic)  # 101 steps
    >>> print(ic.shape)
    (3,)
    >>> print(traj.shape)
    (3, 101)
    >>> # 4 ic
    >>> ic = 0.1 * np.random.randn(4, 3)
    >>> tt, traj = integrate_runge_kutta(fL84, t0=0., t=10., dt=0.1, ic=ic)  # 101 steps
    >>> print(ic.shape)
    (4, 3)
    >>> print(traj.shape)
    (4, 3, 101)
    """

    if ic is None:
        i = 1
        while True:
            ic = np.zeros(i)
            try:
                x = f(0., ic)
            except:
                i += 1
            else:
                break

        i = len(f(0., ic))
        ic = np.zeros(i)

    if len(ic.shape) == 1:
        ic = ic.reshape((1, -1))

    # Default is RK4
    if a is None and b is None and c is None:
        c = np.array([0., 0.5, 0.5, 1.])
        b = np.array([1./6, 1./3, 1./3, 1./6])
        a = np.zeros((len(c), len(b)))
        a[1, 0] = 0.5
        a[2, 1] = 0.5
        a[3, 2] = 1.

    if forward:
        time_direction = 1
    else:
        time_direction = -1

    time = np.concatenate((np.arange(t0, t, dt), np.full((1,), t)))

    recorded_traj, recorded_tend = _integrate_runge_kutta_jit(f, time, ic, time_direction, write_steps, b, c, a)

    if write_steps > 0:
        if forward:
            if time[::write_steps][-1] == time[-1]:
                return time[::write_steps], np.squeeze(recorded_traj), np.squeeze(recorded_tend)
            else:
                return np.concatenate((time[::write_steps], np.full((1,), t))), np.squeeze(recorded_traj), np.squeeze(recorded_tend)
        else:
            rtime = reverse(time[::-write_steps])
            if rtime[0] == time[0]:
                return rtime, np.squeeze(recorded_traj), np.squeeze(recorded_tend)
            else:
                return np.concatenate((np.full((1,), t0), rtime)), np.squeeze(recorded_traj), np.squeeze(recorded_tend)
    else:
        return time[-1], np.squeeze(recorded_traj), np.squeeze(recorded_tend)


# @njit
def _integrate_runge_kutta_jit(f, time, ic, time_direction, write_steps, b, c, a):

    n_traj = ic.shape[0]
    n_dim = ic.shape[1]

    s = len(b)

    if write_steps == 0:
        n_records = 1
    else:
        tot = time[::write_steps]
        n_records = len(tot)
        if tot[-1] != time[-1]:
            n_records += 1

    recorded_traj = np.zeros((n_traj, n_dim, n_records))
    recorded_tend = np.zeros((n_traj, n_dim, n_records))

    if time_direction == -1:
        directed_time = reverse(time)
    else:
        directed_time = time

    for i_traj in range(n_traj):
        y = ic[i_traj].copy()
        k = np.zeros((s, n_dim))
        iw = 0
        for ti, (tt, dt) in enumerate(zip(directed_time[:-1], np.diff(directed_time))):

            if write_steps > 0 and np.mod(ti, write_steps) == 0:
                recorded_traj[i_traj, :, iw] = y
                recorded_tend[i_traj, :, iw] = f(0, y)                
                iw += 1

            k.fill(0.)
            for i in range(s):
                y_s = y + dt * a[i] @ k
                k[i] = f(tt + c[i] * dt, y_s)
            y_new = y + dt * b @ k
            y = y_new

        recorded_traj[i_traj, :, -1] = y
        recorded_tend[i_traj, :, -1] = f(0, y)

    return recorded_traj[:, :, ::time_direction], recorded_tend[:, :, ::time_direction]
