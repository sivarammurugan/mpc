"""
ltisys -- a collection of classes and functions for modeling linear
time invariant systems.
"""
from __future__ import division, print_function, absolute_import

#
# Author: Sivaram Murugan Subramanian
#

import warnings
import numpy as np

#np.linalg.qr fails on some tests with LinAlgError: zgeqrf returns -7
#use scipy's qr until this is solved

from scipy.linalg import qr as s_qr


import numpy
from numpy import (r_, eye, real, atleast_1d, atleast_2d, poly,
                   squeeze, diag, asarray, product, zeros, array,
                   dot, transpose, ones, zeros_like, linspace, nan_to_num)
import copy

from scipy import integrate, interpolate, linalg
from scipy._lib.six import xrange

from .filter_design import tf2zpk, zpk2tf, normalize, freqs


__all__ = ['tf2ss', 'ss2tf', 'abcd_normalize', 'zpk2ss', 'ss2zpk', 'lti',
           'TransferFunction', 'ZerosPolesGain', 'StateSpace', 'lsim',
           'lsim2', 'impulse', 'impulse2', 'step', 'step2', 'bode',
           'freqresp', 'place_poles']


def tf2ss(d):
    """Transfer function to state-space representation.
    Parameters
    ----------
    num, den : array_like
        Sequences representing the numerator and denominator polynomials.
        The denominator needs to be at least as long as the numerator.
    Returns
    -------
    A, B, C, D : ndarray
        State space representation of the system, in controller canonical
        form.
    """
    # Controller canonical state-space representation.
    #  if M+1 = len(num) and K+1 = len(den) then we must have M <= K
    #  states are found by asserting that X(s) = U(s) / D(s)
    #  then Y(s) = N(s) * X(s)
    #
    #   A, B, C, and D follow quite naturally.
    #
    #num, den = normalize(num, den)   # Strips zeros, checks arrays
    nn = len(num.shape)
    if nn == 1:
        num = asarray([num], num.dtype)
    M = num.shape[1]
    K = len(den)
    if M > K:
        msg = "Improper transfer function. `num` is longer than `den`."
        raise ValueError(msg)
    if M == 0 or K == 0:  # Null system
        return (array([], float), array([], float), array([], float),
                array([], float))

    # pad numerator to have same number of columns has denominator
    num = r_['-1', zeros((num.shape[0], K - M), num.dtype), num]

    if num.shape[-1] > 0:
        D = num[:, 0]
    else:
        D = array([], float)

    if K == 1:
        return array([], float), array([], float), array([], float), D

    frow = -array([den[1:]])
    A = r_[frow, eye(K - 2, K - 1)]
    B = eye(K - 1, 1)
    C = num[:, 1:] - num[:, 0] * den[1:]
    return A, B, C, D, d


def lsim(system, U=None, T=None, X0=None, **kwargs):
    """
    Simulate output of a continuous-time linear system, by using
    the ODE solver `scipy.integrate.odeint`.
    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:
        * 2: (num, den)
        * 3: (zeros, poles, gain)
        * 4: (A, B, C, D)
    U : array_like (1D or 2D), optional
        An input array describing the input at each time T.  Linear
        interpolation is used between given times.  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.  If U is not given, the input is assumed
        to be zero.
    T : array_like (1D or 2D), optional
        The time steps at which the input is defined and at which the
        output is desired.  The default is 101 evenly spaced points on
        the interval [0,10.0].
    X0 : array_like (1D), optional
        The initial condition of the state vector.  If `X0` is not
        given, the initial conditions are assumed to be 0.
    kwargs : dict
        Additional keyword arguments are passed on to the function
        `odeint`.  See the notes below for more details.
    Returns
    -------
    T : 1D ndarray
        The time values for the output.
    yout : ndarray
        The response of the system.
    xout : ndarray
        The time-evolution of the state-vector.
    Notes
    -----
    This function uses `scipy.integrate.odeint` to solve the
    system's differential equations.  Additional keyword arguments
    given to `lsim2` are passed on to `odeint`.  See the documentation
    for `scipy.integrate.odeint` for the full list of arguments.
    """

    if X0 is None:
        X0 = zeros(sys.B.shape[0], sys.A.dtype)

    if T is None:
        # XXX T should really be a required argument, but U was
        # changed from a required positional argument to a keyword,
        # and T is after U in the argument list.  So we either: change
        # the API and move T in front of U; check here for T being
        # None and raise an exception; or assign a default value to T
        # here.  This code implements the latter.
        T = linspace(0, 100, 101)

    T = atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError("T must be a rank-1 array.")

    if U is not None:
        U = atleast_1d(U)
        if len(U.shape) == 1:
            U = U.reshape(-1, 1)
        sU = U.shape
        if sU[0] != len(T):
            raise ValueError("U must have the same number of rows "
                             "as elements in T.")

        if sU[1] != sys.inputs:
            raise ValueError("The number of inputs in U (%d) is not "
                             "compatible with the number of system "
                             "inputs (%d)" % (sU[1], sys.inputs))
        # Create a callable that uses linear interpolation to
        # calculate the input at any time.
        ufunc = interpolate.interp1d(T, U, kind='linear',
                                     axis=0, bounds_error=False)

        def fprime(x, t, sys, ufunc):
            """The vector field of the linear system."""
            return dot(sys.A, x) + squeeze(dot(sys.B, nan_to_num(ufunc([t]))))
        xout = integrate.odeint(fprime, X0, T, args=(sys, ufunc), **kwargs)
        yout = dot(sys.C, transpose(xout)) + dot(sys.D, transpose(U))
    else:
        def fprime(x, t, sys):
            """The vector field of the linear system."""
            return dot(sys.A, x)
        xout = integrate.odeint(fprime, X0, T, args=(sys,), **kwargs)
        yout = dot(sys.C, transpose(xout))

    return T, squeeze(transpose(yout)), xout


def _default_response_times(A, n):
    """Compute a reasonable set of time samples for the response time.
    This function is used by `impulse`, `impulse2`, `step` and `step2`
    to compute the response time when the `T` argument to the function
    is None.
    Parameters
    ----------
    A : ndarray
        The system matrix, which is square.
    n : int
        The number of time samples to generate.
    Returns
    -------
    t : ndarray
        The 1-D array of length `n` of time samples at which the response
        is to be computed.
    """
    # Create a reasonable time interval.
    # TODO: This could use some more work.
    # For example, what is expected when the system is unstable?
    vals = linalg.eigvals(A)
    r = min(abs(real(vals)))
    if r == 0.0:
        r = 1.0
    tc = 1.0 / r
    t = linspace(0.0, 7 * tc, n)
    return t

def impulse(system, X0=None, T=None, N=None, **kwargs):
    """
    Impulse response of a single-input, continuous-time linear system.
    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)
    X0 : 1-D array_like, optional
        The initial condition of the state vector.  Default: 0 (the
        zero vector).
    T : 1-D array_like, optional
        The time steps at which the input is defined and at which the
        output is desired.  If `T` is not given, the function will
        generate a set of time samples automatically.
    N : int, optional
        Number of time points to compute.  Default: 100.
    kwargs : various types
        Additional keyword arguments are passed on to the function
        `scipy.signal.lsim2`, which in turn passes them on to
        `scipy.integrate.odeint`; see the latter's documentation for
        information about these arguments.
    Returns
    -------
    T : ndarray
        The time values for the output.
    yout : ndarray
        The output response of the system.
    See Also
    --------
    impulse, lsim2, integrate.odeint
    Notes
    -----
    The solution is generated by calling `scipy.signal.lsim2`, which uses
    the differential equation solver `scipy.integrate.odeint`.
    .. versionadded:: 0.8.0
    Examples
    --------
    Second order system with a repeated root: x''(t) + 2*x(t) + x(t) = u(t)
    >>> from scipy import signal
    >>> system = ([1.0], [1.0, 2.0, 1.0])
    >>> t, y = signal.impulse2(system)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, y)
    """
    if isinstance(system, lti):
        sys = system.to_ss()
    else:
        sys = lti(*system).to_ss()
    B = sys.B
    if B.shape[-1] != 1:
        raise ValueError("impulse2() requires a single-input system.")
    B = B.squeeze()
    if X0 is None:
        X0 = zeros_like(B)
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)

    # Move the impulse in the input to the initial conditions, and then
    # solve using lsim2().
    ic = B + X0
    Tr, Yr, Xr = lsim(sys, T=T, X0=ic, **kwargs)
    return Tr, Yr




def step(system, X0=None, T=None, N=None, **kwargs):
    """Step response of continuous-time system.
    This function is functionally the same as `scipy.signal.step`, but
    it uses the function `scipy.signal.lsim2` to compute the step
    response.
    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:
            * 2 (num, den)
            * 3 (zeros, poles, gain)
            * 4 (A, B, C, D)
    X0 : array_like, optional
        Initial state-vector (default is zero).
    T : array_like, optional
        Time points (computed if not given).
    N : int, optional
        Number of time points to compute if `T` is not given.
    kwargs : various types
        Additional keyword arguments are passed on the function
        `scipy.signal.lsim2`, which in turn passes them on to
        `scipy.integrate.odeint`.  See the documentation for
        `scipy.integrate.odeint` for information about these arguments.
    Returns
    -------
    T : 1D ndarray
        Output time points.
    yout : 1D ndarray
        Step response of system.
    See also
    --------
    scipy.signal.step
    Notes
    -----
    .. versionadded:: 0.8.0
    """
    if isinstance(system, lti):
        sys = system.to_ss()
    else:
        sys = lti(*system).to_ss()
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = asarray(T)
    U = ones(T.shape, sys.A.dtype)
    vals = lsim(sys, U, T, X0=X0, **kwargs)
    return vals[0], vals[1]




