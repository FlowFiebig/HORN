"""A single damped harmonic oscillator (DHO) node -- the building block of a HORN.

The continuous-time model of one node is the driven, damped harmonic oscillator

    x'' + 2*gamma*x' + omega^2 * x = alpha * tanh(I(t)),

with amplitude x, velocity y = x', natural frequency omega, damping gamma and
excitability alpha. HORNs integrate it with the *symplectic Euler* scheme
(step size h, usually h = 1):

    y[t+1] = y[t] + h * ( alpha * tanh(u[t]) - 2*gamma*y[t] - omega^2 * x[t] )
    x[t+1] = x[t] + h * y[t+1]

where u[t] is the total drive entering the tanh nonlinearity. For a single node
with feedback, u[t] = I_ext[t] + v*x[t] + w*y[t] (amplitude feedback v, velocity
feedback w). In a network, u additionally contains the recurrent input from the
other nodes (see horn/network.py).

Everything in this file is deliberately written for a *single* node so that the
dynamics can be studied in isolation before looking at networks.
"""
from __future__ import annotations

import numpy as np


def dho_step(x, y, drive, omega, gamma, alpha=1.0, h=1.0, nonlinearity=np.tanh):
    """One symplectic-Euler step of the driven DHO.

    `drive` is the total pre-activation entering the nonlinearity. Works on
    scalars and on arrays (broadcasting), so the same function can be used
    for many independent nodes at once.
    """
    y_new = y + h * (alpha * nonlinearity(drive) - 2.0 * gamma * y - omega ** 2 * x)
    x_new = x + h * y_new
    return x_new, y_new


def simulate_dho(inp, omega, gamma, alpha=1.0, v=0.0, w=0.0, x0=0.0, y0=0.0,
                 h=1.0, nonlinearity=np.tanh):
    """Simulate one DHO node driven by the input time series `inp` (shape (T,)).

    Feedback terms v*x (amplitude feedback) and w*y (velocity feedback) are added
    to the drive *inside* the nonlinearity, exactly as in a HORN with one node
    (where the recurrent gain 1/sqrt(n) equals 1).

    Returns
    -------
    x, y : arrays of shape (T+1,) -- amplitude and velocity, index 0 is the
           initial condition, index t is the state after the t-th input sample.
    """
    inp = np.asarray(inp, dtype=float)
    T = inp.shape[0]
    x = np.empty(T + 1)
    y = np.empty(T + 1)
    x[0], y[0] = x0, y0
    for t in range(T):
        drive = inp[t] + v * x[t] + w * y[t]
        x[t + 1], y[t + 1] = dho_step(x[t], y[t], drive, omega, gamma, alpha, h, nonlinearity)
    return x, y


def impulse(T, t_on=5, width=1, amplitude=1.0):
    """A rectangular pulse of given width -- the simplest possible stimulus."""
    s = np.zeros(T)
    s[t_on:t_on + width] = amplitude
    return s


def step_input(T, t_on=5, amplitude=1.0):
    """A step that switches on at t_on and stays on."""
    s = np.zeros(T)
    s[t_on:] = amplitude
    return s


def sinusoid(T, freq, amplitude=1.0, phase=0.0):
    """A sinusoid with angular frequency `freq` (radians per time step)."""
    return amplitude * np.sin(freq * np.arange(T) + phase)


# ---------------------------------------------------------------------------
# Linear theory of the discretised node
# ---------------------------------------------------------------------------

def linear_map(omega, gamma, h=1.0):
    """The 2x2 matrix M of the *linear* part of one update step, state = (x, y).

        [x'] = M [x],   M = [[1 - h^2 w^2,  h (1 - 2 g h)],
        [y']     [y]         [   -h w^2  ,     1 - 2 g h ]]

    Its eigenvalues tell us everything about the free (undriven) dynamics:
    |lambda| is the decay factor per step, angle(lambda) the phase advance per
    step (i.e. the oscillation frequency in radians/step).
    """
    return np.array([[1.0 - h * h * omega ** 2, h * (1.0 - 2.0 * gamma * h)],
                     [-h * omega ** 2, 1.0 - 2.0 * gamma * h]])


def eigen_analysis(omega, gamma, h=1.0):
    """Decay factor, oscillation frequency and regime of the linear node.

    Returns a dict with
      eigenvalues : the two eigenvalues of the one-step map
      decay       : |lambda| -- amplitude multiplier per step (< 1 means stable)
      freq        : angular frequency per step (0 if non-oscillatory)
      period      : 2*pi/freq (inf if non-oscillatory)
      regime      : 'underdamped' (oscillating), 'overdamped' (no oscillation)
                    or 'unstable'
    """
    lam = np.linalg.eigvals(linear_map(omega, gamma, h))
    decay = np.abs(lam).max()
    if np.iscomplex(lam).any():
        freq = abs(np.angle(lam[0]))
        regime = 'underdamped'
    else:
        freq = 0.0
        regime = 'overdamped'
    if decay > 1.0 + 1e-12:
        regime = 'unstable'
    period = 2 * np.pi / freq if freq > 0 else np.inf
    return dict(eigenvalues=lam, decay=decay, freq=freq, period=period, regime=regime)


def effective_parameters(omega, gamma, alpha, v=0.0, w=0.0):
    """Feedback shifts the *effective* oscillator parameters (linear regime).

    With tanh(u) ~ u the feedback terms alpha*(v*x + w*y) simply add to the
    restoring and damping forces:

        omega_eff^2 = omega^2 - alpha * v      (amplitude feedback tunes the frequency)
        gamma_eff   = gamma   - alpha * w / 2  (velocity feedback tunes the damping)

    Positive w therefore sharpens the resonance and, for w > 2*gamma/alpha, makes
    the node a self-sustained oscillator (negative effective damping).
    """
    omega_eff_sq = omega ** 2 - alpha * v
    gamma_eff = gamma - alpha * w / 2.0
    omega_eff = np.sqrt(omega_eff_sq) if omega_eff_sq > 0 else np.nan
    return omega_eff, gamma_eff


def gain_linear(freqs, omega, gamma, alpha=1.0, v=0.0, w=0.0, h=1.0):
    """Exact steady-state gain |x| / |input| of the *linear* discretised node.

    Derived from the z-transform of the update equations (z = exp(i*theta),
    theta = input frequency in radians/step):

        H(z) = h^2 alpha z^2 / ( (z-1)(z-1+2 gamma h) + h^2 omega^2 z )

    Feedback v, w enters through the effective parameters (see
    `effective_parameters`). In the paper this gain curve is called the
    *intrinsic receptive field* (IRF) of the node: the frequency band in which
    the node amplifies (gain > 1) or attenuates its input.
    """
    omega_eff_sq = omega ** 2 - alpha * v
    gamma_eff = gamma - alpha * w / 2.0
    z = np.exp(1j * np.asarray(freqs, dtype=float) * h)
    H = (h * h * alpha * z ** 2) / ((z - 1.0) * (z - 1.0 + 2.0 * gamma_eff * h) + h * h * omega_eff_sq * z)
    return np.abs(H)


def gain_continuous(freqs, omega, gamma, alpha=1.0):
    """Textbook gain of the continuous-time driven DHO (for comparison)."""
    wi = np.asarray(freqs, dtype=float)
    return alpha / np.sqrt((omega ** 2 - wi ** 2) ** 2 + (2.0 * gamma * wi) ** 2)


def gain_simulated(freqs, omega, gamma, alpha=1.0, v=0.0, w=0.0, amplitude=1.0,
                   T=4000, settle=0.5, h=1.0, nonlinearity=np.tanh):
    """Measure the gain curve by brute force: drive the (nonlinear) node with a
    sinusoid at each frequency, wait for transients to settle, and read off the
    steady-state amplitude of x divided by the input amplitude.

    Because of the tanh, the measured gain depends on the input amplitude
    (compression for large inputs) -- compare with `gain_linear`.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    gains = np.empty_like(freqs)
    t0 = int(settle * T)
    for i, f in enumerate(freqs):
        x, _ = simulate_dho(sinusoid(T, f, amplitude), omega, gamma, alpha, v, w,
                            h=h, nonlinearity=nonlinearity)
        gains[i] = 0.5 * (x[t0:].max() - x[t0:].min()) / amplitude
    return gains
