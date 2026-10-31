"""Analysis tools used in the paper: synchrony measures, receptive fields,
correlations and gradient flow.

All functions take trajectories X of shape (T, n) (one sample) or (B, T, n)
(a batch of samples), as returned by `model.simulate(S)['X']`.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .dho import gain_linear
from .losses import softmax_cross_entropy


def instantaneous_phase(X):
    """Instantaneous phase of every node via the analytic signal (Hilbert transform),
    computed along the time axis (axis -2 for (…, T, n) arrays)."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=-2, keepdims=True)
    return np.angle(hilbert(X, axis=-2))


def kuramoto_order(X):
    """Time-resolved Kuramoto order parameter r(t) = |mean_k exp(i phi_k(t))|.

    r = 1: all nodes have the same phase (perfect synchrony), r ~ 0: phases are
    spread out. Shape (…, T)."""
    phi = instantaneous_phase(X)
    return np.abs(np.exp(1j * phi).mean(axis=-1))


def plv(x1, x2):
    """Phase-locking value of two time series (1 = constant phase difference)."""
    phi = instantaneous_phase(np.stack([x1, x2], axis=-1))
    return np.abs(np.exp(1j * (phi[..., 0] - phi[..., 1])).mean(axis=-1))


def pairwise_plv(X):
    """Matrix of pairwise phase-locking values, averaged over samples if batched."""
    phi = instantaneous_phase(X)
    if phi.ndim == 2:
        phi = phi[None]
    out = np.zeros((phi.shape[-1], phi.shape[-1]))
    for p in phi:
        z = np.exp(1j * p)                  # (T, n)
        out += np.abs(z.T.conj() @ z) / z.shape[0]
    return out / phi.shape[0]


def pairwise_crosscorr(X):
    """Pearson cross-correlation coefficients between node activities,
    averaged over samples if batched."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[None]
    out = np.zeros((X.shape[-1], X.shape[-1]))
    for x in X:
        xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
        out += xs.T @ xs / x.shape[0]
    return out / X.shape[0]


def upper_triangle(M):
    """The entries above the diagonal of a square matrix (as a flat array)."""
    i, j = np.triu_indices(M.shape[0], k=1)
    return M[i, j]


def intrinsic_receptive_fields(model, freqs):
    """Gain curves (IRFs) of all nodes of a HORN, computed analytically from
    omega, gamma, alpha and the learned feedback terms v (amplitude) and
    w = diag(W_hh) (velocity, scaled by 1/sqrt(n)). Returns (n, len(freqs))."""
    c = model.gain_rec
    w = np.diag(model.params["W_hh"]) * c
    v = model.params["v"] * c
    return np.stack([gain_linear(freqs, model.omega[i], model.gamma[i], model.alpha[i], v[i], w[i], model.h)
                     for i in range(model.n_hidden)])


def network_frequency_response(model, freqs, amplitude=0.1, T=3000, settle=0.5):
    """Effective frequency response of all nodes *inside the network*: drive the
    whole network with a sinusoid and measure each node's steady-state amplitude.
    Unlike the IRF this includes the recurrent interactions. Returns (n, len(freqs))."""
    t = np.arange(T)
    out = np.zeros((model.n_hidden, len(freqs)))
    S = np.stack([amplitude * np.sin(f * t) for f in freqs])[:, :, None]
    S = np.repeat(S, model.n_in, axis=2)
    X = model.simulate(S)["X"][:, 1:]
    t0 = int(settle * T)
    out = 0.5 * (X[:, t0:].max(1) - X[:, t0:].min(1)) / amplitude    # (n_freqs, n)
    return out.T


def gradient_flow(model, S, labels):
    """Norm of dL/dx_t (gradient of the loss with respect to the state) at every
    time step, averaged over the batch. Shows how far back in time the error
    signal from the readout can 'see'. Returns array of shape (T+1,)."""
    logits, cache = model.forward(S)
    _, dlogits, _ = softmax_cross_entropy(logits, labels)
    _, GX, _ = model.backward(cache, dlogits, return_state_grads=True)
    return np.linalg.norm(GX, axis=-1).mean(axis=1)


def power_spectrum(s):
    """One-sided power spectrum of a 1-D signal; returns (freqs in rad/step, power)."""
    s = np.asarray(s, dtype=float)
    s = s - s.mean()
    P = np.abs(np.fft.rfft(s)) ** 2
    f = 2 * np.pi * np.fft.rfftfreq(len(s))
    return f, P
