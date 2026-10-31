"""Gradient checking: compare the analytic BPTT gradients with finite differences.

The brute-force way to get dL/dp for a parameter p is to nudge it up and down
by a tiny eps and measure how the loss changes:

    dL/dp  ~  ( L(p + eps) - L(p - eps) ) / (2 eps)

This needs two full forward passes *per parameter* -- hopeless for training,
but perfect for verifying that the hand-written backward pass is correct.
"""
from __future__ import annotations

import numpy as np

from .losses import softmax_cross_entropy


def loss_fn(model, S, labels):
    logits, _ = model.forward(S)
    return softmax_cross_entropy(logits, labels)[0]


def numerical_gradient(model, S, labels, name, index, eps=1e-6):
    """Central finite-difference estimate of dL/dp for one entry of one parameter."""
    p = model.params[name]
    old = p[index]
    p[index] = old + eps
    lp = loss_fn(model, S, labels)
    p[index] = old - eps
    lm = loss_fn(model, S, labels)
    p[index] = old
    return (lp - lm) / (2 * eps)


def check_gradients(model, S, labels, n_checks=5, eps=1e-6, seed=0, verbose=True):
    """Compare analytic and numerical gradients for `n_checks` random entries of
    every trainable parameter. Returns a dict name -> max relative error."""
    rng = np.random.default_rng(seed)
    logits, cache = model.forward(S)
    _, dlogits, _ = softmax_cross_entropy(logits, labels)
    grads = model.backward(cache, dlogits)
    results = {}
    if verbose:
        print(f"{'parameter':>8s} {'analytic':>13s} {'numerical':>13s} {'rel. error':>11s}")
    for name in model.trainable:
        p = model.params[name]
        worst = 0.0
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            g_num = numerical_gradient(model, S, labels, name, idx, eps)
            g_ana = grads[name][idx]
            rel = abs(g_ana - g_num) / max(abs(g_ana) + abs(g_num), 1e-12)
            worst = max(worst, rel)
            if verbose:
                print(f"{name:>8s} {g_ana:13.6e} {g_num:13.6e} {rel:11.2e}")
        results[name] = worst
    return results
