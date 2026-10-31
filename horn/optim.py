"""Gradient-descent optimisers operating on dicts of parameter arrays.

All optimisers update the arrays *in place*, so a model's `params` dict stays
valid after `step`. `params` should be the dict returned by
`model.trainable_params()`; gradient keys that are not in it are ignored.
"""
from __future__ import annotations

import numpy as np


class SGD:
    """Plain (mini-batch) stochastic gradient descent: p <- p - lr * g."""

    def __init__(self, params, lr=1e-2):
        self.params, self.lr = params, lr

    def step(self, grads):
        for k, p in self.params.items():
            p -= self.lr * grads[k]


class Adam:
    """Adam with decoupled weight decay (= AdamW, the optimiser used in the paper).

    Adam keeps a running mean m (momentum) and a running mean of squared
    gradients s for every parameter and takes steps of roughly size lr in a
    normalised direction m / sqrt(s). This makes the step size insensitive to
    the (very different) gradient scales of W_ih, W_hh and W_out.
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params, self.lr, self.eps, self.wd = params, lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = {k: np.zeros_like(p) for k, p in params.items()}
        self.s = {k: np.zeros_like(p) for k, p in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2, t = self.b1, self.b2, self.t
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.s[k] = b2 * self.s[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** t)          # bias corrections
            s_hat = self.s[k] / (1 - b2 ** t)
            if self.wd:
                p -= self.lr * self.wd * p              # decoupled weight decay
            p -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)
