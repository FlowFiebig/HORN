"""Loss functions, each returning the loss *and* its gradient w.r.t. the logits.

For classification we use softmax + cross-entropy. Its gradient is famously
simple: dL/dlogits = (p - onehot(label)) / batch_size, i.e. "predicted
probability minus target". This is the error signal that backpropagation
through time then carries backwards through the network.
"""
from __future__ import annotations

import numpy as np


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)   # subtract max for numerical stability
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, labels):
    """Mean cross-entropy over the batch.

    Returns (loss, dlogits, probs) where dlogits already includes the 1/batch
    factor of the mean, so it can be passed straight to `model.backward`.
    """
    labels = np.asarray(labels, dtype=int)
    B = logits.shape[0]
    p = softmax(logits)
    loss = -np.log(p[np.arange(B), labels] + 1e-12).mean()
    dlogits = p.copy()
    dlogits[np.arange(B), labels] -= 1.0
    dlogits /= B
    return loss, dlogits, p


def mse_loss(pred, target):
    """Mean squared error (for regression tasks). Returns (loss, dpred)."""
    diff = pred - target
    loss = 0.5 * np.mean(diff ** 2)
    return loss, diff / diff.size


def accuracy(logits, labels):
    return float(np.mean(logits.argmax(1) == np.asarray(labels)))
