"""Small matplotlib helpers used by the tutorial notebook."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .train import smooth


def plot_traces(X, ax=None, t=None, title=None, xlabel="time step", ylabel="amplitude x",
                alpha=0.8, lw=1.0, cmap="viridis", legend=False, **kw):
    """Plot the trajectories of all nodes (X: (T, n)) with a colour per node."""
    X = np.asarray(X)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    t = np.arange(X.shape[0]) if t is None else t
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, X.shape[1]))
    for i in range(X.shape[1]):
        ax.plot(t, X[:, i], color=colors[i], alpha=alpha, lw=lw, label=f"node {i}" if legend else None, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=7, ncol=2)
    return ax


def plot_matrix(M, ax=None, title=None, cmap="RdBu_r", symmetric=True, colorbar=True, **kw):
    """imshow with a symmetric colour scale (useful for weights and correlations)."""
    M = np.asarray(M)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.5))
    if symmetric:
        v = np.abs(M).max()
        kw.setdefault("vmin", -v)
        kw.setdefault("vmax", v)
    im = ax.imshow(M, cmap=cmap, **kw)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    if title:
        ax.set_title(title)
    return im


def plot_history(hist, ax=None, label=None, color=None, k=20, val=True, steps_per_epoch=None):
    """Smoothed training loss and (if available) validation accuracy."""
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax_loss, ax_acc = ax
    step, loss = np.asarray(hist["step"]), np.asarray(hist["loss"])
    sm = smooth(loss, k)
    ax_loss.plot(step[len(step) - len(sm):], sm, label=label, color=color)
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("training loss")
    if val and len(hist.get("val_step", [])):
        ax_acc.plot(hist["val_step"], hist["val_acc"], "o-", ms=3, label=label, color=color)
        ax_acc.set_xlabel("training step")
        ax_acc.set_ylabel("validation accuracy")
        ax_acc.set_ylim(0, 1)
    elif len(hist.get("acc", [])):
        sa = smooth(hist["acc"], k)
        ax_acc.plot(step[len(step) - len(sa):], sa, label=label, color=color)
        ax_acc.set_xlabel("training step")
        ax_acc.set_ylabel("training accuracy (smoothed)")
        ax_acc.set_ylim(0, 1.02)
    if steps_per_epoch:
        for a in ax:
            for e in range(1, int(step[-1] // steps_per_epoch) + 1):
                a.axvline(e * steps_per_epoch, color="gray", ls=":", lw=0.8)
    if label:
        ax_loss.legend()
    return ax


def confusion_matrix(y_true, y_pred, n_classes=10):
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M
