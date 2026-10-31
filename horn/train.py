"""A small training loop: mini-batches, forward, loss, backward, optimiser step.

The loop is intentionally short -- the notebook first writes it out by hand and
only afterwards uses this helper.
"""
from __future__ import annotations

import time

import numpy as np

from .losses import softmax_cross_entropy, accuracy
from .optim import Adam


def evaluate(model, S, labels, batch_size=500):
    """Mean loss and accuracy over a data set (no gradients)."""
    losses, correct = [], 0
    for i in range(0, len(S), batch_size):
        logits, _ = model.forward(S[i:i + batch_size])
        loss, _, _ = softmax_cross_entropy(logits, labels[i:i + batch_size])
        losses.append(loss * len(logits))
        correct += int((logits.argmax(1) == labels[i:i + batch_size]).sum())
    return float(np.sum(losses) / len(S)), correct / len(S)


def train(model, S, labels, epochs=1, batch_size=64, lr=1e-2, optimizer=None,
          S_val=None, y_val=None, eval_every=None, max_steps=None, shuffle=True,
          seed=0, verbose=True, history=None, callback=None, weight_decay=0.0):
    """Train `model` with BPTT + Adam on sequences S (N, T, n_in) and integer labels.

    Returns a history dict with per-step training loss/accuracy and (optionally)
    validation loss/accuracy every `eval_every` steps. Pass an existing history
    to continue a training run.
    """
    S = np.asarray(S)
    labels = np.asarray(labels, dtype=int)
    N = len(S)
    rng = np.random.default_rng(seed)
    if optimizer is None:
        optimizer = Adam(model.trainable_params(), lr=lr, weight_decay=weight_decay)
    if history is None:
        history = dict(step=[], loss=[], acc=[], val_step=[], val_loss=[], val_acc=[], time=[])
    step = history["step"][-1] + 1 if history["step"] else 0
    t_start = time.time()

    try:
        from tqdm.auto import tqdm
    except ImportError:            # tqdm is optional
        tqdm = None

    steps_per_epoch = int(np.ceil(N / batch_size))
    total = epochs * steps_per_epoch if max_steps is None else min(max_steps, epochs * steps_per_epoch)
    pbar = tqdm(total=total, disable=not verbose, leave=False) if tqdm else None
    done = False
    for epoch in range(epochs):
        order = rng.permutation(N) if shuffle else np.arange(N)
        for i in range(0, N, batch_size):
            idx = order[i:i + batch_size]
            logits, cache = model.forward(S[idx])
            loss, dlogits, _ = softmax_cross_entropy(logits, labels[idx])
            grads = model.backward(cache, dlogits)
            optimizer.step(grads)

            history["step"].append(step)
            history["loss"].append(loss)
            history["acc"].append(accuracy(logits, labels[idx]))
            history["time"].append(time.time() - t_start)
            if eval_every and S_val is not None and (step % eval_every == 0 or step == total - 1):
                vl, va = evaluate(model, S_val, y_val)
                history["val_step"].append(step)
                history["val_loss"].append(vl)
                history["val_acc"].append(va)
                if pbar:
                    pbar.set_postfix(loss=f"{loss:.3f}", val_acc=f"{va:.3f}")
            elif pbar and step % 10 == 0:
                pbar.set_postfix(loss=f"{loss:.3f}", acc=f"{history['acc'][-1]:.2f}")
            if callback is not None:
                callback(model, step, history)
            if pbar:
                pbar.update(1)
            step += 1
            if max_steps is not None and step >= max_steps:
                done = True
                break
        if done:
            break
    if pbar:
        pbar.close()
    if verbose:
        msg = f"finished {step} steps in {time.time() - t_start:.1f}s, last train loss {history['loss'][-1]:.3f}"
        if history["val_acc"]:
            msg += f", val acc {history['val_acc'][-1]:.3f}"
        print(msg)
    return history


def smooth(x, k=20):
    """Running mean for plotting noisy training curves."""
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")
