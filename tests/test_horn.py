"""Sanity checks for the toy HORN implementation.

Run from the repository root:

    python tests/test_horn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from horn import (HORN, ElmanRNN, Adam, accuracy, check_gradients, eigen_analysis,  # noqa: E402
                  make_temporal_xor, make_xor, simulate_dho, softmax_cross_entropy, train)
from horn.dho import impulse  # noqa: E402


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_single_node_matches_network():
    omega, gamma, alpha, v, w = 2 * np.pi / 28, 0.01, 0.04, -0.2, 0.3
    net = HORN(1, 1, 2, omega=omega, gamma=gamma, alpha=alpha)
    net.params["W_ih"][:] = 1.0
    net.params["b_ih"][:] = 0.0
    net.params["W_hh"][:] = w
    net.params["b_hh"][:] = 0.0
    net.params["v"][:] = v
    s = impulse(80, 5, 3, 2.0)
    x_net = net.simulate(s[None, :, None])["X"][0, :, 0]
    x_dho, _ = simulate_dho(s, omega, gamma, alpha, v=v, w=w)
    _assert(np.allclose(x_net, x_dho), "HORN(n=1) must match simulate_dho")


def test_regime_boundary():
    omega = 0.2
    _assert(eigen_analysis(omega, 0.01)["regime"] == "underdamped", "low gamma should oscillate")
    _assert(eigen_analysis(omega, 0.5)["regime"] == "overdamped", "high gamma should not oscillate")


def test_bptt_matches_finite_differences():
    rng = np.random.default_rng(0)
    model = HORN.heterogeneous(n_in=2, n_hidden=5, n_out=3, seed=1)
    S, y = rng.normal(size=(3, 12, 2)), rng.integers(0, 3, size=3)
    errors = check_gradients(model, S, y, n_checks=3, eps=1e-6, verbose=False)
    _assert(max(errors.values()) < 1e-6, f"HORN BPTT mismatch: {errors}")
    elman = ElmanRNN(2, 5, 3, seed=1)
    errors = check_gradients(elman, S, y, n_checks=3, verbose=False)
    _assert(max(errors.values()) < 1e-6, f"Elman BPTT mismatch: {errors}")


def test_xor_learns():
    S, y = make_xor(T=30)
    model = HORN(2, 4, 2, seed=0)
    train(model, S, y, epochs=250, batch_size=4, lr=1e-2, verbose=False)
    logits, _ = model.forward(S)
    _assert(accuracy(logits, y) == 1.0, "4-node HORN should solve static XOR")


def test_temporal_xor_learns():
    S, y = make_temporal_xor(T=40, t1=5, t2=19)
    model = HORN(1, 4, 2, seed=0)
    train(model, S, y, epochs=400, batch_size=4, lr=1e-2, verbose=False)
    logits, _ = model.forward(S)
    _assert(accuracy(logits, y) == 1.0, "4-node HORN should solve half-period temporal XOR")


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    model = HORN.heterogeneous(1, 6, 3, seed=4)
    path = Path(tmp_path or tempfile.mkdtemp()) / "m.npz"
    model.save(path)
    loaded = HORN.load(path)
    for k in model.params:
        _assert(np.allclose(model.params[k], loaded.params[k]), f"{k} not restored")
    _assert(np.allclose(model.omega, loaded.omega), "omega not restored")
    _assert(loaded.trainable == model.trainable, "trainable list not restored")


def test_adam_decreases_loss():
    rng = np.random.default_rng(0)
    model = HORN(1, 4, 3, seed=0)
    S, y = rng.normal(size=(8, 15, 1)), rng.integers(0, 3, size=8)
    opt = Adam(model.trainable_params(), lr=1e-2)
    logits, cache = model.forward(S)
    loss0, dlogits, _ = softmax_cross_entropy(logits, y)
    opt.step(model.backward(cache, dlogits))
    logits, _ = model.forward(S)
    loss1, _, _ = softmax_cross_entropy(logits, y)
    _assert(loss1 < loss0, f"Adam should decrease the loss ({loss0:.4f} -> {loss1:.4f})")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
