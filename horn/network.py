"""HORN: a recurrent network of damped harmonic oscillators, in plain NumPy.

Forward pass (Effenberger et al., PNAS 2025, Materials & Methods), for a
network of n nodes with per-node parameters omega, gamma, alpha (vectors):

    u[t]   = (1/sqrt(n)) * ( W_hh y[t] + b_hh + v * x[t] ) + W_ih s[t] + b_ih
    y[t+1] = y[t] + h * ( alpha * tanh(u[t]) - 2*gamma*y[t] - omega^2 * x[t] )
    x[t+1] = x[t] + h * y[t+1]
    logits = W_out x[T] + b_out                       (affine readout, last step)

Nodes are coupled through their *velocities* y (W_hh acts on y). The diagonal
of W_hh is the velocity feedback w of each node, the vector v is the amplitude
feedback. Compared with the authors' PyTorch code, this implementation adds the
optional amplitude feedback v (zero-initialised) and -- most importantly -- a
hand-written backward pass (backpropagation through time, BPTT) so that every
gradient can be inspected. `horn.gradcheck` verifies it against finite
differences.

Conventions
-----------
* Inputs S have shape (batch, T, n_in). Labels are integer class indices.
* States are stored time-major: X, Y have shape (T+1, batch, n), X[0] = x0.
* `forward` returns (logits, cache); `backward(cache, dlogits)` returns a dict
  of gradients with the same keys as `params`.
"""
from __future__ import annotations

import numpy as np


class RecurrentClassifier:
    """Minimal shared machinery for the two models below (params, I/O)."""

    param_names: tuple = ()

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.trainable: list[str] = []

    @property
    def n_params(self) -> int:
        return int(sum(self.params[k].size for k in self.trainable))

    def trainable_params(self) -> dict[str, np.ndarray]:
        return {k: self.params[k] for k in self.trainable}

    def zero_grads(self) -> dict[str, np.ndarray]:
        return {k: np.zeros_like(p) for k, p in self.params.items()}

    def predict(self, S, batch_size=500):
        """Class predictions for a (possibly large) set of sequences."""
        out = []
        for i in range(0, len(S), batch_size):
            logits, _ = self.forward(S[i:i + batch_size])
            out.append(logits.argmax(1))
        return np.concatenate(out)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        meta = {k: v for k, v in self._config().items()}
        np.savez(path, **{f"param_{k}": v for k, v in self.params.items()},
                 **{f"meta_{k}": np.asarray(v) for k, v in meta.items()},
                 trainable=np.array(self.trainable))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            meta = {k[5:]: d[k] for k in d.files if k.startswith("meta_")}
            meta = {k: (v.item() if v.ndim == 0 else v.copy()) for k, v in meta.items()}
            model = cls(**meta)
            for k in d.files:
                if k.startswith("param_"):
                    model.params[k[6:]] = np.array(d[k])
            model.trainable = [str(k) for k in d["trainable"]]
        return model


class HORN(RecurrentClassifier):
    """Harmonic Oscillator Recurrent Network with manual BPTT.

    Parameters
    ----------
    n_in, n_hidden, n_out : layer sizes
    omega, gamma, alpha   : scalars (homogeneous network) or length-n arrays
                            (heterogeneous network)
    h                     : integration step (1.0 in the paper)
    use_v                 : train the amplitude-feedback vector v (default True;
                            v starts at 0, so an untrained network equals the
                            reference model)
    seed                  : seed for the PyTorch-style uniform initialisation
                            U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    """

    def __init__(self, n_in, n_hidden, n_out, omega=2 * np.pi / 28, gamma=0.01,
                 alpha=0.04, h=1.0, use_v=True, seed=0):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out = int(n_in), int(n_hidden), int(n_out)
        n = self.n_hidden
        self.omega = np.broadcast_to(np.asarray(omega, dtype=float), (n,)).copy()
        self.gamma = np.broadcast_to(np.asarray(gamma, dtype=float), (n,)).copy()
        self.alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (n,)).copy()
        self.h = float(h)
        self.use_v = bool(use_v)
        self.seed = int(seed)
        self.gain_rec = 1.0 / np.sqrt(n)        # the 1/sqrt(n) in front of I_rec
        self.mask_hh = None                     # optional binary mask on W_hh

        rng = np.random.default_rng(seed)

        def uniform(shape, fan_in):
            b = 1.0 / np.sqrt(fan_in)
            return rng.uniform(-b, b, size=shape)

        self.params = {
            "W_ih": uniform((n, self.n_in), self.n_in),
            "b_ih": uniform((n,), self.n_in),
            "W_hh": uniform((n, n), n),
            "b_hh": uniform((n,), n),
            "v": np.zeros(n),
            "W_out": uniform((self.n_out, n), n),
            "b_out": uniform((self.n_out,), n),
        }
        self.trainable = ["W_ih", "b_ih", "W_hh", "b_hh", "W_out", "b_out"]
        if self.use_v:
            self.trainable.insert(4, "v")

    @classmethod
    def heterogeneous(cls, n_in, n_hidden, n_out, omega=2 * np.pi / 28, gamma=0.01,
                      alpha=0.04, omega_range=(0.5, 2.0), gamma_range=(0.5, 2.0),
                      seed=0, **kw):
        """Heterogeneous HORN: per-node omega and gamma drawn uniformly from
        [lo*base, hi*base], as in the authors' dynamics.py example."""
        rng = np.random.default_rng(seed + 12345)
        om = rng.uniform(omega_range[0] * omega, omega_range[1] * omega, size=n_hidden)
        ga = rng.uniform(gamma_range[0] * gamma, gamma_range[1] * gamma, size=n_hidden)
        return cls(n_in, n_hidden, n_out, omega=om, gamma=ga, alpha=alpha, seed=seed, **kw)

    def _config(self):
        return dict(n_in=self.n_in, n_hidden=self.n_hidden, n_out=self.n_out,
                    omega=self.omega, gamma=self.gamma, alpha=self.alpha, h=self.h,
                    use_v=self.use_v, seed=self.seed)

    # ------------------------------------------------------------------ forward
    def forward(self, S, x0=None, y0=None):
        """Run the network on a batch of sequences.

        S : array (batch, T, n_in)
        Returns logits (batch, n_out) and a cache with everything the backward
        pass needs: the states X, Y (T+1, batch, n) and tanh outputs A (T, batch, n).
        """
        S = np.asarray(S, dtype=float)
        if S.ndim == 2:                      # (batch, T) -> (batch, T, 1)
            S = S[:, :, None]
        B, T, _ = S.shape
        n, P, h = self.n_hidden, self.params, self.h
        W_hh = P["W_hh"] if self.mask_hh is None else P["W_hh"] * self.mask_hh
        W_hh_T = W_hh.T
        c = self.gain_rec
        alpha, two_gamma, omega_sq = self.alpha, 2.0 * self.gamma, self.omega ** 2

        X = np.zeros((T + 1, B, n))
        Y = np.zeros((T + 1, B, n))
        A = np.zeros((T, B, n))
        if x0 is not None:
            X[0] = x0
        if y0 is not None:
            Y[0] = y0

        # external drive for all time steps at once: (T, B, n)
        I_ext = np.einsum("btk,nk->tbn", S, P["W_ih"]) + P["b_ih"]

        x, y = X[0], Y[0]
        for t in range(T):
            u = c * (y @ W_hh_T + P["b_hh"] + P["v"] * x) + I_ext[t]
            a = np.tanh(u)
            y = y + h * (alpha * a - two_gamma * y - omega_sq * x)
            x = x + h * y
            A[t], X[t + 1], Y[t + 1] = a, x, y

        logits = x @ P["W_out"].T + P["b_out"]
        cache = dict(S=S, X=X, Y=Y, A=A, W_hh=W_hh)
        return logits, cache

    def simulate(self, S, x0=None, y0=None):
        """Convenience wrapper returning batch-major trajectories.

        Returns dict(logits, X, Y) with X, Y of shape (batch, T+1, n).
        """
        logits, cache = self.forward(S, x0, y0)
        return dict(logits=logits, X=cache["X"].transpose(1, 0, 2),
                    Y=cache["Y"].transpose(1, 0, 2))

    # ----------------------------------------------------------------- backward
    def backward(self, cache, dlogits, return_state_grads=False):
        """Backpropagation through time.

        dlogits : dL/dlogits, shape (batch, n_out)  (from the loss function)
        Returns a dict of parameter gradients. With return_state_grads=True it
        also returns GX, GY of shape (T+1, batch, n): the gradient of the loss
        with respect to the state (x_t, y_t) at every time step -- useful for
        visualising how the error signal travels backwards in time.

        Reverse-mode rules, read the forward equations bottom-up:
          logits = W_out x_T + b_out            -> gx_T = W_out^T dlogits
          x_{t+1} = x_t + h y_{t+1}             -> gy_{t+1} += h gx_{t+1}, gx_t += gx_{t+1}
          y_{t+1} = (1-2gh) y_t - h w^2 x_t + h a tanh(u_t)
                                                -> gy_t += (1-2gh) gy_{t+1}
                                                   gx_t += -h w^2 gy_{t+1}
                                                   gu_t  = h a gy_{t+1} (1 - tanh(u_t)^2)
          u_t = c (W_hh y_t + b_hh + v x_t) + W_ih s_t + b_ih
                                                -> gW_hh += c gu_t y_t^T, gb_hh += c gu_t
                                                   gv += c gu_t x_t, gW_ih += gu_t s_t^T, gb_ih += gu_t
                                                   gy_t += c W_hh^T gu_t, gx_t += c v gu_t
        """
        S, X, Y, A, W_hh = cache["S"], cache["X"], cache["Y"], cache["A"], cache["W_hh"]
        T, B, n = A.shape
        P, h, c = self.params, self.h, self.gain_rec
        alpha, two_gamma, omega_sq = self.alpha, 2.0 * self.gamma, self.omega ** 2

        g = self.zero_grads()
        g["W_out"] = dlogits.T @ X[T]
        g["b_out"] = dlogits.sum(0)

        gx = dlogits @ P["W_out"]           # dL/dx_T
        gy = np.zeros_like(gx)              # dL/dy_T (the readout does not see y)
        GU = np.zeros((T, B, n))            # dL/du_t for all t (used for weight grads)
        if return_state_grads:
            GX = np.zeros((T + 1, B, n))
            GY = np.zeros((T + 1, B, n))
            GX[T] = gx

        for t in range(T - 1, -1, -1):
            # x_{t+1} = x_t + h * y_{t+1}
            gy = gy + h * gx
            # y_{t+1} = (1 - 2 gamma h) y_t - h omega^2 x_t + h alpha a_t
            gu = (h * alpha * gy) * (1.0 - A[t] ** 2)
            GU[t] = gu
            gy_prev = (1.0 - two_gamma * h) * gy + c * (gu @ W_hh)
            gx_prev = gx - h * omega_sq * gy + c * gu * P["v"]
            gx, gy = gx_prev, gy_prev
            if return_state_grads:
                GX[t], GY[t] = gx, gy

        # parameter gradients, accumulated over time and batch in one go
        g["W_hh"] = c * np.einsum("tbn,tbm->nm", GU, Y[:-1])
        g["b_hh"] = c * GU.sum((0, 1))
        g["v"] = c * (GU * X[:-1]).sum((0, 1))
        g["W_ih"] = np.einsum("tbn,btk->nk", GU, S)
        g["b_ih"] = GU.sum((0, 1))
        if self.mask_hh is not None:
            g["W_hh"] *= self.mask_hh

        if return_state_grads:
            return g, GX, GY
        return g


class ElmanRNN(RecurrentClassifier):
    """The classic tanh RNN ("tanh" baseline in the paper), same interface.

        h_{t+1} = tanh( W_hh h_t + b_hh + W_ih s_t + b_ih ),   logits = W_out h_T + b_out
    """

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        super().__init__()
        self.n_in, self.n_hidden, self.n_out, self.seed = int(n_in), int(n_hidden), int(n_out), int(seed)
        n = self.n_hidden
        rng = np.random.default_rng(seed)

        def uniform(shape, fan_in):
            b = 1.0 / np.sqrt(fan_in)
            return rng.uniform(-b, b, size=shape)

        self.params = {
            "W_ih": uniform((n, self.n_in), self.n_in), "b_ih": uniform((n,), self.n_in),
            "W_hh": uniform((n, n), n), "b_hh": uniform((n,), n),
            "W_out": uniform((self.n_out, n), n), "b_out": uniform((self.n_out,), n),
        }
        self.trainable = list(self.params)

    def _config(self):
        return dict(n_in=self.n_in, n_hidden=self.n_hidden, n_out=self.n_out, seed=self.seed)

    def forward(self, S, h0=None):
        S = np.asarray(S, dtype=float)
        if S.ndim == 2:
            S = S[:, :, None]
        B, T, _ = S.shape
        n, P = self.n_hidden, self.params
        H = np.zeros((T + 1, B, n))
        if h0 is not None:
            H[0] = h0
        I_ext = np.einsum("btk,nk->tbn", S, P["W_ih"]) + P["b_ih"]
        W_hh_T = P["W_hh"].T
        hstate = H[0]
        for t in range(T):
            hstate = np.tanh(hstate @ W_hh_T + P["b_hh"] + I_ext[t])
            H[t + 1] = hstate
        logits = hstate @ P["W_out"].T + P["b_out"]
        return logits, dict(S=S, H=H)

    def simulate(self, S, h0=None):
        logits, cache = self.forward(S, h0)
        return dict(logits=logits, X=cache["H"].transpose(1, 0, 2))

    def backward(self, cache, dlogits, return_state_grads=False):
        S, H = cache["S"], cache["H"]
        T = S.shape[1]
        P = self.params
        g = self.zero_grads()
        g["W_out"] = dlogits.T @ H[T]
        g["b_out"] = dlogits.sum(0)
        gh = dlogits @ P["W_out"]
        GU = np.zeros_like(H[1:])
        if return_state_grads:
            GH = np.zeros_like(H)
            GH[T] = gh
        for t in range(T - 1, -1, -1):
            gu = gh * (1.0 - H[t + 1] ** 2)
            GU[t] = gu
            gh = gu @ P["W_hh"]
            if return_state_grads:
                GH[t] = gh
        g["W_hh"] = np.einsum("tbn,tbm->nm", GU, H[:-1])
        g["b_hh"] = GU.sum((0, 1))
        g["W_ih"] = np.einsum("tbn,btk->nk", GU, S)
        g["b_ih"] = GU.sum((0, 1))
        if return_state_grads:
            return g, GH, None
        return g
