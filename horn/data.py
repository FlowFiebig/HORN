"""Toy data sets (XOR variants) and the sequential MNIST loader.

All sequence data sets are returned as arrays S of shape (N, T, n_in) together
with integer labels of shape (N,).
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# XOR
# ---------------------------------------------------------------------------

XOR_TABLE = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
XOR_LABELS = np.array([0, 1, 1, 0])


def make_xor(T=30, n_repeat=1, noise=0.0, seed=0):
    """The classic XOR as a sequence task.

    Both bits are presented *simultaneously* on two input channels and held
    constant for T time steps; the network must output XOR(b1, b2) at time T.
    With n_repeat > 1 every pattern is repeated (useful together with noise).
    """
    rng = np.random.default_rng(seed)
    bits = np.repeat(XOR_TABLE, n_repeat, axis=0).astype(float)
    labels = np.repeat(XOR_LABELS, n_repeat)
    S = np.repeat(bits[:, None, :], T, axis=1)
    if noise > 0:
        S = S + noise * rng.standard_normal(S.shape)
    return S, labels


def make_temporal_xor(T=40, t1=5, t2=19, width=1, amplitude=1.0, n_repeat=1,
                      noise=0.0, seed=0):
    """Temporal XOR: the two bits arrive one after the other on a *single* channel.

    Bit b_k = 1 is a pulse of height `amplitude` at time t_k (b_k = 0: no pulse);
    the network must answer XOR(b1, b2) at time T. To solve the task it must
    keep a memory of the first bit until the second one arrives -- in a HORN
    this memory is the ringing of the oscillators, and the second pulse
    interferes with it.
    """
    rng = np.random.default_rng(seed)
    bits = np.repeat(XOR_TABLE, n_repeat, axis=0)
    labels = np.repeat(XOR_LABELS, n_repeat)
    S = np.zeros((len(bits), T, 1))
    for i, (b1, b2) in enumerate(bits):
        S[i, t1:t1 + width, 0] = amplitude * b1
        S[i, t2:t2 + width, 0] = amplitude * b2
    if noise > 0:
        S = S + noise * rng.standard_normal(S.shape)
    return S, labels


def make_frequency_task(n_samples=200, T=150, freqs=(2 * np.pi / 28, 3 * 2 * np.pi / 28),
                        noise=1.0, amp_range=(0.5, 1.5), random_phase=False, seed=0):
    """Noisy sinusoids at one of several frequencies; the label is the frequency index.

    A deliberately simple task for the 'readout only' lesson: a node tuned to
    one of the frequencies amplifies that class through resonance and filters
    out the broadband noise, so even an *untrained* HORN produces final states
    from which a linear readout can decode the class.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(freqs), size=n_samples)
    t = np.arange(T)
    amps = rng.uniform(*amp_range, size=n_samples)
    phases = rng.uniform(0, 2 * np.pi, size=n_samples) if random_phase else np.zeros(n_samples)
    clean = amps[:, None] * np.sin(np.asarray(freqs)[labels][:, None] * t + phases[:, None])
    S = clean + noise * rng.standard_normal((n_samples, T))
    return S[:, :, None], labels, clean[:, :, None]


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def _download(url, path, timeout=30):
    """Download `url` to `path`. Tries IPv4 before IPv6 (IPv6 connections hang on
    some machines) and falls back to curl / wget if urllib fails."""
    import shutil
    import socket
    import subprocess

    orig_getaddrinfo = socket.getaddrinfo

    def ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
        res = orig_getaddrinfo(host, port, family, type, proto, flags)
        return sorted(res, key=lambda r: r[0] != socket.AF_INET)

    socket.getaddrinfo = ipv4_first
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
        return
    except Exception as e:                       # noqa: BLE001 -- fall back to CLI tools
        err = e
    finally:
        socket.getaddrinfo = orig_getaddrinfo
    for cmd in (["curl", "-fsSL", "--max-time", "300", "-o", str(path), url],
                ["wget", "-q", "-T", "60", "-O", str(path), url]):
        if shutil.which(cmd[0]) and subprocess.run(cmd, check=False).returncode == 0:
            return
    raise RuntimeError(f"download failed for {url}: {err!r}")


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(data_dir=None, verbose=True):
    """Download (once) and load MNIST.

    Returns (train_images, train_labels, test_images, test_labels) with images
    as uint8 arrays of shape (N, 28, 28). Cached as data/mnist.npz.
    """
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "mnist.npz"
    if cache.exists():
        d = np.load(cache)
        return d["train_images"], d["train_labels"], d["test_images"], d["test_labels"]

    arrays = {}
    for key, fname in MNIST_FILES.items():
        path = data_dir / fname
        if not path.exists():
            last_err = None
            for mirror in MNIST_MIRRORS:
                try:
                    if verbose:
                        print(f"downloading {mirror}{fname} ...", flush=True)
                    _download(mirror + fname, path)
                    break
                except Exception as e:      # noqa: BLE001 -- try the next mirror
                    last_err = e
                    if path.exists():
                        path.unlink()
            else:
                raise RuntimeError(f"could not download {fname}: {last_err}")
        arrays[key] = _read_idx(path)
    np.savez_compressed(cache, **arrays)
    for fname in MNIST_FILES.values():       # keep only the compact cache
        try:
            os.remove(data_dir / fname)
        except OSError:
            pass
    return arrays["train_images"], arrays["train_labels"], arrays["test_images"], arrays["test_labels"]


def to_sequences(images, permutation=None, dtype=np.float32):
    """Turn images (N, H, W) into sMNIST time series (N, H*W, 1) in [0, 1],
    scanning the pixels row by row. With a fixed `permutation` of the pixel
    indices you get *permuted* sMNIST (psMNIST)."""
    N = images.shape[0]
    S = images.reshape(N, -1).astype(dtype) / 255.0
    if permutation is not None:
        S = S[:, permutation]
    return S[:, :, None]


def downsample(images, factor=2):
    """Average-pool images by an integer factor (e.g. 28x28 -> 14x14) to get
    shorter -- and much faster to train on -- sequences."""
    N, H, W = images.shape
    h, w = H // factor, W // factor
    return images[:, :h * factor, :w * factor].reshape(N, h, factor, w, factor).mean(axis=(2, 4))


def bar_image(orientation="vertical", size=28, pos=None, thickness=1):
    """A single straight bar in an otherwise empty image (for resonance demos)."""
    img = np.zeros((size, size))
    pos = size // 2 if pos is None else pos
    if orientation == "vertical":
        img[:, pos:pos + thickness] = 1.0
    elif orientation == "horizontal":
        img[pos:pos + thickness, :] = 1.0
    elif orientation == "diagonal":
        for i in range(size):
            img[i, min(size - 1, max(0, i + pos - size // 2))] = 1.0
    else:
        raise ValueError(orientation)
    return img


def add_noise(S, sigma, seed=0, clip=True):
    """Additive white Gaussian pixel noise (as in the paper's noisy sMNIST)."""
    rng = np.random.default_rng(seed)
    out = S + sigma * rng.standard_normal(S.shape).astype(S.dtype)
    return np.clip(out, 0.0, 1.0) if clip else out
