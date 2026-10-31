# HORN — a toy Harmonic Oscillator Recurrent Network

A small, didactic **NumPy** implementation of the networks in

> F. Effenberger, P. Carvalho, I. Dubinin & W. Singer,
> *The functional role of oscillatory dynamics in neocortical circuits: A computational perspective*,
> [PNAS 122(4) e2412830122 (2025)](https://doi.org/10.1073/pnas.2412830122).

The authors' PyTorch reference is [github.com/exilef/horn](https://github.com/exilef/horn). This repository reimplements the same update equations with a **hand-written backward pass** (backpropagation through time), so every gradient can be inspected. The tutorial notebook walks from a single oscillator to training a tiny network on XOR and then sequential MNIST.

## Start here

Open [`notebooks/HORN_tutorial.ipynb`](notebooks/HORN_tutorial.ipynb). It is meant to be run **cell by cell**. Sliders (via `ipywidgets`) let you change $\omega$, $\gamma$, coupling, pulse delay, network size, etc.; if widgets do not render, the same functions still run once with the default values.

| Part | Contents |
|---|---|
| 1 | One damped harmonic oscillator: ringing, phase coding, resonance, feedback |
| 2 | Coupling nodes into a HORN; synchrony vs. heterogeneity |
| 3 | Softmax / cross-entropy, gradient descent on the readout only (reservoir computing) |
| 4 | Backpropagation through time, finite-difference gradient check, gradient flow |
| 5 | XOR, then *temporal* XOR (two pulses that interfere) |
| 6 | Sequential MNIST: why $\omega = 2\pi/28$, training, looking inside a trained net |
| 7 | Suggested next experiments (delays, two layers, Hebbian rules, travelling waves) |

Pick the Jupyter kernel **Python (.venv Dropbox/GIT)** (or whatever environment has the packages in `requirements.txt`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks && jupyter notebook HORN_tutorial.ipynb
```

## The model, in one block

Each node is a driven damped harmonic oscillator with amplitude $x$ and velocity $y$, integrated with symplectic Euler ($h = 1$):

```
u[t]   = (1/sqrt(n)) * (W_hh @ y[t] + b_hh + v * x[t]) + W_ih @ s[t] + b_ih
y[t+1] = y[t] + alpha * tanh(u[t]) - 2*gamma*y[t] - omega**2 * x[t]
x[t+1] = x[t] + y[t+1]
logits = W_out @ x[T] + b_out
```

`omega`, `gamma`, `alpha` are **fixed** per node (homogeneous: all equal; heterogeneous: drawn in `[0.5, 2] ×` the base value). Everything else is trained with BPTT. The paper's default for sequential MNIST is `omega = 2π/28`, `gamma = 0.01`, `alpha = 0.04`.

## Layout

```
horn/                 NumPy library (import horn)
  dho.py              single node + linear theory (gain curve, eigenvalues)
  network.py          HORN and Elman RNN, with manual BPTT
  losses.py           softmax cross-entropy
  optim.py            SGD and AdamW
  train.py            mini-batch loop
  data.py             XOR, frequency task, sequential MNIST
  analysis.py         Kuramoto order, PLV, IRFs, gradient flow
  gradcheck.py        finite-difference check of the backward pass
notebooks/HORN_tutorial.ipynb
scripts/train_smnist.py
pretrained/           32-node sMNIST checkpoints the notebook loads if present
tests/test_horn.py
```

## Training sequential MNIST yourself

Two epochs of a 32-node HORN take about five minutes on a CPU and reach ~88% test accuracy (the paper trains for 100 epochs). The notebook will load these checkpoints if they exist, so the slow part can be skipped:

```bash
python scripts/train_smnist.py --model horn --n-hidden 32 --epochs 2
python scripts/train_smnist.py --model horn_hetero --n-hidden 32 --epochs 2
python scripts/train_smnist.py --model elman --n-hidden 32 --epochs 2
python scripts/train_smnist.py --model horn --n-hidden 32 --epochs 2 --gamma 0.5 --name horn_overdamped_32
```

Checkpoints already in `pretrained/` from a 2-epoch run on this machine:

| model | test accuracy | notes |
|---|---|---|
| homogeneous HORN, 32 nodes | 0.877 | $\gamma = 0.01$ |
| heterogeneous HORN, 32 nodes | 0.895 | per-node $\omega, \gamma$ |
| overdamped HORN, 32 nodes | 0.426 | same architecture, $\gamma = 0.5$ |
| Elman tanh RNN, 32 nodes | 0.101 | chance; gradients vanish over 784 steps |

## Tests

```bash
python tests/test_horn.py
```

Verifies that a 1-node HORN matches the isolated DHO, that BPTT agrees with finite differences, and that a 4-node network solves both XOR variants.

## Citation

Effenberger, F., Carvalho, P., Dubinin, I. & Singer, W. (2025). The functional role of oscillatory dynamics in neocortical circuits: A computational perspective. *PNAS* 122(4), e2412830122. https://doi.org/10.1073/pnas.2412830122
