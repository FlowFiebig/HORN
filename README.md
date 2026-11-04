# HORN tutorial: from a single oscillator to sequential MNIST 
A proposed course tutorial for KTH's Advanced Neural Networks course (because they are cool and certainly more advanced that standard ANNs)

This is a hands-on introduction to **Harmonic Oscillator Recurrent Networks (HORNs)**, after
> F. Effenberger, P. Carvalho, I. Dubinin & W. Singer, *The functional role of oscillatory dynamics in neocortical circuits: A computational perspective*, PNAS 122(4) e2412830122 (2025). Reference code: [github.com/exilef/horn](https://github.com/exilef/horn).

For students to be able to run this themselves, everything here runs on a small, pure-NumPy implementation that lives in the `horn/` package of this repository. I purposefully avoid using torch.autograd: the backward pass (backpropagation through time) is written out by hand in `horn/network.py`, meaning we will be able to see the chain rule in action on a simple XOR task which students know from the ANN course (also temporal XOR). My idea was to make this presentation of HORN somewhat more useful for educational purposes by avoiding ML libraries, such that every line of code that is executed can be read and comprehended before students jump to use Pytorch.

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
