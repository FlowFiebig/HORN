"""horn -- a small, didactic NumPy implementation of Harmonic Oscillator
Recurrent Networks (HORNs), after Effenberger, Carvalho, Dubinin & Singer,
"The functional role of oscillatory dynamics in neocortical circuits: A
computational perspective", PNAS 122(4) e2412830122 (2025).

Modules
-------
dho       single damped-harmonic-oscillator node and its linear theory
network   HORN (and an Elman RNN baseline) with hand-written BPTT
losses    softmax cross-entropy (+ gradient)
optim     SGD and Adam(W)
train     mini-batch training loop and evaluation
data      XOR toy tasks, frequency task, (sequential) MNIST loading
analysis  synchrony, receptive fields, correlations, gradient flow
gradcheck finite-difference verification of the backward pass
plotting  matplotlib helpers for the notebook
"""
from .dho import (dho_step, simulate_dho, impulse, step_input, sinusoid, linear_map,
                  eigen_analysis, effective_parameters, gain_linear, gain_continuous,
                  gain_simulated)
from .network import HORN, ElmanRNN
from .losses import softmax, softmax_cross_entropy, mse_loss, accuracy
from .optim import SGD, Adam
from .train import train, evaluate, smooth
from .data import (make_xor, make_temporal_xor, make_frequency_task, load_mnist, to_sequences,
                   downsample, bar_image, add_noise, XOR_TABLE, XOR_LABELS)
from .plotting import plot_traces, plot_matrix, plot_history, confusion_matrix
from .analysis import (instantaneous_phase, kuramoto_order, plv, pairwise_plv,
                       pairwise_crosscorr, upper_triangle, intrinsic_receptive_fields,
                       network_frequency_response, gradient_flow, power_spectrum)
from .gradcheck import check_gradients, numerical_gradient

__version__ = "0.1.0"
