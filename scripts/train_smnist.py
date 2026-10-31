"""Train a HORN (or an Elman RNN baseline) on sequential MNIST and save the result.

Example (from the repository root):

    python scripts/train_smnist.py --model horn --n-hidden 32 --epochs 2
    python scripts/train_smnist.py --model horn_hetero --n-hidden 32 --epochs 2
    python scripts/train_smnist.py --model elman --n-hidden 32 --epochs 2

Writes pretrained/<name>.npz (weights) and pretrained/<name>_history.npz
(training curves). The notebook loads these if they exist, so that the slow
training step can be skipped.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from horn import HORN, ElmanRNN, train, evaluate, load_mnist, to_sequences, downsample  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", default="horn", choices=["horn", "horn_hetero", "elman"])
p.add_argument("--n-hidden", type=int, default=32)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--weight-decay", type=float, default=1e-2)
p.add_argument("--alpha", type=float, default=0.04)
p.add_argument("--omega", type=float, default=2 * np.pi / 28)
p.add_argument("--gamma", type=float, default=0.01)
p.add_argument("--downsample", type=int, default=1, help="pool pixels by this factor (2 -> 14x14 images, 196 steps)")
p.add_argument("--n-train", type=int, default=None, help="use only the first N training samples")
p.add_argument("--permute", action="store_true", help="psMNIST: apply a fixed random pixel permutation")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval-every", type=int, default=100)
p.add_argument("--name", default=None)
p.add_argument("--out", default=str(ROOT / "pretrained"))
args = p.parse_args()

name = args.name or f"{args.model}_{args.n_hidden}" + (f"_ds{args.downsample}" if args.downsample > 1 else "") + ("_perm" if args.permute else "")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
print(args, flush=True)

tr_img, tr_lbl, te_img, te_lbl = load_mnist()
if args.downsample > 1:
    tr_img, te_img = downsample(tr_img, args.downsample) * 255, downsample(te_img, args.downsample) * 255
perm = np.random.default_rng(42).permutation(tr_img.shape[1] * tr_img.shape[2]) if args.permute else None
S_tr, S_te = to_sequences(tr_img, perm), to_sequences(te_img, perm)
y_tr, y_te = tr_lbl.astype(int), te_lbl.astype(int)

# hold out 1000 training samples for validation (as in the reference code)
rng = np.random.default_rng(args.seed)
idx = rng.permutation(len(S_tr))
val_idx, tr_idx = idx[:1000], idx[1000:]
if args.n_train:
    tr_idx = tr_idx[:args.n_train]
S_val, y_val = S_tr[val_idx], y_tr[val_idx]
S_tr, y_tr = S_tr[tr_idx], y_tr[tr_idx]
print(f"train {len(S_tr)}, val {len(S_val)}, test {len(S_te)}, T = {S_tr.shape[1]}", flush=True)

if args.model == "horn":
    model = HORN(1, args.n_hidden, 10, omega=args.omega, gamma=args.gamma, alpha=args.alpha, seed=args.seed)
elif args.model == "horn_hetero":
    model = HORN.heterogeneous(1, args.n_hidden, 10, omega=args.omega, gamma=args.gamma, alpha=args.alpha, seed=args.seed)
else:
    model = ElmanRNN(1, args.n_hidden, 10, seed=args.seed)
print(f"{name}: {model.n_params} trainable parameters", flush=True)
model.save(out / f"{name}_init.npz")

t0 = time.time()
hist = train(model, S_tr, y_tr, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             weight_decay=args.weight_decay, S_val=S_val, y_val=y_val, eval_every=args.eval_every,
             seed=args.seed, verbose=True)
test_loss, test_acc = evaluate(model, S_te, y_te)
print(f"{name}: test accuracy {test_acc:.4f} (loss {test_loss:.3f}) after {time.time() - t0:.0f}s", flush=True)

model.save(out / f"{name}.npz")
np.savez(out / f"{name}_history.npz", **{k: np.asarray(v) for k, v in hist.items()},
         test_acc=test_acc, test_loss=test_loss, n_params=model.n_params)
print(f"saved {out / (name + '.npz')}")
