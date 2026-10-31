Checkpoints from `scripts/train_smnist.py` (32 nodes, 2 epochs, ~1,480 parameters). The tutorial notebook loads them in Part 6 so you can inspect a trained network without waiting.

| file | test accuracy |
|---|---|
| `horn_32.npz` | 0.877 |
| `horn_hetero_32.npz` | 0.895 |
| `horn_overdamped_32.npz` | 0.426 |
| `elman_32.npz` | 0.101 (chance) |

`*_init.npz` is the same network before training; `*_history.npz` holds the learning curves.
