CADRL - Minimal CADRL Reimplementation
=====================================

Quick start
-----------

This repository contains a minimal implementation of CADRL components (env, agent, value network)
and a small training harness. Use the provided runners to start training.

Unix / macOS

```
./run.sh --run-name test_run --lr 0.001 --epochs 10
```

Windows (PowerShell / CMD)

```
run.bat --run-name test_run --lr 0.001 --epochs 10
```

Examples
--------

- Train for 50 epochs with a custom learning rate:

```
./run.sh --run-name exp_lr1e3 --lr 0.001 --epochs 50
```

- Resume from a checkpoint:

```
./run.sh --resume checkpoints/20251008-123456/final.pt
```

Notes
-----

- CLI args override values in `config/env.json` and `config/model.json` where applicable.
- Logging is written into `logs/cadrl.log` by default.
- The training loop and model are minimal and intended as a starting point; please review
	`scripts/train.py` to adapt learning rules or evaluation to your needs.
