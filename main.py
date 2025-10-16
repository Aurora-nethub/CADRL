"""
Entry point for CADRL training and evaluation.

Parses command-line arguments and delegates to functions in `scripts.train`.
Keep argument parsing here; `scripts.train` exposes `run_training` without CLI.
"""
from __future__ import annotations


import argparse
import sys
import random
from typing import Optional

import numpy as np
import torch

from core import get_config_container
from scripts.train import run_training
from scripts.test import run_test


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CADRL training/test entry point")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test")
    parser.add_argument("--case", type=int, default=0, help="Test case id (0~9)")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda[:id])")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--sample-episodes", type=int, default=None, help="Override sample_episodes from config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Override checkpoint_interval from cfg")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)

    cfg = get_config_container()

    # apply overrides from CLI to cfg
    train_overrides = {}
    if args.lr is not None:
        train_overrides.setdefault("train", {})["learning_rate"] = float(args.lr)
    if args.epochs is not None:
        train_overrides.setdefault("train", {})["num_epochs"] = int(args.epochs)
    if args.sample_episodes is not None:
        train_overrides.setdefault("train", {})["sample_episodes"] = int(args.sample_episodes)
    if args.batch_size is not None:
        train_overrides.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.checkpoint_interval is not None:
        train_overrides.setdefault("train", {})["checkpoint_interval"] = int(args.checkpoint_interval)

    if train_overrides:
        cfg.update(train_overrides)

    # set device
    device = None
    if args.device:
        try:
            device = torch.device(args.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            sys.stderr.write(f"Invalid device \"{args.device}\", falling back to config device: {exc}\n")
            device = getattr(cfg, "device", None)
    else:
        device = getattr(cfg, "device", None)

    # set seed
    if args.seed is not None:
        cfg.update({"train": {"random_seed": int(args.seed)}})
        _set_seed(int(args.seed))

    # initialize logger from cfg (will create logs folder)
    try:
        cfg.init_logger()
        cfg.logger.info("Starting run: %s", args.run_name or "auto")
        cfg.logger.info("Config: %s", cfg)
    except Exception as exc:  # pylint: disable=broad-except
        # best-effort logging: if logger fails, print to stderr
        sys.stderr.write(f"Logger init failed: {exc}\n")

    if getattr(args, "mode", "train") == "test":
        # 直接调用测试函数，不用子进程
        # 如果未指定case参数，则run_test内部自动随机case
        if hasattr(args, "case") and args.case is not None:
            run_test._case = args.case
        run_test(cfg, device=device, run_name=args.run_name)
    else:
        # call training routine
        run_training(cfg, device=device, resume=args.resume, run_name=args.run_name)


if __name__ == "__main__":
    main()
