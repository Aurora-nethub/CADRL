"""
Minimal training loop for CADRL.

This module exposes `run_training(cfg, ...)` which can be imported and called
from `main.py` (CLI handling should live in main.py as requested).

The function implements a simple TD(0) training loop using the project"s
`CADRLEnv`, `ActionSpace`, `ValueNetwork`, `ReplayMemory`, and `CADRLAgent`.
"""
from typing import Optional, Dict, List
import os
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from core import ConfigContainer, get_config_container
from env import CADRLEnv
from agent import CADRLAgent
from models import ValueNetwork
from utils import ActionSpace, ReplayMemory
from utils.trajectory import Trajectory
from utils.value_target import ArrivalTimeTarget


def load_trajectories(path: str, *, skip_timed_out: bool = True) -> List[Trajectory]:
    """
    Load trajectories from `path` and return a list of `Trajectory` objects.

    NOTE: placeholder. Implement data loading in your environment and
    return a List[Trajectory].
    """
    trajs: List[Trajectory] = []
    if not os.path.exists(path):
        return trajs

    files = sorted([f for f in os.listdir(path) if f.endswith(".npz")])
    for fn in files:
        fp = os.path.join(path, fn)
        try:
            with np.load(fp) as data:
                # skip trajectories that were timed out or went out of bounds (generator may add these fields)
                if skip_timed_out and (bool(data.get("timed_out", False)) or bool(data.get("out_of_bounds", False))):
                    continue
                times = data["times"]
                positions = data["positions"]  # (T,2,2)
                v_pref = float(data.get("v_pref", 1.0))
                radius = float(data.get("radius", 0.3))
                goal_x = float(data.get("goal_x", 0.0))
                goal_y = float(data.get("goal_y", 0.0))
                kinematic = bool(data.get("kinematic", True))

                # build Trajectory: requires gamma, goal_x, goal_y, radius, v_pref, times, positions
                traj = Trajectory(
                    gamma=float(getattr(get_config_container().model, "gamma", 0.8)),
                    goal_x=float(goal_x),
                    goal_y=float(goal_y),
                    radius=float(radius),
                    v_pref=float(v_pref),
                    times=np.asarray(times, dtype=np.float32),
                    positions=np.asarray(positions, dtype=np.float32),
                    kinematic=bool(kinematic),
                )
                trajs.append(traj)
        except Exception: # pylint: disable=broad-except
            # skip malformed files
            continue
    return trajs


def _make_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_checkpoint(model: nn.Module, path: str) -> None:
    torch.save({"state_dict": model.state_dict()}, path)


def run_training(
    cfg: Optional[Dict] = None,
    *,
    device: Optional[torch.device] = None,
    resume: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    """
    Run a minimal TD training loop.

    Parameters
    ----------
    cfg : Optional[Dict]
        If None, load configuration via `get_config_container()`; otherwise
        expects a ConfigContainer or dict-like mapping with required fields.
    device : Optional[torch.device]
        Device to run training on; if None, use cfg.device.
    resume : Optional[str]
        Path to a checkpoint to resume from (optional).
    run_name : Optional[str]
        Name for this run; used to create checkpoint/log directories.

    Notes
    -----
    This is intentionally minimal: it demonstrates integration with the
    repository"s components. It is not optimized for speed or stability.
    """
    # load config container if needed
    if cfg is None:
        cfg = get_config_container()
    # allow passing a raw dict -> get a ConfigContainer
    if isinstance(cfg, dict):
        cfg = ConfigContainer.from_dict(cfg)
    if not hasattr(cfg, "agent"):
        # fallback to loader
        cfg = get_config_container()

    device = device or getattr(cfg, "device", torch.device("cpu"))

    # build paths
    base_dir = os.path.join("checkpoints", run_name or time.strftime("%Y%m%d-%H%M%S"))
    _make_dirs(base_dir)

    # environment and action space
    env = CADRLEnv(cfg, phase="train")
    action_space = ActionSpace(cfg.agent.v_pref, cfg.agent.kinematic)

    # model and replay
    model = ValueNetwork(state_dim=cfg.model.state_dim, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate))
    replay = ReplayMemory(capacity=int(cfg.train.capacity), state_dim=int(cfg.model.state_dim))

    # -------------------- supervised pretraining (optional) --------------------

    pretrain_epochs = getattr(cfg.train, "pretrain_epochs", 0)
    traj_dir = getattr(cfg.init, "traj_dir", None)
    if pretrain_epochs > 0 and traj_dir:
        print("\n" + "="*70)
        print("🚀 开始预训练阶段 (Supervised Pretraining)")
        print("="*70)
        try:
            trajectories = load_trajectories(traj_dir)
        except NotImplementedError:
            trajectories = []

        if trajectories:
            print(f"✓ 加载了 {len(trajectories)} 条轨迹数据")
            pair_gen = ArrivalTimeTarget(gamma=float(cfg.model.gamma))
            # simple pretraining loop (MSE on arrival time targets)
            preoptimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate))
            model.train()

            # 预训练进度条
            pretrain_pbar = tqdm(range(pretrain_epochs), desc="预训练进度",
                                unit="epoch", colour="blue", ncols=100)

            for pe in pretrain_pbar:
                pairs = []
                for traj in trajectories:
                    pairs.extend(pair_gen.compute_pairs(traj))

                epoch_loss = 0.0
                batch_count = 0

                # small random shuffle and batch
                torch.manual_seed(pe)
                for i in range(0, max(1, len(pairs)), int(cfg.train.batch_size)):
                    batch_pairs = pairs[i:i + int(cfg.train.batch_size)]
                    if not batch_pairs:
                        continue
                    s_batch = torch.stack([p[0].flatten() for p in batch_pairs]).to(device)
                    y_batch = torch.stack([p[1].flatten() for p in batch_pairs]).to(device)
                    pred = model(s_batch)
                    loss_pre = nn.functional.mse_loss(pred, y_batch)
                    preoptimizer.zero_grad()
                    loss_pre.backward()
                    preoptimizer.step()

                    epoch_loss += loss_pre.item()
                    batch_count += 1

                # 更新进度条显示
                avg_loss = epoch_loss / max(1, batch_count)
                pretrain_pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "pairs": len(pairs),
                    "batches": batch_count
                })

            pretrain_pbar.close()
            print(f"✓ 预训练完成！共训练 {pretrain_epochs} 轮")
            # end pretraining
        else:
            print("⚠ 警告：未找到有效的轨迹数据，跳过预训练")
    else:
        if pretrain_epochs == 0:
            print("\n📝 预训练已禁用 (pretrain_epochs=0)")
        else:
            print("\n⚠ 警告：未配置轨迹目录，跳过预训练")

    # agent that uses the value network for action selection
    agent = CADRLAgent(value_net=model, action_space=action_space, device=device, gamma=float(cfg.model.gamma))

    # optionally resume
    if resume is not None and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        try:
            model.load_state_dict(ckpt.get("state_dict", ckpt))
        except Exception: # pylint: disable=broad-except
            model.load_state_dict(ckpt)

    # training loop (minimal)
    num_epochs = int(cfg.train.num_epochs)
    sample_episodes = int(cfg.train.sample_episodes)
    batch_size = int(cfg.train.batch_size)

    generator = torch.Generator()

    print("\n" + "="*70)
    print("🎯 开始强化学习训练阶段 (Reinforcement Learning)")
    print("="*70)
    print("配置信息:")
    print(f"  - 训练轮数: {num_epochs}")
    print(f"  - 每轮采样episodes: {sample_episodes}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 学习率: {cfg.train.learning_rate}")
    print(f"  - Gamma: {cfg.model.gamma}")
    print(f"  - 设备: {device}")
    print(f"  - 检查点间隔: {cfg.train.checkpoint_interval} epochs")
    print("="*70 + "\n")

    # 训练主循环进度条
    train_pbar = tqdm(range(num_epochs), desc="训练进度",
                     unit="epoch", colour="green", ncols=100)

    for epoch in train_pbar:
        # collect some episodes into replay
        episode_rewards = []
        episode_lengths = []

        for _ in range(sample_episodes):
            states = env.reset()
            done = [False, False]
            steps = 0
            ep_reward = 0.0

            while not all(done) and steps < env.max_steps:
                actions = []
                for i in range(2):
                    s = states[i]
                    a = agent.act(s, env, i, mode="train")
                    actions.append(a)
                s_next, rewards, dones = env.step(actions)
                # push transitions for each agent
                for i in range(2):
                    replay.push(s=states[i], r=rewards[i], s_next=s_next[i], done=dones[i] != 0)

                ep_reward += rewards[0]  # 记录agent 0的奖励
                states = s_next
                done = [d != 0 for d in dones]
                steps += 1

            episode_rewards.append(ep_reward)
            episode_lengths.append(steps)

        # learning step if enough data
        train_loss = 0.0
        if len(replay) >= max(1, batch_size):
            batch = replay.sample(batch_size, generator=generator)
            s = batch["s"].to(device)
            r = batch["r"].to(device)
            s_next = batch["s_next"].to(device)
            done = batch["done"].to(device)

            with torch.no_grad():
                v_next = model(s_next).detach()
            v = model(s)

            # TD target: y = r + gamma * v_next * (1 - done)
            gamma = float(cfg.model.gamma)
            y = r + gamma * v_next * (~done)

            loss = nn.functional.mse_loss(v, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

        # 更新进度条显示
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        train_pbar.set_postfix({
            "loss": f"{train_loss:.4f}",
            "reward": f"{avg_reward:.2f}",
            "steps": f"{avg_length:.1f}",
            "replay": len(replay)
        })

        # checkpoint regularly
        if (epoch + 1) % int(cfg.train.checkpoint_interval) == 0:
            ckpt_path = os.path.join(base_dir, f"checkpoint_epoch_{epoch+1}.pt")
            _save_checkpoint(model, ckpt_path)
            tqdm.write(f"💾 已保存检查点: {ckpt_path}")

    train_pbar.close()

    # final save
    final_path = os.path.join(base_dir, "final.pt")
    _save_checkpoint(model, final_path)

    print("\n" + "="*70)
    print("🎉 训练完成！")
    print("="*70)
    print(f"✓ 最终模型已保存至: {final_path}")
    print(f"✓ 检查点目录: {base_dir}")
    print(f"✓ 训练总轮数: {num_epochs}")
    print(f"✓ 经验池大小: {len(replay)}")
    print("="*70 + "\n")


__all__ = ["run_training"]
