"""
Minimal training loop for CADRL.

This module exposes `run_training(cfg, ...)` which can be imported and called
from `main.py` (CLI handling should live in main.py as requested).

The function implements a simple TD(0) training loop using the project's
`CADRLEnv`, `ActionSpace`, `ValueNetwork`, `ReplayMemory`, and `CADRLAgent`.
"""
from typing import Optional, Dict, List
import os
import time

import torch
import torch.nn as nn
import numpy as np

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

    files = sorted([f for f in os.listdir(path) if f.endswith('.npz')])
    for fn in files:
        fp = os.path.join(path, fn)
        try:
            with np.load(fp) as data:
                # skip trajectories that were timed out or went out of bounds (generator may add these fields)
                if skip_timed_out and (bool(data.get('timed_out', False)) or bool(data.get('out_of_bounds', False))):
                    continue
                times = data['times']
                positions = data['positions']  # (T,2,2)
                v_pref = float(data.get('v_pref', 1.0))
                radius = float(data.get('radius', 0.3))
                goal_x = float(data.get('goal_x', 0.0))
                goal_y = float(data.get('goal_y', 0.0))
                kinematic = bool(data.get('kinematic', True))

                # build Trajectory: requires gamma, goal_x, goal_y, radius, v_pref, times, positions
                traj = Trajectory(
                    gamma=float(getattr(get_config_container().model, 'gamma', 0.8)),
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
    torch.save({'state_dict': model.state_dict()}, path)


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
    repository's components. It is not optimized for speed or stability.
    """
    # load config container if needed
    if cfg is None:
        cfg = get_config_container()
    # allow passing a raw dict -> get a ConfigContainer
    if isinstance(cfg, dict):
        cfg = ConfigContainer.from_dict(cfg)
    if not hasattr(cfg, 'agent'):
        # fallback to loader
        cfg = get_config_container()

    device = device or getattr(cfg, 'device', torch.device('cpu'))

    # build paths
    base_dir = os.path.join('checkpoints', run_name or time.strftime('%Y%m%d-%H%M%S'))
    _make_dirs(base_dir)

    # environment and action space
    env = CADRLEnv(cfg, phase='train')
    action_space = ActionSpace(cfg.agent.v_pref, cfg.agent.kinematic)

    # model and replay
    model = ValueNetwork(state_dim=cfg.model.state_dim, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate))
    replay = ReplayMemory(capacity=int(cfg.train.capacity), state_dim=int(cfg.model.state_dim))

    # -------------------- supervised pretraining (optional) --------------------

    pretrain_epochs = getattr(cfg.train, 'pretrain_epochs', 0)
    traj_dir = getattr(cfg.init, 'traj_dir', None)
    if pretrain_epochs > 0 and traj_dir:
        try:
            trajectories = load_trajectories(traj_dir)
        except NotImplementedError:
            trajectories = []

        if trajectories:
            pair_gen = ArrivalTimeTarget(gamma=float(cfg.model.gamma))
            # simple pretraining loop (MSE on arrival time targets)
            preoptimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.learning_rate))
            model.train()
            for pe in range(pretrain_epochs):
                pairs = []
                for traj in trajectories:
                    pairs.extend(pair_gen.compute_pairs(traj))
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
            # end pretraining

    # agent that uses the value network for action selection
    agent = CADRLAgent(value_net=model, action_space=action_space, device=device, gamma=float(cfg.model.gamma))

    # optionally resume
    if resume is not None and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        try:
            model.load_state_dict(ckpt.get('state_dict', ckpt))
        except Exception: # pylint: disable=broad-except
            model.load_state_dict(ckpt)

    # training loop (minimal)
    num_epochs = int(cfg.train.num_epochs)
    sample_episodes = int(cfg.train.sample_episodes)
    batch_size = int(cfg.train.batch_size)

    generator = torch.Generator()

    for epoch in range(num_epochs):
        # collect some episodes into replay
        for _ in range(sample_episodes):
            states = env.reset()
            done = [False, False]
            steps = 0
            while not all(done) and steps < env.max_steps:
                actions = []
                for i in range(2):
                    s = states[i]
                    a = agent.act(s, env, i, mode='train')
                    actions.append(a)
                s_next, rewards, dones = env.step(actions)
                # push transitions for each agent
                for i in range(2):
                    replay.push(s=states[i], r=rewards[i], s_next=s_next[i], done=dones[i] != 0)
                states = s_next
                done = [d != 0 for d in dones]
                steps += 1

        # learning step if enough data
        if len(replay) >= max(1, batch_size):
            batch = replay.sample(batch_size, generator=generator)
            s = batch['s'].to(device)
            r = batch['r'].to(device)
            s_next = batch['s_next'].to(device)
            done = batch['done'].to(device)

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

        # checkpoint regularly
        if (epoch + 1) % int(cfg.train.checkpoint_interval) == 0:
            ckpt_path = os.path.join(base_dir, f'checkpoint_epoch_{epoch+1}.pt')
            _save_checkpoint(model, ckpt_path)

    # final save
    final_path = os.path.join(base_dir, 'final.pt')
    _save_checkpoint(model, final_path)


__all__ = ['run_training']
