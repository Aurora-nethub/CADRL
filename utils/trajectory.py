# utils/trajectory.py
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from utils.state import FullState, BaseState, JointState


@dataclass
class Trajectory:
    """
    仅承载轨迹数据与状态重建的“瘦容器”。

    times: np.ndarray, shape=(T,), 单调非减（单位：步或秒，取决于你的管线）
    positions: np.ndarray, shape=(T, 2, 2),  [t, agent_idx, xy]，agent_idx=0 自车, 1 对手
    radius: 两个体的半径（如不同，可在 state_at 里替换 neighbor 半径）
    v_pref: 自车首选速度（某些目标会用到，比如 gamma**(time_to_goal * v_pref)）
    goal_x, goal_y: 自车目标
    kinematic: 是否差速学动力学（用于估计自车朝向）
    gamma: 仅存放，具体目标策略根据需要使用
    """
    gamma: float
    goal_x: float
    goal_y: float
    radius: float
    v_pref: float
    times: np.ndarray
    positions: np.ndarray
    kinematic: bool = True

    def __post_init__(self):
        self.gamma = float(self.gamma)
        self.goal_x = float(self.goal_x)
        self.goal_y = float(self.goal_y)
        self.radius = float(self.radius)
        self.v_pref = float(self.v_pref)
        self.times = np.asarray(self.times, dtype=np.float32)
        self.positions = np.asarray(self.positions, dtype=np.float32)

        assert self.positions.ndim == 3 and self.positions.shape[1:] == (2, 2), \
            f"positions expected (T,2,2), got {self.positions.shape}"
        assert self.times.ndim == 1 and self.times.shape[0] == self.positions.shape[0], \
            "times and positions length mismatch"

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def time_at(self, idx: int) -> float:
        return float(self.times[idx])

    def _vel_between(self, i_prev: int, i_curr: int, agent_index: int) -> Tuple[float, float]:
        """根据两帧位置估计 vx, vy（简单差分）。"""
        dt = max(self.time_at(i_curr) - self.time_at(i_prev), 1e-6)
        p_prev = self.positions[i_prev, agent_index, :]
        p_curr = self.positions[i_curr, agent_index, :]
        vx = float((p_curr[0] - p_prev[0]) / dt)
        vy = float((p_curr[1] - p_prev[1]) / dt)
        return vx, vy

    def state_at(self, idx: int) -> torch.Tensor:
        """
        构造 idx 时刻的 14 维 JointState（CPU tensor）。
        注意：为了估计速度，要求 idx >= 1。
        """
        assert idx >= 1, "state_at(idx) requires idx >= 1 to estimate velocities."

        # 自车
        px0, py0 = [float(x) for x in self.positions[idx, 0, :]]
        vx0, vy0 = self._vel_between(idx-1, idx, 0)
        theta = math.atan2(vy0, vx0) if self.kinematic else 0.0

        # 对手
        px1, py1 = [float(x) for x in self.positions[idx, 1, :]]
        vx1, vy1 = self._vel_between(idx-1, idx, 1)

        self_state = FullState(
            px=px0, py=py0, vx=vx0, vy=vy0, radius=self.radius,
            pgx=self.goal_x, pgy=self.goal_y, v_pref=self.v_pref, theta=theta
        )
        neighbor_state = BaseState(px=px1, py=py1, vx=vx1, vy=vy1, radius=self.radius)
        joint = JointState(self_state=self_state, neighbor_state=neighbor_state)
        return joint.to_tensor()  # CPU (14,)

__all__ = ["Trajectory"]
