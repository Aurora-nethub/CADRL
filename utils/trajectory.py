import math
import numpy as np

from dataclasses import dataclass
from itertools import pairwise
from typing import List, Tuple

import torch
from core import get_config_container

config = get_config_container()
device = config.device

def compute_value(gamma:float, time_to_goal:float, v_pref:float)->float:
    return gamma**(time_to_goal*v_pref)

@dataclass
class Trajectory:
    times: List[float]
    positions: List[Tuple[float, float]]  # List of [px, py]
    gamma: float
    goal: Tuple[float, float]  # [pgx, pgy]
    v_pref: float
    radius: float
    kinematic: bool = True  # True for robot, False for point mass

    def to_pairs(self)->List[Tuple[torch.Tensor, torch.Tensor]]:
        """Convert trajectory to <state, value> pairs for value function training."""
        pairs = []
        total_time = self.times[-1]

        for (prev_time, curr_time), (prev_pos, curr_pos) in zip(
            pairwise(self.times),
            pairwise(self.positions)
        ):
            dt = curr_time - prev_time

            # 智能体0
            pos0_curr = curr_pos[0]
            vel0 = (curr_pos[0] - prev_pos[0]) / dt
            theta = math.atan2(vel0[1], vel0[0]) if self.kinematic else 0.0

            # 智能体1
            pos1_curr = curr_pos[1]
            vel1 = (curr_pos[1] - prev_pos[1]) / dt

            # 构建状态数组
            state_data = np.array([
                pos0_curr[0], pos0_curr[1], vel0[0], vel0[1], self.radius,
                self.goal[0], self.goal[1], self.v_pref, theta,
                pos1_curr[0], pos1_curr[1], vel1[0], vel1[1], self.radius
            ], dtype=np.float32)

            time_to_goal = total_time - curr_time

            state = torch.tensor(state_data, device=device, dtype=torch.float32)
            value = torch.tensor(
                [self.compute_value(self.gamma, time_to_goal, self.v_pref)],
                device=device,
                dtype=torch.float32
            )

            pairs.append((state, value))

        return pairs

__all__ = ['Trajectory', 'compute_value']
