import itertools
import math
import random
from dataclasses import dataclass
from typing import List

from core import get_config_container

config = get_config_container()
random.seed(config.train.random_seed)

@dataclass
class Action:
    """表示一个机器人动作"""
    velocity: float  # 线速度
    rotation: float  # 旋转角度（弧度）

    def __repr__(self):
        return f"Action(v={self.velocity:.2f}, r={math.degrees(self.rotation):.1f}°)"


class ActionSpace:
    def __init__(self, v_pref: float, kinematic: bool = True):
        self.v_pref = v_pref
        self.kinematic = kinematic
        self.actions = self._build_action_space()

    def _build_action_space(self) -> List[Action]:
        """
        构建动作空间
        25 pre-computed actions (e.g. directed toward an agent’s goal or current heading) 
        and 10 randomly sampled actions.
        """
        actions = []

        if self.kinematic:
            # 运动学模型（差速驱动）
            velocities = [(i + 1) / 5 * self.v_pref for i in range(5)]
            rotations = [i/4 * math.pi/3 - math.pi/6 for i in range(5)]

            # 添加预定义动作
            for v, r in itertools.product(velocities, rotations):
                actions.append(Action(v, r))

            # 添加随机动作
            for _ in range(10):
                v = random.random() * self.v_pref
                r = random.random() * math.pi/3 - math.pi/6
                actions.append(Action(v, r))
        else:
            # 全向模型（全向移动）
            velocities = [(i + 1) / 5 * self.v_pref for i in range(5)]
            rotations = [i / 4 * 2 * math.pi for i in range(5)]

            # 添加预定义动作
            for v, r in itertools.product(velocities, rotations):
                actions.append(Action(v, r))

            # 添加随机动作
            for _ in range(10):
                v = random.random() * self.v_pref
                r = random.random() * 2 * math.pi
                actions.append(Action(v, r))

        # 添加停止动作
        actions.append(Action(0, 0))
        return actions

    def __len__(self) -> int:
        """返回动作空间大小"""
        return len(self.actions)

    def __getitem__(self, index: int) -> Action:
        """通过索引获取动作"""
        return self.actions[index]

    def sample(self) -> Action:
        """随机采样一个动作"""
        return random.choice(self.actions)

    def get_action_index(self, action: Action) -> int:
        """获取动作的索引"""
        for i, a in enumerate(self.actions):
            if a.velocity == action.velocity and a.rotation == action.rotation:
                return i
        return -1

    def get_actions(self) -> list[Action]:
        """获取所有动作列表"""
        return self.actions.copy()

__all__ = ['ActionSpace', 'Action']
