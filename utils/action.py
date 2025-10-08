# utils/action.py
import itertools
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class Action:
    """表示一个机器人动作"""
    velocity: float  # 线速度
    rotation: float  # 旋转角度（弧度）

    def normalized(self) -> "Action":
        """将角度归一到 [-pi, pi]，返回新对象，不修改自身"""
        r = ((self.rotation + math.pi) % (2 * math.pi)) - math.pi
        return Action(float(self.velocity), float(r))

    def to_tuple(self, ndigits: int = 6) -> Tuple[float, float]:
        """量化后的 (v, r) 元组，用于去重/建索引"""
        return (round(float(self.velocity), ndigits), round(float(self.rotation), ndigits))

    def __repr__(self) -> str:
        return f"Action(v={self.velocity:.2f}, r={math.degrees(self.rotation):.1f}°)"


class ActionSpace:
    def __init__(
        self,
        v_pref: float,
        kinematic: bool = True,
        *,
        grid_v: int = 5,         # 速度网格数
        grid_r: int = 5,         # 角度网格数
        rand_k: int = 10,        # 随机动作数
        rng: Optional[random.Random] = None,  # 注入本地 RNG，避免全局依赖
    ):
        self.v_pref = float(v_pref)
        self.kinematic = bool(kinematic)
        self.grid_v = int(grid_v)
        self.grid_r = int(grid_r)
        self.rand_k = int(rand_k)
        self.rng = rng or random.Random()

        self.actions: List[Action] = self._build_action_space()
        self._build_index()  # 立刻可用
        # 记住停止动作索引（若不存在则为 None）
        self.stop_index: Optional[int] = self._act2idx.get(Action(0.0, 0.0).to_tuple())

    # ---------- 构建 ----------

    def _build_action_space(self) -> List[Action]:
        """
        25 个预定义（5×5 网格）+ rand_k 个随机动作 + 一个停止动作。
        - 差速模型：rotation ∈ [-π/6, +π/6]
        - 全向模型：rotation ∈ [0, 2π)
        """
        actions: List[Action] = []

        # 速度网格
        velocities = [((i + 1) / self.grid_v) * self.v_pref for i in range(self.grid_v)]

        # 角度网格
        if self.kinematic:
            # 等距覆盖 [-π/6, +π/6]
            if self.grid_r == 1:
                rotations = [0.0]
            else:
                step = (math.pi / 3) / (self.grid_r - 1)
                rotations = [(-math.pi / 6) + i * step for i in range(self.grid_r)]
        else:
            if self.grid_r == 1:
                rotations = [0.0]
            else:
                step = (2 * math.pi) / self.grid_r
                rotations = [i * step for i in range(self.grid_r)]

        # 预定义动作
        for v, r in itertools.product(velocities, rotations):
            actions.append(Action(v, r).normalized())

        # 随机动作
        for _ in range(self.rand_k):
            v = self.rng.random() * self.v_pref
            if self.kinematic:
                r = self.rng.random() * (math.pi / 3) - (math.pi / 6)
            else:
                r = self.rng.random() * (2 * math.pi)
            actions.append(Action(v, r).normalized())

        # 停止动作
        actions.append(Action(0.0, 0.0))

        # 去重（按量化后的键，保持插入顺序）
        unique = {}
        for a in actions:
            unique.setdefault(a.to_tuple(), a)
        return list(unique.values())

    def _build_index(self) -> None:
        self._idx2act: List[Action] = list(self.actions)
        self._act2idx = {a.to_tuple(): i for i, a in enumerate(self._idx2act)}

    # ---------- 查询/采样 ----------

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> Action:
        return self.actions[index]

    def get_actions(self) -> List[Action]:
        return list(self.actions)

    def sample(self) -> Action:
        """随机采样一个动作（使用实例级 RNG，避免全局随机冲突）"""
        return self.rng.choice(self._idx2act)

    def action_to_index(self, action: Action) -> int:
        """精确映射；若找不到返回 -1"""
        return self._act2idx.get(action.normalized().to_tuple(), -1)

    def index_to_action(self, idx: int) -> Action:
        return self._idx2act[idx]

__all__ = ["Action", "ActionSpace"]
