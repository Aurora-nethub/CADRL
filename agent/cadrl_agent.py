# agent/cadrl_agent.py
import random
from typing import List, Optional, Union

import torch

from utils import Action, ActionSpace, JointState


class CADRLAgent:
    """
    CADRL 决策智能体（只负责“选动作”）：
      - 行为策略：ε-greedy
      - 打分：score(a) = immediate_reward(s, a) + gamma_eff * V(s')
      - 前瞻：通过 env.peek(my_idx, a, other_action=None) 统一复用环境的动力学/奖励

    说明
    ----
    1) gamma_eff:
       - 若 discount_with_vpref=True，则 gamma_eff = gamma ** v_pref（与开源仓库口径一致）
       - 否则 gamma_eff = gamma（固定折扣）
    2) 值网络前向：
       - 兼容两种签名：forward(x) 或 forward(x, device)
       - 统一批量评估，避免多次小张量前向导致的性能抖动
    3) 输入状态：
       - 可以是 JointState 对象或 (14,) CPU Tensor（世界坐标系）
       - 值网络内部若有“坐标对齐/旋转”，由网络自行处理；本类不做旋转
    """

    def __init__(
        self,
        value_net,
        action_space: ActionSpace,
        device: torch.device,
        gamma: float,
        *,
        discount_with_vpref: bool = False,
        default_epsilon: float = 0.1,
    ) -> None:
        self.value_net = value_net
        self.action_space = action_space
        self.device = device
        self.gamma = float(gamma)
        self.discount_with_vpref = bool(discount_with_vpref)
        self.default_epsilon = float(default_epsilon)

    # ---------------- 公有接口 ----------------

    def act(
        self,
        joint_state: Union[JointState, torch.Tensor],
        env,
        my_idx: int,
        *,
        mode: str = "train",
        epsilon: Optional[float] = None,
    ) -> Action:
        """
        选择一个动作。

        参数
        ----
        joint_state : JointState | (14,) Tensor
            当前联合状态（世界系；如果是 JointState 将自动转成 tensor）
        env : CADRLEnv
            环境实例（用于调用 env.peek 做统一的前瞻与奖励计算）
        my_idx : int
            当前智能体在环境中的索引（0 或 1）
        mode : {"train","eval"}
            训练模式可用 ε 探索；评估模式纯贪婪
        epsilon : Optional[float]
            覆盖默认 ε；仅在 mode="train" 时生效

        返回
        ----
        Action
            选中的动作
        """
        if isinstance(joint_state, JointState):
            s14 = joint_state.to_tensor(as_batch=False)  # CPU (14,)
        else:
            s14 = joint_state
            assert s14.dim() == 1 and s14.numel() == 14, "joint_state tensor must be shape (14,)"

        if mode == "train":
            eps = self.default_epsilon if (epsilon is None) else float(epsilon)
            if random.random() < eps:
                return self.action_space.sample()

        # 贪婪动作（批量前瞻 + 批量 V 估计）
        actions = self.action_space.get_actions()
        next_states: List[torch.Tensor] = []
        rewards: List[float] = []

        # 对每个候选动作，使用 env.peek 无副作用地得到 (s', r, end_ratio)
        for a in actions:
            sn, r, _ = env.peek(my_idx, a, other_action=None)
            next_states.append(sn)     # (14,) CPU
            rewards.append(float(r))

        # 批量送入值网络
        ns_batch = torch.stack(next_states, dim=0)  # (N,14) CPU
        with torch.no_grad():
            v = self._predict_v(ns_batch)           # (N,1) on device
            v = v.squeeze(1).cpu().tolist()         # List[float]

        # 折扣（按需与 v_pref 绑定）
        gamma_eff = self._gamma_eff(s14)

        # 评分并选择最优
        scores = [r + gamma_eff * vn for r, vn in zip(rewards, v)]
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return actions[best_idx]

    # ---------------- 内部工具 ----------------

    def _gamma_eff(self, s14: torch.Tensor) -> float:
        if self.discount_with_vpref:
            v_pref = float(s14[7].item())
            return self.gamma ** v_pref
        return self.gamma

    def _predict_v(self, s_batch: torch.Tensor) -> torch.Tensor:
        """
        兼容值网络两种前向签名：
          - forward(x) -> (B,1)
          - forward(x, device) -> (B,1)
        """
        try:
            out = self.value_net(s_batch.to(self.device))
        except TypeError:
            out = self.value_net(s_batch.to(self.device), self.device)
        return out  # (B,1)

