# agent/cadrl_agent.py
import random
from typing import List, Optional, Union

import torch

from utils import Action, ActionSpace, JointState
from sim.reward import compute_lookahead_score


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
                # 修正：排除零速/低速动作（velocity < 0.1）
                all_actions = self.action_space.get_actions()
                non_zero_actions = [a for a in all_actions if abs(a.velocity) > 0.1]
                if non_zero_actions:
                    return random.choice(non_zero_actions)
                else:
                    return self.action_space.sample()

        # 贪婪动作（批量前瞻 + 批量 V 估计）
        # 修正：25预设动作 + 10随机动作（符合论文）
        preset_actions = self.action_space.get_actions()[:25]  # 前25个预设
        random_actions = [self.action_space.sample() for _ in range(10)]
        actions = preset_actions + random_actions

        next_states: List[torch.Tensor] = []
        rewards: List[float] = []
        lookahead_scores: List[float] = []

        # 对每个候选动作：
        # 1. 使用 env.peek 得到一步后的 (s', r) - 环境层的即时 reward
        # 2. 使用 compute_lookahead_score 做前瞻评估 - 策略层的前瞻评分
        for a in actions:
            sn, r, _ = env.peek(my_idx, a, other_action=None)
            next_states.append(sn)     # (14,) CPU
            rewards.append(float(r))
            
            # 策略层前瞻：评估该动作在 1s 窗口内的安全性
            lh_score = compute_lookahead_score(
                s14, a, action_other=None,
                kinematic=getattr(env, "kinematic", True),
                collision_mode=getattr(env, "collision_mode", "analytic"),
                lookahead_time=1.0,
                lookahead_steps=10,
                near_penalty=-0.1,
                near_gap_threshold=0.2,
            )
            lookahead_scores.append(float(lh_score))

        # 批量送入值网络
        ns_batch = torch.stack(next_states, dim=0)  # (N,14) CPU
        with torch.no_grad():
            v = self._predict_v(ns_batch)           # (N,1) on device
            v = v.squeeze(1).cpu().tolist()         # List[float]

        # 折扣（按需与 v_pref 绑定）
        gamma_eff = self._gamma_eff(s14)

        # 评分并选择最优
        # score = 环境层即时reward + 前瞻评分 + 折扣未来价值
        scores = [r + lh + gamma_eff * vn for r, lh, vn in zip(rewards, lookahead_scores, v)]
        max_score = max(scores)

        # 修正：平手时随机打破（论文要求）
        best_indices = [i for i, s in enumerate(scores) if abs(s - max_score) < 1e-6]
        best_idx = random.choice(best_indices)
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

