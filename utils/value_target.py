# utils/value_target.py
from typing import List, Tuple, Optional, Callable
import torch

from .trajectory import Trajectory
from sim.reward import RewardSpec, generate_step_rewards_for_trajectory


class PairGenerator:
    """把轨迹转成 (state, value) 对列表。"""
    def compute_pairs(self, traj: Trajectory) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


def _delta_ts(traj: Trajectory, times: Optional[List[float]]) -> List[float]:
    T = len(traj) # pylint: disable=invalid-name
    if T < 2:
        return []
    # 优先使用外部传入 times；否则用 traj.times；再否则 Δt=1
    if times is None:
        if getattr(traj, "times", None) is not None:
            times = traj.times.tolist()
        else:
            return [1.0] * (T - 1)
    return [float(times[k + 1] - times[k]) for k in range(T - 1)]


def _discount(gamma: float, dt: float, v_pref: float) -> float:
    # 论文中的 γ^{Δt * v_pref}
    return float(gamma) ** (max(0.0, float(dt)) * float(v_pref))


# ----------------------------------------------------------------------
# 初始化/模仿阶段：returns-to-go 监督标签（论文公式(6)）
# ----------------------------------------------------------------------
class ArrivalTimeTarget(PairGenerator):
    """
    保留类名以减小改动，但语义为“折扣回报监督信号”（不再用 gamma**time 的伪目标）。
    """
    def __init__(self,
                 gamma: float,
                 reward_spec: Optional[RewardSpec] = None,
                 keep_failed: bool = False,
                 goal_tolerance: Optional[float] = None):
        self.gamma = float(gamma)
        self.reward_spec = reward_spec or RewardSpec()
        self.keep_failed = bool(keep_failed)
        self.goal_tolerance = goal_tolerance

    def compute_pairs(self,
                      traj: Trajectory,
                      times: Optional[List[float]] = None,
                      terminal: Optional[str] = None
                      ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        T = len(traj) # pylint: disable=invalid-name
        if T < 2:
            return []

        # 失败轨迹是否保留
        if terminal is None:
            px = float(traj.positions[-1, 0, 0]); py = float(traj.positions[-1, 0, 1])
            tol = float(self.goal_tolerance) if self.goal_tolerance is not None else float(traj.radius)
            success = ( (px - traj.goal_x) ** 2 + (py - traj.goal_y) ** 2 ) ** 0.5 <= tol
            terminal = "success" if success else "failure"
        if terminal == "failure" and not self.keep_failed:
            return []

        deltas = _delta_ts(traj, times)

        # 与在线规则一致，从 Trajectory 直接生成每步奖励（离线不做过近 shaping）
        rewards = generate_step_rewards_for_trajectory(
            traj,
            spec=self.reward_spec,
            goal_tolerance=self.goal_tolerance,
            terminal_hint=terminal,
            times=times
        )

        # 折扣
        g = [_discount(self.gamma, deltas[k], float(traj.v_pref)) for k in range(T - 1)]

        # 后向累计 returns-to-go
        rtg = [0.0] * (T - 1)
        acc = 0.0
        for k in reversed(range(T - 1)):
            acc = float(rewards[k]) + g[k] * acc
            rtg[k] = acc

        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        # 对每个中间状态 s_i，监督标签为 y_i = RTG[i-1]
        for i in range(1, T):
            s_i = traj.state_at(i)  # (14,)
            target = rtg[i - 1] if (i - 1) < len(rtg) else 0.0
            pairs.append((s_i, torch.tensor([target], dtype=torch.float32)))
        return pairs


# ----------------------------------------------------------------------
# 强化阶段：TD(0) 目标（y = r + γ^{Δt v_pref} V(s')）
# ----------------------------------------------------------------------
class TDTarget(PairGenerator):
    def __init__(self,
                 gamma: float,
                 model_or_fn,
                 device: Optional[torch.device] = None,
                 reward_spec: Optional[RewardSpec] = None,
                 terminal_bootstrap: float = 0.0,
                 penalty_fn: Optional[Callable[[Trajectory, Optional[Trajectory], int, int], float]] = None):
        self.gamma = float(gamma)
        self.model_or_fn = model_or_fn
        self.device = device
        self.reward_spec = reward_spec or RewardSpec()
        self.terminal_bootstrap = float(terminal_bootstrap)
        self.penalty_fn = penalty_fn

    def _predict_v(self, s_batch: torch.Tensor) -> torch.Tensor:
        m = self.model_or_fn
        try:
            out = m(s_batch if self.device is None else s_batch.to(self.device))
        except TypeError:
            out = m(s_batch if self.device is None else s_batch.to(self.device), self.device)
        return out  # (B,1)

    def compute_pairs(self,
                      traj: Trajectory,
                      times: Optional[List[float]] = None,
                      terminals: Optional[List[bool]] = None,
                      terminal: Optional[str] = None,
                      other_traj: Optional[Trajectory] = None
                      ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        T = len(traj) # pylint: disable=invalid-name
        if T < 2:
            return []

        deltas = _delta_ts(traj, times)
        # 与初始化一致的奖励规则（在线时通常由 env.step 返回；此处为对齐与容错）
        rewards = generate_step_rewards_for_trajectory(
            traj,
            spec=self.reward_spec,
            terminal_hint=terminal,
            times=times
        )

        if terminals is None:
            terminals = [False] * (T - 2) + [True]

        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(1, T - 0 - 1):  # 1..T-2
            s_i = traj.state_at(i)
            s_ip1 = traj.state_at(i + 1)

            r_i = float(rewards[i - 1])
            g_i = _discount(self.gamma, deltas[i - 1], float(traj.v_pref))

            if terminals[i - 1]:
                v_next = self.terminal_bootstrap
            else:
                with torch.no_grad():
                    v_next = float(self._predict_v(s_ip1.unsqueeze(0)).squeeze().item())

            y = r_i + g_i * v_next

            if self.penalty_fn is not None:
                y += float(self.penalty_fn(traj, other_traj, i, T))

            pairs.append((s_i, torch.tensor([y], dtype=torch.float32)))
        return pairs


__all__ = ["RewardSpec", "ArrivalTimeTarget", "TDTarget"]

