# sim/reward.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch

from utils import Action, Trajectory
from utils.state import JointState, unpack_joint
from .dynamics import _self_velocity_from_action, _other_velocity_from_action
from .collision import closest_approach_analytic, closest_approach_sample3


# ---------------------------
# 奖励规格（初始化与强化统一使用）
# ---------------------------
@dataclass
class RewardSpec:
    step_time_penalty: float = -0.01  # 每步时间惩罚（living cost）
    goal_reward: float = 1.0          # 到达终端奖励
    collision_penalty: float = -0.25  # 碰撞终端惩罚
    failure_penalty: float = 0.0      # 超时/失败（离线标注时使用）



# ----------------------------------------------------------------------
# 在线环境：一步即时奖励
# ----------------------------------------------------------------------
def compute_reward(
    joint: torch.Tensor,
    action_self: Action,
    action_other: Optional[Action],
    *,
    dt: float,
    kinematic: bool,
    collision_mode: str = "analytic",  # "analytic" or "sample3"
    goal_tolerance: Optional[float] = None,
    # 可改为从 config 读取，这里给默认值
    step_time_penalty: float = -0.01,
    goal_reward: float = 1.0,
    collision_penalty: float = -0.25,
) -> Tuple[float, float]:
    """
    返回: (reward, step_ratio)
      - 若本步内碰撞，reward 含 collision_penalty，step_ratio = t*/dt ∈ (0,1]
      - 否则 step_ratio = 1.0
    """
    js: JointState = unpack_joint(joint)
    s0, s1 = js.self_state, js.neighbor_state

    # 由动作得到本步恒定速度（差速/全向）= vx0n, vy0n, _theta_next
    vx0n, vy0n, _ = _self_velocity_from_action(s0.theta, action_self, kinematic)
    # 关键：把 kinematic 传入对手侧
    vx1n, vy1n = _other_velocity_from_action(s1.vx, s1.vy, action_other, kinematic)

    # 最近接近（解析或采样）
    if collision_mode == "analytic":
        dmin, t_star = closest_approach_analytic(s0.px, s0.py, vx0n, vy0n,
                                                 s1.px, s1.py, vx1n, vy1n, dt=dt)
    else:
        dmin, t_star = closest_approach_sample3(s0.px, s0.py, vx0n, vy0n,
                                                s1.px, s1.py, vx1n, vy1n, dt=dt)

    step_ratio = (t_star / dt) if (0.0 < t_star < dt and dt > 0) else 1.0
    reward = step_time_penalty * step_ratio  # 每步时间惩罚（按有效推进比例）

    sum_r = s0.radius + s1.radius
    # 碰撞（本步内）
    if dmin < sum_r:
        reward += collision_penalty
        return reward, step_ratio

    # （可选）过近 shaping：若你不需要可删除
    if dmin < (sum_r + 0.2):
        reward += (-0.1 - dmin / 2.0)
        return reward, 1.0

    # 步末到达
    nx = s0.px + vx0n * dt
    ny = s0.py + vy0n * dt
    tol = float(goal_tolerance) if goal_tolerance is not None else float(s0.radius)
    if math.hypot(nx - s0.pgx, ny - s0.pgy) <= tol:
        reward += goal_reward
        return reward, 1.0

    # 普通过渡
    return reward, 1.0


# ----------------------------------------------------------------------
# 离线（初始化/模仿）：从 Trajectory 生成每步奖励序列
#   —— 对外签名只接收 traj（不含 v_pref 参数）
# ----------------------------------------------------------------------
def generate_step_rewards_for_trajectory(
    traj: Trajectory,
    *,
    spec: Optional[RewardSpec] = None,
    goal_tolerance: Optional[float] = None,
    terminal_hint: Optional[str] = None,  # 'success' / 'failure'
    times: Optional[List[float]] = None,  # 可用外部 times 覆写；默认用 traj.times
) -> List[float]:
    """
    每步奖励 = step_time_penalty（按“步”计） + 终端奖励(最后一步叠加)。
    注：离线阶段不复现“过近 shaping”（在线一步前瞻更合适），避免几何重放不一致。
    """
    spec = spec or RewardSpec()

    T = len(traj) # pylint: disable=invalid-name
    if T < 2:
        return []

    # 时间序列（只用于一致性；此处按“步”计惩罚，不乘 dt）
    if times is None:
        if getattr(traj, "times", None) is not None:
            times = traj.times.tolist()
        else:
            times = list(range(T))

    rewards = [spec.step_time_penalty for _ in range(T - 1)]

    # 终端奖励（最后一步）
    if terminal_hint is None:
        px = float(traj.positions[-1, 0, 0])
        py = float(traj.positions[-1, 0, 1])
        tol = float(goal_tolerance) if goal_tolerance is not None else float(traj.radius)
        success = math.hypot(px - traj.goal_x, py - traj.goal_y) <= tol
        terminal_hint = "success" if success else "failure"

    if terminal_hint == "success":
        rewards[-1] += spec.goal_reward
    elif terminal_hint == "failure":
        rewards[-1] += spec.failure_penalty

    return rewards
