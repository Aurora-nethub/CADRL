# sim/reward.py
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
import torch

from utils import Action, Trajectory
from utils.state import JointState, unpack_joint
from .dynamics import _self_velocity_from_action, _other_velocity_from_action
from .collision import closest_approach_analytic, closest_approach_sample3


# ---------------------------
# 奖励规格（保留原字段，新增少量注释）
# ---------------------------
@dataclass
class RewardSpec:
    # 初始化阶段（imitation-style）会用到的时间惩罚 + 终端项
    step_time_penalty: float = -0.01  # 每秒(或每步)时间惩罚（仅用于初始化阶段）
    goal_reward: float = 1.0          # 到达终端奖励（两阶段可共用）
    collision_penalty: float = -0.25  # 碰撞终端惩罚（训练阶段用；离线可选）
    failure_penalty: float = 0.0      # 超时/失败（初始化阶段可用；训练阶段通常不用）

    # 训练阶段（经典 CADRL）过近整形阈值（对 gap = dmin - (r0+r1)）
    near_gap_threshold: float = 0.2   # 0 <= gap < 0.2 时的线性惩罚区


# ----------------------------------------------------------------------
# 训练阶段：一步即时奖励（经典 CADRL 口径）
#   - 无“每步时间惩罚”（living cost）
#   - 使用 gap = dmin - (r0+r1) 判定“过近”
#   - 分段：
#       碰撞: -0.25
#       过近: -0.1 - gap/2
#       到达: +1
#       其他: 0
# 返回: (reward, step_ratio)
#   step_ratio 用于标注碰撞等在本步内的发生时刻比例；经典 CADRL 不依赖它做时间惩罚，但保留给外层使用。
# ----------------------------------------------------------------------
def compute_reward_cadrl_train(
    joint: torch.Tensor,
    action_self: Action,
    action_other: Optional[Action],
    *,
    dt: float,
    kinematic: bool,
    collision_mode: str = "analytic",  # "analytic" or "sample3"
    goal_tolerance: Optional[float] = None,
    goal_reward: float = 1.0,
    collision_penalty: float = -0.25,
    near_gap_threshold: float = 0.2,
) -> Tuple[float, float]:
    """
    经典 CADRL 训练阶段的一步即时奖励。
    """
    js: JointState = unpack_joint(joint)
    s0, s1 = js.self_state, js.neighbor_state

    # 由动作得到本步恒定速度
    vx0n, vy0n, _ = _self_velocity_from_action(s0.theta, action_self, kinematic)
    vx1n, vy1n = _other_velocity_from_action(s1.vx, s1.vy, action_other, kinematic)

    # 最近接近（解析或采样）
    if collision_mode == "analytic":
        dmin, t_star = closest_approach_analytic(
            s0.px, s0.py, vx0n, vy0n,
            s1.px, s1.py, vx1n, vy1n, dt=dt
        )
    else:
        dmin, t_star = closest_approach_sample3(
            s0.px, s0.py, vx0n, vy0n,
            s1.px, s1.py, vx1n, vy1n, dt=dt
        )

    step_ratio = (t_star / dt) if (0.0 < t_star < dt and dt > 0) else 1.0
    reward = 0.0  # 经典 CADRL 没有 living cost

    sum_r = s0.radius + s1.radius
    gap = dmin - sum_r

    # 碰撞
    if dmin < sum_r:
        reward += collision_penalty
        return reward, step_ratio

    # 过近整形（线性）
    if 0.0 <= gap < float(near_gap_threshold):
        reward += (-0.1 - gap / 2.0)
        return reward, 1.0

    # 步末到达（注意：经典 CADRL里到达与碰撞/过近的优先顺序实现上通常先判安全性）
    nx = s0.px + vx0n * dt
    ny = s0.py + vy0n * dt
    tol = float(goal_tolerance) if goal_tolerance is not None else float(s0.radius)
    if math.hypot(nx - s0.pgx, ny - s0.pgy) <= tol:
        reward += goal_reward
        return reward, 1.0

    # 其他
    return reward, 1.0


# ----------------------------------------------------------------------
# 初始化阶段：一步“时间型”奖励（用于离线生成标签 / imitation-style）
#   - 与论文思路对齐：更快到达更好；通常不对“过近/碰撞”做 shaping
#   - 我们按 Δt 计负的时间代价 + 末步终端（成功 +1，失败/failure_penalty）
# 返回: (reward, step_ratio) —— 为了接口兼容，step_ratio=1.0（初始化时一般不需要 t*）
# ----------------------------------------------------------------------
def compute_reward_init_time(
    *,
    dt: float,
    step_time_penalty: float = -0.01,
    terminal_event: Optional[str] = None,  # 'success' / 'failure' / None
    goal_reward: float = 1.0,
    failure_penalty: float = 0.0,
) -> Tuple[float, float]:
    """
    初始化阶段的一步“时间型”奖励：
      - 非终端步：reward = step_time_penalty * dt
      - 终端步：   再叠加 success/failure 的终端项
    """
    reward = step_time_penalty * float(max(dt, 0.0))

    if terminal_event == "success":
        reward += goal_reward
    elif terminal_event == "failure":
        reward += failure_penalty

    return reward, 1.0


# ----------------------------------------------------------------------
# 兼容接口：保留原 compute_reward 名称（默认走“训练阶段/经典 CADRL”）
#   - 若你需要显式用初始化口径，请直接调用 compute_reward_init_time 或在
#     generate_step_rewards_for_trajectory(..., phase="init") 中使用。
# ----------------------------------------------------------------------
def compute_reward(
    joint: torch.Tensor,
    action_self: Action,
    action_other: Optional[Action],
    *,
    dt: float,
    kinematic: bool,
    collision_mode: str = "analytic",
    goal_tolerance: Optional[float] = None,
    # 训练期的默认（经典 CADRL）
    goal_reward: float = 1.0,
    collision_penalty: float = -0.25,
) -> Tuple[float, float]:
    """
    为了尽量保持你原先的接口不报错，这里保留 step_time_penalty 形参，但在经典 CADRL 训练里不会使用。
    """
    return compute_reward_cadrl_train(
        joint, action_self, action_other,
        dt=dt, kinematic=kinematic, collision_mode=collision_mode,
        goal_tolerance=goal_tolerance,
        goal_reward=goal_reward, collision_penalty=collision_penalty
    )


# ----------------------------------------------------------------------
# 离线（初始化/训练）从 Trajectory 生成每步奖励序列
#   - phase="init": 使用“时间型”口径（Δt * step_time_penalty + 末步终端）
#   - phase="train": 近似按“经典 CADRL”口径生成（基于离散帧做 gap 判定，
#                    无法做一步内解析 t*，作为离线近似）
#   —— 对外签名尽量与原来一致，仅新增 phase 参数
# ----------------------------------------------------------------------
def generate_step_rewards_for_trajectory(
    traj: Trajectory,
    *,
    spec: Optional[RewardSpec] = None,
    goal_tolerance: Optional[float] = None,
    terminal_hint: Optional[str] = None,  # 'success' / 'failure' / 'collision' / 'timeout'
    times: Optional[List[float]] = None,  # 默认用 traj.times
    phase: str = "init",                  # 'init' or 'train'
    # —— 可选：若提供该回调，train 分支将用“在线同口径”的单步奖励 —— #
    per_step_inputs: Optional[
        Callable[[int], Tuple[torch.Tensor, Action, Optional[Action]]]
    ] = None,
    dt_provider: Optional[Callable[[int], float]] = None,
    kinematic: bool = True,
    collision_mode: str = "analytic",
) -> List[float]:
    """
    返回长度 (T-1) 的每步奖励列表。

    - phase='init'  ：Δt * step_time_penalty + 末步终端（逐步调用 compute_reward_init_time）
    - phase='train' ：
        * 若 per_step_inputs 提供：逐步调用 compute_reward_cadrl_train（与在线一致）
        * 否则：使用离散近似（帧间中心距估 gap），三段式 CADRL 奖励
    """
    spec = spec or RewardSpec()

    T = len(traj) # pylint: disable=invalid-name
    if T < 2:
        return []

    # 时间戳与 Δt
    if times is None:
        if getattr(traj, "times", None) is not None:
            times = traj.times.tolist()
        else:
            times = list(range(T))
    dts = [float(times[i+1] - times[i]) for i in range(T-1)]
    dts = [dt if dt > 0 else 0.0 for dt in dts]

    # 终端事件（用于 init 末步；train 分支可作为参考）
    term = terminal_hint
    if term is None:
        px = float(traj.positions[-1, 0, 0])
        py = float(traj.positions[-1, 0, 1])
        tol = float(goal_tolerance) if goal_tolerance is not None else float(traj.radius)
        term = "success" if (math.hypot(px - traj.goal_x, py - traj.goal_y) <= tol) else "failure"

    rewards: List[float] = [0.0 for _ in range(T - 1)]

    # =========================== INIT 分支（解耦：显式调用） ===========================
    if phase.lower() == "init":
        for i in range(T - 1):
            terminal_event = None
            if i == T - 2:
                # 将 collision/timeout 统一视为 failure（也可自定义映射）
                if term in ("success", "failure"):
                    terminal_event = term
                elif term in ("collision", "timeout"):
                    terminal_event = "failure"

            # 逐步调用时间型奖励
            r_i, _ = compute_reward_init_time(
                dt=(dts[i] if dt_provider is None else float(dt_provider(i))),
                step_time_penalty=spec.step_time_penalty,
                terminal_event=terminal_event,
                goal_reward=spec.goal_reward,
                failure_penalty=spec.failure_penalty,
            )
            rewards[i] = r_i
        return rewards

    # =========================== TRAIN 分支 ===========================
    # A) 若提供 per_step_inputs：逐步调用“在线同口径”奖励（强一致）
    if per_step_inputs is not None:
        for i in range(T - 1):
            joint_i, a_self_i, a_other_i = per_step_inputs(i)
            dt_i = (dts[i] if dt_provider is None else float(dt_provider(i)))
            r_i, _ = compute_reward_cadrl_train(
                joint_i, a_self_i, a_other_i,
                dt=dt_i, kinematic=kinematic, collision_mode=collision_mode,
                goal_tolerance=goal_tolerance,
                goal_reward=spec.goal_reward,
                collision_penalty=spec.collision_penalty,
                near_gap_threshold=spec.near_gap_threshold,
            )
            rewards[i] = float(r_i)
        return rewards

    # B) 否则：离散近似 CADRL 三段式（默认行为）
    # 默认假设 self/other 半径相同；若不同需按你的 Trajectory 定义修改
    sum_r = float(traj.radius * 2.0)
    near_thr = float(spec.near_gap_threshold)

    def reached_at(i: int) -> bool:
        px = float(traj.positions[i+1, 0, 0])
        py = float(traj.positions[i+1, 0, 1])
        tol = float(goal_tolerance) if goal_tolerance is not None else float(traj.radius)
        return math.hypot(px - traj.goal_x, py - traj.goal_y) <= tol

    for i in range(T - 1):
        p0x, p0y = float(traj.positions[i+1, 0, 0]), float(traj.positions[i+1, 0, 1])
        p1x, p1y = float(traj.positions[i+1, 1, 0]), float(traj.positions[i+1, 1, 1])
        center_dist = math.hypot(p0x - p1x, p0y - p1y)
        gap = center_dist - sum_r

        r = 0.0
        if center_dist < sum_r:
            r += spec.collision_penalty
        elif 0.0 <= gap < near_thr:
            r += (-0.1 - gap / 2.0)
        elif reached_at(i):
            r += spec.goal_reward
        rewards[i] = r

    return rewards
