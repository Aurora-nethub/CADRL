# sim/reward.py
import math
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass

import torch

from utils.state import JointState, unpack_joint
from utils.action import Action
from utils.trajectory import Trajectory
from sim.collision import closest_approach_analytic, closest_approach_sample3


@dataclass
class RewardSpec:
    """奖励函数超参数规格"""
    goal_reward: float = 1.0
    collision_penalty: float = -0.25
    failure_penalty: float = 0.0
    step_time_penalty: float = -0.01
    near_gap_threshold: float = 0.2


def _self_velocity_from_action(theta: float, action: Action, kinematic: bool) -> Tuple[float, float, float]:
    """将动作转换为自车速度（世界坐标系）"""
    if kinematic:
        vx = action.velocity * math.cos(theta + action.rotation)
        vy = action.velocity * math.sin(theta + action.rotation)
        new_theta = theta + action.rotation
    else:
        vx = action.velocity * math.cos(action.rotation)
        vy = action.velocity * math.sin(action.rotation)
        new_theta = action.rotation
    return vx, vy, new_theta


def _other_velocity_from_action(vx_old: float, vy_old: float, action: Optional[Action], kinematic: bool) -> Tuple[float, float]:
    """对手动作转速度（若为 None 则保持当前速度）"""
    if action is None:
        return vx_old, vy_old
    if kinematic:
        theta_old = math.atan2(vy_old, vx_old)
        return (
            action.velocity * math.cos(theta_old + action.rotation),
            action.velocity * math.sin(theta_old + action.rotation)
        )
    else:
        return (
            action.velocity * math.cos(action.rotation),
            action.velocity * math.sin(action.rotation)
        )


def compute_reward_cadrl_train(
    joint: torch.Tensor,
    action_self: Action,
    action_other: Optional[Action],
    *,
    dt: float,
    kinematic: bool,
    collision_mode: str = "analytic",
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

    # 1. living cost参数
    step_time_penalty = -0.008  # 可调参数
    safe_dt = max(1e-6, min(dt, 1.0))

    # 2. 由动作得到本步恒定速度（只需要自车的速度）
    vx0n, vy0n, _ = _self_velocity_from_action(s0.theta, action_self, kinematic)

    # Keep unused params referenced for API stability and to silence linters
    # (these are intentionally part of the signature for compatibility).
    _ = action_other
    _ = collision_mode
    _ = near_gap_threshold

    # 3. 计算当前真实距离（环境层：用于判定碰撞和 done）
    current_dist = math.hypot(s0.px - s1.px, s0.py - s1.py)
    sum_r = s0.radius + s1.radius
    eps_c = 1e-8 * float(max(1.0, sum_r))

    # 4. 预测下一步位置，判定到达
    nx = s0.px + vx0n * safe_dt
    ny = s0.py + vy0n * safe_dt
    tol = float(goal_tolerance) if goal_tolerance is not None else float(s0.radius)
    arrived = math.hypot(nx - s0.pgx, ny - s0.pgy) <= tol
    
    # 5. 判定状态：完全基于当前真实几何（不用 lookahead）
    collided = current_dist <= (sum_r + eps_c)

    # 6. 每步都扣 living cost（时间代价）
    reward = step_time_penalty * safe_dt
    
    # 7. 只在终止步叠加终止奖励（collision/goal 为附加项）
    if collided:
        reward += collision_penalty
    elif arrived:
        reward += goal_reward
    # 注意：near 奖励已移除，环境层不使用 lookahead
    # near 的判断应该在策略层（agent 选动作时）通过前瞻处理

    # 8. 安全检查：确保reward在合理范围内
    if reward < -1.0 or reward > 1.0:
        sep = "=" * 70
        error_msg = (
            f"\n{sep}\n"
            "⚠️ REWARD OUT OF RANGE ⚠️\n"
            f"{sep}\n"
            f"reward = {reward:.6f}\n"
            f"current_dist = {current_dist:.4f}\n"
            f"sum_r = {sum_r:.4f}\n"
            f"collision_penalty = {collision_penalty}\n"
            f"goal_reward = {goal_reward}\n"
            f"step_time_penalty = {step_time_penalty}\n"
            f"safe_dt = {safe_dt}\n"
            f"{sep}\n"
        )
        print(error_msg, flush=True)
        reward = max(-1.0, min(1.0, reward))

    # 9. 返回 reward 和 step_ratio（环境层：总是完整步长）
    return reward, 1.0
#   - 我们按 Δt 计负的时间代价 + 末步终端（成功 +1，失败/failure_penalty）
# 返回: (reward, step_ratio) —— 为了接口兼容，step_ratio=1.0（初始化时一般不需要 t*）
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 策略层专用：lookahead 前瞻（用于动作选择时评估候选动作）
# ----------------------------------------------------------------------
def compute_lookahead_score(
    joint: torch.Tensor,
    action_self: Action,
    action_other: Optional[Action],
    *,
    kinematic: bool,
    collision_mode: str = "analytic",
    lookahead_time: float = 1.0,
    lookahead_steps: int = 10,  # pylint: disable=unused-argument
    near_penalty: float = -0.1,
    near_gap_threshold: float = 0.2,
) -> float:
    """
    策略层的前瞻评分（不是 reward，是动作候选的附加评分）。
    
    用途：在 agent 选择动作时，对每个候选动作做短时前瞻（例如 1s），
         计算该动作下与邻居的最小分离距离，避免"逼近"或"碰撞"。
    
    返回：附加评分（负值表示惩罚）
         - 如果前瞻期间会碰撞或接近，返回负值
         - 如果安全，返回 0
    """
    js: JointState = unpack_joint(joint)
    s0, s1 = js.self_state, js.neighbor_state
    
    # 由动作得到速度
    vx0n, vy0n, _ = _self_velocity_from_action(s0.theta, action_self, kinematic)
    vx1n, vy1n = _other_velocity_from_action(s1.vx, s1.vy, action_other, kinematic)
    
    # 计算前瞻窗口内的最小距离
    if collision_mode == "analytic":
        dmin, _ = closest_approach_analytic(
            s0.px, s0.py, vx0n, vy0n,
            s1.px, s1.py, vx1n, vy1n, dt=lookahead_time
        )
    else:
        dmin, _ = closest_approach_sample3(
            s0.px, s0.py, vx0n, vy0n,
            s1.px, s1.py, vx1n, vy1n, dt=lookahead_time
        )
    
    # 计算 gap
    sum_r = s0.radius + s1.radius
    gap = dmin - sum_r
    
    # 评分逻辑：接近时给予负分
    if gap < 0:
        # 会碰撞
        return -0.25  # 与 collision_penalty 对应
    elif 0.0 <= gap < near_gap_threshold:
        # 接近但未碰撞
        return near_penalty - gap / 2.0
    else:
        # 安全
        # reference unused parameter to avoid linter warnings
        _ = lookahead_steps
        return 0.0


# ----------------------------------------------------------------------
# 初始化阶段的时间型奖励（保持不变）
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
