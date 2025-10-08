# sim/dynamics.py
import math
from typing import Optional, Tuple
import torch

from utils import Action

def _infer_heading(vx: float, vy: float) -> float:
    # 当对手没给出朝向时，用速度方向近似；速度极小则视作 0
    if abs(vx) + abs(vy) < 1e-8:
        return 0.0
    return math.atan2(vy, vx)

def _self_velocity_from_action(theta: float, action: Action, kinematic: bool) -> Tuple[float, float, float]:
    """返回 (vx, vy, theta_next)"""
    if kinematic:
        theta_next = theta + float(action.rotation)
        vx = float(action.velocity) * math.cos(theta_next)
        vy = float(action.velocity) * math.sin(theta_next)
    else:
        theta_next = 0.0  # 全向模型不跟踪朝向
        vx = float(action.velocity) * math.cos(float(action.rotation))
        vy = float(action.velocity) * math.sin(float(action.rotation))
    return vx, vy, theta_next

def _other_velocity_from_action(
        vx1: float, vy1: float,
        action_other: Optional[Action],
        kinematic: bool
    ) -> Tuple[float, float]:
    """对手：若无动作，保持当前速度；若有动作且为差速，则以当前速度方向为基准增量旋转。"""
    if action_other is None:
        return float(vx1), float(vy1)
    if kinematic:
        # base heading should be atan2(vy1, vx1) -> _infer_heading(vx, vy) expects (vx, vy)
        base_heading = _infer_heading(float(vx1), float(vy1))
        heading = base_heading + float(action_other.rotation)
        v = float(action_other.velocity)
        return v * math.cos(heading), v * math.sin(heading)
    # 全向
    return (float(action_other.velocity) * math.cos(float(action_other.rotation)),
            float(action_other.velocity) * math.sin(float(action_other.rotation)))

def rollout_state(
    joint: torch.Tensor,
    action_self: Action,
    action_other: Optional[Action],
    *,
    dt: float,
    kinematic: bool,
) -> torch.Tensor:
    """
    输入:
      joint: (14,)  当前联合状态 [px,py,vx,vy,r, pgx,pgy,v_pref,theta,  px1,py1,vx1,vy1,r1]
      action_self:   当前体动作
      action_other:  另一体动作（None 表示保持原速度）
    输出:
      next_joint: (14,)  dt 后的联合状态；张量保持在 CPU。
    """
    assert joint.numel() == 14 and joint.dim() == 1, "joint must be (14,) tensor"
    # 读取
    px, py, _, _, r = [float(joint[i].item()) for i in range(5)] # px, py, vx, vy, r
    pgx, pgy = float(joint[5].item()), float(joint[6].item())
    v_pref, theta = float(joint[7].item()), float(joint[8].item())
    px1, py1, vx1, vy1, r1 = [float(joint[i].item()) for i in range(9, 14)]

    # 由动作得到恒定速度
    vx0n, vy0n, theta_next = _self_velocity_from_action(theta, action_self, kinematic)
    vx1n, vy1n = _other_velocity_from_action(vx1, vy1, action_other, kinematic)

    # 位置推进
    pxn = px + vx0n * dt
    pyn = py + vy0n * dt
    px1n = px1 + vx1n * dt
    py1n = py1 + vy1n * dt

    # 拼装
    out = torch.tensor([
        pxn, pyn, vx0n, vy0n, r,
        pgx, pgy, v_pref, (theta_next if kinematic else 0.0),
        px1n, py1n, vx1n, vy1n, r1
    ], dtype=torch.float32)
    return out
