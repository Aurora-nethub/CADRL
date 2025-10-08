# sim/collision.py
import math
from typing import Tuple

def _dist_at_time(px, py, vx, vy, px1, py1, vx1, vy1, t: float) -> float:
    dx = (px + vx * t) - (px1 + vx1 * t)
    dy = (py + vy * t) - (py1 + vy1 * t)
    return math.hypot(dx, dy)

def closest_approach_analytic(
    px, py, vx, vy, px1, py1, vx1, vy1, *, dt: float
) -> Tuple[float, float]:
    """
    返回: (d_center_min, t_star) 其中 t_star ∈ [0, dt]
    假设本步内速度恒定。
    """
    dx, dy = (px - px1), (py - py1)
    dvx, dvy = (vx - vx1), (vy - vy1)
    dv2 = dvx * dvx + dvy * dvy
    if dv2 < 1e-12:
        t_star = 0.0
    else:
        t_star = max(0.0, min((-(dx * dvx + dy * dvy) / dv2), dt))
    cx = dx + dvx * t_star
    cy = dy + dvy * t_star
    dmin = math.hypot(cx, cy)
    return dmin, t_star

def closest_approach_sample3(
    px, py, vx, vy, px1, py1, vx1, vy1, *, dt: float
) -> Tuple[float, float]:
    """
    三点采样近似：t ∈ {0, 0.5*dt, dt}
    返回: (d_center_min, t_star)
    """
    candidates = (0.0, 0.5 * dt, dt)
    dmin = float("inf")
    t_star = dt
    for t in candidates:
        d = _dist_at_time(px, py, vx, vy, px1, py1, vx1, vy1, t)
        if d < dmin:
            dmin = d
            t_star = t
    return dmin, t_star
