from dataclasses import dataclass

@dataclass
class FullState:
    px: float
    py: float
    vx: float
    vy: float
    radius: float
    pgx: float
    pgy: float
    v_pref: float
    theta: float

@dataclass
class ObservableState:
    px: float
    py: float
    vx: float
    vy: float
    radius: float

@dataclass
class JointState:
    # self状态
    px: float
    py: float
    vx: float
    vy: float
    radius: float
    pgx: float
    pgy: float
    v_pref: float
    theta: float

    # neighbor状态
    px1: float
    py1: float
    vx1: float
    vy1: float
    radius1: float

@dataclass
class Velocity:
    vx: float
    vy: float

@dataclass
class Action:
    # v is velocity, under kinematic constraints, r is rotation angle otherwise it's speed direction
    v: float
    r: float

__all__ = ['FullState', 'ObservableState', 'JointState', 'Velocity', 'Action']
