from dataclasses import dataclass, fields
import torch

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

    def to_tensor(self) -> torch.Tensor:
        """Convert all fields to a PyTorch tensor in declaration order."""
        # Collect all field values in the order they appear in the class
        values = [getattr(self, field.name) for field in fields(self)]
        return torch.tensor(values, dtype=torch.float32)

@dataclass
class Velocity:
    vx: float
    vy: float


__all__ = ['FullState', 'ObservableState', 'JointState', 'Velocity']
