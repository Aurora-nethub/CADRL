# utils/state.py
from dataclasses import dataclass, fields
import torch

@dataclass
class BaseState:
    px: float
    py: float
    vx: float
    vy: float
    radius: float

    def to_tensor(self, as_batch: bool = False) -> torch.Tensor:
        """
        默认返回 CPU float32 tensor；需要 GPU 时训练阶段自行 to(device)。
        """
        values = [getattr(self, f.name) for f in fields(self)]
        t = torch.tensor(values, dtype=torch.float32)  # 默认 device=cpu
        return t.unsqueeze(0) if as_batch else t

@dataclass
class FullState(BaseState):
    pgx: float
    pgy: float
    v_pref: float
    theta: float

@dataclass
class ObservableState(BaseState):
    pass

@dataclass
class JointState:
    self_state: FullState
    neighbor_state: BaseState

    def to_tensor(self, as_batch: bool = False) -> torch.Tensor:
        """Concatenate self and neighbor tensors into one tensor."""
        return torch.cat([
            self.self_state.to_tensor(as_batch),
            self.neighbor_state.to_tensor(as_batch),
        ])

    @classmethod
    def from_tensor(cls, joint: torch.Tensor) -> 'JointState':
        t = joint.detach().flatten().to(dtype=torch.float32, copy=False).cpu()
        if t.numel() != 14:
            raise ValueError(f"JointState.from_tensor expects 14-D, got {t.numel()}")
        s0 = FullState(px=float(t[0]), py=float(t[1]), vx=float(t[2]), vy=float(t[3]), radius=float(t[4]),
                       pgx=float(t[5]), pgy=float(t[6]), v_pref=float(t[7]), theta=float(t[8]))
        s1 = BaseState(px=float(t[9]), py=float(t[10]), vx=float(t[11]), vy=float(t[12]), radius=float(t[13]))
        return cls(self_state=s0, neighbor_state=s1)


def unpack_joint(joint: torch.Tensor) -> JointState:
    return JointState.from_tensor(joint)


__all__ = ['FullState', 'ObservableState', 'JointState', 'unpack_joint']
