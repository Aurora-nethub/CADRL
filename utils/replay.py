# utils/replay.py
from __future__ import annotations
from typing import Dict, Optional
import threading
import torch


class ReplayMemory:
    """
    环形经验回放（CPU 常驻）
    - 预分配 CPU 张量，避免频繁分配
    - 不做设备迁移（由调用方决定 .to(device)）
    - sample() 支持传入 torch.Generator，便于多卡可复现
    """
    def __init__(self, capacity: int, state_dim: int = 14) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)

        self.s      = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.r      = torch.zeros((capacity, 1),       dtype=torch.float32)
        self.s_next = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.done   = torch.zeros((capacity, 1),       dtype=torch.bool)
        self.a_idx  = torch.full((capacity,), -1,      dtype=torch.long)

        self.position = 0
        self.size = 0
        self._lock = threading.Lock()  # 如有多线程采样/写入时更安全

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        with self._lock:
            self.position = 0
            self.size = 0
            # 数据是否清零看需求；通常仅需重置有效标记即可
            self.done.zero_()
            self.a_idx.fill_(-1)

    def push(
        self,
        s: torch.Tensor,
        r: float,
        s_next: torch.Tensor,
        done: bool,
        a_idx: Optional[int] = None,
    ) -> None:
        """写入单条经验；输入张量会被复制到 CPU 缓冲。"""
        s_cpu = torch.as_tensor(s, dtype=torch.float32, device="cpu").flatten()
        sn_cpu = torch.as_tensor(s_next, dtype=torch.float32, device="cpu").flatten()
        if s_cpu.numel() != self.state_dim or sn_cpu.numel() != self.state_dim:
            raise ValueError(f"state dim mismatch: expect {self.state_dim}, got {s_cpu.numel()} and {sn_cpu.numel()}")

        with self._lock:
            i = self.position
            self.s[i]      = s_cpu
            self.r[i, 0]   = float(r)
            self.s_next[i] = sn_cpu
            self.done[i, 0]= bool(done)
            self.a_idx[i]  = -1 if a_idx is None else int(a_idx)

            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, *, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """
        随机采样一个 batch（CPU 张量）：
          - s:      (B, state_dim) float32
          - r:      (B, 1)         float32
          - s_next: (B, state_dim) float32
          - done:   (B, 1)         bool
          - a_idx:  (B,)           long   （-1 表示无效）
          - a_mask: (B,)           bool   （True=有有效动作索引）
        """
        if self.size == 0:
            raise RuntimeError("ReplayMemory is empty.")

        b = min(int(batch_size), self.size)
        # 关键：使用可注入的 generator，保障多卡/多进程的可复现 & 去相关
        idx = torch.randperm(self.size, generator=generator)[:b]

        # index_select + clone，避免外部不小心原地修改底层缓冲
        s      = self.s.index_select(0, idx).clone()
        r      = self.r.index_select(0, idx).clone()
        s_next = self.s_next.index_select(0, idx).clone()
        done   = self.done.index_select(0, idx).clone()
        a_idx  = self.a_idx.index_select(0, idx).clone()
        a_mask = a_idx >= 0

        return {"s": s, "r": r, "s_next": s_next, "done": done, "a_idx": a_idx, "a_mask": a_mask}

    # --------- 可选：持久化，便于断点续训 ---------

    def save(self, path: str) -> None:
        torch.save({
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "position": self.position,
            "size": self.size,
            "s": self.s,
            "r": self.r,
            "s_next": self.s_next,
            "done": self.done,
            "a_idx": self.a_idx,
        }, path)

    @classmethod
    def load(cls, path: str) -> "ReplayMemory":
        data = torch.load(path, map_location="cpu")
        mem = cls(capacity=int(data["capacity"]), state_dim=int(data.get("state_dim", 14)))
        mem.position = int(data["position"])
        mem.size = int(data["size"])
        mem.s.copy_(data["s"])
        mem.r.copy_(data["r"])
        mem.s_next.copy_(data["s_next"])
        mem.done.copy_(data["done"])
        mem.a_idx.copy_(data["a_idx"])
        return mem


__all__ = ["ReplayMemory"]
