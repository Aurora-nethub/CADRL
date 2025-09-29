from typing import Any, Dict, List

from torch.utils.data import Dataset

class ReplayMemory(Dataset):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: List[Dict[str, Any]] = []
        self.position = 0
