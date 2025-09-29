from typing import Any, Dict, List

from torch.utils.data import Dataset

class ReplayMemory(Dataset):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: List[Dict[str, Any]] = []
        self.position = 0

    def push(self, **kwargs: Any) -> None:
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append({})
        self.memory[self.position] = kwargs
        self.position = (self.position + 1) % self.capacity

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)
