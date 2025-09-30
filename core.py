from dataclasses import dataclass, field
from typing import Any, Dict

import json

import torch

@dataclass
class AgentConfig:
    """代理相关配置"""
    radius: float = 0.3
    v_pref: float = 1.0
    kinematic: bool = True

@dataclass
class SimulationConfig:
    """模拟环境配置"""
    agent_num: int = 2
    crossing_radius: float = 2.0
    xmin: float = -8.0
    xmax: float = 8.0
    ymin: float = -8.0
    ymax: float = 8.0
    max_time: float = 100.0

@dataclass
class VisualizationConfig:
    """可视化配置"""
    xmin: float = -9.0
    xmax: float = 9.0
    ymin: float = -9.0
    ymax: float = 9.0

@dataclass
class ModelConfig:
    """模型配置"""
    state_dim: int = 14
    gamma: float = 0.8

@dataclass
class InitConfig:
    """初始化配置"""
    traj_dir: str = "data/multi_sim"
    num_epochs: int = 250

@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 100
    learning_rate: float = 0.01
    step_size: int = 150
    train_episodes: int = 200
    sample_episodes: int = 10
    test_interval: int = 10
    test_episodes: int = 100
    capacity: int = 40000
    epsilon_start: float = 0.5
    epsilon_end: float = 0.1
    epsilon_decay: int = 150
    num_epochs: int = 30
    checkpoint_interval: int = 30

@dataclass
class ConfigContainer:
    """统一配置容器"""
    agent: AgentConfig = AgentConfig()
    sim: SimulationConfig = SimulationConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    model: ModelConfig = ModelConfig()
    init: InitConfig = InitConfig()
    train: TrainConfig = TrainConfig()
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConfigContainer":
        """从字典创建配置容器"""
        return cls(
            agent=AgentConfig(**config_dict.get("agent", {})),
            sim=SimulationConfig(**config_dict.get("sim", {})),
            visualization=VisualizationConfig(**config_dict.get("visualization", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            init=InitConfig(**config_dict.get("init", {})),
            train=TrainConfig(**config_dict.get("train", {})),
        )

    def to_dict(self) -> Dict:
        """将配置容器转换为字典"""
        return {
            "agent": self.agent.__dict__,
            "sim": self.sim.__dict__,
            "visualization": self.visualization.__dict__,
            "model": self.model.__dict__,
            "init": self.init.__dict__,
            "train": self.train.__dict__,
        }

    def update(self, config_dict: dict):
        """使用字典更新配置"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

    def __str__(self) -> str:
        """返回格式化的配置字符串"""
        sections = [
            f"[agent]\n{self._format_section(self.agent)}",
            f"[sim]\n{self._format_section(self.sim)}",
            f"[visualization]\n{self._format_section(self.visualization)}",
            f"[model]\n{self._format_section(self.model)}",
            f"[init]\n{self._format_section(self.init)}",
            f"[train]\n{self._format_section(self.train)}",
            f"[device]\n{self.device}",
        ]
        return "\n\n".join(sections)

    def _format_section(self, section_obj) -> str:
        """格式化单个配置部分"""
        return "\n".join(f"{k} = {v}" for k, v in section_obj.__dict__.items())


def _load_config(
    env_config_path: str="config/env.json",
    model_config_path: str="config/model.json",
) -> ConfigContainer:
    """加载配置文件并返回配置容器"""
    with open(env_config_path, "r",encoding="utf-8") as f:
        env_config = json.load(f)
    with open(model_config_path, "r",encoding="utf-8") as f:
        model_config = json.load(f)

    config = ConfigContainer()
    config.update(env_config)
    config.update(model_config)
    return config

def get_config_container() -> ConfigContainer:
    """获取完整配置"""
    return _load_config()

__all__ = ["get_config_container"]
