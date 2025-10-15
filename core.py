from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import json

import torch
import logging
import os
import sys

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
    pretrain_epochs: int = 10
    pretrain_batch_size: int = 500
    pretrain_total_iters: int = 10000

@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 100
    learning_rate: float = 0.01
    step_size: int = 150
    train_episodes: int = 1000  # 总episode数，建议主控
    sample_episodes: int = 1    # 每轮采样episode数，建议设为1
    test_interval: int = 10
    test_episodes: int = 100
    capacity: int = 40000
    epsilon_start: float = 0.5
    epsilon_end: float = 0.1
    epsilon_decay: int = 400    # epsilon衰减episode数
    num_epochs: int = 1000      # 若用epoch主控，等于train_episodes/sample_episodes
    checkpoint_interval: int = 50
    random_seed: int = 42

@dataclass
class ConfigContainer:
    """统一配置容器"""
    agent: AgentConfig = field(default_factory=AgentConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitConfig = field(default_factory=InitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger: Optional[logging.Logger] = field(default=None)

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

    def init_logger(self, *, log_dir: str = "logs", log_file: str = "cadrl.log") -> None:
        """
        Initialize a file logger for the config container. Creates `log_dir` if missing.
        """
        if getattr(self, "logger", None) is not None:
            return
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        logger = logging.getLogger("cadrl")
        logger.setLevel(logging.INFO)
        # avoid adding multiple handlers in interactive runs
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == os.path.abspath(log_path)
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        # also add a StreamHandler once if none exists (so logs show during runs)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(sh)
        self.logger = logger

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
    env_config_path: str = "config/env.json",
    model_config_path: str = "config/model.json",
) -> ConfigContainer:
    """加载配置文件并返回配置容器"""
    with open(env_config_path, "r", encoding="utf-8") as f:
        env_config = json.load(f)
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    config = ConfigContainer()
    config.update(env_config)
    config.update(model_config)

    # 尝试初始化日志并记录配置
    try:
        config.init_logger()
        config.logger.info("Loaded configuration from %s and %s", env_config_path, model_config_path)
        config.logger.info("Device: %s", config.device)
    except Exception as exc:  # pylint: disable=broad-except
        # 日志不应导致配置加载失败；捕获错误但提供基本反馈
        sys.stderr.write(f"日志初始化失败: {type(exc).__name__} - {exc}\n")

    return config

def get_config_container() -> ConfigContainer:
    """获取完整配置"""
    return _load_config()

__all__ = ["get_config_container"]
