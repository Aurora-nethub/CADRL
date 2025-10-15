
"""
env 模块
----------------
提供 CADRL 的仿真环境实现。

主要类：
- CADRLEnv：双智能体环境，支持 reset/step 等接口。

用法：
	from env import CADRLEnv
	env = CADRLEnv(cfg)
	states = env.reset()
	next_states, rewards, dones = env.step(actions)
"""
from .cadrl_env import CADRLEnv

__all__ = ["CADRLEnv"]
