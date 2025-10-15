
"""
models 模块
----------------
包含神经网络模型定义。

主要类：
- ValueNetwork：状态价值网络，输入状态向量输出价值。

用法：
	from models import ValueNetwork
	net = ValueNetwork(state_dim=14)
	value = net(state_tensor)
"""
from .value_network import ValueNetwork
