import random
import torch

from core import get_config_container
from models import ValueNetwork
from utils import ActionSpace, JointState

config = get_config_container()
device = config.device
random.seed(config.train.random_seed)

class CADRLAgent:
    def __init__(self, mode='train'):
        self.mode = mode  # 'train' 或 'test'
        self.epsilon = config.train.epsilon_start if mode == 'train' else 0.0
        self.value_net = ValueNetwork().to(device)
        self.action_space = ActionSpace(config.agent.v_pref, config.agent.kinematic)

    def act(self, state: JointState):
        if self.mode == 'train' and random.random() < self.epsilon:
            return self._explore()
        else:
            return self._exploit(state)

    def _explore(self):
        """探索：随机选择动作"""
        return self.action_space.sample()

    def _exploit(self, state: JointState):
        """利用：选择价值最高的动作"""
        state_tensor = state.to_tensor().to(device)
        with torch.no_grad():
            q_values = self.value_net(state_tensor)
        return q_values.argmax().item()

    def set_mode(self, mode):
        """切换模式（训练/测试）"""
        self.mode = mode
        if mode == 'test':
            self.epsilon = 0.0  # 测试模式下禁用探索
