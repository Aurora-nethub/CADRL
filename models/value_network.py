import torch
import torch.nn as nn
from typing import Optional, List

class ValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 14,
        fc_layers: Optional[List[int]] = None,
        kinematic: bool = True,
        reparametrization: bool = True,
        device: Optional[torch.device] = None
    ) -> None:
        super(ValueNetwork, self).__init__()
        if fc_layers is None:
            fc_layers = [256, 128, 64]
        self.reparametrization = reparametrization
        self.kinematic = kinematic
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 根据是否使用重新参数化调整输入维度
        input_dim = 15 if reparametrization else state_dim
        # remember the declared (original) state_dim for save/load compatibility
        self.declared_state_dim = int(state_dim)

        # 构建多层感知机
        layers = []
        prev_dim = input_dim
        for dim in fc_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.value_network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化网络权重"""
        for module in self.value_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)

    def rotate(self, state: torch.Tensor) -> torch.Tensor:
        """
        重新参数化状态表示
        
        参数:
            state: 输入状态张量, 形状为 (batch_size, 14)
            包含: [px, py, vx, vy, radius, pgx, pgy, v_pref, theta, px1, py1, vx1, vy1, radius1]
        
        返回:
            重新参数化后的状态张量, 形状为 (batch_size, 15)
        """
        # 使用张量切片代替IndexTranslator
        px = state[:, 0:1]
        py = state[:, 1:2]
        vx = state[:, 2:3]
        vy = state[:, 3:4]
        radius = state[:, 4:5]
        pgx = state[:, 5:6]
        pgy = state[:, 6:7]
        v_pref = state[:, 7:8]
        theta = state[:, 8:9]
        px1 = state[:, 9:10]
        py1 = state[:, 10:11]
        vx1 = state[:, 11:12]
        vy1 = state[:, 12:13]
        radius1 = state[:, 13:14]

        # 计算目标方向和旋转角度
        dx = pgx - px
        dy = pgy - py
        rot = torch.atan2(dy, dx)
        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)

        # 计算目标距离
        dg = torch.sqrt(dx**2 + dy**2)

        # 旋转自身速度（标准旋转矩阵）
        vx_rot = vx * cos_rot + vy * sin_rot
        vy_rot = -vx * sin_rot + vy * cos_rot

        # 调整角度
        theta_rot = theta - rot if self.kinematic else theta

        # 旋转其他智能体速度（标准旋转矩阵）
        vx1_rot = vx1 * cos_rot + vy1 * sin_rot
        vy1_rot = -vx1 * sin_rot + vy1 * cos_rot

        # 旋转其他智能体位置（相对位置，标准旋转矩阵）
        px1_rel = px1 - px
        py1_rel = py1 - py
        px1_rot = px1_rel * cos_rot + py1_rel * sin_rot
        py1_rot = -px1_rel * sin_rot + py1_rel * cos_rot

        # 计算其他特征
        radius_sum = radius + radius1
        cos_theta = torch.cos(theta_rot)
        sin_theta = torch.sin(theta_rot)
        da = torch.sqrt((px - px1)**2 + (py - py1)**2)

        # 组合新状态
        new_state = torch.cat([
            dg, v_pref, vx_rot, vy_rot, radius, theta_rot,
            vx1_rot, vy1_rot, px1_rot, py1_rot,
            radius1, radius_sum, cos_theta, sin_theta, da
        ], dim=1)

        return new_state

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.reparametrization:
            state = self.rotate(state)
        value = self.value_network(state)
        return value

    def save(self, path: str) -> None:
        """保存模型到文件"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                # preserve the original declared state_dim (14 by default)
                'state_dim': self.declared_state_dim,
                'fc_layers': [layer.out_features for layer in self.value_network if isinstance(layer, nn.Linear)][:-1],
                'kinematic': self.kinematic,
                'reparametrization': self.reparametrization
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'ValueNetwork':
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(
            state_dim=config['state_dim'],
            fc_layers=config['fc_layers'],
            kinematic=config['kinematic'],
            reparametrization=config['reparametrization'],
            device=device
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model

__all__=['ValueNetwork']
