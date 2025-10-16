# -*- coding: utf-8 -*-
"""
脚本：使用训练好的模型 rollout 一条新轨迹并可视化
"""
from typing import Optional
import os
import random
import torch
import numpy as np
from core import get_config_container, ConfigContainer
from models import ValueNetwork
from agent import CADRLAgent
from env import CADRLEnv
from utils import ActionSpace
from utils.trajectory import Trajectory
from vis.visualize import plot_trajectory

def rollout_episode(env, agent, max_steps=200):
    states = env.reset()
    done = [False, False]
    steps = 0

    # 记录实际的目标（考虑cr扰动）
    actual_goal_x = states[0].self_state.pgx
    actual_goal_y = states[0].self_state.pgy
    # Agent 1（邻居）的目标
    neighbor_goal_x = states[1].self_state.pgx
    neighbor_goal_y = states[1].self_state.pgy

    ep_times = [0.0]
    ep_positions = [[
        [states[0].self_state.px, states[0].self_state.py],
        [states[0].neighbor_state.px, states[0].neighbor_state.py]
    ]]
    t_acc = 0.0
    while not all(done) and steps < max_steps:
        actions = []
        for i in range(2):
            s = states[i]
            a = agent.act(s, env, i, mode='eval')
            actions.append(a)
        s_next, _, dones = env.step(actions)
        eff_dt = float(env.last_step_ratio) * float(env.dt)
        t_acc += eff_dt
        ep_times.append(t_acc)
        ep_positions.append([
            [s_next[0].self_state.px, s_next[0].self_state.py],
            [s_next[0].neighbor_state.px, s_next[0].neighbor_state.py]
        ])
        states = s_next
        done = [d != 0 for d in dones]
        steps += 1
    traj = Trajectory(
        gamma=env.cfg.model.gamma,
        goal_x=actual_goal_x,  # 使用实际目标（考虑扰动）
        goal_y=actual_goal_y,
        radius=env.cfg.agent.radius,
        v_pref=env.cfg.agent.v_pref,
        times=np.array(ep_times, dtype=np.float32),
        positions=np.array(ep_positions, dtype=np.float32),
        kinematic=env.cfg.agent.kinematic
    )
    # 附加邻居目标信息（用于可视化）
    traj.neighbor_goal_x = neighbor_goal_x
    traj.neighbor_goal_y = neighbor_goal_y
    return traj


def run_test(
    cfg: Optional[ConfigContainer] = None,
    *,
    device: Optional[torch.device] = None,
    run_name: Optional[str] = None,
    case: int = None
) -> None:
    """
    运行测试：加载训练好的模型，rollout新轨迹并可视化
    
    Parameters
    ----------
    cfg : Optional[ConfigContainer]
        配置容器，如果为None则自动加载
    device : Optional[torch.device]
        运行设备，如果为None则自动选择
    run_name : Optional[str]
        运行名称，用于定位checkpoint目录
    """
    n_cases = 20
    if case is None:
        case = random.randint(0, n_cases - 1)
    if cfg is None:
        cfg = get_config_container()
    if isinstance(cfg, dict):
        cfg = ConfigContainer.from_dict(cfg)

    device = device or getattr(cfg, 'device', torch.device('cpu'))

    # 定位checkpoint
    if run_name:
        ckpt_dir = os.path.join('checkpoints', run_name)
    else:
        # 使用最新的checkpoint
        ckpt_base = 'checkpoints'
        if not os.path.exists(ckpt_base) or not os.listdir(ckpt_base):
            raise FileNotFoundError(f"未找到checkpoint目录: {ckpt_base}")
        ckpt_dir = os.path.join(ckpt_base, sorted(os.listdir(ckpt_base))[-1])

    model_path = os.path.join(ckpt_dir, 'final.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    print(f"\n{'='*70}")
    print('🧪 开始测试模型')
    print(f"{'='*70}")
    print(f'模型路径: {model_path}')
    print(f'设备: {device}')

    # 加载模型（直接加载state_dict，不需要config）
    ckpt = torch.load(model_path, map_location=device)
    model = ValueNetwork(state_dim=cfg.model.state_dim, device=device).to(device)
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    model.eval()
    print('✓ 模型加载成功')

    # 构造agent和环境
    action_space = ActionSpace(cfg.agent.v_pref, cfg.agent.kinematic)
    agent = CADRLAgent(value_net=model, action_space=action_space, device=device, gamma=cfg.model.gamma)

    # 测试时关闭cr扰动，保证起点-目标距离一致性
    env = CADRLEnv(cfg, phase='test')
    env.disturb_range = (1.0, 1.0)  # 测试时固定cr，避免目标不匹配

    # rollout函数（支持指定case）
    def rollout_episode_case(env, agent, max_steps=200, case=0):
        # 传递case和角度，reset内部自动处理
        states = env.reset(case=case)
        done = [False, False]
        steps = 0
        # Agent 0的目标
        actual_goal_x = states[0].self_state.pgx
        actual_goal_y = states[0].self_state.pgy
        # Agent 1（邻居）的目标
        neighbor_goal_x = states[1].self_state.pgx
        neighbor_goal_y = states[1].self_state.pgy
        ep_times = [0.0]
        ep_positions = [[
            [states[0].self_state.px, states[0].self_state.py],
            [states[0].neighbor_state.px, states[0].neighbor_state.py]
        ]]
        t_acc = 0.0
        while not all(done) and steps < max_steps:
            actions = []
            for i in range(2):
                s = states[i]
                a = agent.act(s, env, i, mode='eval')
                actions.append(a)
            s_next, _, dones = env.step(actions)
            eff_dt = float(env.last_step_ratio) * float(env.dt)
            t_acc += eff_dt
            ep_times.append(t_acc)
            ep_positions.append([
                [s_next[0].self_state.px, s_next[0].self_state.py],
                [s_next[0].neighbor_state.px, s_next[0].neighbor_state.py]
            ])
            states = s_next
            done = [d != 0 for d in dones]
            steps += 1
        traj = Trajectory(
            gamma=env.cfg.model.gamma,
            goal_x=actual_goal_x,
            goal_y=actual_goal_y,
            radius=env.cfg.agent.radius,
            v_pref=env.cfg.agent.v_pref,
            times=np.array(ep_times, dtype=np.float32),
            positions=np.array(ep_positions, dtype=np.float32),
            kinematic=env.cfg.agent.kinematic
        )
        # 附加邻居目标信息（用于可视化）
        traj.neighbor_goal_x = neighbor_goal_x
        traj.neighbor_goal_y = neighbor_goal_y
        # 附加done原因
        traj.done_reason = dones
        return traj

    traj = rollout_episode_case(env, agent, max_steps=cfg.sim.max_time, case=case)
    print('✓ Rollout完成')
    print(f'  - 轨迹长度: {len(traj)} 步')
    print(f'  - 自车起点: ({traj.positions[0,0,0]:.2f}, {traj.positions[0,0,1]:.2f})')
    print(f'  - 自车目标: ({traj.goal_x:.2f}, {traj.goal_y:.2f})')
    print(f'  - 自车终点: ({traj.positions[-1,0,0]:.2f}, {traj.positions[-1,0,1]:.2f})')
    print(f'  - 邻居起点: ({traj.positions[0,1,0]:.2f}, {traj.positions[0,1,1]:.2f})')
    print(f'  - 邻居终点: ({traj.positions[-1,1,0]:.2f}, {traj.positions[-1,1,1]:.2f})')
    
    # 附加done原因的输出
    if hasattr(traj, 'done_reason'):
        done_map = {0: 'Active', 1: 'Reached', 2: 'Collision', 3: 'Out-of-Bound', 4: 'Timeout'}
        print(f'  - Done原因: Agent0={done_map.get(traj.done_reason[0], "Unknown")}, Agent1={done_map.get(traj.done_reason[1], "Unknown")}')

    # 可视化
    vis_dir = 'vis'
    os.makedirs(vis_dir, exist_ok=True)
    rname = run_name or 'test'
    save_path = os.path.join(vis_dir, f'rollout_{rname}.png')
    plot_trajectory(traj, save_path=save_path, value_fn=model)
    print(f'\n✓ 可视化已保存至: {save_path}')
    print(f"{'='*70}\n")


if __name__ == '__main__':
    run_test()
