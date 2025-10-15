# -*- coding: utf-8 -*-
"""
脚本：使用训练好的模型 rollout 一条新轨迹并可视化
"""
import os
import torch
import numpy as np
from core import get_config_container
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
        goal_x=env.cfg.sim.crossing_radius,
        goal_y=0.0,
        radius=env.cfg.agent.radius,
        v_pref=env.cfg.agent.v_pref,
        times=np.array(ep_times, dtype=np.float32),
        positions=np.array(ep_positions, dtype=np.float32),
        kinematic=env.cfg.agent.kinematic
    )
    return traj

def main():
    cfg = get_config_container()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    ckpt_dir = os.path.join('checkpoints', sorted(os.listdir('checkpoints'))[-1])
    model_path = os.path.join(ckpt_dir, 'final.pt')
    model = ValueNetwork.load(model_path, device=device)
    model.eval()
    # 构造 agent
    action_space = ActionSpace(cfg.agent.v_pref, cfg.agent.kinematic)
    agent = CADRLAgent(value_net=model, action_space=action_space, device=device, gamma=cfg.model.gamma)
    # 构造环境
    env = CADRLEnv(cfg, phase='test')
    # rollout 一条新轨迹
    traj = rollout_episode(env, agent, max_steps=cfg.sim.max_time)
    print(f"rollout轨迹长度: {len(traj)}")
    print(f"起点: {traj.positions[0,0,:]}, 终点: {traj.goal_x, traj.goal_y}")
    # 可视化
    vis_dir = 'vis'
    os.makedirs(vis_dir, exist_ok=True)
    save_path = os.path.join(vis_dir, 'rollout_traj.png')
    plot_trajectory(traj, save_path=save_path, value_fn=model)
    print(f"已保存可视化至: {save_path}")

if __name__ == '__main__':
    main()
