
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.trajectory import Trajectory

def plot_trajectory(traj: Trajectory, save_path=None, value_fn=None):
    """
    可视化一条轨迹，支持叠加值函数曲线。
    """
    T = len(traj) # pylint: disable=invalid-name
    px = traj.positions[:, 0, 0]
    py = traj.positions[:, 0, 1]
    px1 = traj.positions[:, 1, 0]
    py1 = traj.positions[:, 1, 1]

    plt.figure(figsize=(8, 8))

    # Plot agent trajectories
    plt.plot(px, py, '-', color='blue', linewidth=2, alpha=0.6, label='Agent 0 Path')
    plt.plot(px1, py1, '--', color='orange', linewidth=2, alpha=0.6, label='Agent 1 Path')

    # Mark start positions
    plt.scatter([px[0]], [py[0]], c='blue', marker='o', s=100, edgecolors='black', linewidths=1.5, zorder=5, label='Agent 0 Start')
    plt.scatter([px1[0]], [py1[0]], c='orange', marker='s', s=100, edgecolors='black', linewidths=1.5, zorder=5, label='Agent 1 Start')

    # Mark goal positions (both agents' goals)
    # Agent 0的目标
    plt.scatter([traj.goal_x], [traj.goal_y], c='green', marker='*', s=300, edgecolors='black', linewidths=1.5, label='Agent 0 Goal', zorder=6)

    # Agent 1（邻居）的目标 - 从traj中读取真实目标
    if hasattr(traj, 'neighbor_goal_x') and hasattr(traj, 'neighbor_goal_y'):
        goal1_x, goal1_y = traj.neighbor_goal_x, traj.neighbor_goal_y
    else:
        # 如果没有记录邻居目标，使用镜像假设
        goal1_x, goal1_y = -px1[0], -py1[0]
    plt.scatter([goal1_x], [goal1_y], c='red', marker='*', s=300, edgecolors='black', linewidths=1.5, label='Agent 1 Goal', zorder=6)

    # Draw final position as circles (based on radius)
    radius = traj.radius
    circle0 = patches.Circle((px[-1], py[-1]), radius, color='blue', alpha=0.3, zorder=4)
    circle1 = patches.Circle((px1[-1], py1[-1]), radius, color='orange', alpha=0.3, zorder=4)
    plt.gca().add_patch(circle0)
    plt.gca().add_patch(circle1)

    plt.grid(True, alpha=0.3)
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title(f'CADRL Rollout Trajectory (T={T} steps)', fontsize=14)
    plt.legend(loc='upper left', fontsize=9, ncol=2)

    # 固定地图范围为 (-4, 4)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 注意：value_fn被禁用，因为它会在空间坐标图上绘制step-value曲线导致混淆
    # 如需要value可视化，应创建单独的子图
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

