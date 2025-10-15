
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(6, 6))
    plt.plot(px, py, 'o-', label='Agent 0')
    plt.plot(px1, py1, 's--', label='Agent 1')
    plt.scatter([traj.goal_x], [traj.goal_y], c='r', marker='*', s=120, label='Goal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.legend()
    plt.axis('equal')
    if value_fn is not None:
        values = [value_fn(traj.state_at(i).unsqueeze(0)).item() for i in range(1, T)]
        ax2 = plt.gca().twinx()
        ax2.plot(range(1, T), values, 'k-', label='Value')
        ax2.set_ylabel('Value')
        ax2.legend(loc='upper right')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

