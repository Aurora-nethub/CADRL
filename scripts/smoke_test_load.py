import os
import sys

from scripts.train import load_trajectories

# ensure repo root is on sys.path so we can import project modules when running
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


p = os.path.join('data', 'multi_sim')
trajectories = load_trajectories(p)
print(f'Loaded {len(trajectories)} trajectories from {p}')

for i, t in enumerate(trajectories):
    times = getattr(t, 'times', None)
    pos = getattr(t, 'positions', None)
    vel = getattr(t, 'velocities', None)
    T = len(times) if times is not None else (len(pos) if pos is not None else None)
    print(f'-- traj {i}: length T={T}')
    if pos is not None:
        print('   positions[0]:', pos[0].tolist())
    if vel is not None:
        print('   velocities[0]:', vel[0].tolist())
