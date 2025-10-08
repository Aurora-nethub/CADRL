"""
Generate two-agent trajectories using a simple ORCA-like local planner.

This script produces .npz files under the output directory (default: data/multi_sim/)
each containing the following fields:
 - times: shape (T,), float32
 - positions: shape (T, 2, 2) float32 (t, agent_idx=0/1, x/y)
 - velocities: shape (T, 2, 2) float32 (vx, vy)
 - radius: float
 - goal_x, goal_y: float (for agent 0)
 - v_pref: float
 - kinematic: bool
 - gamma: float

The planner implemented here is a simple, deterministic local collision-avoidance
policy inspired by ORCA principles (reciprocal reactive avoidance). It is not a
full-featured ORCA implementation but produces reasonable crossing trajectories
for training/testing. The loader in `scripts/train.py` expects `Trajectory`-compatible
arrays (times, positions, etc.).
"""
from __future__ import annotations

import argparse
import os
import math
import random
from typing import Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)


def orca_step(px: np.ndarray, py: np.ndarray, vx: np.ndarray, vy: np.ndarray,
              goal: Tuple[float, float], v_pref: float, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute next velocities for two agents using a simple reciprocal avoidance rule.

    Returns arrays vx_next (2,), vy_next (2,).
    """
    # preferred velocities towards goal (agent 0 uses provided goal, agent1 uses opposite)
    prefs = np.zeros((2, 2), dtype=np.float32)
    # agent 0 goal is provided, agent 1 goal is negative of agent0's position (symmetric)
    prefs[0] = _unit(np.array([goal[0] - px[0], goal[1] - py[0]])) * v_pref
    prefs[1] = _unit(np.array([-goal[0] - px[1], -goal[1] - py[1]])) * v_pref

    pos = np.vstack([[px[0], py[0]], [px[1], py[1]]])  # (2,2)
    vel = np.vstack([[vx[0], vy[0]], [vx[1], vy[1]]])  # (2,2)

    # simple reciprocal avoidance: if predicted approach is too close, steer perpendicular
    rel = pos[1] - pos[0]
    dist = np.linalg.norm(rel)
    rsum = 2 * radius

    vx_next = np.array([prefs[0, 0], prefs[1, 0]], dtype=np.float32)
    vy_next = np.array([prefs[0, 1], prefs[1, 1]], dtype=np.float32)

    # if agents are on collision course within horizon, modify preferred velocities
    # compute relative velocity
    rel_vel = vel[0] - vel[1]
    approaching = np.dot(rel, rel_vel) < 0

    if dist < (rsum + 0.1) or (approaching and dist < 5.0):
        # generate avoidance direction perpendicular to rel
        perp = np.array([-rel[1], rel[0]])
        perp_unit = _unit(perp)
        sidestep = perp_unit * v_pref
        # reciprocal: agent 0 and 1 take opposite sidesteps
        vx_next[0], vy_next[0] = sidestep
        vx_next[1], vy_next[1] = -sidestep

    return vx_next, vy_next


def simulate_episode(cr: float = 2.0, v_pref: float = 1.0, radius: float = 0.3,
                     dt: float = 1.0, max_steps: int = 100, seed: int = None) -> dict:
    """Simulate a single crossing episode and return dict of arrays."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # initialize positions as in env.reset: agent0 at (-cr,0), agent1 at angle on circle
    angle = random.random() * math.pi
    # avoid too small separation seen in original code
    while math.sin((math.pi - angle) / 2.0) < 0.3 / 2.0:
        angle = random.random() * math.pi

    px = np.zeros(2, dtype=np.float32)
    py = np.zeros(2, dtype=np.float32)
    vx = np.zeros(2, dtype=np.float32)
    vy = np.zeros(2, dtype=np.float32)

    px[0], py[0] = -cr, 0.0
    px[1], py[1] = cr * math.cos(angle), cr * math.sin(angle)

    goal_x0, goal_y0 = cr, 0.0

    times = []
    positions = []
    velocities = []
    timed_out = False
    out_of_bounds = False

    # environment bounds (default large box; generator doesn't read config here)
    xmin, xmax, ymin, ymax = -8.0, 8.0, -8.0, 8.0

    for t in range(max_steps):
        times.append(t * dt)
        positions.append([[px[0], py[0]], [px[1], py[1]]])
        velocities.append([[vx[0], vy[0]], [vx[1], vy[1]]])

        # compute next velocities
        vx_n, vy_n = orca_step(px, py, vx, vy, (goal_x0, goal_y0), v_pref, radius)

        # integrate
        px += vx_n * dt
        py += vy_n * dt
        vx, vy = vx_n, vy_n

        # check bounds
        if not (xmin <= px[0] <= xmax and ymin <= py[0] <= ymax and xmin <= px[1] <= xmax and ymin <= py[1] <= ymax):
            out_of_bounds = True
            # record the out-of-bounds step and stop
            times.append((t + 1) * dt)
            positions.append([[px[0], py[0]], [px[1], py[1]]])
            velocities.append([[vx[0], vy[0]], [vx[1], vy[1]]])
            break

        # check goal for agent 0
        if math.hypot(px[0] - goal_x0, py[0] - goal_y0) <= radius:
            # mark final step and break
            times.append((t + 1) * dt)
            positions.append([[px[0], py[0]], [px[1], py[1]]])
            velocities.append([[vx[0], vy[0]], [vx[1], vy[1]]])
            break

    else:
        # loop completed without break -> timed out
        timed_out = True

    data = {
        'times': np.asarray(times, dtype=np.float32),
        'positions': np.asarray(positions, dtype=np.float32),  # (T,2,2)
        'velocities': np.asarray(velocities, dtype=np.float32),
        'radius': float(radius),
        'goal_x': float(goal_x0),
        'goal_y': float(goal_y0),
        'v_pref': float(v_pref),
        'kinematic': True,
        'gamma': 0.8,
        'timed_out': bool(timed_out),
        'out_of_bounds': bool(out_of_bounds),
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate multi-agent crossing trajectories (simple ORCA-like)')
    parser.add_argument('--out-dir', type=str, default=os.path.join('data', 'multi_sim'))
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--v-pref', type=float, default=1.0)
    parser.add_argument('--radius', type=float, default=0.3)
    parser.add_argument('--crossing-radius', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.n_episodes):
        data = simulate_episode(cr=args.crossing_radius, v_pref=args.v_pref,
                                radius=args.radius, dt=args.dt, max_steps=args.max_steps,
                                seed=(None if args.seed is None else args.seed + i))
        out_path = os.path.join(args.out_dir, f'traj_{i:05d}.npz')
        np.savez_compressed(out_path, **data)
        print(f"Saved {out_path} (T={data['positions'].shape[0]})")


if __name__ == '__main__':
    main()
