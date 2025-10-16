import os
import sys
import math

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core import get_config_container
from env import CADRLEnv
from utils.action import Action


def main():
    cfg = get_config_container()
    env = CADRLEnv(cfg, phase="train")
    # Disable crossing radius disturbance for determinism
    env.disturb_range = (1.0, 1.0)

    states = env.reset()
    print(f"Start positions: A0=({states[0].self_state.px:.3f},{states[0].self_state.py:.3f}) -> goal=({states[0].self_state.pgx:.3f},{states[0].self_state.pgy:.3f})")
    print(f"                 A1=({states[1].self_state.px:.3f},{states[1].self_state.py:.3f}) -> goal=({states[1].self_state.pgx:.3f},{states[1].self_state.pgy:.3f})")

    # Simple policy: Agent 0 moves forward; Agent 1 stays still to avoid collision
    a0 = Action(cfg.agent.v_pref, 0.0)  # agent 0 maintain heading, move forward
    a1 = Action(0.0, 0.0)               # agent 1 hold still

    reached = [False, False]
    for step in range(env.max_steps):
        s_next, rewards, dones = env.step([a0, a1])
        reached = [d == 1 for d in dones]
        if any(reached):
            print(f"Reached at step {step}: done={dones}, rewards={rewards}")
            break
    else:
        print("Did not reach within max steps. Final dones:", dones)

    # Geometric check for agent 0
    s0 = s_next[0].self_state
    dist0 = math.hypot(s0.px - s0.pgx, s0.py - s0.pgy)
    print(f"Agent0 distance to goal: {dist0:.6f} (radius={s0.radius})")

    # Exit code to reflect success
    if not any(reached):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
