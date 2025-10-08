Multi-sim trajectory data
=========================

This directory contains tools and data for generating two-agent crossing trajectories
used by the CADRL training pipeline.

Generator
---------

`generate_multi_sim_orca.py` is a small script that generates synthetic two-agent
crossing episodes using a simple ORCA-inspired local avoidance rule. It writes
one compressed `.npz` file per episode into the output folder (default `data/multi_sim`).

Each `.npz` file contains the following fields:

- `times`: float32 array, shape (T,)  -- timestamps (in units of dt)
- `positions`: float32 array, shape (T, 2, 2) -- positions[t, agent_idx, (x,y)]
- `velocities`: float32 array, shape (T, 2, 2) -- velocities[t, agent_idx, (vx,vy)]
- `radius`: float -- agent radius
- `goal_x`, `goal_y`: float -- goal for agent 0
- `v_pref`: float -- preferred speed used in generation
- `kinematic`: bool
- `gamma`: float

New fields added for data quality control:

- `timed_out`: bool -- True if the episode ran to the `max_steps` limit without agent0 reaching its goal. Use this to filter out truncated episodes for supervised pretraining.
- `out_of_bounds`: bool -- True if any agent left the environment bounds (default bounds used by generator: xmin=-8, xmax=8, ymin=-8, ymax=8). Out-of-bounds episodes may be invalid for training in bounded environments.

These `.npz` files are easily loaded with numpy (`np.load`) or pandas/polars
by first reading the arrays and converting to DataFrame if desired. For example:

```python
import numpy as np

data = np.load('traj_00000.npz')
times = data['times']
positions = data['positions']  # shape (T,2,2)
```

Notes
-----

- The generator is intentionally simple and deterministic (except for random seed).
- If you need a different format (csv, parquet), you can transform the `.npz` files
  using pandas or polars and save them to the desired format.
