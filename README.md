# Hexapod RL Locomotion on Rough Terrain

A six-legged robot trained to walk with reinforcement learning (PPO) in MuJoCo Warp. It follows velocity commands (forward/back, strafe, turn) over randomized rough terrain, and you can drive it live from the keyboard.

## Demo Video
 
[![Demo video](https://img.youtube.com/vi/sgxTRLx2zf8/maxresdefault.jpg)](https://youtu.be/sgxTRLx2zf8)
 
▶️ **[Click here to watch the full video on YouTube](https://youtu.be/sgxTRLx2zf8)**


## Highlights

- 18-DoF hexapod (3 joints per leg), position-controlled
- Massively parallel training: 2048 environments on one GPU with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- PPO from [RSL-RL](https://github.com/leggedrobotics/rsl_rl), small actor and critic MLPs (`54-42-30`, ELU)
- One policy tracks commands $(v_x, v_y, \omega_z)$ on flat ground, pebbles, rocks and slopes
- Domain randomization: friction, random pushes, action delay, observation noise
- Keyboard teleop in the MuJoCo viewer, no extra dependencies

## How it works

**Observation (63 values):** angular velocity (3), projected gravity (3), command (3), joint position offsets (18), joint velocities (18), previous action (18).

**Action (18 values):** joint position offsets added to the standing pose, scaled by `0.2` and clipped to $[-1, 1]$ rad.

**Commands** are resampled every 10 s: $v_x, v_y \in [-0.4, 0.4]$ m/s and $\omega_z \in [-0.8, 0.8]$ rad/s.

**Reward** (per step, scaled by the control timestep):

$$
r = 8\,e^{-\lVert \mathbf{v}^{cmd}_{xy}-\mathbf{v}_{xy}\rVert^2/0.05}
  + 6\,e^{-(\omega^{cmd}_z-\omega_z)^2/0.03}
  - 2.5\!\times\!10^{-5}\lVert\ddot{\mathbf{q}}\rVert^2
  - 0.02\lVert\Delta\mathbf{a}\rVert^2
  - 0.01\lVert\mathbf{q}-\mathbf{q}_0\rVert^2
  - 5\lVert\mathbf{g}_{xy}\rVert^2
$$

The terms reward velocity tracking and penalize joint acceleration, action rate, deviation from the default pose and body tilt. Falling gives a terminal penalty of $-200$.

**Terrain:** a $12\times12$ m heightfield made of $3\times3$ m patches. Each patch is randomly flat, small pebbles, larger rocks, or a pyramid hill (3 to 12 degrees).

**Randomization:** ground friction in $[0.4, 1.25]$, a random push of up to $0.3$ m/s every 4 s, a one-step action delay in half of the episodes, and Gaussian noise on all observations.

## Project structure

```
hexapod/
├── hexa_env.py          # environment, terrain generation, rewards
├── cfg.py               # PPO / network config
├── train.py             # training
├── play.py              # evaluate the policy / watch it in the viewer
├── teleop.py            # keyboard control
├── export_policy.py     # export weights to .npz + .json for deployment
├── check_terrain.py     # quick terrain sanity check
├── mk_mjcf.py           # URDF -> MuJoCo XML converter
├── mk_terrain.py        # adds the heightfield to an XML
├── hexapod_terrain.xml  # robot + terrain model
├── hexapod.xml          # robot on flat ground
├── meshes/              # STL files
├── logs/hexa/<run>/     # trained checkpoint (model_*.pt), new runs are saved here too
└── optional/            # URDF and alternate env files
```

## Requirements

- Linux with an NVIDIA GPU (the code uses `cuda:0`)
- Python 3.12
- `mujoco`, `mujoco_warp`, `warp-lang`, `torch`, `tensordict`, `rsl-rl-lib`

Install these following the instructions of each project (MuJoCo Warp in particular has its own GPU requirements).

## Usage

Run everything from inside this folder.

**Watch the trained policy**

```bash
python3 play.py --view                 # walks forward in the MuJoCo viewer
python3 play.py --view --cmd 0 0.3 0   # fixed command: vx vy wz
python3 play.py                        # headless test of 6 commands, prints tracking and falls
python3 play.py -c path/to/model.pt    # use a specific checkpoint
```

By default the newest checkpoint in `logs/hexa/` is loaded. A trained checkpoint is included in this repo, so you can run these commands right away without training.

**Drive it with the keyboard**

```bash
python3 teleop.py
```

Type in the terminal you launched it from:

| Key | Action |
|-----|--------|
| Up / Down | forward / backward |
| Left / Right | turn left / right |
| W / S | strafe left / right |
| R | reset the robot |
| Q | quit |

The robot walks while a key is held and stands still when you let go. Noise, random pushes and the episode timeout are disabled in this mode so it is easy to record.

**Train from scratch**

```bash
python3 train.py                       # 2048 envs, 1500 iterations
python3 train.py -e 4096 -i 3000       # more envs / iterations
python3 train.py -r logs/hexa/<run>/model_1000.pt   # resume
```

Checkpoints are saved every 50 iterations in `logs/hexa/<timestamp>/`, with TensorBoard logs.

**Export the policy**

```bash
python3 export_policy.py logs/hexa/<run>/model_1500.pt hexa_policy
```

This writes `hexa_policy.npz` (weights) and `hexa_policy.json` (joint names, default pose, scaling) and checks that a plain NumPy forward pass matches the trained policy.

## Regenerating the model (optional)

The MuJoCo XML files are generated from the URDF in `optional/`:

```bash
python3 mk_mjcf.py optional/hexapod.urdf hexapod.xml meshes
python3 mk_terrain.py hexapod.xml hexapod_terrain.xml
```

## Notes

- This project is simulation only.
- `hexapod_terrain.xml` expects the STL files in `meshes/` next to it.

## Author

Your Name · [GitHub](https://github.com/YOUR_USERNAME)
