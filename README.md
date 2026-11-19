# Hexapod RL Locomotion on Rough Terrain

A six-legged robot trained to walk with reinforcement learning (PPO) in MuJoCo Warp. It follows velocity commands (forward/back, strafe, turn) over randomized rough terrain, and you can drive it live from the keyboard.

This repo also contains the code for the technical report **"Which Randomizations Matter? A Seeded Ablation for Blind Hexapod Locomotion in MuJoCo Warp"** (draft): 6 training configurations x 3 seeds, evaluated under 24 paired test conditions.

## Demo Video

[![Demo video](https://img.youtube.com/vi/sgxTRLx2zf8/maxresdefault.jpg)](https://youtu.be/sgxTRLx2zf8)

▶️ **[Click here to watch the full video on YouTube](https://youtu.be/sgxTRLx2zf8)**


## Highlights

- 18-DoF hexapod (3 joints per leg), position-controlled, blind (proprioception only)
- Massively parallel training: 2048 environments on one GPU with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp); about 20 minutes per run on a laptop RTX 4050
- PPO from [RSL-RL](https://github.com/leggedrobotics/rsl_rl), small actor and critic MLPs (`54-42-30`, ELU)
- One policy tracks commands $(v_x, v_y, \omega_z)$ on flat ground, pebbles, rocks and slopes
- Domain randomization: friction, random pushes, action delay, observation noise
- Seeded ablation of each randomization component with a paired evaluation protocol
- Keyboard teleop in the MuJoCo viewer, no extra dependencies

## Main findings

Linear and yaw velocity tracking scores (1 is perfect), mean over the six standard test terrains and 3 seeds:

| Config | What changes | Linear | Yaw |
|--------|--------------|--------|-----|
| `full` | all randomization, mixed terrain | 0.842 | 0.681 |
| `flat` | flat terrain only | 0.815 | 0.693 |
| `nofric` | fixed friction 1.0 | 0.862 | 0.698 |
| `nopush` | no pushes | 0.856 | 0.690 |
| `nodelay` | no action delay | 0.649 | 0.556 |
| `nonoise` | no observation noise | 0.854 | 0.623 |

- **Action-delay randomization matters most.** Without it, linear tracking drops by 0.36 when every test episode is delayed by one control step.
- **Terrain diversity** matters mainly outside the training distribution (0.12 lower on 15 to 20 degree slopes for the flat-only policy).
- **Observation-noise randomization** affects yaw tracking, not linear tracking.
- **Friction and push randomization** showed no measurable benefit in the ranges tested, even at friction 0.1 and 3 m/s pushes.

This is a simulation-only study with three seeds per configuration. See the report for the limitations.

## How it works

**Observation (63 values):** angular velocity (3), projected gravity (3), command (3), joint position offsets (18), joint velocities (18), previous action (18).

**Action (18 values):** joint position offsets added to the standing pose, scaled by `0.2` and clipped to $[-1, 1]$ rad. The policy acts every 16 ms (4 physics steps of 4 ms), and an episode lasts 20 s.

**Commands** are resampled every 10 s: $v_x, v_y \in [-0.4, 0.4]$ m/s and $\omega_z \in [-0.8, 0.8]$ rad/s.

**Reward** (per step, every term scaled by the control timestep $\Delta t = 16$ ms):

  $$
  r = 8\,e^{-\lVert \mathbf{v}^{cmd}_{xy}-\mathbf{v}_{xy}\rVert^2/0.05}
    + 6\,e^{-(\omega^{cmd}_z-\omega_z)^2/0.03}
    - 2.5\times 10^{-5}\lVert\ddot{\mathbf{q}}\rVert^2
    - 0.02\lVert\Delta\mathbf{a}\rVert^2
    - 0.01\lVert\mathbf{q}-\mathbf{q}_0\rVert^2
    - 5\lVert\mathbf{g}_{xy}\rVert^2
  $$

The terms reward velocity tracking and penalize joint acceleration, action rate, deviation from the default pose, body tilt and falling. Because the fall penalty is scaled by $\Delta t$ as well, one fall costs $200 \times 0.016 = 3.2$ reward units.

**Terrain:** a $12\times12$ m heightfield made of $3\times3$ m patches. Each patch is randomly flat (30%), small pebbles (25%), larger rocks (25%), or a pyramid hill (20%, 3 to 12 degrees). Training uses one fixed map.

**Randomization:** ground friction in $[0.4, 1.25]$, a random push of up to $0.3$ m/s every 4 s, a one-step action delay in half of the episodes, and Gaussian noise on angular velocity, projected gravity, joint positions and joint velocities.

## Project structure

```
hexa_mj/
├── hexa_env.py          # environment, terrain generation, rewards
├── cfg.py               # PPO / network config
├── expcfg.py            # ablation configurations (full, flat, nofric, nopush, nodelay, nonoise)
├── train.py             # training (config overrides with --set key=value)
├── run_all.sh           # trains all 6 configurations x 3 seeds
├── post.sh              # helper: waits for run_all.sh, then runs eval.py and aggregate.py
├── eval.py              # paired evaluation under 24 test conditions
├── aggregate.py         # tables and figures for the standard / stress conditions
├── aggregate_hard.py    # tables for the harder combined-stress conditions
├── play.py              # run one policy / watch it in the viewer
├── teleop.py            # keyboard control
├── export_policy.py     # export weights to .npz + .json for deployment
├── check_terrain.py     # quick terrain sanity check
├── mk_mjcf.py           # URDF -> MuJoCo XML converter
├── mk_terrain.py        # adds the heightfield to an XML
├── hexapod_terrain.xml  # robot + terrain model
├── hexapod.xml          # robot on flat ground
├── results/             # evaluation CSVs, tables and figures used in the report
├── legacy/              # earlier versions of the environment
├── meshes/              # STL files (not in this repo, see "Robot model")
└── logs/exp/<cfg>_s<seed>/   # checkpoints (not in this repo, see "Trained policies")
```

## Requirements

- Linux with an NVIDIA GPU (the code uses `cuda:0`)
- Python 3.10 or newer
- `mujoco`, `mujoco_warp`, `warp-lang`, `torch`, `tensordict`, `rsl-rl-lib`

The results in the report were produced with Python 3.10.12, MuJoCo 3.9.0 and PyTorch 2.3.0 on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB). Install the packages following the instructions of each project (MuJoCo Warp in particular has its own GPU requirements).

## Robot model

The robot's STL meshes come from the robot's ROS 2 description package (`hexa_description`) and are not stored in this repo. Put them in a folder called `meshes/` next to `hexapod_terrain.xml`, because the XML files expect them there.

## Trained policies

The 18 trained policies (6 configurations x 3 seeds: checkpoints, configs and TensorBoard logs) are not stored in this repo, because `logs/` is git-ignored. They will be archived on Zenodo once the report is final. Until then, train them yourself with `./run_all.sh` (see below).

Each run is saved in `logs/exp/<cfg>_s<seed>/`, for example `logs/exp/full_s0/model_799.pt`.

## Usage

Run everything from inside this folder.

**Watch a trained policy**

```bash
python3 play.py --view -c logs/exp/full_s0/model_799.pt                 # walks forward in the MuJoCo viewer
python3 play.py --view --cmd 0 0.3 0 -c logs/exp/full_s0/model_799.pt   # fixed command: vx vy wz
python3 play.py -c logs/exp/full_s0/model_799.pt                        # headless test of 6 commands, prints tracking and falls
```

Always pass the checkpoint with `-c`. Without it, `play.py` and `teleop.py` look for checkpoints in `logs/hexa/`, which the current `train.py` no longer writes to.

**Drive it with the keyboard**

```bash
python3 teleop.py -c logs/exp/full_s0/model_799.pt
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

**Train a single policy**

```bash
python3 train.py                                   # default: config "full", seed 0 -> logs/exp/full_s0/
python3 train.py -n nodelay -s 1 --set pdl=0.0     # a named ablation run with seed 1
python3 train.py -e 4096 -i 3000                   # more envs / iterations
python3 train.py -r logs/exp/full_s0/model_500.pt  # resume
```

Checkpoints are saved every 50 iterations, with TensorBoard logs. Any configuration value can be overridden on the command line with `--set key=value`. The overrides used for the ablations are listed in `run_all.sh`.

**Reproduce the ablation study**

```bash
./run_all.sh                                     # 6 configs x 3 seeds, 800 iterations each (about 20 min per run)
python3 eval.py logs/exp/*_s?/                   # standard + stress conditions  -> results/eval.csv
python3 eval.py logs/exp/*_s?/ --hard            # harder combined stress        -> results/eval_hard.csv
python3 aggregate.py results/eval.csv            # tables and figures
python3 aggregate_hard.py results/eval_hard.csv
```

`eval.py` evaluates the final checkpoint (`model_799.pt`) of every run.

**Export the policy**

```bash
python3 export_policy.py logs/exp/full_s0/model_799.pt hexa_policy
```

This writes `hexa_policy.npz` (weights) and `hexa_policy.json` (joint names, default pose, scaling) and checks that a plain NumPy forward pass matches the trained policy.

## Regenerating the model (optional)

The MuJoCo XML files are generated from the robot's URDF (in the `hexa_description` package):

```bash
python3 mk_mjcf.py path/to/hexapod.urdf hexapod.xml meshes
python3 mk_terrain.py hexapod.xml hexapod_terrain.xml
```

## Notes

- This project is simulation only.
- `hexapod_terrain.xml` expects the STL files in `meshes/` next to it.

## Author

Pranjal Pandey (UpstairsEngineer48) · [GitHub](https://github.com/UpstairsEngineer48) · [LinkedIn](https://www.linkedin.com/in/pranjal-pandey-ug48/)
