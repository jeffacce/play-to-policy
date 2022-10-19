# From Play to Policy: Conditional Behavior Generation from Uncurated Robot Data
[[Paper]](https://arxiv.org/abs/2210.10047) [[Project Website]](https://play-to-policy.github.io/)

[Zichen Jeff Cui](https://jeffcui.com/), Yibin Wang, [Nur Muhammad (Mahi) Shafiullah](https://mahis.life), and [Lerrel Pinto](https://www.lerrelpinto.com/), New York University

This repo contains code for reproducing sim environment experiments, and the real-world robotic experiment gym environment, and data collection tools. Datasets for the simulated environments will be uploaded soon.

## Getting started
The following assumes our current working directory is the root folder of this project repository; tested on Ubuntu 20.04 LTS (amd64).
### Setting up the project environments
- Install the project environment:
  ```
  conda env create --file=conda_env.yml
  ```
- Activate the environment:
  ```
  conda activate cbet
  ```
- Clone the Relay Policy Learning repo:
  ```
  git clone https://github.com/google-research/relay-policy-learning
  ```
- Install MuJoCo 2.1.0: https://github.com/openai/mujoco-py#install-mujoco
- Install CARLA server 0.9.13: https://carla.readthedocs.io/en/0.9.13/start_quickstart/#a-debian-carla-installation
- To enable logging, log in with a `wandb` account:
  ```
  wandb login
  ```
  Alternatively, to disable logging altogether, set the environment variable `WANDB_MODE`:
  ```
  export WANDB_MODE=disabled
  ```

### Getting the training datasets
Datasets used for training will be uploaded soon.
- Download and unzip the datasets.
- In `./config/env_vars/env_vars.yaml`, set the dataset paths to the unzipped directories.
  - `carla_multipath_town04_merge`: CARLA environment
  - `relay_kitchen`: Franka kitchen environment
  - `multimodal_push_fixed_target`: Block push environment

## Reproducing experiments
The following assumes our current working directory is the root folder of this project repository.

To reproduce the experiment results, the overall steps are:
1. Activate the conda environment with
   ```
   conda activate cbet
   ```
2. Train with `python3 train.py`. A model snapshot will be saved to `./exp_local/...`;
3. In the corresponding environment config, set the `load_dir` to the absolute path of the snapshot directory above;
4. Eval with `python3 run_on_env.py`.

See below for detailed steps for each environment.

### CARLA
- Train:
  ```
  python3 train.py --config-name=train_carla_future_cond
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_carla_train`
- In `configs/env/carla_multipath_merge_town04_traj_rep.yaml`, set `load_dir` to the absolute path of the directory above.
- Evaluation:
  ```
  python3 run_on_env.py --config-name=eval_carla_future_cond
  ```

### Franka kitchen
- Train:
  ```
  python3 train.py --config-name=train_kitchen_future_cond
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_kitchen_train`
- In `configs/env/relay_kitchen_traj.yaml`, set `load_dir` to the absolute path of the directory above.
- Evaluation:
  ```
  export PYTHONPATH=$PYTHONPATH:$(pwd)/relay-policy-learning/adept_envs
  python3 run_on_env.py --config-name=eval_kitchen_future_cond
  ```
  (Evaluation requires including the relay policy learning repo in `PYTHONPATH`.)

### Block push
- Train:
  ```
  python3 train.py --config-name=train_blockpush_future_cond
  ```
  Snapshots will be saved to a new timestamped directory `./exp_local/{date}/{time}_blockpush_train`
- In `configs/env/block_pushing_multimodal_fixed_target.yaml`, set `load_dir` to the absolute path of the directory above.
- Evaluation:
  ```
  ASSET_PATH=$(pwd) python3 run_on_env.py --config-name=eval_blockpush_future_cond
  ```
  (Evaluation requires including this repository in `ASSET_PATH`.)

### Speeding up evaluation
- Rendering can be disabled for the kitchen and block pushing environments: set `enable_render: False` in `configs/eval_kitchen.yaml`, `configs/eval_blockpush.yaml`.
  
  (This option does not affect CARLA, as it requires rendering for RGB camera observations.)
- CARLA (Unreal Engine 4) renders on GPU 0 by default. If multiple GPUs are available, running the evaluated model on other GPUs can speed up evaluation: e.g. set `device: cuda:1` in `configs/eval_carla.yaml`.
