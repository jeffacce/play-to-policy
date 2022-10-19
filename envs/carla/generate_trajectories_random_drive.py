import sys

sys.path.append("../..")

import os
import gym
import json
import envs
import torch
import einops
import torchvision
import numpy as np
from typing import Dict
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from envs.carla.utils import Town10_spawns
from envs.carla.builtin_agent import BuiltinAgent

N_TRAJECTORIES = 4000
T_MAX = 1000
SAVE_DIR = Path("/path/to/carla_dataset/random_drive")


def save_trajectory_helper(
    agent: BuiltinAgent,
    path: os.PathLike,
    metadata: Dict,
    save_video: bool = False,
):
    trajectory_name = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    trajectory_dir = Path(path) / trajectory_name
    trajectory_dir.mkdir(parents=True)
    print("Saving trajectory to {}".format(trajectory_dir))
    with open(trajectory_dir / "actions.json", "w") as f:
        json.dump(agent.actions, f)
    with open(trajectory_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    torch.save(agent.vehicle_transforms, trajectory_dir / "vehicle_transforms.pt")

    buffer = agent.states
    sensor_name = "0"
    if save_video:
        video_path = trajectory_dir / (sensor_name + ".mp4")
        video_array = np.array(buffer)
        video_array = (video_array * 255).astype(np.uint8)
        video_array = einops.rearrange(video_array, "T C H W -> T H W C")
        torchvision.io.write_video(
            str(video_path),
            video_array=video_array,
            fps=1 / agent.env.dt,
        )
    else:
        for elem in buffer:
            if elem is not None:
                relative_frame = elem.frame - agent.start_frame
                save_path = trajectory_dir / sensor_name / f"{relative_frame:04d}.png"
                if relative_frame >= 0:
                    elem.save_to_disk(str(save_path))
    print("Saved trajectory to", trajectory_dir)
    return trajectory_dir


if __name__ == "__main__":
    env = gym.make("carla-random-drive-v0")
    processes = []

    existing_trajs = SAVE_DIR.glob("2022*")
    existing_seeds = []
    for traj in existing_trajs:
        with open(traj / "metadata.json") as f:
            metadata = json.load(f)
        existing_seeds.append(metadata["seed"])
    seeds_to_run = sorted(set(range(N_TRAJECTORIES)) - set(existing_seeds))
    print("%s trajectories to run" % len(seeds_to_run))

    for i in seeds_to_run:
        try:
            obs, info = env.reset(
                return_info=True, seed=i, action_exec_noise_std=0.1, spawn_noise_std=0.5
            )
            metadata = info.copy()
            rng = np.random.RandomState(seed=i)

            valid_dests = [
                i for i in range(len(Town10_spawns)) if i != info["spawn_idx"]
            ]
            dest_idx = rng.choice(valid_dests)
            dest = Town10_spawns[dest_idx].location

            agent = BuiltinAgent(env, [dest], seed=i)
            for j in range(T_MAX):
                action = agent.step(obs, info)
                action = np.array([action.throttle - action.brake, action.steer])
                action = np.clip(action, -1, 1)
                obs, reward, done, info = env.step(action)

                if agent.done():
                    dest_idx = np.random.choice(valid_dests)
                    dest = Town10_spawns[dest_idx].location
                    agent.set_route_points([dest])

            last_frame = info["frame"]
            while info["obs_frame"] < last_frame:
                action = agent.step(obs, info)
                action = np.array([action.throttle - action.brake, action.steer])
                action = np.clip(action, -1, 1)
                obs, reward, done, info = env.step(action)
            metadata["seed"] = i

            # async save
            p = mp.Process(
                target=save_trajectory_helper,
                args=(
                    agent,
                    SAVE_DIR,
                    metadata,
                    True,
                ),
            )
            p.start()
            processes.append(p)
        except Exception as e:
            print(e)
            continue

    env.close()
