import sys

sys.path.append("../..")

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import envs
from envs.carla.builtin_agent import BuiltinAgent
from envs.carla.v0 import preproc_carla_img
import gym
import multiprocessing as mp
from typing import Dict
import torchvision


def save_trajectory_helper(
    agent: BuiltinAgent,
    path: os.PathLike,
    metadata: Dict,
):
    trajectory_name = datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    trajectory_dir = Path(path) / trajectory_name
    trajectory_dir.mkdir(parents=True)
    print("Saving trajectory to {}".format(trajectory_dir))
    with open(trajectory_dir / "actions.json", "w") as f:
        json.dump(agent.actions, f)
    with open(trajectory_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    np.save(trajectory_dir / "states.npy", agent.states)
    print("Saved trajectory to", trajectory_dir)
    return trajectory_dir


if __name__ == "__main__":
    env = gym.make(
        "carla-multipath-state-v0",
        noise_std=0.5,
    )

    for i in range(1000):
        obs, info = env.reset(return_info=True, seed=i)
        print(env.world.get_weather())
        frame = info["frame"]
        done = False
        rng = np.random.RandomState(seed=i)
        destination = rng.choice(list(env.DESTINATIONS.values()))
        agent = BuiltinAgent(
            env,
            destination,
            action_mult_noise_range=(1.0, 1.0),
            seed=i,
        )
        while not done:
            action = agent.step(obs, info)
            print(len(agent.states), len(agent.actions))
            action = np.array([action.throttle - action.brake, action.steer])
            action = np.clip(action, -1, 1)
            obs, reward, done, info = env.step(action)

        last_frame = info["frame"]
        while info["obs_frame"] < last_frame:
            action = agent.step(obs, info)
            action = np.array([action.throttle - action.brake, action.steer])
            action = np.clip(action, -1, 1)
            obs, reward, done, info = env.step(action)

        save_trajectory_helper(agent, "/path/to/carla_dataset/runs_state", {"seed": i})

    env.close()
