import matplotlib.pyplot as plt
import numpy as np
import hydra

import envs
import logging
from workspaces import base
from utils import get_split_idx
from dataloaders.trajectory_loader import CarlaTrajectoryDataset

carla_traj = CarlaTrajectoryDataset("/path/to/carla_dataset", onehot_goals=True)


class CarlaMultipathWorkspace(base.Workspace):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _setup_plots(self):
        pass

    def _setup_starting_state(self):
        self.known_seeds = list(range(100))

    def _start_from_known(self):
        # quick dirty hack to force spawn noise
        obs = self.env.reset(
            seed=None,
            spawn_noise_std=0.5,
        )
        return obs

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        pass

    def _report_result_upon_completion(self, goal_idx=None):
        if goal_idx is not None:
            train_idx, val_idx = get_split_idx(
                len(carla_traj),
                seed=self.cfg.seed,
                train_fraction=self.cfg.train_fraction,
            )
            _, _, _, onehot_labels = carla_traj[train_idx[goal_idx]]
            condition_on_end_overlap = (goal_idx % 3) == 0
            routes_used = self.env.last_info["routes_used"]
            expected_route = onehot_labels.max(0).values.bool().numpy()
            logging.info("Condition on end overlap: %s", condition_on_end_overlap)
            logging.info("Routes used: {}".format(routes_used))
            if condition_on_end_overlap:
                return int(any(routes_used)) * self.env.last_info["reward"]
            else:
                logging.info("Expected route: {}".format(expected_route))
                return (
                    int(all(routes_used == expected_route))
                    * self.env.last_info["reward"]
                )
        else:
            return 0


class CarlaMultipathStateWorkspace(CarlaMultipathWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
