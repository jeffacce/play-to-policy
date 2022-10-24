import einops
import gym
import hydra
import joblib
import torch
import wandb
import logging

import utils
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from workspaces import base
import envs
from dataloaders.trajectory_loader import PushTrajectoryDataset
from utils import get_split_idx


class MaskOutTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x[..., 10:] = 0
        return x


class BlockPushWorkspace(base.Workspace):
    def __init__(self, cfg):
        self.push_traj = PushTrajectoryDataset(
            cfg.env_vars.datasets.multimodal_push_fixed_target, onehot_goals=True
        )
        super().__init__(cfg)
        # NOTE: only for future conditional
        self.obs_encoding_net = MaskOutTarget()

    def _setup_plots(self):
        pass

    def _setup_starting_state(self):
        pass

    def _start_from_known(self):
        pass

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        pass

    def _report_result_upon_completion(self, goal_idx=None):
        if goal_idx is not None:
            train_idx, val_idx = get_split_idx(
                len(self.push_traj),
                seed=self.cfg.seed,
                train_fraction=self.cfg.train_fraction,
            )
            _, _, _, onehot_goals = self.push_traj[train_idx[goal_idx]]
            onehot_mask, first_frame = onehot_goals.max(0)
            goals = [(first_frame[i], i) for i in range(4) if onehot_mask[i]]
            goals = sorted(goals, key=lambda x: x[0])
            goals = [g[1] for g in goals]
            logging.info(f"Expected tasks {goals}")
            expected_tasks = set(goals)
            conditional_done = set(self.env.all_completions).intersection(
                expected_tasks
            )
            return len(conditional_done) / 2
        else:
            return len(self.env.all_completions) / 2
