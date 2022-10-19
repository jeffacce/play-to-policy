import torch
import logging
from typing import Optional
from dataloaders.trajectory_loader import RelayKitchenTrajectoryDataset
from utils import get_split_idx
from . import OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES


def get_goal_fn(
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
):
    relay_traj = RelayKitchenTrajectoryDataset(
        "/path/to/relay_kitchen_dataset", onehot_goals=True
    )
    train_idx, val_idx = get_split_idx(
        len(relay_traj),
        seed=seed,
        train_fraction=train_fraction,
    )
    if goal_conditional is None:
        goal_fn = lambda state: None
    elif goal_conditional == "future":
        assert (
            goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"

        def goal_fn(state, goal_idx, frame_idx):
            logging.info(f"goal_idx: {train_idx[goal_idx]}")
            obs, _, _, _ = relay_traj[train_idx[goal_idx]]  # seq_len x obs_dim
            obs = obs[-goal_seq_len:]
            return obs

    elif goal_conditional == "onehot":

        def goal_fn(state, goal_idx, frame_idx):
            if frame_idx == 0:
                logging.info(f"goal_idx: {train_idx[goal_idx]}")
            _, _, _, onehot_goals = relay_traj[train_idx[goal_idx]]  # seq_len x obs_dim
            return onehot_goals[min(frame_idx, len(onehot_goals) - 1)]

    return goal_fn
