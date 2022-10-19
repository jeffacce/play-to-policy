import torch
from typing import Optional
from dataloaders.trajectory_loader import (
    FrankaPlayKitchenDataset,
)


goal_traj = FrankaPlayKitchenDataset(
    "/path/to/franka-play-kitchen-play-dataset-byol-rgbx2-samplex5"
)


def get_goal_fn(
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
):
    if goal_conditional is None:
        goal_fn = lambda state: None
    elif goal_conditional == "future":
        assert (
            goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"

        def goal_fn(state):
            obs, act, mask = goal_traj[0]  # seq_len x obs_dim
            obs = obs[-goal_seq_len:]
            return obs

    return goal_fn
