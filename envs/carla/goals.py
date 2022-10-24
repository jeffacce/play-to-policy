from typing import Optional
from dataloaders.trajectory_loader import CarlaTrajectoryDataset
from utils import get_split_idx
import logging


def get_goal_fn_town04_merge(
    cfg,
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
):
    if goal_conditional is None:
        goal_fn = lambda state: None
    else:
        carla_traj = CarlaTrajectoryDataset(
            cfg.env_vars.datasets.carla_multipath_town04_merge, onehot_goals=True
        )
        train_idx, val_idx = get_split_idx(
            len(carla_traj), seed=seed, train_fraction=train_fraction
        )
        if goal_conditional == "future":
            assert (
                goal_seq_len is not None
            ), "goal_seq_len must be provided if goal_conditional is 'future'"

            def goal_fn(state, goal_idx, frame_idx):
                condition_on_end_overlap = (goal_idx % 3) == 0  # both routes count
                if condition_on_end_overlap:
                    logging.info("goal: overlapping end goal")
                    obs, _, _, _ = carla_traj[0]
                    obs = obs[-goal_seq_len:]
                else:
                    logging.info(f"goal_idx: {train_idx[goal_idx]}")
                    obs, _, _, _ = carla_traj[train_idx[goal_idx]]  # seq_len x obs_dim
                    obs = obs[-goal_seq_len - 100 : -100]
                return obs

        elif goal_conditional == "onehot":

            def goal_fn(state, goal_idx, frame_idx):
                _, _, _, onehot_goals = carla_traj[train_idx[goal_idx]]
                onehot_goals, _ = onehot_goals.max(0)
                return onehot_goals

    return goal_fn
