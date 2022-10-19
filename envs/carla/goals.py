import torch
from typing import Optional
from models.resnet import resnet18
from dataloaders.trajectory_loader import CarlaTrajectoryDataset
from utils import get_split_idx
import logging


# encoder = resnet18(pretrained=True, freeze_pretrained=True).eval()


def get_goal_fn_random_drive(
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
):
    if goal_conditional is None:
        goal_fn = lambda state: None
    elif goal_conditional == "future":
        DATA_DIR = "/path/to/carla_dataset/random_drive_2x"
        carla_traj = CarlaTrajectoryDataset(DATA_DIR)
        lookup_table = torch.load(f"{DATA_DIR}/waypoint_traj_frame.pth")
        assert (
            goal_seq_len is not None
        ), "goal_seq_len must be provided if goal_conditional is 'future'"
        waypoint_idx = 111
        _, traj_idx, frame_idx = list(
            filter(lambda x: x[0] == waypoint_idx, lookup_table)
        )[0]

        obs, act, mask = carla_traj[traj_idx]  # seq_len x obs_dim
        obs = obs[frame_idx : frame_idx + goal_seq_len]
        # with torch.no_grad():
        #     obs = encoder(obs)

        def goal_fn(state):
            return obs

    elif goal_conditional == "onehot":

        def goal_fn(state):
            return torch.Tensor([1, 0])

    return goal_fn


def get_goal_fn_town04_merge(
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
):
    if goal_conditional is None:
        goal_fn = lambda state: None
    else:
        DATA_DIR = "/path/to/carla_dataset/town04_merge_2x"
        carla_traj = CarlaTrajectoryDataset(DATA_DIR, onehot_goals=True)
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
