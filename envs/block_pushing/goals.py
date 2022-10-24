import torch
import logging
import numpy as np
from typing import Optional
from dataloaders.trajectory_loader import PushTrajectoryDataset
from utils import get_split_idx


def get_goal_fn(
    cfg,
    goal_conditional: Optional[str] = None,
    goal_seq_len: Optional[int] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
):
    push_traj = PushTrajectoryDataset(
        cfg.env_vars.datasets.multimodal_push_fixed_target, onehot_goals=True
    )
    train_idx, val_idx = get_split_idx(
        len(push_traj),
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
            # assuming at this point the state hasn't been masked yet by the obs_encoder
            obs, _, _, _ = push_traj[train_idx[goal_idx]]
            obs[..., [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = 0
            # obs = obs[-goal_seq_len:]
            obs = obs[-1:].repeat(goal_seq_len, 1)
            return obs

    elif goal_conditional == "onehot":

        def goal_fn(state, goal_idx, frame_idx):
            _, _, _, onehot_goals = push_traj[train_idx[goal_idx]]
            onehot_mask, first_frame = onehot_goals.max(0)
            goals = [(first_frame[i], i) for i in range(4) if onehot_mask[i]]
            goals = sorted(goals, key=lambda x: x[0])
            goals = [g[1] for g in goals]
            last_goal = goals[-1]  # in case everything is done, return the last goal
            if frame_idx == 0:
                logging.info(f"goal_idx: {train_idx[goal_idx]}")
                logging.info(f"goals: {goals}")

            # determine which goals are already done
            block_idx = [[0, 1], [3, 4]]
            target_idx = [[10, 11], [13, 14]]
            close_eps = 0.05
            for b in range(2):
                for t in range(2):
                    blk = state[block_idx[b]]
                    tgt = state[target_idx[t]]
                    dist = np.linalg.norm(blk - tgt)
                    if dist < close_eps:
                        if (2 * b + t) in goals:
                            goals.remove(2 * b + t)
            result = torch.zeros(4)
            if len(goals) > 0:
                result[goals[0]] = 1
            else:
                result[last_goal] = 1
            return result

    return goal_fn
