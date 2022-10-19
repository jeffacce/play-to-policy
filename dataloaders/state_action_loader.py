import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, Dataset
from pathlib import Path
from dataloaders.trajectory_loader import CarlaMultipathTrajectoryDataset
from envs.multi_route import multi_route
from utils import split_datasets


class RelayKitchenDataset(TensorDataset):
    def __init__(self, data_directory, device="cpu"):
        data_directory = Path(data_directory)
        observations = np.load(data_directory / "all_observations.npy")
        actions = np.load(data_directory / "all_actions.npy")

        super().__init__(
            torch.from_numpy(observations).to(device).float(),
            torch.from_numpy(actions).to(device).float(),
        )


class MultiPathDataset(TensorDataset):
    def __init__(
        self,
        path_waypoints=multi_route.MULTI_PATH_WAYPOINTS_1,
        path_probs=multi_route.PATH_PROBS_1,
        num_samples=20_000,
        device="cpu",
    ):
        path_generator = multi_route.PathGenerator(
            waypoints=path_waypoints,
            step_size=1,
            num_draws=100,
            noise_scale=0.05,
        )
        observations, actions = path_generator.get_memoryless_dataset(
            num_paths=num_samples, probabilities=path_probs
        )
        super().__init__(
            torch.from_numpy(observations).to(device).float(),
            torch.from_numpy(actions).to(device).float(),
        )


class GridDataset(TensorDataset):
    def __init__(
        self,
        grid_size=5,
        device="cpu",
        num_samples=1_000_000,
        top_prob=0.4,
        noise_scale=0.05,
        random_seed=42,
        scale_factor=1.0,
    ):
        rng = np.random.default_rng(random_seed)
        observations = rng.integers(
            grid_size,
            size=(num_samples, 2),
        ).astype(float)
        observations += rng.normal(loc=0.0, scale=noise_scale, size=observations.shape)
        actions = rng.binomial(n=1, p=top_prob, size=(num_samples, 1))
        actions = np.concatenate([1 - actions, actions], axis=1).astype(float)
        actions += rng.normal(loc=0.0, scale=noise_scale, size=observations.shape)
        actions = np.clip(actions * (1 - 2 * noise_scale), 0.0, 1.0)

        # Scale the actions to be between 0 and scale_factor
        observations, actions = scale_factor * observations, scale_factor * actions
        super().__init__(
            torch.from_numpy(observations).to(device).float(),
            torch.from_numpy(actions).to(device).float(),
        )


class CarlaMultipathDataset(Dataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
    ):
        self.device = device
        self.data_directory = Path(data_directory)
        self.trajectory_dataset = CarlaMultipathTrajectoryDataset(
            self.data_directory, self.device
        )
        self.valid_idx = []
        for i in range(len(self.trajectory_dataset.seq_lengths)):
            N = self.trajectory_dataset.seq_lengths[i]
            idx = np.array(
                [
                    np.ones(N) * i,
                    np.arange(N),
                ],
                dtype=int,
            ).T
            self.valid_idx.append(idx)
        self.valid_idx = np.concatenate(self.valid_idx)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        # observations: Tensor[N, C, H, W]
        # actions: Tensor[N, C]
        N, T = self.valid_idx[idx]
        return (
            self.trajectory_dataset.observations[N, T, :, :, :] / 255,
            self.trajectory_dataset.actions[N, T, :],
        )


def get_relay_kitchen_train_val(
    data_directory, train_fraction=0.95, random_seed=42, device="cpu"
):
    return split_datasets(
        RelayKitchenDataset(data_directory),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )


def get_multiroute_dataset(train_fraction=0.95, random_seed=42, device="cpu"):
    return split_datasets(
        MultiPathDataset(), train_fraction=train_fraction, random_seed=random_seed
    )


def get_grid_dataset(train_fraction=0.95, random_seed=42, device="cpu"):
    return split_datasets(
        GridDataset(random_seed=random_seed),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )


def get_carla_multipath_dataset(
    data_directory, train_fraction=0.95, random_seed=42, device="cpu"
):
    return split_datasets(
        CarlaMultipathDataset(data_directory),
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
