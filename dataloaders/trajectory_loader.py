import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, Subset
from pathlib import Path
import numpy as np
from envs.multi_route import multi_route
from utils import (
    shuffle_along_axis,
    transpose_batch_timestep,
    eval_mode,
)
from typing import Union, Callable, Optional, Sequence, List, Any
from tqdm import tqdm
import abc
from torch import default_generator, randperm
from torch._utils import _accumulate


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_all_actions(self):
        return self.dataset.get_all_actions()


class RelayKitchenTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory, device="cpu", onehot_goals=False):
        data_directory = Path(data_directory)
        observations = torch.from_numpy(
            np.load(data_directory / "observations_seq.npy")
        )
        actions = torch.from_numpy(np.load(data_directory / "actions_seq.npy"))
        masks = torch.from_numpy(np.load(data_directory / "existence_mask.npy"))
        goals = torch.load(data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        observations, actions, masks, goals = transpose_batch_timestep(
            observations, actions, masks, goals
        )
        self.masks = masks
        tensors = [observations, actions, masks]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


class RelayKitchenVisionTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(self, data_directory, device="cpu", onehot_goals=False):
        data_directory = Path(data_directory)
        states = torch.from_numpy(np.load(data_directory / "observations_seq.npy"))
        actions = torch.from_numpy(np.load(data_directory / "actions_seq.npy"))
        masks = torch.from_numpy(np.load(data_directory / "existence_mask.npy"))
        goals = torch.load(data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        states, actions, masks, goals = transpose_batch_timestep(
            states, actions, masks, goals
        )
        # only take the joint angles (7 DoF), ignore grippers and env state
        states = states[:, :, :7]
        imgs_embedding = torch.load(data_directory / "observations_seq_embedding.pth")
        observations = torch.cat([imgs_embedding, states], dim=2)
        self.masks = masks
        tensors = [observations, actions, masks]
        if onehot_goals:
            tensors.append(goals)
        tensors = [t.to(device).float() for t in tensors]
        TensorDataset.__init__(self, *tensors)
        self.actions = self.tensors[1]

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


class FrankaPlayKitchenDataset(TrajectoryDataset):
    def __init__(self, data_directory, device="cpu"):
        data_directory = Path(data_directory)
        self.observations = torch.load(data_directory / "all_observations.pth")
        self.actions = torch.load(data_directory / "all_actions.pth")
        self.seq_lengths = torch.load(data_directory / "seq_lengths.pth")
        # augment_theta = self._joint_states_augmentation(obses, 1.02, 73)

    def _joint_states_augmentation(self, obs, base_val, repeat_num):
        # perform repeat joint augmentation on each frame
        power = torch.arange(repeat_num) + 1
        augment_val = (base_val**power).reshape(
            1, 1, repeat_num, 1
        )  # shape [1, 1, repeat_num, 1]

        sin_theta = obs[..., -14:-7]
        cos_theta = obs[..., -7:]
        theta = torch.atan2(sin_theta, cos_theta).unsqueeze(2)  # shape: [N, T, 1, 7]
        theta = torch.repeat_interleave(
            theta, repeat_num, dim=2
        )  # [N, T, repeat_num, 7]

        augment_theta = augment_val * theta
        sin_theta = torch.sin(augment_theta)
        cos_theta = torch.cos(augment_theta)

        # change shape of augment_theta [N, T, repeat_num, 7] -> [N, T, repeat_num, 14] -> [N, T, repeat_num * 14]
        augment_theta = torch.cat((sin_theta, cos_theta), dim=3).reshape(
            obs.shape[0], obs.shape[1], repeat_num * 14
        )  # shape: [N, T, repeat_num*14]

        return augment_theta

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        T = self.seq_lengths[idx]
        observations = self.observations[idx, :T]
        joint_states = observations[:, -14:]
        joint_states = joint_states.repeat(1, 73)
        observations = torch.cat([observations, joint_states], dim=1)
        actions = self.actions[idx, :T]
        mask = torch.ones(T, dtype=torch.float32)  # existence mask
        return (
            observations,
            actions,
            mask,
        )


class CarlaMultipathTrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        subset_fraction: float = 1.0,
        onehot_goals: bool = False,
        preprocess_to_float: bool = False,
    ):
        assert 0.0 < subset_fraction <= 1.0, "subset_fraction must be in (0, 1]"
        self.data_directory = Path(data_directory)
        self.onehot_goals = onehot_goals

        print("CARLA loading: started")
        self.seq_lengths_all = torch.load(self.data_directory / "seq_lengths.pth")
        self.observations_all = torch.load(self.data_directory / "all_observations.pth")
        self.actions_all = torch.load(self.data_directory / "all_actions_pm1.pth")
        if onehot_goals:
            self.goals = torch.load(self.data_directory / "onehot_goals.pth")
        else:
            self.goals = None
        print("CARLA loading: done")

        N = int(len(self.seq_lengths_all) * subset_fraction)
        self.seq_lengths = self.seq_lengths_all[:N]
        self.observations = self.observations_all[:N, :, :, :, :]
        self.actions = self.actions_all[:N, :, :]
        del self.seq_lengths_all, self.observations_all, self.actions_all

        self.preprocess_to_float = preprocess_to_float
        # NOTE: dividing by 255 might explode memory if image size is large
        if self.preprocess_to_float:
            self.observations = self.observations / 255.0

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        # observations: Tensor[N, T, C, H, W]
        # actions: Tensor[N, T, C]
        T = self.seq_lengths[idx]
        observations = self.observations[idx, :T, :, :, :]
        if not self.preprocess_to_float:
            observations = observations / 255.0
        actions = self.actions[idx, :T, :]
        mask = torch.ones(T, dtype=torch.float32)  # existence mask
        if self.onehot_goals:
            goals = self.goals[idx, :T]
            return (observations, actions, mask, goals)
        else:
            return (observations, actions, mask)

    def get_seq_length(self, idx) -> int:
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class CarlaMultipathStateTrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        subset_fraction: float = 1.0,
        obs_noise_scale: float = 0.0,
        device="cpu",
    ):
        assert 0.0 < subset_fraction <= 1.0, "subset_fraction must be in (0, 1]"
        self.device = device
        self.data_directory = Path(data_directory)
        self.seq_lengths_all = torch.load(self.data_directory / "seq_lengths.pth")
        self.observations_all = torch.load(
            self.data_directory / "all_observations.pth"
        ).float()
        self.actions_all = torch.load(
            self.data_directory / "all_actions_pm1.pth"
        ).float()
        # normalize observations
        self.observations_mean = torch.load(
            self.data_directory / "observations_mean.pth"
        )
        self.observations_std = torch.load(self.data_directory / "observations_std.pth")
        # don't normalize the last col of affine (always 0,0,0,1)
        self.observations_std[12:16] = 1.0
        self.observations_all -= self.observations_mean
        self.observations_all /= self.observations_std
        self.obs_noise_scale = obs_noise_scale

        N = int(len(self.seq_lengths_all) * subset_fraction)
        self.seq_lengths = self.seq_lengths_all[:N]
        self.observations = self.observations_all[:N, :, :]
        self.actions = self.actions_all[:N, :, :]
        del self.seq_lengths_all, self.observations_all, self.actions_all

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        # observations: Tensor[batch seq state_dim(23)]
        # actions: Tensor[batch seq action_dim(2)]
        T = self.seq_lengths[idx]
        observations = self.observations[idx, :T, :]
        observations += torch.randn_like(observations) * self.obs_noise_scale
        actions = self.actions[idx, :T, :]
        mask = torch.ones(T, dtype=torch.float32)  # existence mask
        return (
            observations,
            actions,
            mask,
        )

    def get_seq_length(self, idx) -> int:
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class CarlaTrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        onehot_goals: bool = False,
    ):
        self.data_directory = Path(data_directory)
        self.onehot_goals = onehot_goals

        print("CARLA loading: started")
        self.seq_lengths = torch.load(self.data_directory / "seq_lengths.pth")
        self.observations = torch.load(self.data_directory / "all_observations.pth")
        self.actions = torch.load(self.data_directory / "all_actions_pm1.pth")
        if onehot_goals:
            self.goals = torch.load(self.data_directory / "onehot_goals.pth")
        else:
            self.goals = None
        print("CARLA loading: done")

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        T = self.seq_lengths[idx]
        observations = self.observations[idx, :T]
        actions = self.actions[idx, :T]
        mask = torch.ones(T, dtype=torch.float32)  # existence mask
        if self.onehot_goals:
            goals = self.goals[idx, :T]
            return (observations, actions, mask, goals)
        else:
            return (observations, actions, mask)

    def get_seq_length(self, idx) -> int:
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)


class PushTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        onehot_goals=False,
    ):
        self.device = device
        self.data_directory = Path(data_directory)
        logging.info("Multimodal loading: started")
        self.observations = np.load(
            self.data_directory / "multimodal_push_observations.npy"
        )
        self.actions = np.load(self.data_directory / "multimodal_push_actions.npy")
        self.masks = np.load(self.data_directory / "multimodal_push_masks.npy")
        self.observations = torch.from_numpy(self.observations).to(device).float()
        self.actions = torch.from_numpy(self.actions).to(device).float()
        self.masks = torch.from_numpy(self.masks).to(device).float()
        self.goals = (
            torch.load(self.data_directory / "onehot_goals.pth").to(device).float()
        )
        logging.info("Multimodal loading: done")
        tensors = [self.observations, self.actions, self.masks]
        if onehot_goals:
            tensors.append(self.goals)
        # The current values are in shape N x T x Dim, so all is good in the world.
        TensorDataset.__init__(self, *tensors)

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def __getitem__(self, idx):
        T = self.masks[idx].sum().int().item()
        return tuple(x[idx, :T] for x in self.tensors)


class MultiPathTrajectoryDataset(TensorDataset, TrajectoryDataset):
    def __init__(
        self,
        path_waypoints=multi_route.MULTI_PATH_WAYPOINTS_1,
        path_probs=multi_route.PATH_PROBS_1,
        num_samples=200_000,
        device="cpu",
    ):
        path_generator = multi_route.PathGenerator(
            waypoints=path_waypoints,
            step_size=1,
            num_draws=100,
            noise_scale=0.05,
        )
        observations, actions, length_mask = path_generator.get_sequence_dataset(
            num_paths=num_samples, probabilities=path_probs
        )
        self.length_mask = length_mask
        TensorDataset.__init__(
            self,
            torch.from_numpy(observations).to(device).float(),
            torch.from_numpy(actions).to(device).float(),
            torch.from_numpy(length_mask).to(device).float(),
        )

    def get_seq_length(self, idx) -> int:
        return int(self.length_mask[idx].sum().item())

    def get_all_actions(self):
        return self.tensors[1]


class GridTrajectoryDataset(TensorDataset, TrajectoryDataset):
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
        total_grid_size = grid_size * 2
        top_length = int(total_grid_size * top_prob)
        side_length = total_grid_size - top_length

        all_up_actions = np.concatenate(
            [np.ones((num_samples, top_length)), np.zeros((num_samples, side_length))],
            axis=-1,
        ).astype(float)
        all_up_actions = shuffle_along_axis(all_up_actions, axis=-1)
        all_side_actions = 1.0 - all_up_actions
        all_actions = np.stack([all_up_actions, all_side_actions], axis=-1)
        all_observations = np.cumsum(all_actions, axis=1)  # [N, T, 2]
        all_actions += rng.normal(scale=noise_scale, size=all_actions.shape)
        all_observations += rng.normal(scale=noise_scale, size=all_observations.shape)

        # Scale the actions to be between 0 and scale_factor
        all_observations, all_actions = (
            scale_factor * all_observations,
            scale_factor * all_actions,
        )
        # All cells are valid
        mask = np.ones(all_observations.shape[:-1])
        self.mask = mask

        TensorDataset.__init__(
            self,
            torch.from_numpy(all_observations).to(device).float(),
            torch.from_numpy(all_actions).to(device).float(),
            torch.from_numpy(mask).to(device).float(),
        )

    def get_seq_length(self, idx) -> int:
        return int(self.mask[idx].sum().item())

    def get_all_actions(self):
        return self.tensors[1]


class TrajectorySlicerDataset(TrajectoryDataset):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Slice a trajectory dataset into unique (but overlapping) sequences of length `window`.

        dataset: a trajectory dataset that satisfies:
            dataset.get_seq_length(i) is implemented to return the length of sequence i
            dataset[i] = (observations, actions, mask)
            observations: Tensor[T, ...]
            actions: Tensor[T, ...]
            mask: Tensor[T]
                0: invalid
                1: valid
        window: int
            number of timesteps to include in each slice
        future_conditional: bool = False
            if True, observations will be augmented with future observations sampled from the same trajectory
        min_future_sep: int = 0
            minimum number of timesteps between the end of the current sequence and the start of the future sequence
            for the future conditional
        future_seq_len: Optional[int] = None
            the length of the future conditional sequence;
            required if future_conditional is True
        only_sample_tail: bool = False
            if True, only sample future sequences from the tail of the trajectory
        transform: function (observations, actions, mask[, goal]) -> (observations, actions, mask[, goal])
        """
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.slices = []
        min_seq_length = np.inf
        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)
            if T - window < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={window}")
            else:
                self.slices += [
                    (i, start, start + window) for start in range(T - window)
                ]  # slice indices follow convention [start, end)

        if min_seq_length < window:
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def get_all_actions(self) -> torch.Tensor:
        return self.dataset.get_all_actions()

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        values = [
            x[start:end] for x in self.dataset[i]
        ]  # [observations, actions, mask]

        if self.future_conditional:
            valid_start_range = (
                end + self.min_future_sep,
                self.dataset.get_seq_length(i) - self.future_seq_len,
            )
            if valid_start_range[0] < valid_start_range[1]:
                if self.only_sample_tail:
                    future_obs = self.dataset[i][0][-self.future_seq_len :]
                else:
                    start = np.random.randint(*valid_start_range)
                    end = start + self.future_seq_len
                    future_obs = self.dataset[i][0][start:end]
            else:
                # zeros placeholder T x obs_dim
                _, obs_dim = values[0].shape
                future_obs = torch.zeros((self.future_seq_len, obs_dim))

            # [observations, actions, mask, future_obs (goal conditional)]
            values.append(future_obs)

        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return tuple(values)


class TrajectoryRepDataset(TrajectoryDataset):
    def __init__(
        self,
        trajectory_dataset: TrajectoryDataset,
        encoder: nn.Module,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Union[torch.device, str] = "cuda",
        batch_size: Optional[int] = 128,
    ):
        """
        Given a trajectory dataset, encode its states into representations.
        Inputs:
            trajectory_dataset: a trajectory dataset that satisfies:
                dataset[i] = (observations, *)
                observations: Tensor[T, ...]
                `*` (any following tensors): Tensor[T, ...]
            encoder: a module that accepts observations and returns a representation
            device: encoder will be run on this device
            batch_size: if not None, will batch frames into batches of this size (to avoid OOM)
        """
        self.device = device
        encoder = encoder.to(device)  # not saving encoder to lower VRAM usage
        self.data = []
        self.postprocess = postprocess
        self.seq_lengths = [
            trajectory_dataset.get_seq_length(i) for i in range(len(trajectory_dataset))
        ]
        self.all_actions = trajectory_dataset.get_all_actions()
        with eval_mode(encoder, no_grad=True):
            for i in tqdm(range(len(trajectory_dataset))):
                elem = trajectory_dataset[i]
                obs = elem[0]
                if preprocess is not None:
                    obs = preprocess(obs)
                if batch_size is not None:
                    obs_enc = []
                    for t in range(0, obs.shape[0], batch_size):
                        batch = obs[t : t + batch_size].to(self.device)
                        obs_enc.append(encoder(batch).cpu())
                    obs_enc = torch.cat(obs_enc, dim=0)
                else:
                    obs_enc = encoder(obs.to(self.device)).cpu()
                self.data.append((obs_enc, *elem[1:]))
        del encoder
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        obs = elem[0]
        if self.postprocess is not None:
            obs = self.postprocess(obs)
        return (obs, *elem[1:])

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return self.all_actions


def get_train_val_sliced(
    traj_dataset: TrajectoryDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    device: Union[str, torch.device] = "cpu",
    window_size: int = 10,
    future_conditional: bool = False,
    min_future_sep: int = 0,
    future_seq_len: Optional[int] = None,
    only_sample_tail: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    traj_slicer_kwargs = {
        "window": window_size,
        "future_conditional": future_conditional,
        "min_future_sep": min_future_sep,
        "future_seq_len": future_seq_len,
        "only_sample_tail": only_sample_tail,
        "transform": transform,
    }
    train_slices = TrajectorySlicerDataset(train, **traj_slicer_kwargs)
    val_slices = TrajectorySlicerDataset(val, **traj_slicer_kwargs)
    return train_slices, val_slices


def random_split_traj(
    dataset: TrajectoryDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajectorySubset]:
    """
    (Modified from torch.utils.data.dataset.random_split)

    Randomly split a trajectory dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split_traj(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (TrajectoryDataset): TrajectoryDataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def get_relay_kitchen_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    transform: Optional[Callable[[Any], Any]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        RelayKitchenTrajectoryDataset(
            data_directory, onehot_goals=(goal_conditional == "onehot")
        ),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
    )


def get_relay_kitchen_vision_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        RelayKitchenVisionTrajectoryDataset(
            data_directory, onehot_goals=(goal_conditional == "onehot")
        ),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
    )


def get_franka_play_kitchen_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    sample_num: int = 1,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future"]
    return get_train_val_sliced(
        FrankaPlayKitchenDataset(data_directory),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
    )


def get_push_train_val(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    only_sample_tail: bool = False,
    transform: Optional[Callable[[Any], Any]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        PushTrajectoryDataset(
            data_directory, onehot_goals=(goal_conditional == "onehot")
        ),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        only_sample_tail=only_sample_tail,
        transform=transform,
    )


def get_multiroute_dataset(
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
):
    return get_train_val_sliced(
        MultiPathTrajectoryDataset(num_samples=20_000),
        train_fraction,
        random_seed,
        device,
        window_size,
    )


def get_grid_dataset(train_fraction=0.9, random_seed=42, device="cpu", window_size=10):
    return get_train_val_sliced(
        GridTrajectoryDataset(random_seed=random_seed),
        train_fraction,
        random_seed,
        device,
        window_size,
    )


def get_carla_multipath_dataset(
    data_directory,
    subset_fraction=1.0,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    preprocess_to_float: bool = False,
):
    return get_train_val_sliced(
        CarlaMultipathTrajectoryDataset(
            data_directory,
            subset_fraction=subset_fraction,
            device=device,
            preprocess_to_float=preprocess_to_float,
        ),
        train_fraction,
        random_seed,
        device,
        window_size,
    )


def get_carla_multipath_state_dataset(
    data_directory,
    subset_fraction=1.0,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
):
    return get_train_val_sliced(
        CarlaMultipathStateTrajectoryDataset(
            data_directory,
            subset_fraction=subset_fraction,
            device=device,
        ),
        train_fraction,
        random_seed,
        device,
        window_size,
    )


def get_carla_trajectory_dataset(
    data_directory,
    train_fraction=0.9,
    random_seed=42,
    device="cpu",
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    transform: Optional[Callable[[Any], Any]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    return get_train_val_sliced(
        CarlaTrajectoryDataset(
            data_directory,
            onehot_goals=(goal_conditional == "onehot"),
        ),
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
        transform=transform,
    )


def get_carla_multipath_rep_dataset(
    data_directory,
    subset_fraction=1.0,
    train_fraction=0.9,
    random_seed=42,
    device="cuda",
    batch_size=None,
    window_size=10,
    goal_conditional: Optional[str] = None,
    future_seq_len: Optional[int] = None,
    min_future_sep: int = 0,
    encoder: nn.Module = nn.Identity,
    preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    if goal_conditional is not None:
        assert goal_conditional in ["future", "onehot"]
    dataset = TrajectoryRepDataset(
        CarlaMultipathTrajectoryDataset(
            data_directory,
            subset_fraction=subset_fraction,
            onehot_goals=(goal_conditional == "onehot"),
            preprocess_to_float=False,
        ),
        encoder,
        preprocess=preprocess,
        postprocess=postprocess,
        device=device,
        batch_size=batch_size,
    )
    return get_train_val_sliced(
        dataset,
        train_fraction,
        random_seed,
        device,
        window_size,
        future_conditional=(goal_conditional == "future"),
        min_future_sep=min_future_sep,
        future_seq_len=future_seq_len,
    )
