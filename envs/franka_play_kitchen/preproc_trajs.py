import sys

sys.path.append("../..")

import torch
import einops
import traceback
import numpy as np
import torchvision
from tqdm import tqdm
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Union, Tuple, List
from envs.franka_play_kitchen import preproc, utils, streams
import random

SAVE_STRUCTURE = [
    ("cam_0", streams.Camera, {"read_rgb": True, "read_depth": False}),
    ("cam_1", streams.Camera, {"read_rgb": True, "read_depth": False}),
    ("robot_state.pkl", streams.RobotStateStream, {}),
    ("gripper_state.pkl", streams.GripperStateStream, {}),
]

# assuming this is run in envs/franka_play_kitchen
encoder_0, normalizer_0 = utils.get_resnet18_encoder(
    Path(sys.path[0]) / ".." / ".." / "models" / "weights", "byol_rgb0"
)
encoder_1, normalizer_1 = utils.get_resnet18_encoder(
    Path(sys.path[0]) / ".." / ".." / "models" / "weights", "byol_rgb1"
)
camera_preproc_0 = preproc.camera.CameraPreproc(
    device="cuda:0",
    encoder=encoder_0,
    crop=utils.CAMERA_0_CROP,
    additional_transforms=[normalizer_0],
)
camera_preproc_1 = preproc.camera.CameraPreproc(
    device="cuda:1",
    encoder=encoder_1,
    crop=utils.CAMERA_1_CROP,
    additional_transforms=[normalizer_1],
)

OBSES_STRUCT = [
    ("cam_0", camera_preproc_0),  # RGB (512) embeddings
    ("cam_1", camera_preproc_1),
    ("robot_state.pkl", preproc.robot),  # sin, cos of joint angles (14)
]

ACTIONS_STRUCT = [
    ("robot_state.pkl", preproc.robot),  # d(joint_thetas) (7)
    ("gripper_state.pkl", preproc.gripper),  # target (1)
]


class FrankaRawTrajDataset(Dataset):
    def __init__(
        self,
        paths,
        fps: float,
        sample_num: int,
        frame_jitter: int,
        frame_filter: Optional[Callable[[Dict, int, int], List]] = None,
        visualizer: Optional[Callable[[Path, Dict, int], None]] = None,
    ):
        self.paths = paths
        self.fps = fps
        self.frame_filter = frame_filter
        self.visualizer = visualizer
        self.sample_num = sample_num
        self.frame_jitter = frame_jitter

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path, self._extract_raw(
            path, self.fps, self.frame_filter, self.visualizer
        )

    def __len__(self):
        return len(self.paths)

    def _read_traj(self, traj_path: Union[str, Path]) -> Tuple[Dict, float, float]:
        data = {}
        traj_path = Path(traj_path)
        for path, stream_cls, kwargs in SAVE_STRUCTURE:
            data[path] = stream_cls.read_buffer(traj_path / path, **kwargs)
        with open(traj_path / "start_stop.txt") as f:
            start, stop = (float(x.strip()) for x in f.readlines())

        return data, start, stop

    def _extract_raw(
        self,
        path: Union[str, Path],
        fps: float,
        frame_filter: Optional[Callable[[Dict, int, int], List]] = None,
        visualizer: Optional[Callable[[Path, Dict, int], None]] = None,
    ) -> Optional[List[Dict]]:
        print("Extracting from: %s" % path)
        try:
            path = Path(path)
            data, start, stop = self._read_traj(path)
        except Exception as e:
            print("Error extracting from %s; returning None" % path)
            traceback.print_exc()
            return None
        data = preproc.sync(data, fps, start, stop)
        if frame_filter is not None:
            sample_idx = frame_filter(data, self.sample_num, self.frame_jitter)
        else:
            # ! NOTE: not tested!
            sample_idx = [
                list(range(len(data["robot_state.pkl"][1]))),  # all frames
            ]
        data_list = []
        for idx in sample_idx:
            sampled_data = {k: utils.subset_each(v, idx) for k, v in data.items()}
            data_list.append(sampled_data)

        for i in range(len(data_list)):
            if visualizer is not None:
                visualizer(path, data_list[i], i)
        return data_list


def drop_idle(data, sample_num: int, frame_jitter: int):
    robot_state = data["robot_state.pkl"][1]
    gripper_target = preproc.gripper.get_actions(data["gripper_state.pkl"])[1]
    gripper_change_mask = torch.diff(gripper_target, 1, dim=0) != 0
    gripper_change_idx = torch.where(gripper_change_mask)[0]
    gripper_change_idx = torch.cat(
        (gripper_change_idx, gripper_change_idx + 1)
    ).tolist()
    thetas = torch.Tensor(np.stack([x.joint_positions for x in robot_state]))
    sample_idx = []

    # the original trajectory without any frame jitter
    start_idx = 0
    idx = [0]
    idx += gripper_change_idx
    for i in range(start_idx + 1, len(thetas)):
        diff = thetas[i] - thetas[start_idx]
        if diff.norm() > IDLE_THRESH:
            idx.append(i)
            start_idx = i
    idx = sorted(set(idx))
    sample_idx.append(idx)

    for i in range(sample_num - 1):
        traj_idx = []
        traj_idx += gripper_change_idx
        last_frame = -1
        for j in range(len(idx)):
            lower = max(last_frame + 1, idx[j] - frame_jitter)
            if j + 1 >= len(idx):
                upper = idx[j]
            else:
                upper = min(idx[j] + frame_jitter, idx[j + 1] - 1)
            traj_idx.append(random.randint(lower, upper))  # inclusive both ends
            last_frame = traj_idx[-1]
        traj_idx = sorted(set(traj_idx))
        sample_idx.append(traj_idx)
    return sample_idx


def visualizer(path, data, index):
    cam_preprocs = {
        "cam_0": camera_preproc_0,
        "cam_1": camera_preproc_1,
    }
    for stream_name in ["cam_0", "cam_1"]:
        rgb = data[stream_name][1]
        camera_preproc = cam_preprocs[stream_name]
        raw_rgb = camera_preproc.preproc_rgb(rgb)
        torch.save(
            raw_rgb, str(path / f"{stream_name}_raw_rgb_{index}.pt")
        )  # shape: T C H W
        rgb = einops.rearrange(rgb, "t c h w -> t h w c")
        torchvision.io.write_video(
            str(path / f"{stream_name}_vis_{index}.mp4"), rgb, fps=5
        )


def is_valid_path(p: Path) -> bool:
    if not p.is_dir():
        return False
    invalid_keywords = ["ignore", "reset"]
    return all(x not in str(p) for x in invalid_keywords)


def preproc_traj(data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    obses = {}
    actions = {}
    for k, preproc_cls in OBSES_STRUCT:
        obses[k] = preproc_cls.get_obses(data[k])
        obses[k] = obses[k][1:]  # drop timestamp series
    for k, preproc_cls in ACTIONS_STRUCT:
        actions[k] = preproc_cls.get_actions(data[k])
        actions[k] = actions[k][1:]  # drop timestamp series

    flat_obses = []
    flat_actions = []
    for k, _ in OBSES_STRUCT:
        flat_obses += obses[k]
    for k, _ in ACTIONS_STRUCT:
        flat_actions += actions[k]
    flat_obses = torch.cat(flat_obses, dim=1)
    flat_actions = torch.cat(flat_actions, dim=1)

    return flat_obses, flat_actions


NUM_WORKERS = 8
FPS = 30
IDLE_THRESH = 0.1  # threshold for action norm considered idle
FILTER_FUNC = drop_idle
VISUALIZER_FUNC = None
SAMPLE_NUM = 5
FRAME_JITTER = 5


if __name__ == "__main__":
    invalid_paths = []
    if len(sys.argv) < 2:
        print("Usage: preproc_trajs.py <path_to_folder_of_trajectories>")
        sys.exit(1)
    else:
        traj_root_dir = Path(sys.argv[1])
        traj_paths = [x for x in traj_root_dir.glob("*")]
        traj_paths = sorted(filter(is_valid_path, traj_paths))
        traj_ds = FrankaRawTrajDataset(
            traj_paths, FPS, SAMPLE_NUM, FRAME_JITTER, FILTER_FUNC, VISUALIZER_FUNC
        )
        traj_loader = DataLoader(
            traj_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=NUM_WORKERS,
            # prefetch_factor=1,
        )
        for path, data in tqdm(traj_loader):
            if data is not None:
                try:
                    for i in range(SAMPLE_NUM):
                        obses, actions = preproc_traj(data[i])
                        torch.save(obses, path / f"obses_{i}.pt")
                        torch.save(actions, path / f"actions_{i}.pt")
                except Exception as e:
                    print("Invalid data in %s" % path)
                    traceback.print_exc()
                    invalid_paths.append(path)
            else:
                print("Invalid data in %s" % path)
                invalid_paths.append(path)

    print("All invalid paths:")
    print(invalid_paths)

    all_obses = []
    all_actions = []
    seq_lengths = []
    for path in tqdm(traj_paths):
        path = Path(path)
        if path not in invalid_paths:
            for i in range(SAMPLE_NUM):
                obses = torch.load(path / f"obses_{i}.pt")
                actions = torch.load(path / f"actions_{i}.pt")
                assert len(obses) == len(actions)
                all_obses.append(obses)
                all_actions.append(actions)
                seq_lengths.append(len(obses))
    all_obses = pad_sequence(all_obses, batch_first=True)
    all_actions = pad_sequence(all_actions, batch_first=True)
    torch.save(all_obses, traj_root_dir / "all_observations.pth")
    torch.save(all_actions, traj_root_dir / "all_actions.pth")
    torch.save(seq_lengths, traj_root_dir / "seq_lengths.pth")
