import abc
import torch
import pickle
import einops
import polymetis
import numpy as np
import torchvision
import pyrealsense2 as rs
from pathlib import Path
from typing import Any, List
from .. import preproc, utils


class AbstractStream(abc.ABC):
    @abc.abstractmethod
    def get_data(self) -> Any:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    def save_buffer(self, buf: Any, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump(buf, f)

    @staticmethod
    def read_buffer(filepath: Path):
        with open(filepath, "rb") as f:
            result = pickle.load(f)
        return result


class Camera(AbstractStream):
    def __init__(
        self, idx, resolution=(1280, 720), fps=30, warmup_frames=30, align_rgbd=True
    ):
        ctx = rs.context()
        devices = ctx.query_devices()
        if not (idx in range(len(devices))):
            raise ValueError(
                f"Invalid index: found {len(devices)} cameras but given idx={idx}"
            )

        self.resolution = resolution
        self.fps = fps
        self.cam = devices[idx]
        self.idx = idx
        self.sn = self.cam.get_info(rs.camera_info.serial_number)
        self.pipe = None
        config = rs.config()
        config.enable_device(self.sn)
        config.enable_stream(rs.stream.depth, *self.resolution, rs.format.z16, self.fps)
        config.enable_stream(
            rs.stream.color, *self.resolution, rs.format.rgb8, self.fps
        )
        self.config = config
        self.warmup_frames = warmup_frames

        if align_rgbd:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

    def get_data(self):
        # multiprocessing requires lazy instantiation to feed the right process
        if self.pipe is None:
            self.pipe = rs.pipeline()
            self.profile = self.pipe.start(self.config)
            for _ in range(self.warmup_frames):
                frames = self.pipe.wait_for_frames()
                del frames

        frames = self.pipe.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)
        rgb = np.array(frames.get_color_frame().get_data())
        depth = np.array(frames.get_depth_frame().get_data())
        data = (frames.get_timestamp() / 1000, rgb, depth)
        del frames
        return data

    def close(self):
        self.pipe.stop()

    def save_buffer(self, buf: List, filepath: Path):
        filepath.mkdir(parents=True)

        ts, rgb, depth = list(zip(*buf))
        rgb = torch.Tensor(np.array(rgb))
        depth = np.array(depth)

        ts_path = filepath / "ts.pkl"
        rgb_path = filepath / "rgb.mp4"
        depth_path = filepath / "depth.npz"

        with open(ts_path, "wb") as f:
            pickle.dump(ts, f)
        torchvision.io.write_video(str(rgb_path), rgb, self.fps)
        np.savez_compressed(depth_path, depth)

    @staticmethod
    def read_buffer(filepath: Path, read_rgb: bool = True, read_depth: bool = True):
        ts_path = filepath / "ts.pkl"
        rgb_path = filepath / "rgb.mp4"
        depth_path = filepath / "depth.npz"

        result = []

        with open(ts_path, "rb") as f:
            ts = pickle.load(f)
        result.append(ts)

        if read_rgb:
            rgb, _, _ = torchvision.io.read_video(str(rgb_path))
            rgb = einops.rearrange(rgb, "N H W C -> N C H W")
            result.append(rgb)
        if read_depth:
            depth = np.load(depth_path)["arr_0"]
            result.append(depth)
        return result

    def __str__(self):
        return f"Camera{self.idx}([{self.sn}] @{self.resolution}, {self.fps}fps)"


class RobotStateStream(AbstractStream):
    def __init__(self, ip_address):
        self.robot = None
        self.ip_address = ip_address

    def get_data(self):
        # multiprocessing rpc requires lazy instantiation
        # to end up in the right process
        if self.robot is None:
            self.robot = polymetis.RobotInterface(
                ip_address=self.ip_address, enforce_version=False
            )
        # get one Franka robot state frame
        state = self.robot.get_robot_state()
        return state

    def close(self):
        pass

    def __str__(self):
        return f"RobotStateStream(ip_address={self.ip_address})"

    @staticmethod
    def read_buffer(filepath: Path):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        timestamps = [utils.to_sec(elem.timestamp) for elem in data]
        return (timestamps, data)


class GripperStateStream(AbstractStream):
    def __init__(self, ip_address):
        self.gripper = None
        self.ip_address = ip_address

    def get_data(self):
        if self.gripper is None:
            self.gripper = polymetis.GripperInterface(ip_address=self.ip_address)
        state = self.gripper.get_state()
        return state

    def close(self):
        pass

    def __str__(self):
        return f"GripperStateStream(ip_address={self.ip_address})"

    @staticmethod
    def read_buffer(filepath: Path, preprocess=True):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        timestamps = [utils.to_sec(elem.timestamp) for elem in data]
        if preprocess:
            raw = preproc.gripper.gripper_rec_to_pd_series(data)
            closed = preproc.gripper.is_closed(raw).tolist()
            moving = preproc.gripper.is_moving(raw).tolist()
            target = preproc.gripper.target(raw).tolist()
            return (timestamps, data, closed, moving, target)
        else:
            return (timestamps, data)
