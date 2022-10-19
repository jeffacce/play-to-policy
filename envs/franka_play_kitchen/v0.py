import sys
import gym
import grpc
import time
import torch
import atexit
import einops
import logging
import polymetis
import numpy as np
from pathlib import Path
from gym.spaces import Box
from collections import deque
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
from . import utils, preproc, streams
from .streams.recorder import IPCRecorder
from .control.gripper import GripperController
from .control.robot import RobotController


class FrankaPlayKitchenV0(gym.Env):
    GRIPPER_ROLLING_WINDOW = 1
    ERR_RETRIES = 3
    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(
        self,
        robot_ip=utils.ROBOT_IP_ADDR,
    ):
        """
        Arguments:
            robot_ip: IP address of the robot
        """
        super().__init__()
        self.robot_ip = robot_ip
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2062,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(8,))
        stream_kwargs = {
            "freq_limit": 10,
            "manual_get_data": True,
        }
        self.cams = [
            IPCRecorder(streams.Camera(0), **stream_kwargs),
            IPCRecorder(streams.Camera(1), **stream_kwargs),
        ]
        self.robot = RobotController(ip_address=robot_ip, enforce_version=False)
        self.robot.set_home_pose(torch.Tensor(utils.ROBOT_HOME_HORIZONTAL))
        self.gripper = GripperController(ip_address=robot_ip, timeout=2.0)

        # assuming this is eventually called from project root
        encoder_0, normalizer_0 = utils.get_resnet18_encoder(
            Path(sys.path[0]) / "models" / "weights", "byol_rgb0"
        )
        encoder_1, normalizer_1 = utils.get_resnet18_encoder(
            Path(sys.path[0]) / "models" / "weights", "byol_rgb1"
        )
        self.cam_preprocs = [
            preproc.camera.CameraPreproc(
                device="cuda:0",
                encoder=encoder_0,
                crop=utils.CAMERA_0_CROP,
                additional_transforms=[normalizer_0],
            ),
            preproc.camera.CameraPreproc(
                device="cuda:1",
                encoder=encoder_1,
                crop=utils.CAMERA_1_CROP,
                additional_transforms=[normalizer_1],
            ),
        ]
        self.rendered_img = None
        self.theta_err = torch.zeros(7)
        self.error_count = 0
        self.gripper_cmds_rolling = deque(
            [1] * self.GRIPPER_ROLLING_WINDOW, maxlen=self.GRIPPER_ROLLING_WINDOW
        )

        # start stream daemons
        for elem in self.cams:
            elem.start()
        atexit.register(self.close)

    def reset(self):
        try:
            self.gripper.open()
            self.gripper_cmds_rolling = deque(
                [1] * self.GRIPPER_ROLLING_WINDOW, maxlen=self.GRIPPER_ROLLING_WINDOW
            )
            time.sleep(0.2)
            self.robot.start_joint_impedance(Kq=1.5 * self.robot.Kq_default)
            self.robot.reset_I_term()
            self.robot.move_to_joint_positions_bezier(
                torch.Tensor(utils.ROBOT_HOVER_MIDAIR),
                time_to_go=4.0,
            )
            self.robot.move_to_joint_positions_bezier(
                torch.Tensor(utils.ROBOT_HOME_HORIZONTAL),
                converge_pos_err=True,
            )
            obs = self._get_obs()
            return obs
        except grpc.RpcError as e:
            self._handle_robot_error(e)
        except RuntimeError as e:
            self._handle_robot_error(e)

    def _handle_robot_error(self, e):
        logging.exception(e)
        self.robot.start_joint_impedance(Kq=1.5 * self.robot.Kq_default)
        self.error_count += 1

    def _playback_trajectory(self, traj: torch.Tensor, fps: float = 30.0):
        # playback a trajectory of joint_angles at a given fps
        # this is used for mechanical env resets (e.g. closing all the doors)
        period = 1.0 / fps
        self.robot.move_to_joint_positions_bezier(traj[0], time_to_go=2.0)
        self.robot.start_joint_impedance()
        for i in range(1, traj.shape[0]):
            self.robot.update_desired_joint_positions(traj[i])
            time.sleep(period)

    def estimate_state(self, obs):
        raise NotImplementedError

    def calculate_reward(self, obs):
        return 0

    def _get_obs(self):
        imgs = [stream.get_data() for stream in self.cams]
        result = []
        for i in range(len(imgs)):
            ts, rgb, d = imgs[i]
            rgb = einops.rearrange(torch.Tensor(rgb), "H W C -> 1 C H W")
            _, *embeddings = self.cam_preprocs[i].get_obses((ts, rgb))
            result += embeddings

        _, sintheta, costheta = preproc.robot.get_obses(
            (0, [self.robot.get_robot_state()])  # placeholder timestamp
        )
        result += [sintheta.squeeze(), costheta.squeeze()]
        result = torch.cat(result)
        joint_states = result[-14:]
        joint_states = joint_states.repeat(73)
        result = torch.cat((result, joint_states))
        return result.numpy()

    def close(self):
        for elem in self.cams:
            elem.stop()
        self.gripper.open()
        time.sleep(0.2)
        self.robot.go_home()

    def render(self, mode="rgb_array"):
        _, rgb0, _ = self.cams[0].get_data()
        _, rgb1, _ = self.cams[1].get_data()
        rgb = np.concatenate((rgb0, rgb1), axis=1)
        if mode == "human":
            if self.rendered_img is None:
                self.rendered_img = plt.imshow(rgb)
            else:
                self.rendered_img.set_data(rgb)
            plt.draw()
            plt.pause(0.00001)
        return rgb

    def step(self, act: np.ndarray):
        # action: Tensor[8] (7: d(joint_angles), 1: gripper command (grasp=0, open=1))
        logging.info(f"action: {act}")
        dtheta = torch.Tensor(act[:7])
        closed, _ = self.gripper.get_state()
        this_gripper_cmd = int(act[7] > 0.5)
        self.gripper_cmds_rolling.append(this_gripper_cmd)
        gripper_cmd = round(np.mean(self.gripper_cmds_rolling))
        done = False
        if gripper_cmd == 0:
            self.gripper.grasp(blocking=True)
        else:
            self.gripper.open(blocking=True)
        try:
            theta = self.robot.get_joint_positions()
            theta_cmd = torch.Tensor(theta + dtheta + self.theta_err)
            self.robot.move_to_joint_positions_bezier(
                theta_cmd,
                time_to_go=1.0,
                converge_pos_err=True,
            )
            self.theta_err = self.robot.get_joint_positions() - theta_cmd
            logging.info(f"theta_err: {self.theta_err}")
            self.error_count = max(0, self.error_count - 1)
        except grpc.RpcError as e:
            self._handle_robot_error(e)
        except RuntimeError as e:
            self._handle_robot_error(e)
        if self.error_count >= self.ERR_RETRIES:
            done = True
        obs = self._get_obs()
        reward = self.calculate_reward(obs)
        info = {"robot_state": self.robot.get_robot_state()}
        return obs, reward, done, info
