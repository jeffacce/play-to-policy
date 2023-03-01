import time
import grpc
import torch
import logging
import polymetis
import numpy as np
from .. import utils
from .vr_controller import VRSystem
from .gripper import GripperController
from scipy.spatial.transform import Rotation


class Teleop:
    SLOW_MODE_OFF_THRESH = 0.05  # meters; if close enough, disable slow mode
    SLOW_MODE_ON_THRESH = 0.15  # meters; if too far away, robot enters slow mode (approach desired pose with EWA)

    def __init__(self, ip_address, fall_protection=False):
        self.ip_address = ip_address
        self.robot = polymetis.RobotInterface(
            ip_address=ip_address, enforce_version=False
        )
        self.robot.set_home_pose(torch.Tensor(utils.ROBOT_HOME_HORIZONTAL))
        # if True, pause teleop if controller is likely falling
        self.fall_protection = fall_protection

        self.gripper = GripperController(ip_address=ip_address)

        self.vrsystem = VRSystem()
        left, right = self.vrsystem.get_controller_ids()

        if (left is None) and (right is None):
            raise RuntimeError("No controllers found")
        else:
            controller_idx = right if right is not None else left
            self.controller = self.vrsystem.get_controller(controller_idx)

        self.controller.set_flip_axes(True, False, True)  # flip x, z axes
        self.controller.set_permute_axes(2, 0, 1)  # zxy -> xyz

        self.slow_mode = True
        self.running = False
        self.freeze_pos = False
        self.buttons = self.controller.get_button_state()

    def calibrate_ref_frame(self):
        print("Calibration. Press down on the touchpad to set reference frame base.")

        cal = np.eye(4)
        while utils.is_eye(cal):
            if self.controller.get_button_state()["touchpad"]:
                cal = self.controller.set_calibration()
                if utils.is_eye(cal):
                    logging.warning(
                        "Calibration failed. Make sure the camera sees the controller."
                    )
            time.sleep(0.01)

    def calibrate_start_pose(self):
        self.go_home()
        print(
            "Calibration. Match your controller start pose to the robot hand, and press down on the touchpad."
        )

        cal = np.eye(3)
        while utils.is_eye(cal):
            if self.controller.get_button_state()["touchpad"]:
                cal = self.controller.set_rotation_calibration()
                if utils.is_eye(cal):
                    logging.warning(
                        "Calibration failed. Make sure the camera sees the controller."
                    )
            time.sleep(0.01)

    def go_home(self):
        self.robot.go_home()
        self.update_robot_pose()
        self.robot_rot0 = self.robot_rot
        self.cmd_pos = self.robot_pos
        self.cmd_rot = self.robot_rot0

    def reset(self):
        self.gripper.open(blocking=True)
        self.go_home()
        self.robot.start_cartesian_impedance(
            Kx=2 * self.robot.Kx_default, Kxd=self.robot.Kxd_default
        )
        self.running = True
        self.slow_mode = True
        self.freeze_pos = False
        self.controller.reset_pos_buffer()

    def update_robot_pose(self):
        self.robot_pos, self.robot_quat = self.robot.get_ee_pose()
        self.robot_rot = Rotation.from_quat(self.robot_quat).as_matrix()

    def update_user_state(self):
        affine = self.controller.get_pose()
        self.user_pos = torch.Tensor(affine[:3, 3])
        self.user_rot = affine[:3, :3]
        self.buttons = self.controller.get_button_state()
        self.falling = self.fall_protection and self.controller.is_likely_falling()
        self.trigger = self.controller.get_trigger_state()

    def update_modes(self):
        if self.falling:
            self.running = False
            logging.error("Controller might be falling; terminating teleop.")
        else:
            if self.buttons["menu"]:
                if self.running:
                    self.reset()  # blocks long enough to debounce the button
                    self.running = False
                else:
                    self.reset()
        self.freeze_pos = self.buttons["touchpad"]

        pos_diff = (self.robot_pos - self.user_pos).norm()
        is_far = pos_diff > self.SLOW_MODE_ON_THRESH
        is_close = pos_diff < self.SLOW_MODE_OFF_THRESH
        if self.slow_mode and is_close:
            self.slow_mode = False
        elif (not self.slow_mode and is_far) or self.freeze_pos:
            self.slow_mode = True

    def update_cmd_pose(self):
        self.cmd_rot = self.user_rot @ self.robot_rot0
        self.cmd_quat = torch.Tensor(Rotation.from_matrix(self.cmd_rot).as_quat())

        if self.freeze_pos:
            self.cmd_pos = self.robot_pos
        else:
            if self.slow_mode:
                self.cmd_pos = 0.99 * self.cmd_pos + 0.01 * self.user_pos
            else:
                self.cmd_pos = self.user_pos
        self.robot.update_desired_ee_pose(self.cmd_pos, self.cmd_quat)

    def update_gripper(self):
        if self.trigger > 0.75:
            self.gripper.grasp()
        elif self.trigger < 0.25:
            self.gripper.open()

    def step(self):
        self.update_robot_pose()
        self.update_user_state()
        self.update_modes()
        if self.running:
            try:
                self.update_gripper()
                self.update_cmd_pose()
            except grpc.RpcError:
                logging.warning("Error. Recovering...")
                self.reset()

    def run(self):
        self.reset()
        while self.running:
            self.step()

    def close(self):
        self.vrsystem.shutdown()
