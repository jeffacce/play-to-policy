import openvr
import numpy as np
import time
from typing import Optional


# pad 3x4 pose matrix to 4x4 affine
def affine_from_pose(pose: openvr.TrackedDevicePose_t) -> np.ndarray:
    return np.block(
        [
            [pose.mDeviceToAbsoluteTracking.m],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ]
    )


class VRSystem:
    def __init__(self, freq: float = 60.0):
        self.controllers = []
        self.freq = freq
        self.backend = None
        self.reset()

    def reset(self):
        self.controllers = []
        openvr.shutdown()
        openvr.init(openvr.VRApplication_Scene)
        self.backend = openvr.VRSystem()

    def get_controller(self, controller_id):
        if controller_id is None:
            raise ValueError("Expected a valid controller id, got None")
        controller_ids = [x.controller_id for x in self.controllers]
        if controller_id in controller_ids:
            controller = self.controllers[controller_ids.index(controller_id)]
        else:
            controller = VRController(self, controller_id, freq=self.freq)
            self.controllers.append(controller)
        return controller

    def get_controller_ids(self):
        left, right = None, None
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.backend.getTrackedDeviceClass(i)
            if device_class == openvr.TrackedDeviceClass_Controller:
                role = self.backend.getControllerRoleForTrackedDeviceIndex(i)
                if role == openvr.TrackedControllerRole_RightHand:
                    right = i
                if role == openvr.TrackedControllerRole_LeftHand:
                    left = i
        return left, right

    def shutdown(self):
        openvr.shutdown()


class VRController:
    BUTTONS = {
        "menu": 2,  # hamburger menu
        "grip": 4,  # side grip buttons
        "touchpad": 2**32,  # touchpad pressed
        "trigger": 2**33,  # trigger pressed
    }

    def __init__(self, vrsys: openvr.VRSystem, controller_id: int, freq: float = 60.0):
        self.vrsys = vrsys
        self.controller_id = controller_id
        self.freq = freq
        self.period = 1.0 / freq
        self.last_refresh = -1
        self.last_state_pose = None
        # last 5 positions + timestamp for fall detection
        self.POS_BUFFER_SIZE = 5
        self.reset_pos_buffer()

        self.calibration = np.eye(4)  # identity affine
        self.rotation_calibration = np.eye(3)  # identity rotation
        self.permute_axes = np.eye(4)
        self.flip_axes = np.eye(4)
        self.axes_transform = np.eye(4)

    def set_calibration(self, affine: Optional[np.ndarray] = None):
        # Set a pose as the origin of the reference frame
        if affine is None:
            _, affine = self._get_raw_state_and_pose()
        self.calibration = np.linalg.pinv(affine)
        return self.calibration

    def set_rotation_calibration(self, affine: Optional[np.ndarray] = None):
        # Set a pose as the reference identity rotation, calculated separately from translation
        # Rotation calibration is calculated in the calibration frame
        if affine is None:
            affine = self.get_pose(apply_rotation_calibration=False)
        self.rotation_calibration = np.linalg.pinv(affine[:3, :3])
        return self.rotation_calibration

    def set_flip_axes(self, flip_x: bool, flip_y: bool, flip_z: bool):
        # Flip the axes of the controller
        # True => -1, False => 1
        flip_axes = np.array([flip_x, flip_y, flip_z, False]).astype(int) * (-2) + 1
        self.flip_axes = np.diag(flip_axes)
        self.axes_transform = self.permute_axes @ self.flip_axes

    def set_permute_axes(self, x: int, y: int, z: int):
        # from (0, 1, 2) to `(x, y, z)`
        p = np.diag([0, 0, 0, 1])
        permutation = (x, y, z)

        # row-major
        for i in range(3):
            p[i, permutation[i]] = 1
        self.permute_axes = p
        self.axes_transform = self.permute_axes @ self.flip_axes

    def get_pose(self, apply_rotation_calibration=True):
        # Get the pose of the controller in the calibration frame
        _, affine = self._get_raw_state_and_pose()
        affine = self.axes_transform @ self.calibration @ affine @ self.axes_transform.T
        if apply_rotation_calibration:
            affine[:3, :3] = self.rotation_calibration @ affine[:3, :3]

        return affine

    def get_button_state(self):
        state, _ = self._get_raw_state_and_pose()
        result = {}
        for button, bit in self.BUTTONS.items():
            result[button] = bool(state.ulButtonPressed & bit)
        return result

    def get_trigger_state(self):
        state, _ = self._get_raw_state_and_pose()
        return state.rAxis[1].x

    def is_likely_falling(self):
        # defined as 3 frames with 8~16 m/s^2 acceleration
        pos_buffer = np.roll(self.pos_buffer, -self.pos_buffer_ptr, axis=0)
        has_empty_frames = (pos_buffer == 0).all(axis=1).any()
        if has_empty_frames:
            return False
        dt = np.diff(pos_buffer[:, 3]).reshape(-1, 1)
        if dt.max() > 2 * self.period:
            return False
        dt = dt[1:]  # match length of the second difference
        positions = pos_buffer[:, :3]
        accel = np.diff(np.diff(positions, axis=0), axis=0) / dt / dt
        accel = np.linalg.norm(accel, axis=1)
        fall = (accel > 8.0) & (accel < 16.0)
        return fall.sum() >= 3

    def reset_pos_buffer(self):
        self.pos_buffer = np.zeros((self.POS_BUFFER_SIZE, 4))
        self.pos_buffer_ptr = 0

    def _get_raw_state_and_pose(self):
        now = time.time()
        if (self.last_state_pose is None) or (now - self.last_refresh > self.period):
            _, state, pose = self.vrsys.backend.getControllerStateWithPose(
                0, self.controller_id
            )
            self.last_state_pose = (state, affine_from_pose(pose))
            self.pos_buffer[self.pos_buffer_ptr, :3] = self.last_state_pose[1][:3, 3]
            self.pos_buffer[self.pos_buffer_ptr, 3] = now
            self.pos_buffer_ptr = (self.pos_buffer_ptr + 1) % self.POS_BUFFER_SIZE
            self.last_refresh = now
        return self.last_state_pose
