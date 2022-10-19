from .. import utils
import time
import torch
import bezier
import logging
import polymetis
import numpy as np


BEZIER_NODES = np.array(
    [
        [-1, 1],
        [-0.5, 1],
        [0, 1],
        [0.45, 1],
        [0.5, 0.5],
        [0.55, 0],
        [1, 0],
        [1.5, 0],
        [2, 0],
    ]
)
BEZIER_NODES[:, 0] = utils.normalize_0_1(-1, 2)(BEZIER_NODES[:, 0])
MOTION_PROFILE = bezier.Curve(BEZIER_NODES.T, degree=len(BEZIER_NODES) - 1)


# linear interpolation: w_a * a + (1 - w_a) * b
def lerp(a: torch.Tensor, b: torch.Tensor, w_a: torch.Tensor):
    w_a = w_a.unsqueeze(1)
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
    w_b = 1 - w_a
    return w_a * a + w_b * b


class RobotController(polymetis.RobotInterface):
    def move_to_joint_positions_bezier(
        self,
        joint_positions: torch.Tensor,
        time_to_go: float = 4.0,
        fps: int = 60,
        converge_pos_err: bool = False,
        eps: float = 1e-2,
        I_gain: float = 0.035,
        I_clip: float = 0.1,
        torque_limit: float = 5.0,
    ):
        if not hasattr(self, "I_term"):
            self.I_term = torch.zeros(7)
        sleep_period = 1.0 / fps
        n_points = round(time_to_go * fps)
        xs = np.linspace(0, 1, n_points)
        xs, ys = MOTION_PROFILE.evaluate_multi(xs)
        ys = torch.from_numpy(ys)
        start = self.get_joint_positions()
        joint_cmds = lerp(start, joint_positions, ys)
        if converge_pos_err:
            joint_cmds = joint_cmds + self.I_term
        for i in range(len(joint_cmds)):
            tstart = time.time()
            cmd = joint_cmds[i]
            self.update_desired_joint_positions(cmd)
            tau = torch.Tensor(self.get_robot_state().motor_torques_external)
            if tau.norm() > torque_limit * 2:
                raise RuntimeError("Torque limit exceeded during motion")
            while time.time() - tstart < sleep_period:
                pass
        if converge_pos_err:
            # integral controller to drive down steady state error
            err = torch.ones(1)  # position error placeholder
            tau = torch.zeros(7)  # torque placeholder
            while err.norm() > eps and tau.norm() < torque_limit:
                err = self.get_joint_positions() - joint_positions
                tau = torch.Tensor(self.get_robot_state().motor_torques_external)
                logging.info(f"tau: {tau.norm()}")
                self.I_term += -I_gain * err
                self.I_term = torch.clamp(self.I_term, -I_clip, I_clip)
                cmd = joint_positions + self.I_term
                self.update_desired_joint_positions(cmd)
                while time.time() - tstart < sleep_period:
                    pass
            # # minimize external torque
            # while tau.norm() > torque_limit:
            #     tau = torch.Tensor(self.get_robot_state().motor_torques_external)
            #     logging.info(f"tau: {tau.norm()}")
            #     self.I_term += -torque_comp_gain * tau
            #     self.I_term = torch.clamp(self.I_term, -I_clip, I_clip)
            #     cmd = joint_positions + self.I_term
            #     self.update_desired_joint_positions(cmd)
            #     time.sleep(sleep_period)

    def reset_I_term(self):
        self.I_term = torch.zeros(7)
