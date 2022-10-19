import hydra
import logging
import numpy as np
import matplotlib.pyplot as plt

import envs
from workspaces import base


class FrankaPlayKitchenWorkspace(base.Workspace):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _setup_plots(self):
        pass

    def _setup_starting_state(self):
        pass

    def _setup_action_sampler(self):
        def sampler(actions):
            # actions: [N, 8]
            assert (actions.shape[-1] == 8) and (
                len(actions.shape) == 2
            ), "Expected action batch to be [N, 8]"
            idx = np.random.randint(len(actions))
            dtheta = actions[idx, :7]

            # denoise gripper prediction
            logging.info("Gripper actions and mean:")
            logging.info(actions[:, 7])
            logging.info(np.mean(actions[:, 7]))

            gripper = (np.mean(actions[:, 7]) > 0.5).astype(float)[np.newaxis]
            return np.concatenate([dtheta, gripper])

        self.sampler = sampler

    def _start_from_known(self):
        obs = self.env.reset()
        return obs

    def _plot_obs_and_actions(self, obs, chosen_action, done, all_actions=None):
        pass
