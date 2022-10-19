import torch
import numpy as np


def get_obses(raw):
    ts, data = raw
    theta = np.stack([np.array(x.joint_positions) for x in data])
    theta = torch.Tensor(theta)
    # ? should we return raw here as well?
    return ts, torch.sin(theta), torch.cos(theta)


def get_actions(raw):
    # s = theta[t], a = theta[t+1] - theta[t] (forward diff)
    # a[T] := 0 (zero-pad last frame)
    ts, data = raw
    theta = np.stack([np.array(x.joint_positions) for x in data])
    dtheta = np.diff(theta, axis=0)
    dtheta = np.pad(dtheta, ((0, 1), (0, 0)))  # zero-pad last frame
    dtheta = torch.Tensor(dtheta)
    return ts, dtheta
