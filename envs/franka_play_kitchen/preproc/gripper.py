import torch
import numpy as np
import pandas as pd
from ..utils import to_sec
from typing import Tuple, List


WINDOW = "200ms"  # gripper is non-realtime; ~10Hz rate
EPS = 1e-2
OPEN_POS = 0.99  # 99% of max_width


def serialize(x):
    return [
        to_sec(x.timestamp),
        x.width / x.max_width,
    ]


def gripper_rec_to_pd_series(arr):
    df = pd.DataFrame([serialize(x) for x in arr])
    df.columns = ["ts", "width"]
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.set_index("ts")
    series = df.width
    return series


def is_moving(raw):
    signal = (raw.diff().abs() > EPS).astype(int)
    signal = signal.rolling(WINDOW).max().astype(int)
    signal.name = "moving"
    return signal


def is_closing(raw):
    diff = np.sign(raw.diff().rolling(WINDOW).mean())
    filter_ = is_moving(raw)
    signal = ((diff * filter_) < 0).astype(int)
    signal.name = "closing"
    return signal


def is_opening(raw):
    signal = ~is_closing(raw).astype(bool)
    signal &= is_moving(raw).astype(bool)
    signal.name = "opening"
    return signal.astype(int)


def is_open(raw):
    return (raw > OPEN_POS).astype(int)


def target(raw):
    signal = is_open(raw).astype(bool) | is_opening(raw).astype(bool)
    signal.name = "target"
    return signal.astype(int)


# NOTE: an approximation; gripper might grasp halfway, get stuck, then close some more.
# Here we consider it not closed whenenver there is movement; probably fine for now.
def is_closed(raw):
    # closed widths vary depending on grasped objects; open width is reliably near max_width
    signal = ~is_open(raw).astype(bool) & (~is_moving(raw).astype(bool))
    # gripper state updates on movement success (control.gripper.GripperController)
    signal |= is_opening(raw).astype(bool)
    signal.name = "closed"
    return signal.astype(int)


def get_obses(preprocessed: Tuple) -> Tuple[List[float], torch.Tensor, torch.Tensor]:
    # Given preprocessed gripper states (ts, data, closed, moving, target)
    # Return (ts, closed, moving)
    return (
        preprocessed[0],
        torch.Tensor(preprocessed[2]).unsqueeze(-1),
        torch.Tensor(preprocessed[3]).unsqueeze(-1),
    )


def get_actions(preprocessed: Tuple) -> Tuple[List[float], torch.Tensor]:
    # Given preprocessed gripper states (ts, data, closed, moving, target)
    # Return (ts, target)
    # where d_target in [-1 (close), 0 (no movement), 1 (open)]
    d_target = torch.Tensor(preprocessed[4]).diff()
    d_target = torch.cat([d_target, torch.zeros(1)])  # no movement in the last frame
    return (
        preprocessed[0],
        # d_target.unsqueeze(-1),
        torch.Tensor(preprocessed[4]).unsqueeze(-1),
    )
