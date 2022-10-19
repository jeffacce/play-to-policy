import sys
import torch
import torchvision
import numpy as np
from typing import Tuple, Callable
from pathlib import Path
from models.libraries.r3m import load_r3m


ROBOT_IP_ADDR = "10.19.229.238"
RECORDER_IP_ADDR = "100.78.202.26"
# ROBOT_HOME_HORIZONTAL = [0.1574, -1.6434, 1.6109, -2.7333, 0.0914, 2.5389, -0.8529]
# ROBOT_HOME_HORIZONTAL = [0.1574, -1.4434, 1.6109, -2.7333, 0.0914, 2.5389, -0.8529]
ROBOT_HOME_HORIZONTAL = [
    -0.0357,
    -1.4777,
    1.6052,
    -2.5431,
    -0.0443,
    2.5617,
    -0.6895,
]  # average of all starts
ROBOT_HOVER_MIDAIR = [0.0378, -0.6499, 1.0105, -1.7983, -0.7753, 1.8119, -0.7809]

RESNET_RESIZE = torchvision.transforms.Resize((224, 224))
RESNET_NORMALIZE = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
BYOL_RGB0_NORMALIZE = torchvision.transforms.Normalize(
    mean=[0.4966, 0.4850, 0.4686],
    std=[0.1880, 0.1822, 0.1881],
)
BYOL_RGB1_NORMALIZE = torchvision.transforms.Normalize(
    mean=[0.4908, 0.4666, 0.4358],
    std=[0.1735, 0.1544, 0.1614],
)
CAMERA_0_CROP = (150, 420, 570, 570)
CAMERA_1_CROP = (40, 380, 680, 680)
DEPTH_RANGE = (0.001, 2.5)


def to_sec(x):
    return x.seconds + x.nanos / 1e9


def to_nanos(x):
    return x.seconds * 1e9 + x.nanos


def get_resnet18_encoder(
    model_weights_path, encoder_type
) -> Tuple[torch.nn.Module, Callable]:
    model_weights_path = Path(model_weights_path)
    # return encoder, normalization
    if encoder_type == "r3m":
        return (load_r3m("resnet18").eval(), lambda x: x)
    elif encoder_type == "imagenet":
        resnet = torchvision.models.resnet18(pretrained=True)
        return (
            torch.nn.Sequential(*list(resnet.children())[:-1]).eval(),
            RESNET_NORMALIZE,
        )
    elif encoder_type == "random":
        # NOTE: assuming run starts from the project root
        return (
            torch.load(model_weights_path / "random_resnet.pt").eval(),
            RESNET_NORMALIZE,
        )
    elif encoder_type == "byol_rgb0":
        return (
            torch.load(model_weights_path / "franka_byol_rgb0.pt").eval(),
            BYOL_RGB0_NORMALIZE,
        )
    elif encoder_type == "byol_rgb1":
        return (
            torch.load(model_weights_path / "franka_byol_rgb1.pt").eval(),
            BYOL_RGB1_NORMALIZE,
        )
    else:
        raise ValueError("Unknown encoder type: {}".format(encoder_type))


def normalize_0_1(lower: float = 0.0, upper: float = 255.0):
    def f(arr):
        arr = arr.clip(lower, upper)
        arr -= arr.min()
        arr /= arr.max()
        return arr

    return f


def is_eye(matrix: np.ndarray) -> bool:
    if len(matrix.shape) != 2 or (matrix.shape[0] != matrix.shape[1]):
        return False
    eye = np.eye(matrix.shape[0])
    return np.allclose(matrix, eye)


# for a collection of sequences, index each seq with idx
def subset_each(seqs, idx):
    result = []
    for seq in seqs:
        if isinstance(seq, (torch.Tensor, np.ndarray)):
            result.append(seq[idx])
        else:
            result.append([seq[i] for i in idx])
    return result
