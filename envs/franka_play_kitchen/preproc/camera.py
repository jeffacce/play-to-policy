import cv2
import torch
import einops
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple, Sequence, List, Union, Optional
from .. import utils


class CameraPreproc:
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder: torch.nn.Module,
        crop: Optional[Tuple[int, int, int, int]] = None,
        additional_transforms: List = [],
    ):
        """
        Arguments:
            device: for the encoder
            encoder: RGB encoder
            crop: (top, left, height, width)
            additional_transforms: list of additional transforms,
                to be applied after the default ones
        """
        self.device = device
        self.encoder = encoder.eval().to(device)
        self.crop = crop
        self.additional_transforms = additional_transforms

    def preproc_rgb(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        transform = transforms.Compose(
            [
                lambda x: transforms.functional.crop(x, *self.crop)
                if self.crop is not None
                else x,
                utils.RESNET_RESIZE,
                utils.normalize_0_1(0, 255),
                *self.additional_transforms,
            ]
        )
        return transform(rgb_tensor.type(torch.float))

    def get_obses(
        self,
        raw: Tuple[Sequence[float], torch.Tensor],
    ) -> Tuple[List[float], torch.Tensor]:
        """
        Arguments:
            raw: (timestamp, rgb)
                timestamp: List[float]
                rgb: torch.Tensor[T C H W]
        Output:
            obses: (timestamp, rgb_embedding)
                timestamp: List[float]
                rgb_embedding: torch.Tensor[T E]
        """
        # raw: (timestamp, rgb)
        # output: (timestamp, rgb_embedding)
        ts, rgb = raw
        rgb_input = self.preproc_rgb(rgb).to(self.device)

        with torch.no_grad():
            rgb_embedding = self.encoder(rgb_input).squeeze().cpu()
        return ts, rgb_embedding  # type: ignore
