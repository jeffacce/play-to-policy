import torch
from typing import Union


# Whitepaper: https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
# Note: the whitepaper seems inaccurate at places; the colormap is inconsistent with the librealsense code,
# and a few boundary conditions in depth image recovery are handled incorrectly.
# This implementation is based on the librealsense hue cmap, referenced below:
# https://github.com/IntelRealSense/librealsense/blob/master/src/proc/colorizer.cpp#L16-L24
class DepthColorizer:
    HUE_CMAP_SEGMENTS = [
        (0, 256),  #   0 ≤ d ≤ 255
        (256, 511),  # 255 < d ≤ 510
        (511, 766),  # 510 < d ≤ 765
        (766, 1021),  # 765 < d ≤ 1020
        (1021, 1276),  # 1020 < d ≤ 1275
        (1276, 1530),  # 1275 < d ≤ 1529
    ]

    def __init__(
        self,
        d_min: int = 0,
        d_max: int = 65535,
        disparity: bool = False,
        device: Union[torch.device, str] = "cpu",
    ):
        self.d_min = d_min
        self.d_max = d_max
        self.disparity = disparity
        self.disp_min = 1 / self.d_max
        self.disp_max = 1 / self.d_min
        self.device = device

    def _normalize(self, d: torch.Tensor):
        if self.disparity:
            disp = (1 / d).clip(self.disp_min, self.disp_max)
            result = (disp - self.disp_min) / (self.disp_max - self.disp_min) * 1529
        else:
            d = d.clip(self.d_min, self.d_max)
            result = (d - self.d_min) / (self.d_max - self.d_min) * 1529
        return result

    def _unnormalize(self, d_normal: torch.Tensor):
        if self.disparity:
            result = 1529 / (
                1529 * self.disp_min + (self.disp_max - self.disp_min) * d_normal
            )
        else:
            result = self.d_min + (self.d_max - self.d_min) * d_normal / 1529
        return result

    def _hue_cmap(self, d_normal: torch.Tensor):
        segment_masks = [
            (d_normal >= s[0]) & (d_normal < s[1]) for s in self.HUE_CMAP_SEGMENTS
        ]
        result = torch.zeros(*d_normal.shape, 3)
        result[..., 0] = (
            255 * (segment_masks[0] | segment_masks[5])
            + (510 - d_normal) * segment_masks[1]
            + (d_normal - 1020) * segment_masks[4]
        )
        result[..., 1] = (
            d_normal * segment_masks[0]
            + 255 * (segment_masks[1] | segment_masks[2])
            + (1020 - d_normal) * segment_masks[3]
        )
        result[..., 2] = (
            (d_normal - 510) * segment_masks[2]
            + 255 * (segment_masks[3] | segment_masks[4])
            + (1530 - d_normal) * segment_masks[5]
        )
        return result

    def _recover_d_rnormal(self, rgb):
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        intermediate0 = (r > g) & (r >= b)
        g_minus_b = g - b
        intermediate1 = g_minus_b >= 0
        mask_cond0 = intermediate0 & intermediate1
        mask_cond1 = intermediate0 & ~intermediate1
        mask_cond2 = (g >= r) & (g > b)
        mask_cond3 = (b >= g) & (b > r)

        d_rnormal = (
            mask_cond0 * g_minus_b
            + mask_cond1 * (g_minus_b + 1530)
            + mask_cond2 * (b - r + 510)
            + mask_cond3 * (r - g + 1020)
        )
        return d_rnormal

    def d2rgb(self, d: torch.Tensor):
        d = d.to(self.device)
        d_normal = self._normalize(d).int()
        del d
        rgb = self._hue_cmap(d_normal).int()
        del d_normal
        return rgb

    def rgb2d(self, rgb: torch.Tensor):
        rgb = rgb.to(self.device).int()
        d_rnormal = self._recover_d_rnormal(rgb).int()
        del rgb
        d = self._unnormalize(d_rnormal).int()
        del d_rnormal
        return d
