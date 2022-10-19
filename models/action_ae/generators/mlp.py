import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Tuple, Dict

from models.mlp import MLP
from models.action_ae.generators.base import AbstractGenerator
from typing import Optional, Any, Union

from omegaconf import DictConfig
import hydra


class MLPGenerator(MLP, AbstractGenerator):
    def __init__(
        self,
        optimizer_cfg: Optional[DictConfig] = None,
        train_cfg: Optional[DictConfig] = None,
        output_mod: Optional[DictConfig] = None,
        *args,
        **kwargs
    ):
        super(MLPGenerator, self).__init__(
            *args, **kwargs, output_mod=hydra.utils.instantiate(output_mod)
        )
        self._optimizer = (
            hydra.utils.instantiate(optimizer_cfg, self.parameters())
            if optimizer_cfg
            else None
        )
        self.train_cfg = train_cfg

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def calculate_encodings_and_loss(
        self,
        act: torch.Tensor,
        obs_enc: torch.Tensor,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pred_action = self.forward(obs_enc)
        loss = F.mse_loss(pred_action, act)
        loss_components = (
            OrderedDict(
                total_loss=loss,
            )
            if return_loss_components
            else OrderedDict()
        )
        return (
            torch.zeros(len(pred_action), 1),
            loss,
            loss_components,
        )

    @property
    def latent_dim(self) -> int:
        return 1

    def decode_actions(
        self,
        action_latents: Optional[torch.Tensor],
        input_rep_batch: Optional[torch.Tensor] = None,
    ):
        return self.forward(input_rep_batch)

    def sample_latents(self, num_latents: int):
        return torch.zeros(num_latents, 1)

    def encode_into_latent(
        self, input_action: torch.Tensor, input_rep: Optional[torch.Tensor]
    ) -> Any:
        return None

    @property
    def num_latents(self) -> Union[int, float]:
        return 1
