from torchvision import transforms
import torch
from typing import List, Union


def random_noise(noise: float = 0.1):
    """
    Returns a transform that applies random noise to the input image.
    """
    return lambda x: x + torch.randn_like(x) * noise


def random_noise_affine(
    noise: float = 0.1,
    degrees: float = 10,
    translate: float = 0.1,
    scale: float = 0.1,
):
    """
    Returns a transform that applies random noise and affine transformations to the input image.
    """
    return transforms.Compose(
        [
            random_noise(noise),
            transforms.RandomAffine(
                degrees=degrees,
                translate=(translate, translate),
                scale=(1 - scale, 1 + scale),
            ),
        ]
    )


def random_dropout(dropout: Union[List[float], float] = 0.1, drop_whole: bool = False):
    """
    Returns a transform that randomly set elements of the input to zero.
    Argument:
        dropout: The dropout probability.
            If dropout is a float, all elements have the same dropout probability.
            If dropout is a list, len(dropout) must be equal to len(input);
            specifying the dropout probability for each input element.
        drop_whole: If True, do dropout on whole tensors instead of element-wise (i.e. set the whole tensor to zero).
            default: False
    Returns:
        transform: A transform that randomly set elements of the input to zero.
        Input can be a tensor or a list of tensors.
            Argument: input: Union[List[Tensor], Tensor]
            Returns: Union[List[Tensor], Tensor]
    """
    if isinstance(dropout, float):

        def transform(input):
            if isinstance(input, list):
                if drop_whole:
                    return [(torch.rand(1) > dropout) * x for x in input]
                else:
                    return [(torch.rand_like(x) > dropout) * x for x in input]
            else:
                if drop_whole:
                    return (torch.rand(1) > dropout) * input
                else:
                    return (torch.rand_like(input) > dropout) * input

    else:

        def transform(input):
            assert len(dropout) == len(
                input
            ), "Got {} dropout values for {} inputs".format(len(dropout), len(input))
            if drop_whole:
                return [(torch.rand(1) > dropout[i]) * x for i, x in enumerate(input)]
            else:
                return [
                    (torch.rand_like(x) > dropout[i]) * x for i, x in enumerate(input)
                ]

    return transform


def mask_goal_at_idx(idx, mask_value=0):
    def transform(input):
        assert len(input) == 4, "Input length must be 4: (obs, act, mask, goal)"
        goal = input[-1].clone()
        goal[..., idx] = mask_value
        result = (*input[:-1], goal)
        return result

    return transform


def blockpush_mask_targets():
    def transform(input):
        assert len(input) == 4, "Input length must be 4: (obs, act, mask, goal)"
        # assume obs, act, mask, goal
        obs = input[0].clone()
        obs[..., 10:] = 0
        goal = input[-1].clone()
        goal[..., [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = 0
        return (obs, *input[1:-1], goal)

    return transform
