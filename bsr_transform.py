from __future__ import annotations
import math
import random
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def _get_lengths(length: int, num_block: int) -> List[int]:
    rand = np.random.rand(num_block)
    rand /= rand.sum()

    raw = np.floor(rand * length).astype(np.int32)
    raw[raw < 1] = 1

    diff = length - raw.sum()
    if diff > 0:
        for i in np.random.choice(num_block, diff, replace=True):
            raw[i] += 1
    elif diff < 0:
        for i in np.random.choice(num_block, -diff, replace=True):
            if raw[i] > 1:
                raw[i] -= 1

    total = raw.sum()
    if total != length:
        raw[-1] += length - total

    assert all(raw_i > 0 for raw_i in raw), f"Invalid split: {raw}"
    return raw.tolist()



def _rotate_batch(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Rotate each image in the (B, C, H, W) tensor `x` by the given angles (radians).
    """
    rot_imgs = [
        TF.rotate(
            img,
            angle=float(a * 180.0 / math.pi),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
        )
        for img, a in zip(x, angles)
    ]
    return torch.stack(rot_imgs, dim=0)


class BSRTransform(torch.nn.Module):

    def __init__(
        self,
        num_block: int = 2,
        num_copies: int = 20,
        max_angles: float = 0.2,
    ):
        super().__init__()
        self.num_block = num_block
        self.num_copies = num_copies
        self.max_angles = max_angles

    def _shuffle_rotate(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w_lens = _get_lengths(W, self.num_block)
        h_lens = _get_lengths(H, self.num_block)

        x_split_w = torch.split(x, w_lens, dim=3)

        x_spilt_h_lists = [torch.split(x_split_w[i], h_lens, dim=2) for i in range(self.num_block)]
        perm_w = np.random.permutation(self.num_block)
        perm_h = np.random.permutation(self.num_block)
        stripes: List[torch.Tensor] = []
        for wi in perm_w:
            block_col = x_spilt_h_lists[wi]
            rotated_blocks = []
            angles = torch.empty(B, device=x.device).uniform_(-self.max_angles, self.max_angles)

            for hj in perm_h:
                blk = block_col[hj]
                rotated = _rotate_batch(blk, angles)
                rotated_blocks.append(rotated)
            stripe = torch.cat(rotated_blocks, dim=2)
            stripes.append(stripe)

        out = torch.cat(stripes, dim=3)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        copies = [self._shuffle_rotate(x) for _ in range(self.num_copies)]
        return torch.cat(copies, dim=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dummy = torch.rand(5, 3, 299, 299)
    bsr = BSRTransform(num_block=2, num_copies=3, max_angles=0.2)
    out = bsr(dummy)
    print("Input:", dummy.shape, "→ Output:", out.shape)
