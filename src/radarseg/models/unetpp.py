from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolution layers used in UNet++ decoder nodes."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """UNet++ for binary radargram hyperbola semantic segmentation.

    UNet++ keeps the encoder-decoder idea of U-Net but adds nested dense skip
    pathways between encoder and decoder stages. This often improves boundary
    recovery compared with a plain U-Net while remaining easy to train from
    scratch on domain-specific images such as radargrams.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision

        nb = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = ConvBlock(in_channels, nb[0])
        self.conv1_0 = ConvBlock(nb[0], nb[1])
        self.conv2_0 = ConvBlock(nb[1], nb[2])
        self.conv3_0 = ConvBlock(nb[2], nb[3])
        self.conv4_0 = ConvBlock(nb[3], nb[4])

        self.conv0_1 = ConvBlock(nb[0] + nb[1], nb[0])
        self.conv1_1 = ConvBlock(nb[1] + nb[2], nb[1])
        self.conv2_1 = ConvBlock(nb[2] + nb[3], nb[2])
        self.conv3_1 = ConvBlock(nb[3] + nb[4], nb[3])

        self.conv0_2 = ConvBlock(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = ConvBlock(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = ConvBlock(nb[2] * 2 + nb[3], nb[2])

        self.conv0_3 = ConvBlock(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = ConvBlock(nb[1] * 3 + nb[2], nb[1])

        self.conv0_4 = ConvBlock(nb[0] * 4 + nb[1], nb[0])

        if deep_supervision:
            self.final1 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb[0], out_channels, kernel_size=1)

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample_like(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample_like(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._upsample_like(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample_like(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._upsample_like(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._upsample_like(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample_like(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._upsample_like(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._upsample_like(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample_like(x1_3, x0_0)], dim=1))

        if self.deep_supervision:
            # Average the auxiliary outputs so the public forward contract stays
            # identical to the other semantic models: logits with shape B x C x H x W.
            return torch.stack([
                self.final1(x0_1),
                self.final2(x0_2),
                self.final3(x0_3),
                self.final4(x0_4),
            ], dim=0).mean(dim=0)

        return self.final(x0_4)
