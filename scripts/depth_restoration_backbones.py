import torch
import torch.nn as nn
import torch.nn.functional as F

from train_depth_completion import ResidualUNet


def _group_count(channels):
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        blocks = [ResidualConvBlock(in_channels, out_channels)]
        for _ in range(max(0, int(num_blocks) - 1)):
            blocks.append(ResidualConvBlock(out_channels, out_channels))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, channels, heads=8, mlp_ratio=4.0):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(round(channels * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialTransformerBottleneck(nn.Module):
    def __init__(self, channels, layers=2, heads=8, mlp_ratio=4.0, pool_stride=2):
        super().__init__()
        self.pool_stride = max(1, int(pool_stride))
        # Convolutional positional encoding keeps spatial locality before flattening.
        self.pos = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    channels,
                    heads=int(heads),
                    mlp_ratio=float(mlp_ratio),
                )
                for _ in range(int(layers))
            ]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        residual = x
        y = x + self.pos(x)
        original_hw = y.shape[-2:]
        if self.pool_stride > 1:
            y = F.avg_pool2d(y, kernel_size=self.pool_stride, stride=self.pool_stride)

        b, c, h, w = y.shape
        tokens = y.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        y = tokens.transpose(1, 2).reshape(b, c, h, w)

        if y.shape[-2:] != original_hw:
            y = F.interpolate(y, size=original_hw, mode="bilinear", align_corners=False)
        return residual + self.out(y)


class ResidualUNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels=32,
        out_channels=1,
        blocks_per_stage=2,
        transformer_layers=0,
        transformer_heads=8,
        transformer_mlp_ratio=4.0,
        transformer_pool=2,
    ):
        super().__init__()
        c = int(base_channels)
        blocks = int(blocks_per_stage)
        self.enc1 = ResidualStack(in_channels, c, blocks)
        self.enc2 = ResidualStack(c, c * 2, blocks)
        self.enc3 = ResidualStack(c * 2, c * 4, blocks)
        self.enc4 = ResidualStack(c * 4, c * 8, blocks)
        self.pool = nn.MaxPool2d(2)

        bottleneck = [ResidualStack(c * 8, c * 8, blocks)]
        if int(transformer_layers) > 0:
            bottleneck.append(
                SpatialTransformerBottleneck(
                    c * 8,
                    layers=int(transformer_layers),
                    heads=int(transformer_heads),
                    mlp_ratio=float(transformer_mlp_ratio),
                    pool_stride=int(transformer_pool),
                )
            )
        self.bottleneck = nn.Sequential(*bottleneck)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualStack(c * 8, c * 4, blocks)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualStack(c * 4, c * 2, blocks)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ResidualStack(c * 2, c, blocks)
        self.out = nn.Conv2d(c, out_channels, kernel_size=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(e4)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


class PropagationRefinementBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels=32,
        out_channels=1,
        blocks_per_stage=2,
        propagation_steps=6,
        propagation_hidden_scale=1.0,
        refine_dilate_radius=3,
        residual_scale=1.5,
        global_refine=False,
    ):
        super().__init__()
        c = int(base_channels)
        hidden_c = max(8, int(round(c * float(propagation_hidden_scale))))
        blocks = int(blocks_per_stage)
        self.encoder = ResidualUNetBackbone(
            in_channels=in_channels,
            base_channels=c,
            out_channels=hidden_c,
            blocks_per_stage=blocks,
            transformer_layers=0,
        )
        self.coarse_head = nn.Sequential(
            nn.Conv2d(hidden_c, hidden_c, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_c), hidden_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_c, out_channels, kernel_size=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_c, hidden_c, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_c), hidden_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_c, 9, kernel_size=1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(hidden_c, hidden_c, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_c), hidden_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_c, out_channels, kernel_size=1),
        )
        self.propagation_steps = max(1, int(propagation_steps))
        self.refine_dilate_radius = max(0, int(refine_dilate_radius))
        self.residual_scale = float(residual_scale)
        self.global_refine = bool(global_refine)

        nn.init.zeros_(self.coarse_head[-1].weight)
        nn.init.zeros_(self.coarse_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(self.confidence_head[-1].bias, 4.0 if self.global_refine else 0.0)

    def _refine_region(self, hole_mask):
        if self.global_refine:
            return torch.ones_like(hole_mask)
        region = hole_mask.clamp(0.0, 1.0)
        if self.refine_dilate_radius > 0:
            kernel_size = 2 * self.refine_dilate_radius + 1
            region = F.max_pool2d(region, kernel_size=kernel_size, stride=1, padding=self.refine_dilate_radius)
        return region.clamp(0.0, 1.0)

    def _propagate(self, depth, gate_logits, confidence, anchor, refine_region):
        b, _, h, w = depth.shape
        weights = torch.softmax(gate_logits, dim=1)
        source = anchor * (1.0 - refine_region) + depth * refine_region
        unfold = F.unfold(source, kernel_size=3, padding=1).view(b, 1, 9, h, w)
        propagated = (weights[:, None] * unfold).sum(dim=2)
        blend = torch.sigmoid(confidence)
        updated = blend * depth + (1.0 - blend) * propagated
        return anchor * (1.0 - refine_region) + updated * refine_region

    def forward(self, x):
        anchor = x[:, 0:1]
        hole = x[:, 2:3]
        refine_region = self._refine_region(hole)
        features = self.encoder(x)
        coarse_delta = self.coarse_head(features)
        gate_logits = self.gate_head(features)
        confidence = self.confidence_head(features)
        coarse = anchor + self.residual_scale * torch.tanh(coarse_delta) * refine_region
        coarse = anchor * (1.0 - refine_region) + coarse * refine_region
        refined = coarse
        for _ in range(self.propagation_steps):
            refined = self._propagate(refined, gate_logits, confidence, anchor, refine_region)
        return {
            "coarse": coarse,
            "refined": refined,
            "gate_logits": gate_logits,
            "confidence": confidence,
            "refine_region": refine_region,
        }


def build_depth_backbone(
    backbone,
    in_channels,
    base_channels,
    out_channels=1,
    res_blocks=2,
    transformer_layers=2,
    transformer_heads=8,
    transformer_mlp_ratio=4.0,
    transformer_pool=2,
    propagation_steps=6,
    propagation_hidden_scale=1.0,
    refine_dilate_radius=3,
    residual_scale=1.5,
    global_refine=False,
):
    if backbone == "resunet":
        return ResidualUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=out_channels,
        )
    if backbone == "large_resunet":
        return ResidualUNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=out_channels,
            blocks_per_stage=res_blocks,
        )
    if backbone == "transformer_bottleneck":
        return ResidualUNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=out_channels,
            blocks_per_stage=res_blocks,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
            transformer_pool=transformer_pool,
        )
    if backbone == "propagation_refine":
        return PropagationRefinementBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=out_channels,
            blocks_per_stage=res_blocks,
            propagation_steps=propagation_steps,
            propagation_hidden_scale=propagation_hidden_scale,
            refine_dilate_radius=refine_dilate_radius,
            residual_scale=residual_scale,
            global_refine=global_refine,
        )
    raise ValueError(f"Unknown backbone: {backbone}")
