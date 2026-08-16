"""Fourier + relative space-time transformer for P-ONE direction.

This is a PyTorch-2.x adaptation of the ``DeepIceModel`` described in the
second-place IceCube Kaggle solution.  It deliberately uses only PyTorch and
GraphNeT already present in the production container; the historical
repository's torch-1.11/fastai/timm environment is not installed.

Upstream reference
------------------
Repository: https://github.com/DrHB/icecube-2nd-place
Revision inspected: 484cdcfed01af5255dce148122b095a7427ec1cb
Paper: Bukhari et al., EPJC 84 (2024) 646, arXiv:2310.15674

The upstream implementation is MIT licensed.  Its copyright and license are
preserved in ``THIRD_PARTY_NOTICES.md`` next to this module.  This adaptation
keeps the central scientific ingredients while integrating them with the
existing routed P-ONE train/validation, weighting, loss, metric, and checkpoint
contracts:

* Fourier encoding of position, time, and charge;
* signed Minkowski-like space-time interval as relative attention information;
* relative-attention blocks followed by CLS-token transformer blocks; and
* a 3-vector head whose norm is interpreted as vMF concentration.

P-ONE-specific adaptations are explicit config fields.  In particular,
events are edgeless pulse sequences, times are made event-relative, long
events are sampled to a configured train/evaluation length, and PMT direction
can be added as an opt-in continuous feature.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from graphnet.models.gnn.gnn import GNN
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.callbacks import PiecewiseLinearLR

from energy_weighting import EnergyWeightManifest
from losses import EnergyWeightedDirectionLoss
from model import (
    JointDirectionModel,
    PREDICTION_LABELS,
    TARGET_LABELS,
)


MODEL_NAME = "fourier_spacetime_transformer"


class SinusoidalFourierEmbedding(nn.Module):
    """Encode a continuous scalar with logarithmically spaced frequencies."""

    def __init__(self, dimension: int, maximum_period: float = 10_000.0) -> None:
        super().__init__()
        if dimension <= 0 or dimension % 2:
            raise ValueError("Fourier embedding dimension must be positive and even")
        if not math.isfinite(maximum_period) or maximum_period <= 1.0:
            raise ValueError("maximum_period must be finite and greater than one")
        self.dimension = int(dimension)
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(float(maximum_period))
            * torch.arange(half, dtype=torch.float32)
            / float(half)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, value: Tensor) -> Tensor:
        phase = value.unsqueeze(-1) * self.frequencies.to(value)
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class FeedForward(nn.Module):
    def __init__(self, dimension: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(round(float(ratio) * dimension))
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, dimension),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class RelativeAttention(nn.Module):
    """Multi-head attention with vector-valued pairwise relative features."""

    def __init__(
        self,
        dimension: int,
        head_size: int,
        *,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dimension <= 0 or head_size <= 0 or dimension % head_size:
            raise ValueError("dimension must be a positive multiple of head_size")
        self.dimension = int(dimension)
        self.head_size = int(head_size)
        self.num_heads = self.dimension // self.head_size
        self.scale = self.head_size**-0.5
        self.query = nn.Linear(self.dimension, self.dimension, bias=False)
        self.key = nn.Linear(self.dimension, self.dimension, bias=False)
        self.value = nn.Linear(self.dimension, self.dimension, bias=False)
        self.output = nn.Linear(self.dimension, self.dimension)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.projection_dropout = nn.Dropout(float(projection_dropout))

    def _heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_size).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        valid_mask: Tensor,
        relative: Tensor | None,
    ) -> Tensor:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        query = self._heads(self.query(x))
        key = self._heads(self.key(x))
        value = self._heads(self.value(x))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if relative is not None:
            expected = (x.shape[0], x.shape[1], x.shape[1], self.head_size)
            if tuple(relative.shape) != expected:
                raise ValueError(
                    f"relative tensor must have shape {expected}, got {tuple(relative.shape)}"
                )
            scores = scores + torch.einsum("bhid,bijd->bhij", query, relative)

        scores = scores.masked_fill(~valid_mask[:, None, None, :], -torch.inf)
        # Float32 softmax is important under bf16 mixed precision for the long
        # tails of sparse P-ONE events.
        attention = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attention = self.attention_dropout(attention)
        result = torch.matmul(attention, value).transpose(1, 2)
        if relative is not None:
            result = result + torch.einsum("bhij,bijd->bihd", attention, relative)
        result = result.reshape(x.shape[0], x.shape[1], self.dimension)
        return self.projection_dropout(self.output(result))


class RelativeTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        head_size: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(dimension)
        self.attention = RelativeAttention(
            dimension,
            head_size,
            attention_dropout=dropout,
            projection_dropout=dropout,
        )
        self.norm_mlp = nn.LayerNorm(dimension)
        self.mlp = FeedForward(dimension, ratio=mlp_ratio, dropout=dropout)
        self.gamma_attention = nn.Parameter(
            torch.full((dimension,), float(layer_scale))
        )
        self.gamma_mlp = nn.Parameter(torch.full((dimension,), float(layer_scale)))

    def forward(self, x: Tensor, valid_mask: Tensor, relative: Tensor | None) -> Tensor:
        x = x + self.gamma_attention * self.attention(
            self.norm_attention(x), valid_mask, relative
        )
        x = x + self.gamma_mlp * self.mlp(self.norm_mlp(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        head_size: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dimension % head_size:
            raise ValueError("dimension must be divisible by head_size")
        self.norm_attention = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            dimension // head_size,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(dimension)
        self.mlp = FeedForward(dimension, ratio=mlp_ratio, dropout=dropout)
        self.gamma_attention = nn.Parameter(
            torch.full((dimension,), float(layer_scale))
        )
        self.gamma_mlp = nn.Parameter(torch.full((dimension,), float(layer_scale)))

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        normalized = self.norm_attention(x)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )[0]
        x = x + self.gamma_attention * attended
        x = x + self.gamma_mlp * self.mlp(self.norm_mlp(x))
        return x


class PulseFourierExtractor(nn.Module):
    """Paper-inspired pulse encoder with an optional P-ONE PMT orientation."""

    def __init__(
        self,
        dimension: int,
        base_dimension: int,
        *,
        use_pmt_direction: bool,
        position_fourier_scale: float,
        time_fourier_scale: float,
        charge_fourier_scale: float,
    ) -> None:
        super().__init__()
        if base_dimension <= 0 or base_dimension % 4:
            raise ValueError("base_dimension must be a positive multiple of four")
        self.use_pmt_direction = bool(use_pmt_direction)
        self.position_fourier_scale = float(position_fourier_scale)
        self.time_fourier_scale = float(time_fourier_scale)
        self.charge_fourier_scale = float(charge_fourier_scale)
        self.continuous = SinusoidalFourierEmbedding(base_dimension)
        self.event_length = SinusoidalFourierEmbedding(base_dimension // 2)
        self.pmt_projection = (
            nn.Sequential(
                nn.Linear(3, base_dimension // 2),
                nn.LayerNorm(base_dimension // 2),
                nn.GELU(),
            )
            if self.use_pmt_direction
            else None
        )
        input_width = 5 * base_dimension + base_dimension // 2
        if self.use_pmt_direction:
            input_width += base_dimension // 2
        self.projection = nn.Sequential(
            nn.Linear(input_width, input_width),
            nn.LayerNorm(input_width),
            nn.GELU(),
            nn.Linear(input_width, dimension),
        )

    def forward(
        self,
        position: Tensor,
        time: Tensor,
        charge: Tensor,
        original_length: Tensor,
        pmt_direction: Tensor | None,
    ) -> Tensor:
        encoded = [
            self.continuous(self.position_fourier_scale * position).flatten(-2),
            self.continuous(self.time_fourier_scale * time),
            self.continuous(self.charge_fourier_scale * charge),
            self.event_length(torch.log10(original_length.to(position.dtype)))
            .unsqueeze(1)
            .expand(-1, position.shape[1], -1),
        ]
        if self.use_pmt_direction:
            if pmt_direction is None or self.pmt_projection is None:
                raise ValueError("PMT-direction transformer requires PMT directions")
            encoded.append(self.pmt_projection(pmt_direction))
        elif pmt_direction is not None:
            raise ValueError("PMT direction was supplied to a transformer that disables it")
        return self.projection(torch.cat(encoded, dim=-1))


class RelativeSpaceTimeEncoding(nn.Module):
    def __init__(
        self,
        head_size: int,
        *,
        speed_of_light_m_per_ns: float,
        position_scale_m: float,
        time_scale_ns: float,
        fourier_scale: float,
        clip: float,
    ) -> None:
        super().__init__()
        self.speed_coefficient = (
            float(time_scale_ns)
            / float(position_scale_m)
            * float(speed_of_light_m_per_ns)
        )
        self.fourier_scale = float(fourier_scale)
        self.clip = float(clip)
        self.embedding = SinusoidalFourierEmbedding(head_size)
        self.projection = nn.Linear(head_size, head_size)

    def forward(self, position: Tensor, time: Tensor) -> Tensor:
        spatial_squared = (
            position[:, :, None, :] - position[:, None, :, :]
        ).square().sum(dim=-1)
        temporal_squared = (
            (time[:, :, None] - time[:, None, :]) * self.speed_coefficient
        ).square()
        interval_squared = spatial_squared - temporal_squared
        signed_interval = torch.sign(interval_squared) * torch.sqrt(
            interval_squared.abs()
        )
        signed_interval = signed_interval.clamp(-self.clip, self.clip)
        return self.projection(
            self.embedding(self.fourier_scale * signed_interval)
        )


class FourierSpaceTimeTransformer(GNN):
    """GraphNeT-compatible event backbone operating on edgeless pulse data."""

    def __init__(
        self,
        input_feature_names: Sequence[str],
        *,
        dimension: int = 192,
        base_dimension: int = 128,
        depth_relative: int = 4,
        relative_bias_blocks: int = 4,
        depth_transformer: int = 12,
        head_size: int = 32,
        train_max_pulses: int = 192,
        eval_max_pulses: int = 512,
        train_selection: str = "random",
        eval_selection: str = "uniform_time",
        use_pmt_direction: bool = False,
        position_scale_m: float = 500.0,
        time_scale_ns: float = 30_000.0,
        charge_log10_scale: float = 3.0,
        position_fourier_scale: float = 4096.0,
        time_fourier_scale: float = 4096.0,
        charge_fourier_scale: float = 1024.0,
        relative_fourier_scale: float = 1024.0,
        relative_clip: float = 4.0,
        speed_of_light_m_per_ns: float = 0.3,
        dropout: float = 0.0,
    ) -> None:
        names = tuple(str(value) for value in input_feature_names)
        required = ("pmt_x", "pmt_y", "pmt_z", "dom_time", "charge")
        if names[:5] != required:
            raise ValueError(
                "Transformer input must start with canonical raw P-ONE features "
                f"{required}, got {names}"
            )
        expected = (*required, "pmt_dir_x", "pmt_dir_y", "pmt_dir_z")
        if use_pmt_direction and names != expected:
            raise ValueError(f"PMT transformer requires features {expected}, got {names}")
        if not use_pmt_direction and names != required:
            raise ValueError(f"Base transformer requires features {required}, got {names}")
        if dimension <= 0 or head_size <= 0 or dimension % head_size:
            raise ValueError("dimension must be a positive multiple of head_size")
        if depth_relative <= 0 or not 0 <= relative_bias_blocks <= depth_relative:
            raise ValueError(
                "relative_bias_blocks must lie between zero and depth_relative"
            )
        if depth_transformer <= 0:
            raise ValueError("depth_transformer must be positive")
        if train_max_pulses <= 1 or eval_max_pulses < train_max_pulses:
            raise ValueError("Require 1 < train_max_pulses <= eval_max_pulses")
        if train_selection != "random" or eval_selection != "uniform_time":
            raise ValueError(
                "Approved first experiment requires random train selection and "
                "deterministic uniform_time evaluation selection"
            )
        positive = {
            "position_scale_m": position_scale_m,
            "time_scale_ns": time_scale_ns,
            "charge_log10_scale": charge_log10_scale,
            "position_fourier_scale": position_fourier_scale,
            "time_fourier_scale": time_fourier_scale,
            "charge_fourier_scale": charge_fourier_scale,
            "relative_fourier_scale": relative_fourier_scale,
            "relative_clip": relative_clip,
            "speed_of_light_m_per_ns": speed_of_light_m_per_ns,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in positive.values()):
            raise ValueError(f"Transformer scales must be finite and positive: {positive}")

        super().__init__(nb_inputs=len(names), nb_outputs=int(dimension))
        self.input_feature_names = names
        self.dimension = int(dimension)
        self.train_max_pulses = int(train_max_pulses)
        self.eval_max_pulses = int(eval_max_pulses)
        self.train_selection = train_selection
        self.eval_selection = eval_selection
        self.use_pmt_direction = bool(use_pmt_direction)
        self.position_scale_m = float(position_scale_m)
        self.time_scale_ns = float(time_scale_ns)
        self.charge_log10_scale = float(charge_log10_scale)
        self.relative_bias_blocks = int(relative_bias_blocks)

        self.extractor = PulseFourierExtractor(
            self.dimension,
            int(base_dimension),
            use_pmt_direction=self.use_pmt_direction,
            position_fourier_scale=float(position_fourier_scale),
            time_fourier_scale=float(time_fourier_scale),
            charge_fourier_scale=float(charge_fourier_scale),
        )
        self.relative_encoding = RelativeSpaceTimeEncoding(
            int(head_size),
            speed_of_light_m_per_ns=float(speed_of_light_m_per_ns),
            position_scale_m=self.position_scale_m,
            time_scale_ns=self.time_scale_ns,
            fourier_scale=float(relative_fourier_scale),
            clip=float(relative_clip),
        )
        self.relative_blocks = nn.ModuleList(
            [
                RelativeTransformerBlock(
                    self.dimension,
                    int(head_size),
                    dropout=float(dropout),
                    layer_scale=1.0,
                )
                for _ in range(int(depth_relative))
            ]
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.dimension))
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.dimension,
                    int(head_size),
                    dropout=float(dropout),
                    layer_scale=1.0,
                )
                for _ in range(int(depth_transformer))
            ]
        )
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _event_slices(data: Data) -> list[tuple[int, int]]:
        if hasattr(data, "ptr"):
            ptr = data.ptr.detach().cpu().tolist()
            return [(int(a), int(b)) for a, b in zip(ptr[:-1], ptr[1:])]
        if not hasattr(data, "batch"):
            return [(0, int(data.x.shape[0]))]
        counts = torch.bincount(data.batch).detach().cpu().tolist()
        result: list[tuple[int, int]] = []
        start = 0
        for count in counts:
            result.append((start, start + int(count)))
            start += int(count)
        return result

    def _select_indices(self, length: int, limit: int, device: torch.device) -> Tensor:
        if length <= limit:
            return torch.arange(length, device=device)
        if self.training:
            return torch.randperm(length, device=device)[:limit].sort().values
        # EdgelessGraph sorts each event by raw dom_time.  Uniform positions
        # therefore preserve the full observed time span deterministically.
        return torch.linspace(0, length - 1, limit, device=device).round().long()

    def _pack(self, data: Data) -> dict[str, Tensor | None]:
        if data.x.ndim != 2 or data.x.shape[1] != self.nb_inputs:
            raise ValueError(
                f"Expected node matrix [N,{self.nb_inputs}], got {tuple(data.x.shape)}"
            )
        if not bool(torch.isfinite(data.x).all()):
            raise ValueError("Transformer input contains NaN or infinite values")
        slices = self._event_slices(data)
        if not slices or any(end <= start for start, end in slices):
            raise ValueError("Every transformer event must contain at least one pulse")
        limit = self.train_max_pulses if self.training else self.eval_max_pulses
        selected_lengths = [min(end - start, limit) for start, end in slices]
        dense_length = max(selected_lengths)
        dense = data.x.new_zeros((len(slices), dense_length, self.nb_inputs))
        mask = torch.zeros(
            (len(slices), dense_length), dtype=torch.bool, device=data.x.device
        )
        original_length = torch.empty(
            len(slices), dtype=data.x.dtype, device=data.x.device
        )
        event_first_time = torch.empty_like(original_length)

        for event_index, (start, end) in enumerate(slices):
            event = data.x[start:end]
            length = end - start
            indices = self._select_indices(length, limit, data.x.device)
            selected = event[indices]
            dense[event_index, : selected.shape[0]] = selected
            mask[event_index, : selected.shape[0]] = True
            original_length[event_index] = float(length)
            event_first_time[event_index] = event[:, 3].min()

        position = dense[:, :, :3] / self.position_scale_m
        raw_time = dense[:, :, 3]
        time = (raw_time - event_first_time[:, None]) / self.time_scale_ns
        time = time.masked_fill(~mask, 0.0)
        raw_charge = dense[:, :, 4]
        if bool((raw_charge[mask] <= 0.0).any()):
            raise ValueError("Raw pulse charge must be strictly positive")
        charge = torch.zeros_like(raw_charge)
        charge[mask] = torch.log10(raw_charge[mask]) / self.charge_log10_scale
        pmt_direction = dense[:, :, 5:8] if self.use_pmt_direction else None
        return {
            "position": position,
            "time": time,
            "charge": charge,
            "pmt_direction": pmt_direction,
            "mask": mask,
            "original_length": original_length,
        }

    def forward(self, data: Data) -> Tensor:
        packed = self._pack(data)
        position = packed["position"]
        time = packed["time"]
        charge = packed["charge"]
        pmt_direction = packed["pmt_direction"]
        mask = packed["mask"]
        original_length = packed["original_length"]
        assert isinstance(position, Tensor)
        assert isinstance(time, Tensor)
        assert isinstance(charge, Tensor)
        assert isinstance(mask, Tensor)
        assert isinstance(original_length, Tensor)
        assert pmt_direction is None or isinstance(pmt_direction, Tensor)

        x = self.extractor(
            position,
            time,
            charge,
            original_length,
            pmt_direction,
        )
        relative = self.relative_encoding(position, time)
        for index, block in enumerate(self.relative_blocks):
            x = block(
                x,
                mask,
                relative if index < self.relative_bias_blocks else None,
            )

        batch = x.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls, x), dim=1)
        mask = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=mask.device),
                mask,
            ),
            dim=1,
        )
        for block in self.transformer_blocks:
            x = block(x, mask)
        return x[:, 0]


class Float32DirectionReconstructionWithKappa(DirectionReconstructionWithKappa):
    """Keep spherical loss math in float32 under bf16 mixed precision."""

    def _forward(self, x: Tensor) -> Tensor:
        return super()._forward(x.float())


def build_fourier_transformer_model(
    config: Mapping[str, Any],
    stage_name: str,
    data_representation: Any,
    energy_manifest: EnergyWeightManifest,
    steps_per_optimizer_epoch: int,
) -> JointDirectionModel:
    """Build a transformer inside the established joint-direction wrapper."""

    model_config = config["model"]
    if str(model_config.get("name")) != MODEL_NAME:
        raise ValueError(f"Expected model.name={MODEL_NAME}")
    transformer = dict(model_config["transformer"])
    output_features = tuple(
        str(value) for value in data_representation.output_feature_names
    )
    backbone = FourierSpaceTimeTransformer(
        output_features,
        dimension=int(transformer["dimension"]),
        base_dimension=int(transformer["base_dimension"]),
        depth_relative=int(transformer["depth_relative"]),
        relative_bias_blocks=int(transformer["relative_bias_blocks"]),
        depth_transformer=int(transformer["depth_transformer"]),
        head_size=int(transformer["head_size"]),
        train_max_pulses=int(transformer["train_max_pulses"]),
        eval_max_pulses=int(transformer["eval_max_pulses"]),
        train_selection=str(transformer["train_selection"]),
        eval_selection=str(transformer["eval_selection"]),
        use_pmt_direction=bool(transformer["use_pmt_direction"]),
        position_scale_m=float(transformer["position_scale_m"]),
        time_scale_ns=float(transformer["time_scale_ns"]),
        charge_log10_scale=float(transformer["charge_log10_scale"]),
        position_fourier_scale=float(transformer["position_fourier_scale"]),
        time_fourier_scale=float(transformer["time_fourier_scale"]),
        charge_fourier_scale=float(transformer["charge_fourier_scale"]),
        relative_fourier_scale=float(transformer["relative_fourier_scale"]),
        relative_clip=float(transformer["relative_clip"]),
        speed_of_light_m_per_ns=float(transformer["speed_of_light_m_per_ns"]),
        dropout=float(transformer.get("dropout", 0.0)),
    )

    stage = config["training"][stage_name]
    objective = str(stage["objective"])
    vmf_factor = float(config["loss"]["vmf_factor"])
    direction_loss = EnergyWeightedDirectionLoss.from_manifest(
        energy_manifest,
        objective=objective,
        angular_surrogate=str(config["loss"]["angular_surrogate"]),
        vmf_factor=vmf_factor,
        weighting_mode="train_only",
    )
    task = Float32DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        loss_function=direction_loss,
        target_labels=TARGET_LABELS,
        prediction_labels=PREDICTION_LABELS,
        loss_weight=None,
    )

    optimizer = config["optimizer"]
    if str(optimizer["name"]).lower() != "adamw":
        raise ValueError("Transformer optimizer.name must be adamw")
    base_lr = float(stage["base_lr"])
    peak_lr = float(stage["peak_lr"])
    max_epochs = int(stage["max_epochs"])
    total_steps = max(1, int(steps_per_optimizer_epoch) * max_epochs)
    warmup_steps = max(
        1,
        int(float(stage.get("warmup_fraction", 0.01)) * total_steps),
    )
    return JointDirectionModel(
        tasks=task,
        data_representation=data_representation,
        backbone=backbone,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": base_lr,
            "weight_decay": float(optimizer["weight_decay"]),
            "eps": float(optimizer["eps"]),
        },
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [0, warmup_steps, total_steps],
            "factors": [1.0, peak_lr / base_lr, 1.0],
        },
        scheduler_config={"interval": "step"},
        metric_bin_edges=config["metrics"]["log10_energy_bin_edges"],
        minimum_events_per_bin=int(config["metrics"]["minimum_events_per_bin"]),
        vmf_factor=vmf_factor,
        objective=objective,
    )


__all__ = [
    "MODEL_NAME",
    "FourierSpaceTimeTransformer",
    "SinusoidalFourierEmbedding",
    "build_fourier_transformer_model",
]
