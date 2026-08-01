"""Architecture-only definition of the CA-CLIP framework.

This module intentionally contains no dataset handling, optimization logic,
loss definitions, threshold selection, checkpointing, or training entry point.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CAClipConfig:
    """Structural settings required to instantiate the CA-CLIP head."""

    dropout: float = 0.2
    adapter_reduction: int = 4
    adapter_ratio: float = 0.2
    ca_projection_dim: int = 512
    ca_fusion_dropout: float = 0.1
    ca_text_modulation_dropout: float = 0.1
    prompt_attention_temperature: float = 1.0


# Retained as a compatibility alias for previously released import statements.
Config = CAClipConfig


class TextImageEncoder(nn.Module):
    """Minimal interface expected from the shared vision-language encoder."""

    feature_dim: int

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text_features(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OpenAIClipEncoder(TextImageEncoder):
    """Thin architecture wrapper around an OpenAI CLIP model instance."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.feature_dim = model.visual.output_dim

    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images).float()

    def encode_text_features(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(tokenized_text).float()


class ResidualAdapter(nn.Module):
    """Bottleneck residual adapter applied to CLIP image features."""

    def __init__(self, dim: int, reduction: int = 4, ratio: float = 0.2):
        super().__init__()
        hidden = max(dim // reduction, 1)
        self.ratio = ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.ratio * self.net(features)


class PromptClassAwareResidualGate(nn.Module):
    """Class-aware gate that modulates image features with text prototypes."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        img_proj: torch.Tensor,
        txt_proto: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_proj.ndim not in (2, 3) or txt_proto.ndim != 2:
            raise ValueError(
                "img_proj must be [B, D] or [B, C, D], and txt_proto must be [C, D]."
            )
        if img_proj.shape[-1] != txt_proto.shape[-1]:
            raise ValueError("Image and text projection dimensions must match.")

        batch_size = img_proj.shape[0]
        num_classes = txt_proto.shape[0]
        if img_proj.ndim == 2:
            img_expand = img_proj.unsqueeze(1).expand(batch_size, num_classes, -1)
        else:
            if img_proj.shape[1] != num_classes:
                raise ValueError(
                    "Class-conditioned img_proj must have shape [B, C, D]."
                )
            img_expand = img_proj

        txt_expand = txt_proto.unsqueeze(0).expand(batch_size, num_classes, -1)
        difference = torch.abs(img_expand - txt_expand)
        gate_input = torch.cat(
            [img_expand, txt_expand, difference, img_expand * txt_expand], dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input))
        fused = self.out_norm(img_expand + gate * (txt_expand - img_expand))
        return fused, gate


class NativeAlignmentClassifier(nn.Module):
    """CA-CLIP classification head with prompt-conditioned class-aware gating.

    Args:
        encoder: Shared CLIP image-text encoder wrapper.
        prompt_features: Encoded prompt tensor shaped [C, K, D] or [C, D].
        cfg: Architecture configuration.
        prompt_mask: Optional validity mask shaped [C, K] or [C].
        use_adapter: Whether to include the residual image adapter.
    """

    def __init__(
        self,
        encoder: TextImageEncoder,
        prompt_features: torch.Tensor,
        cfg: CAClipConfig,
        prompt_mask: Optional[torch.Tensor] = None,
        use_adapter: bool = True,
    ):
        super().__init__()
        if prompt_features.ndim not in (2, 3):
            raise ValueError("prompt_features must have shape [C, D] or [C, K, D].")

        self.encoder = encoder
        self.register_buffer("prompt_features", prompt_features.float())
        if prompt_mask is None:
            if prompt_features.ndim == 3:
                prompt_mask = torch.ones(prompt_features.shape[:2], dtype=torch.bool)
            else:
                prompt_mask = torch.ones(prompt_features.shape[0], dtype=torch.bool)
        self.register_buffer("prompt_mask", prompt_mask.bool())

        feature_dim = prompt_features.shape[-1]
        projection_dim = cfg.ca_projection_dim
        self.prompt_attention_temperature = cfg.prompt_attention_temperature
        self.adapter = (
            ResidualAdapter(feature_dim, cfg.adapter_reduction, cfg.adapter_ratio)
            if use_adapter
            else None
        )
        self.image_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.text_modulator = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.ca_text_modulation_dropout),
            nn.Linear(projection_dim * 2, projection_dim * 2),
        )
        self.modulation_norm = nn.LayerNorm(projection_dim)
        self.fusion_gate = PromptClassAwareResidualGate(
            projection_dim, dropout=cfg.ca_fusion_dropout
        )
        self.class_bias = nn.Parameter(torch.zeros(prompt_features.shape[0]))
        self.ca_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.align_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.last_aux = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.encoder.encode_image_features(images)
        if self.adapter is not None:
            image_features = self.adapter(image_features)
        img_proj = self.image_projection(image_features.float())

        raw_prompt_features = self.prompt_features.to(img_proj.device)
        if raw_prompt_features.ndim == 2:
            raw_prompt_features = raw_prompt_features.unsqueeze(1)
        prompt_mask = self.prompt_mask.to(img_proj.device)
        if prompt_mask.ndim == 1:
            prompt_mask = prompt_mask.unsqueeze(1)

        num_classes, num_prompts, _ = raw_prompt_features.shape
        txt_proj = self.text_projection(raw_prompt_features)
        flat_txt = txt_proj.reshape(num_classes * num_prompts, -1)

        gamma_beta = self.text_modulator(txt_proj)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma)
        img_prompt = self.modulation_norm(
            img_proj[:, None, None, :] * gamma.unsqueeze(0) + beta.unsqueeze(0)
        )
        flat_img_prompt = img_prompt.reshape(
            img_proj.shape[0], num_classes * num_prompts, -1
        )

        fused, gate = self.fusion_gate(flat_img_prompt, flat_txt)
        fused = fused.reshape(
            img_proj.shape[0], num_classes, num_prompts, -1
        )
        gate = gate.reshape(img_proj.shape[0], num_classes, num_prompts, -1)

        fused_norm = F.normalize(fused, dim=-1)
        txt_norm = F.normalize(txt_proj, dim=-1)
        prompt_logits = self.ca_logit_scale.exp().clamp(max=100.0) * (
            fused_norm * txt_norm.unsqueeze(0)
        ).sum(dim=-1)
        masked_prompt_logits = prompt_logits.masked_fill(
            ~prompt_mask.unsqueeze(0), -1e4
        )
        temperature = max(self.prompt_attention_temperature, 1e-6)
        prompt_weights = F.softmax(masked_prompt_logits / temperature, dim=-1)
        prompt_weights = prompt_weights * prompt_mask.unsqueeze(0).float()
        prompt_weights = prompt_weights / prompt_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        logits = (prompt_weights * prompt_logits).sum(dim=-1) + self.class_bias
        align_prompt_logits = self.align_logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bd,ckd->bck",
            F.normalize(img_proj, dim=-1),
            txt_norm,
        )
        align_logits = (prompt_weights * align_prompt_logits).sum(dim=-1)
        self.last_aux = {
            "gate": gate,
            "img_proj": img_proj,
            "txt_proj": txt_proj,
            "prompt_mask": prompt_mask,
            "prompt_weights": prompt_weights,
            "prompt_logits": prompt_logits,
            "align_logits": align_logits,
        }
        return logits


CAClipModel = NativeAlignmentClassifier


__all__ = [
    "CAClipConfig",
    "CAClipModel",
    "Config",
    "NativeAlignmentClassifier",
    "OpenAIClipEncoder",
    "PromptClassAwareResidualGate",
    "ResidualAdapter",
    "TextImageEncoder",
]
