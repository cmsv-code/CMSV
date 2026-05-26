import logging
import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmsv_platform import NUM_PLATFORMS

try:
    from mamba_ssm import Mamba
    from mamba_ssm.modules import mamba_simple as mamba_simple_module
    from mamba_ssm.ops import selective_scan_interface as selective_scan_module
    try:
        from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref
    except ImportError:
        causal_conv1d_ref = None
    MAMBA_IMPORT_ERROR = None
except ImportError as exc:
    Mamba = None
    mamba_simple_module = None
    selective_scan_module = None
    causal_conv1d_ref = None
    MAMBA_IMPORT_ERROR = exc


logger = logging.getLogger(__name__)
_CPU_FALLBACK_PATCHED = False


def _enable_mamba_cpu_fallback() -> bool:
    """Switch mamba_ssm to reference CPU kernels when CUDA kernels are unavailable."""
    global _CPU_FALLBACK_PATCHED
    if _CPU_FALLBACK_PATCHED:
        return True

    if mamba_simple_module is None or selective_scan_module is None:
        return False
    if causal_conv1d_ref is None:
        return False

    try:
        # Patch symbols used inside mamba_simple.forward()
        mamba_simple_module.causal_conv1d_fn = causal_conv1d_ref
        mamba_simple_module.selective_scan_fn = selective_scan_module.selective_scan_ref
        _CPU_FALLBACK_PATCHED = True
        logger.warning("Enabled CPU fallback for mamba_ssm (reference kernels, slower inference).")
        return True
    except Exception:  # pragma: no cover - defensive guard for runtime env differences
        logger.exception("Failed to enable CPU fallback for mamba_ssm.")
        return False


# ===================================================================
# Common Components
# ===================================================================

class SEBlock(nn.Module):
    """Squeeze-Excitation channel reweighting."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x))
        return x * weights


class MultiScaleResBlock(nn.Module):
    """
    Multi-scale residual block with 4 parallel branches and SE attention.

    Branches:
      1) 1x1 conv
      2) 3x3 conv
      3) 5x1 dilated(2) conv (long-range in sequence direction)
      4) 1x5 conv (cross-feature direction)

    Each branch outputs out_channels // 4 channels. Concatenated result
    is projected via 1x1 conv, followed by BN + GELU + SE + residual add.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_ch = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=(5, 1), padding=(4, 0),
                      dilation=(2, 1), bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.se = SEBlock(out_channels)

        # Shortcut: 1x1 conv if channel dimensions differ, else identity
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.project(out)
        out = self.se(out)
        out = out + self.shortcut(x)
        return out


class MultiScaleCNNEncoder(nn.Module):
    """
    Multi-scale residual CNN encoder for 200bp segments.

    Shared-weight CNN that processes each 200bp segment independently,
    outputting a single feature vector per segment.

    Early-downsample design: the stem itself strides by 4 in the sequence
    direction so that the expensive multi-scale blocks run at reduced
    resolution, greatly cutting inference latency.

    Architecture:
      Stem:  Conv2d(1->32, k=5, s=(4,1)) + BN + GELU  -> (B, 32, 50, 20)
      Block1: MultiScaleResBlock(32->64)                -> (B, 64, 50, 20)
      Down1:  stride Conv(5,1)                          -> (B, 64, 10, 20)
      Block2: MultiScaleResBlock(64->128)               -> (B, 128, 10, 20)
      Block3: MultiScaleResBlock(128->128)              -> (B, 128, 10, 20)
      AdaptiveAvgPool2d(1) + flatten                    -> (B, 128)

    Input:  (B, 1, 200, 20)
    Output: (B, 128)
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=(4, 1),
                      padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.block1 = MultiScaleResBlock(32, 64)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(5, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.block2 = MultiScaleResBlock(64, embed_dim)
        self.block3 = MultiScaleResBlock(embed_dim, embed_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)       # (B, 32, 50, 20)
        x = self.block1(x)     # (B, 64, 50, 20)
        x = self.down1(x)      # (B, 64, 10, 20)
        x = self.block2(x)     # (B, 128, 10, 20)
        x = self.block3(x)     # (B, 128, 10, 20)
        x = self.pool(x)       # (B, 128, 1, 1)
        x = x.flatten(1)       # (B, 128)
        return x


class MambaResidualBlock(nn.Module):
    """Pre-norm Mamba block with residual Mamba and FFN sublayers."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        use_fast_path: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=use_fast_path,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.mamba(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


# ===================================================================
# Model Definitions
# ===================================================================

class CMSVMamba(nn.Module):
    """CNN-Mamba Hybrid Model for SV Detection"""
    def __init__(
        self,
        window_size: int = 2000,
        feature_dim: int = 20,
        d_model: int = 128,   # embed_dim
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 5,
        dropout: float = 0.3,
        classifier_dropout: float = 0.4,
        num_platforms: int = NUM_PLATFORMS,
    ):
        super().__init__()
        self.use_fast_path = torch.cuda.is_available()
        if Mamba is None:
            raise RuntimeError(
                "mamba_ssm is unavailable; CMSV now requires the Mamba backend. "
                f"Original import error: {MAMBA_IMPORT_ERROR}"
            )
        if not self.use_fast_path and not _enable_mamba_cpu_fallback():
            raise RuntimeError(
                "CUDA is unavailable and mamba_ssm CPU reference kernels could not be enabled. "
                "CMSV is configured to require Mamba."
            )
        self.uses_mamba = True

        self.window_size = int(window_size)
        self.feature_dim = int(feature_dim)
        self.segment_len = 200
        if self.window_size % self.segment_len != 0:
            raise ValueError(f"window_size={self.window_size} must be divisible by segment_len={self.segment_len}")
        self.num_segments = self.window_size // self.segment_len
        self.d_model = d_model
        self.num_platforms = int(num_platforms)

        self.cnn_encoder = MultiScaleCNNEncoder(in_channels=1, embed_dim=d_model)
        self.raw_segment_proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.segment_len * self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.platform_embedding = nn.Embedding(self.num_platforms, d_model)
        nn.init.trunc_normal_(self.platform_embedding.weight, std=0.02)

        self.segment_pos_embed = nn.Parameter(torch.zeros(1, self.num_segments, d_model))
        nn.init.trunc_normal_(self.segment_pos_embed, std=0.02)

        self.mamba_layers = nn.ModuleList(
            [
                MambaResidualBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    use_fast_path=self.use_fast_path,
                )
                for _ in range(n_layers)
            ]
        )
        self.mamba_only_layers = nn.ModuleList(
            [
                MambaResidualBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    use_fast_path=self.use_fast_path,
                )
                for _ in range(n_layers)
            ]
        )
        self.sequence_layers = None

        self.norm = nn.LayerNorm(d_model)
        self.mamba_only_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cnn_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.mamba_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        platform_ids: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        # --- Normalize input to (B, 1, window_size, feature_dim) ---
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] == self.window_size:
            x = x.permute(0, 3, 1, 2)

        if x.dim() != 4:
            raise ValueError(f"Expected 3D/4D input, got shape {tuple(x.shape)}")
        if int(x.shape[-2]) != self.window_size or int(x.shape[-1]) != self.feature_dim:
            raise ValueError(
                f"Expected input shape (*, {self.window_size}, {self.feature_dim}), got {tuple(x.shape)}"
            )

        B = x.shape[0]
        if platform_ids is None:
            platform_ids = torch.zeros(B, dtype=torch.long, device=x.device)
        else:
            platform_ids = platform_ids.to(device=x.device, dtype=torch.long).reshape(B)
        platform_ids = platform_ids.clamp(min=0, max=self.num_platforms - 1)
        platform_context = self.platform_embedding(platform_ids)

        # --- Slice into N × 200bp segments ---
        x = x.reshape(B, 1, self.num_segments, self.segment_len, self.feature_dim)
        segment_grid = x

        x = segment_grid.reshape(B * self.num_segments, 1, self.segment_len, self.feature_dim)

        # --- Shared-weight CNN encodes each segment ---
        x = self.cnn_encoder(x)  # (B*10, 128)

        # --- Reshape to sequence of segment features ---
        x = x.reshape(B, self.num_segments, self.d_model)  # (B, 10, 128)
        cnn_sequence = x

        # --- Add learnable segment positional encoding ---
        x = x + self.segment_pos_embed + platform_context.unsqueeze(1)

        # --- Sequence layers model inter-segment dependencies ---
        for layer in self.mamba_layers:
            x = layer(x)

        x = self.norm(x)
        pooled_sequence = x.mean(dim=1)

        # --- Classify using pooled sequence context across all segments ---
        cnn_mamba_pred = self.classifier(pooled_sequence)
        if return_aux:
            # Retained for compatibility with legacy ablation checkpoints; the
            # default CMSV training/inference path uses cnn_mamba_pred only.
            raw_segments = segment_grid.reshape(B * self.num_segments, self.segment_len * self.feature_dim)
            raw_tokens = self.raw_segment_proj(raw_segments)
            raw_tokens = raw_tokens.reshape(B, self.num_segments, self.d_model)
            raw_tokens = raw_tokens + self.segment_pos_embed + platform_context.unsqueeze(1)
            for layer in self.mamba_only_layers:
                raw_tokens = layer(raw_tokens)
            raw_tokens = self.mamba_only_norm(raw_tokens)
            mamba_pred = self.mamba_classifier(raw_tokens.mean(dim=1))
            cnn_pred = self.cnn_classifier(cnn_sequence.mean(dim=1) + platform_context)
            return {
                'cnn_pred': cnn_pred,
                'mamba_pred': mamba_pred,
                'cnn_mamba_pred': cnn_mamba_pred,
            }
        return cnn_mamba_pred


class CMSVLoss(nn.Module):
    """Binary Cross Entropy Loss for SV Detection"""
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight else None

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.nan_to_num(predictions.float(), nan=0.5, posinf=1.0, neginf=0.0)
        predictions = predictions.clamp(1e-6, 1.0 - 1e-6)
        targets = targets.float()
        loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        if self.pos_weight is not None:
            pw = self.pos_weight.to(predictions.device)
            # 正样本乘以 pos_weight，负样本权重为 1
            weight = torch.where(targets == 1, pw, torch.ones_like(loss))
            loss = loss * weight
        return loss.mean()


def _normalize_state_dict_keys(state_dict):
    """Normalize checkpoint dict and strip DataParallel prefixes if needed."""
    if isinstance(state_dict, dict) and 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
        state_dict = state_dict['state_dict']
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must be a state_dict dictionary.")

    if len(state_dict) == 0:
        return state_dict

    cleaned = OrderedDict()
    for key, value in state_dict.items():
        normalized = key[7:] if isinstance(key, str) and key.startswith('module.') else key
        normalized = re.sub(r'^mamba_layers\.(\d+)\.0\.', r'mamba_layers.\1.norm1.', normalized)
        normalized = re.sub(r'^mamba_layers\.(\d+)\.1\.', r'mamba_layers.\1.mamba.', normalized)
        cleaned[normalized] = value
    return cleaned


def _summarize_keys(keys, max_items: int = 8) -> str:
    if not keys:
        return ""
    if len(keys) <= max_items:
        return ", ".join(keys)
    return ", ".join(keys[:max_items]) + f", ... (+{len(keys) - max_items} more)"


def create_model(
    model_type: str = 'mamba',
    pretrained_weights: Optional[str] = None,
    device: str = 'cuda',
    feature_dim: int = 20,
    window_size: int = 2000,
):
    """
    Create CMSV Mamba model with optional pretrained weights.

    Args:
        model_type: The model architecture to create (only 'mamba' is supported).
        pretrained_weights: Path to pretrained weights file.
        device: Device to load model on.

    Returns:
        An nn.Module instance.
    """
    if model_type == 'mamba':
        model = CMSVMamba(feature_dim=feature_dim, window_size=window_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Only 'mamba' is supported.")

    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights, map_location=device, weights_only=True)
        state_dict = _normalize_state_dict_keys(state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing_suffixes = (
            '.norm2.weight',
            '.norm2.bias',
            '.ffn.0.weight',
            '.ffn.0.bias',
            '.ffn.3.weight',
            '.ffn.3.bias',
        )
        unexpected_keys = list(incompatible.unexpected_keys)
        missing_keys = list(incompatible.missing_keys)
        hard_missing = [
            key for key in missing_keys
            if key != 'platform_embedding.weight'
            and not key.startswith('raw_segment_proj.')
            and not key.startswith('mamba_only_layers.')
            and not key.startswith('mamba_only_norm.')
            and not key.startswith('mamba_classifier.')
            and not (
                key.startswith('mamba_layers.')
                and key.endswith(allowed_missing_suffixes)
            )
        ]
        if hard_missing or unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible with the current model. "
                f"Missing keys: {_summarize_keys(hard_missing)}; "
                f"Unexpected keys: {_summarize_keys(unexpected_keys)}"
            )
        if missing_keys:
            logger.warning(
                "Loaded checkpoint with newly initialized parameters: %s",
                _summarize_keys(missing_keys),
            )

    return model.to(device)


if __name__ == '__main__':
    # Test Mamba model
    if Mamba is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"--- Testing Mamba Model (MultiScale CNN) on {device} ---")
        mamba_model = CMSVMamba().to(device)
        total_params = sum(p.numel() for p in mamba_model.parameters())
        print(f"Mamba Total Parameters: {total_params:,}")
        dummy_input_mamba = torch.randn(4, mamba_model.window_size, mamba_model.feature_dim, device=device)
        dummy_platform = torch.tensor([0, 1, 2, 0], device=device)
        output_mamba = mamba_model(dummy_input_mamba, platform_ids=dummy_platform)
        print(f"Mamba Input: {dummy_input_mamba.shape}, Output: {output_mamba.shape}")
    else:
        print("Mamba-ssm library not installed. Please install it to test the model.")
