import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PLMConfig

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None


if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    compiler_disable = torch.compiler.disable
else:
    compiler_disable = lambda fn: fn


@compiler_disable
def build_flex_key_padding_block_mask(
    attention_mask: torch.Tensor,
    num_heads: int,
) -> Any:
    if create_block_mask is None:
        raise RuntimeError("FlexAttention is not available in this PyTorch build")
    if attention_mask.dim() == 4:
        attention_mask = attention_mask[:, 0, 0, :]
    if attention_mask.dim() != 2:
        raise ValueError("FlexAttention key padding mask must have shape (batch, length)")

    key_padding_mask = attention_mask.bool()
    batch_size, sequence_length = key_padding_mask.shape

    def key_padding_mask_mod(b, h, q_idx, kv_idx):
        return key_padding_mask[b, kv_idx]

    return create_block_mask(
        mask_mod=key_padding_mask_mod,
        B=batch_size,
        H=num_heads,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=key_padding_mask.device,
    )


class Rotary(nn.Module):
    def __init__(self, head_size: int, theta: float = 10000.0):
        super().__init__()
        assert head_size % 2 == 0, "rope requires an even head size"
        self.head_size = head_size
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.sequence_length_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cache(self, sequence_length: int, device: torch.device) -> None:
        cache_is_valid = (
            self.sequence_length_cached == sequence_length
            and self.cos_cached is not None
            and self.sin_cached is not None
            and self.cos_cached.device == device
        )
        if cache_is_valid:
            return

        positions = torch.arange(sequence_length, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq.to(device))
        self.sequence_length_cached = sequence_length
        self.cos_cached = angles.cos().view(1, 1, sequence_length, self.head_size // 2)
        self.sin_cached = angles.sin().view(1, 1, sequence_length, self.head_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, sequence_length, head_size = x.shape
        assert head_size == self.head_size, "unexpected head size"
        self._update_cache(sequence_length, x.device)

        x_float = x.float()
        x_pair = x_float.reshape(batch_size, num_heads, sequence_length, head_size // 2, 2)
        x_even, x_odd = x_pair.unbind(dim=-1)
        x_rotated = torch.stack(
            (
                x_even * self.cos_cached - x_odd * self.sin_cached,
                x_even * self.sin_cached + x_odd * self.cos_cached,
            ),
            dim=-1,
        )
        return x_rotated.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size # d
        self.head_size = config.head_size # h
        self.num_heads = self.hidden_size // self.head_size # n
        self.scale = 1.0 / math.sqrt(self.head_size)
        self.attention_backend = config.attention_backend

        self.layernorm_Wqkv = nn.Sequential(
            nn.LayerNorm(self.hidden_size, bias=False),
            nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        )
        self.q_ln = nn.LayerNorm(self.hidden_size, bias=False)
        self.k_ln = nn.LayerNorm(self.hidden_size, bias=False)
        self.rotary = Rotary(self.head_size)
        if config.value_embeddings:
            self.value_lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            self.value_lambdas = None
        self.Wo = nn.Linear(self.hidden_size, self.hidden_size)

        if self.attention_backend == "flex":
            if flex_attention is None:
                raise RuntimeError("attention_backend='flex' requires PyTorch FlexAttention")
            if not hasattr(torch, "compile"):
                raise RuntimeError("attention_backend='flex' requires torch.compile")
            self.flex_attention = torch.compile(flex_attention)

    def add_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = x.shape
        assert hidden_size == self.hidden_size, "unexpected hidden size"
        return x.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)

    def remove_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, sequence_length, head_size = x.shape
        assert num_heads == self.num_heads, "unexpected number of heads"
        assert head_size == self.head_size, "unexpected head size"
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_size)

    def prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Any:
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                device=device,
                dtype=torch.bool,
            )
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape (batch, length)")
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask shape must match input batch and length")

        attention_mask = attention_mask.bool()
        if self.attention_backend == "flex":
            return build_flex_key_padding_block_mask(
                attention_mask,
                num_heads=self.num_heads,
            )
        return attention_mask[:, None, None, :]

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[Any] = None,
        value_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        attention_mask = self.prepare_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=x.device,
        )
        QKV = self.layernorm_Wqkv(x) # (b, l, 3d)
        Q, K, V = torch.chunk(QKV, 3, dim=-1) # (b, l, d)
        Q, K = self.q_ln(Q).to(Q.dtype), self.k_ln(K).to(K.dtype)
        Q, K, V = map(self.add_heads, (Q, K, V)) # (b, n, l, h)
        if value_embedding is not None:
            if self.value_lambdas is None:
                raise ValueError("value embeddings were provided but config.value_embeddings is false")
            value_embedding = self.add_heads(value_embedding.to(dtype=V.dtype))
            V = self.value_lambdas[0] * V + self.value_lambdas[1] * value_embedding
        Q, K = self.rotary(Q), self.rotary(K)
        if self.attention_backend == "flex":
            A = self.flex_attention(
                Q,
                K,
                V,
                score_mod=None,
                block_mask=attention_mask,
            ) # (b, n, l, h)
        else:
            A = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attention_mask, scale=self.scale
            ) # (b, n, l, h)
        A = self.remove_heads(A) # (b, l, d)
        O = self.Wo(A)
        return O


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    ### Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, l, d = 8, 64, 512
    X = torch.rand((b, l, d)).to(device)

    config = PLMConfig(hidden_size=d)

    attention_layer = Attention(config=config).to(device)

    Y = attention_layer(X)
    print(Y.shape)

    rope_length, rope_head_size = l, attention_layer.head_size
    rope_input = torch.zeros((1, 1, rope_length, rope_head_size), device=device)
    rope_input[..., 0::2] = 1.0
    rope_output = attention_layer.rotary(rope_input)

    assert torch.allclose(
        rope_input.norm(dim=-1),
        rope_output.norm(dim=-1),
        atol=1e-5,
    ), "rope should preserve each token vector norm"
    assert torch.allclose(
        rope_input[:, :, 0, :],
        rope_output[:, :, 0, :],
        atol=1e-5,
    ), "position zero should not rotate"

    positions = torch.arange(rope_length).cpu()
    rope_values = rope_output[0, 0].detach().cpu()
    relative_similarity = (rope_values @ rope_values[0]) / rope_values[0].dot(rope_values[0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    scatter = axes[0].scatter(
        rope_values[:, 0],
        rope_values[:, 1],
        c=positions,
        s=14,
        cmap="viridis",
    )
    axes[0].plot(rope_values[:, 0], rope_values[:, 1], color="black", alpha=0.25, linewidth=1)
    axes[0].set_title("First RoPE Pair")
    axes[0].set_xlabel("dim 0")
    axes[0].set_ylabel("dim 1")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(scatter, ax=axes[0], label="position")

    heatmap = axes[1].imshow(
        rope_values.T,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        origin="lower",
    )
    axes[1].set_title("Rotated Head Features")
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("head dim")
    fig.colorbar(heatmap, ax=axes[1], label="value")

    axes[2].plot(positions, relative_similarity)
    axes[2].set_title("Similarity to Position 0")
    axes[2].set_xlabel("position offset")
    axes[2].set_ylabel("cosine similarity")
    axes[2].set_ylim(-1.05, 1.05)

    fig.tight_layout()
    plot_path = Path("rope_effect.png").resolve()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"saved {plot_path}")
