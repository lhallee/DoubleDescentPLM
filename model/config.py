import sys
from typing import Optional

from transformers import PretrainedConfig


class PLMConfig(PretrainedConfig):
    model_type = "PLM"
    def __init__(
        self,
        vocab_size: int = 33,
        hidden_size: int = 768,
        head_size: int = 64,
        num_hidden_layers: int = 12,
        expansion_ratio: float = 2.0,
        unet: bool = False,
        attention_backend: str = "sdpa",
        value_embeddings: bool = False,
        soft_logit_cap: Optional[float] = None,
        **kwargs  
    ):
        super().__init__(**kwargs)
        if attention_backend not in {"sdpa", "flex"}:
            raise ValueError("attention_backend must be 'sdpa' or 'flex'")
        if attention_backend == "flex" and not sys.platform.startswith("linux"):
            raise ValueError("attention_backend='flex' is only supported on Linux")
        if soft_logit_cap is not None and soft_logit_cap <= 0:
            raise ValueError("soft_logit_cap must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_hidden_layers = num_hidden_layers
        self.expansion_ratio = expansion_ratio
        self.unet = unet
        self.attention_backend = attention_backend
        self.value_embeddings = value_embeddings
        self.soft_logit_cap = soft_logit_cap
