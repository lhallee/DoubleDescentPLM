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
        **kwargs  
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_hidden_layers = num_hidden_layers
        self.expansion_ratio = expansion_ratio
