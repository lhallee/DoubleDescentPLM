import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from model.config import PLMConfig
from model.attention import Attention
from model.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attention(x=x, attention_mask=attention_mask)
        x = x + self.mlp(x=x)
        return x


class LMHead(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size, bias=True),
            nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


@dataclass
class PLMOutput(ModelOutput):
    logits: torch.Tensor = None
    loss: Optional[torch.Tensor] = None


class PLM(PreTrainedModel):
    config_class = PLMConfig
    def __init__(self, config: PLMConfig):
        super().__init__(config=config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config=config)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_size, bias=False)
        self.lm_head = LMHead(config=config)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        assert len(input_ids.shape) == 2, "Input ids should be (b, l)"
        b, l = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((b, l), device=x.device) 
        assert len(attention_mask.shape) == 2, "Expecting 2d attention mask (b, l)"
        assert input_ids.shape == attention_mask.shape, "Input ids and attention mask should be the same shape"
        attention_mask = attention_mask[:, None, None, :].bool() # (b, l, l)

        x = self.embedding(input_ids) # (b, l, d)
        for block in self.blocks:
            x = block(x=x, attention_mask=attention_mask)
        last_hidden_state = self.norm(x)
        logits = self.lm_head(last_hidden_state)

        if labels is not None:
            loss = self.ce_loss(logits.view(-1, self.vocab_size), labels.view(-1))
        else:
            loss = None

        return PLMOutput(
            logits=logits,
            loss=loss
        )


if __name__ == "__main__":
    ### Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, l, vocab_size, num_layers, hidden_size = 8, 64, 33, 2, 512
    input_ids = torch.randint(0, vocab_size, (b, l)).to(device)

    config = PLMConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        vocab_size=vocab_size
    )

    model = PLM(config=config).to(device)
    print(model)

    Y = model(input_ids)
    print(Y.logits.shape)
    print(Y.loss)
