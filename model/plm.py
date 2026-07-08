from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from model.config import PLMConfig
from model.attention import Attention
from model.mlp import MLP


class ValueEmbedding(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.embed = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size, config.hidden_size)
                for _ in range(config.num_hidden_layers // 2)
            ]
        )

    def forward(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        encoder_values = [embedding(input_ids) for embedding in self.embed]
        return encoder_values + list(reversed(encoder_values))


class TransformerBlock(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        value_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            x=x,
            attention_mask=attention_mask,
            value_embedding=value_embedding,
        )
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
        self.soft_logit_cap = config.soft_logit_cap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        if self.soft_logit_cap is not None:
            logits = self.soft_logit_cap * torch.tanh(logits / self.soft_logit_cap)
        return logits


@dataclass
class PLMOutput(ModelOutput):
    logits: torch.Tensor = None
    loss: Optional[torch.Tensor] = None


class PLM(PreTrainedModel):
    config_class = PLMConfig
    def __init__(self, config: PLMConfig):
        super().__init__(config=config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.unet = config.unet
        if self.unet and config.num_hidden_layers % 2 != 0:
            raise ValueError("UNet PLM requires an even number of hidden layers")
        self.use_value_embeddings = config.value_embeddings
        if self.use_value_embeddings and not self.unet:
            raise ValueError("value_embeddings requires unet=True")
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        if self.use_value_embeddings:
            self.value_embeddings = ValueEmbedding(config)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config=config)
                for i in range(config.num_hidden_layers)
            ]
        )
        if self.unet:
            self.skip_weights = nn.Parameter(torch.ones(config.num_hidden_layers // 2))
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
            attention_mask = torch.ones((b, l), device=input_ids.device, dtype=torch.bool)
        assert len(attention_mask.shape) == 2, "Expecting 2d attention mask (b, l)"
        assert input_ids.shape == attention_mask.shape, "Input ids and attention mask should be the same shape"
        attention_mask = attention_mask.bool()

        x = self.embedding(input_ids) # (b, l, d)

        if self.unet:
            num_encoder_layers = len(self.blocks) // 2
            value_embeddings = (
                self.value_embeddings(input_ids)
                if self.use_value_embeddings
                else [None] * len(self.blocks)
            )
            skip_connections = []
            for layer_index, block in enumerate(self.blocks[:num_encoder_layers]):
                x = block(
                    x=x,
                    attention_mask=attention_mask,
                    value_embedding=value_embeddings[layer_index],
                )
                skip_connections.append(x)
            for i, block in enumerate(self.blocks[num_encoder_layers:]):
                x = x + self.skip_weights[i] * skip_connections.pop()
                layer_index = num_encoder_layers + i
                x = block(
                    x=x,
                    attention_mask=attention_mask,
                    value_embedding=value_embeddings[layer_index],
                )
        else:
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
