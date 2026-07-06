import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import PLMConfig


def get_intermediate_size(hidden_size: int, expansion_ratio: float) -> int:
    return int(((hidden_size * expansion_ratio) + 255) // 256 * 256)


class MLP(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size # d
        self.intermediate_size = get_intermediate_size(config.hidden_size, config.expansion_ratio) # c
        self.norm = nn.LayerNorm(self.hidden_size)
        self.Wup = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.Wdown = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3, "Expect (b, l, d) tensor"
        x = self.norm(x)
        x1, x2 = self.Wup(x).chunk(2, dim=-1) # (b, l, c)
        x = F.silu(x1) * x2 # (b, l, c)
        x = self.Wdown(x) # (b, l, d)
        return x 


if __name__ == "__main__":
    ### Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b, l, d = 8, 64, 512
    X = torch.rand((b, l, d)).to(device)

    config = PLMConfig(hidden_size=d)

    mlp = MLP(config=config).to(device)

    Y = mlp(X)
    print(Y.shape)
