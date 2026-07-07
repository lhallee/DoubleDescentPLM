from model.config import PLMConfig


MODEL_CONFIGS = {
    "small": PLMConfig(
        hidden_size=256,
        num_hidden_layers=2,
        expansion_ratio=2.0
    ),
    "normal": PLMConfig(
        hidden_size=768,
        num_hidden_layers=12,
        expansion_ratio=8/3,
    ),
    "large": PLMConfig(
        hidden_size=1536,
        num_hidden_layers=24,
        expansion_ratio=3.0
    ),
    "wide": PLMConfig(
        hidden_size=2048,
        num_hidden_layers=8,
        expansion_ratio=4.0
    ),
    "deep": PLMConfig(
        hidden_size=512,
        num_hidden_layers=48,
        expansion_ratio=8/3
    )
}