import argparse
import math
import random
import sys
from typing import Dict, List

import numpy as np
import torch


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def round_up(value: int, multiple: int) -> int:
    return int(((value + multiple - 1) // multiple) * multiple)


def resolve_attention_backend(args: argparse.Namespace) -> str:
    if args.max_length > 512 and sys.platform.startswith("linux"):
        return "flex"
    return "sdpa"


def apply_model_overrides(model_config, args: argparse.Namespace):
    if args.hidden_size is not None:
        model_config.hidden_size = round_up(args.hidden_size, args.round_hidden_to)
    if args.head_size is not None:
        model_config.head_size = args.head_size
    if args.num_hidden_layers is not None:
        model_config.num_hidden_layers = args.num_hidden_layers
    if args.expansion_ratio is not None:
        model_config.expansion_ratio = args.expansion_ratio
    if args.unet:
        model_config.unet = True
    model_config.attention_backend = resolve_attention_backend(args)
    if args.value_embeddings:
        model_config.value_embeddings = True
    if args.soft_logit_cap is not None:
        model_config.soft_logit_cap = args.soft_logit_cap
    if model_config.hidden_size % model_config.head_size != 0:
        raise ValueError("hidden_size must be divisible by head_size")
    if model_config.unet and model_config.num_hidden_layers % 2 != 0:
        raise ValueError("UNet requires an even number of hidden layers")
    if model_config.value_embeddings and not model_config.unet:
        raise ValueError("value_embeddings requires --unet")
    return model_config


def zero_grad(optimizers: List[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def step_optimizers(optimizers: List[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.step()


def build_schedulers(
    optimizers: List[torch.optim.Optimizer],
    args: argparse.Namespace,
    total_steps: int,
) -> List[torch.optim.lr_scheduler.LambdaLR]:
    if args.scheduler == "none":
        return []

    warmup_steps = max(0, args.lr_warmup_steps)
    total_steps = max(1, total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        cooldown_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / cooldown_steps))
        if args.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    return [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]


def step_schedulers(schedulers: List[torch.optim.lr_scheduler.LambdaLR]) -> None:
    for scheduler in schedulers:
        scheduler.step()


def update_muon_momentum(
    optimizers: List[torch.optim.Optimizer],
    step: int,
    warmup_steps: int,
) -> None:
    if warmup_steps <= 0:
        return
    frac = min(step / warmup_steps, 1.0)
    momentum = (1.0 - frac) * 0.85 + frac * 0.95
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            if "momentum" in group:
                group["momentum"] = momentum


def compile_default() -> bool:
    return sys.platform.startswith("linux")


def should_compile(args: argparse.Namespace) -> bool:
    if args.compile is None:
        return compile_default()
    return args.compile


def dataloader_kwargs(
    args: argparse.Namespace,
    device: torch.device,
    drop_last: bool = False,
) -> Dict:
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": drop_last,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return kwargs
