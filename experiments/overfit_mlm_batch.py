import argparse
import copy
import csv
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import entrypoint_setup

import matplotlib.pyplot as plt
import torch

from configs import MODEL_CONFIGS
from data.sampler import PDBClusteredDataset, TokenizeCollator
from model.plm import PLM
from optimizer import build_optimizers


def seed_all(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        choices=sorted(MODEL_CONFIGS),
        default="wide",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--mask_rate",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--round_hidden_to",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--head_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--expansion_ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--unet",
        action="store_true",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "muon"],
        default="adam",
    )
    parser.add_argument(
        "--muon_lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--muon_momentum",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--switch_to_adam_loss",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
    )
    parser.add_argument(
        "--dynamic_masks",
        action="store_true",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
    )
    return parser.parse_args()


def round_up(value: int, multiple: int) -> int:
    return int(((value + multiple - 1) // multiple) * multiple)


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
    if model_config.hidden_size % model_config.head_size != 0:
        raise ValueError("hidden_size must be divisible by head_size")
    if model_config.unet and model_config.num_hidden_layers % 2 != 0:
        raise ValueError("UNet requires an even number of hidden layers")
    return model_config


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    masked = labels != -100
    mask_count = int(masked.sum().item())
    if mask_count == 0:
        return math.nan, 0
    predictions = logits.argmax(dim=-1)
    correct = (predictions[masked] == labels[masked]).sum()
    return float((correct.float() / mask_count).item()), mask_count


def grad_norm(model: torch.nn.Module) -> float:
    total = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm(2)
        squared = norm * norm
        total = squared if total is None else total + squared
    if total is None:
        return 0.0
    return float(total.sqrt().item())


def zero_grad(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def step_optimizers(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.step()


def validate_batch(batch: dict[str, torch.Tensor], vocab_size: int) -> None:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    if int(input_ids.max().item()) >= vocab_size:
        raise ValueError(f"input_ids exceed model vocab size {vocab_size}")
    valid_labels = labels[labels != -100]
    if valid_labels.numel() == 0:
        raise ValueError("batch contains no masked labels")
    if int(valid_labels.max().item()) >= vocab_size:
        raise ValueError(f"labels exceed model vocab size {vocab_size}")


def make_plot(rows: list[dict[str, float]], plot_path: pathlib.Path) -> None:
    steps = [row["step"] for row in rows]
    losses = [row["loss"] for row in rows]
    accuracies = [row["accuracy"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(steps, losses)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Fixed-Batch MLM Loss")

    axes[1].plot(steps, accuracies)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Masked-token accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Masked Accuracy")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = get_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = apply_model_overrides(copy.deepcopy(MODEL_CONFIGS[args.config]), args)

    train_dataset = PDBClusteredDataset(mode="single", split="train", seed=args.seed)
    train_dataset.set_epoch(0)
    raw_batch = [train_dataset[index] for index in range(args.batch_size)]
    collator = TokenizeCollator(max_length=args.max_length, mask_rate=args.mask_rate)
    fixed_batch = collator(raw_batch)
    validate_batch(fixed_batch, model_config.vocab_size)

    mode = "dynamic" if args.dynamic_masks else "fixed"
    results_dir = ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = (
        f"overfit_mlm_{mode}_{args.config}_b{args.batch_size}_l{args.max_length}_"
        f"m{args.mask_rate:g}_lr{args.lr:g}_s{args.steps}_{stamp}"
    )
    csv_path = results_dir / f"{stem}.csv"
    plot_path = results_dir / f"{stem}.png"

    model = PLM(config=model_config).to(device)
    if args.compile:
        model = torch.compile(model)
    using_muon = args.optimizer == "muon"
    optimizers = build_optimizers(
        model,
        adam_lr=args.lr,
        use_muon=using_muon,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
    )

    fixed_batch = move_to_device(fixed_batch, device)
    model.train()
    rows = []
    start = time.perf_counter()

    print(f"device={device}")
    print(f"mode={mode}")
    print(f"config={args.config}")
    print(f"batch_size={args.batch_size}")
    print(f"max_length={args.max_length}")
    print(f"mask_rate={args.mask_rate}")
    print(f"hidden_size={model_config.hidden_size}")
    print(f"head_size={model_config.head_size}")
    print(f"num_hidden_layers={model_config.num_hidden_layers}")
    print(f"expansion_ratio={model_config.expansion_ratio}")
    print(f"unet={model_config.unet}")
    print(f"optimizer={args.optimizer}")
    print(f"steps={args.steps}")
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")

    for step in range(1, args.steps + 1):
        if args.dynamic_masks:
            batch = move_to_device(collator(raw_batch), device)
            validate_batch(batch, model_config.vocab_size)
        else:
            batch = fixed_batch

        zero_grad(optimizers)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        current_grad_norm = grad_norm(model)
        step_optimizers(optimizers)

        accuracy, mask_count = masked_accuracy(output.logits.detach(), batch["labels"])
        elapsed_seconds = time.perf_counter() - start
        row = {
            "step": step,
            "loss": float(loss.item()),
            "accuracy": accuracy,
            "mask_count": mask_count,
            "grad_norm": current_grad_norm,
            "lr": args.lr,
            "elapsed_seconds": elapsed_seconds,
        }
        rows.append(row)
        if (
            using_muon
            and args.switch_to_adam_loss is not None
            and row["loss"] <= args.switch_to_adam_loss
        ):
            optimizers = build_optimizers(model, adam_lr=args.lr)
            using_muon = False

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step={step} loss={row['loss']:.6f} acc={accuracy:.4f} "
                f"mask_count={mask_count} grad_norm={current_grad_norm:.4f} "
                f"elapsed={elapsed_seconds:.1f}s",
                flush=True,
            )

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    make_plot(rows, plot_path)
    print(f"saved_csv={csv_path}")
    print(f"saved_plot={plot_path}")


if __name__ == "__main__":
    main()
