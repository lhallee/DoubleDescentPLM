import argparse
import csv
import math
import pathlib
import random
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import entrypoint_setup

import torch
from torch.utils.data import DataLoader

from data.sampler import PDBClusteredDataset, TokenizeCollator
from model.config import PLMConfig
from model.plm import PLM
from optimizer import build_optimizers


@dataclass
class TrialConfig:
    trial_id: int
    hidden_size: int
    head_size: int
    num_hidden_layers: int
    expansion_ratio: float
    mask_rate: float
    batch_size: int
    grad_accum: int
    effective_batch_size: int
    lr: float
    muon_lr: float
    muon_momentum: float
    unet: bool


def seed_all(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--train_batches",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--target_acc",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--target_loss",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--prune_after_step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--prune_min_eval_acc",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min_hidden_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_hidden_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--hidden_multiple",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--head_size_choices",
        type=str,
        default="64,128",
    )
    parser.add_argument(
        "--layer_choices",
        type=str,
        default="2,4,6,8,10,12",
    )
    parser.add_argument(
        "--expansion_choices",
        type=str,
        default="2.0,2.5,3.0,4.0",
    )
    parser.add_argument(
        "--batch_size_choices",
        type=str,
        default="16,32,64,128,256",
    )
    parser.add_argument(
        "--grad_accum_choices",
        type=str,
        default="1,2,4",
    )
    parser.add_argument(
        "--max_effective_batch",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min_mask_rate",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--max_mask_rate",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--mask_rate_multiple",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=3e-3,
    )
    parser.add_argument(
        "--min_muon_lr",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--max_muon_lr",
        type=float,
        default=4e-3,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["muon", "adam"],
        default="muon",
    )
    parser.add_argument(
        "--no_unet",
        action="store_true",
    )
    parser.add_argument(
        "--switch_to_adam_loss",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fixed_hidden_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fixed_head_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fixed_num_hidden_layers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fixed_expansion_ratio",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fixed_mask_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fixed_batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fixed_grad_accum",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fixed_lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fixed_muon_lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fixed_muon_momentum",
        type=float,
        default=None,
    )
    return parser.parse_args()


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def parse_int_choices(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_choices(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def rounded_mask_rate(value: float, multiple: float) -> float:
    if multiple <= 0:
        return value
    return round(round(value / multiple) * multiple, 4)


def sample_trial(rng: random.Random, trial_id: int, args: argparse.Namespace) -> TrialConfig:
    hidden_choices = list(
        range(args.min_hidden_size, args.max_hidden_size + 1, args.hidden_multiple)
    )
    hidden_size = args.fixed_hidden_size or rng.choice(hidden_choices)
    head_choices = [
        head_size
        for head_size in parse_int_choices(args.head_size_choices)
        if hidden_size % head_size == 0
    ]
    if not head_choices:
        raise ValueError("No head_size_choices divide hidden_size")
    head_size = args.fixed_head_size or rng.choice(head_choices)
    layer_choices = parse_int_choices(args.layer_choices)
    if not args.no_unet:
        layer_choices = [layers for layers in layer_choices if layers % 2 == 0]
    if not layer_choices:
        raise ValueError("No valid layer_choices remain")
    num_hidden_layers = args.fixed_num_hidden_layers or rng.choice(layer_choices)
    batch_size = args.fixed_batch_size or rng.choice(
        parse_int_choices(args.batch_size_choices)
    )
    grad_accum_choices = parse_int_choices(args.grad_accum_choices)
    if args.max_effective_batch > 0:
        grad_accum_choices = [
            grad_accum
            for grad_accum in grad_accum_choices
            if batch_size * grad_accum <= args.max_effective_batch
        ]
    if not grad_accum_choices:
        raise ValueError("No valid grad_accum_choices remain")
    grad_accum = args.fixed_grad_accum or rng.choice(grad_accum_choices)
    return TrialConfig(
        trial_id=trial_id,
        hidden_size=hidden_size,
        head_size=head_size,
        num_hidden_layers=num_hidden_layers,
        expansion_ratio=args.fixed_expansion_ratio
        or rng.choice(parse_float_choices(args.expansion_choices)),
        mask_rate=args.fixed_mask_rate
        or rounded_mask_rate(
            rng.uniform(args.min_mask_rate, args.max_mask_rate),
            args.mask_rate_multiple,
        ),
        batch_size=batch_size,
        grad_accum=grad_accum,
        effective_batch_size=batch_size * grad_accum,
        lr=args.fixed_lr or log_uniform(rng, args.min_lr, args.max_lr),
        muon_lr=args.fixed_muon_lr
        or log_uniform(rng, args.min_muon_lr, args.max_muon_lr),
        muon_momentum=args.fixed_muon_momentum or rng.choice([0.90, 0.95]),
        unet=not args.no_unet,
    )


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def masked_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    masked = labels != -100
    mask_count = int(masked.sum().item())
    if mask_count == 0:
        return math.nan, 0
    predictions = logits.argmax(dim=-1)
    correct = (predictions[masked] == labels[masked]).sum()
    return float((correct.float() / mask_count).item()), mask_count


def zero_grad(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def step_optimizers(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.step()


def make_loader(
    dataset: PDBClusteredDataset,
    trial: TrialConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=trial.batch_size,
        shuffle=False,
        collate_fn=TokenizeCollator(
            max_length=args.max_length,
            mask_rate=trial.mask_rate,
        ),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=device.type == "cuda",
    )


def next_batch(
    loader: DataLoader,
    iterator,
    batch_index: int,
    max_batches: int,
):
    if iterator is None or (max_batches > 0 and batch_index >= max_batches):
        return iter(loader), 0
    return iterator, batch_index


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float, int]:
    model.eval()
    loss_tally = 0.0
    accuracy_tally = 0.0
    mask_tally = 0
    batch_count = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        output = model(**batch)
        accuracy, mask_count = masked_metrics(output.logits, batch["labels"])
        loss_tally += output.loss.item() * mask_count
        accuracy_tally += accuracy * mask_count
        mask_tally += mask_count
        batch_count += 1
        if max_batches > 0 and batch_count >= max_batches:
            break
    model.train()
    return loss_tally / mask_tally, accuracy_tally / mask_tally, mask_tally


def is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def run_trial(
    trial: TrialConfig,
    args: argparse.Namespace,
    dataset: PDBClusteredDataset,
    device: torch.device,
    summary_writer: csv.DictWriter,
    eval_writer: csv.DictWriter,
) -> None:
    trial_start = time.perf_counter()
    status = "running"
    best_acc = 0.0
    best_loss = math.inf
    reached_step = None
    reached_seconds = None
    parameter_count = 0
    optimizer_name = args.optimizer

    try:
        config = PLMConfig(
            hidden_size=trial.hidden_size,
            head_size=trial.head_size,
            num_hidden_layers=trial.num_hidden_layers,
            expansion_ratio=trial.expansion_ratio,
            unet=trial.unet,
        )
        model = PLM(config=config).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if args.compile:
            model = torch.compile(model)
        optimizers = build_optimizers(
            model,
            adam_lr=trial.lr,
            use_muon=args.optimizer == "muon",
            muon_lr=trial.muon_lr,
            muon_momentum=trial.muon_momentum,
        )
        loader = make_loader(dataset, trial, args, device)
        eval_loader = make_loader(dataset, trial, args, device)
        iterator = None
        batch_index = 0
        switched_to_adam = False

        model.train()
        for step in range(1, args.max_steps + 1):
            zero_grad(optimizers)
            train_loss = 0.0
            train_acc = 0.0
            train_masks = 0
            for _ in range(trial.grad_accum):
                iterator, batch_index = next_batch(
                    loader,
                    iterator,
                    batch_index,
                    args.train_batches,
                )
                try:
                    batch = next(iterator)
                    batch_index += 1
                except StopIteration:
                    iterator = iter(loader)
                    batch_index = 0
                    batch = next(iterator)
                    batch_index += 1
                batch = move_to_device(batch, device)
                output = model(**batch)
                accuracy, mask_count = masked_metrics(output.logits.detach(), batch["labels"])
                (output.loss / trial.grad_accum).backward()
                train_loss += output.loss.item() * mask_count
                train_acc += accuracy * mask_count
                train_masks += mask_count
            step_optimizers(optimizers)

            train_loss /= train_masks
            train_acc /= train_masks
            if (
                args.optimizer == "muon"
                and args.switch_to_adam_loss is not None
                and not switched_to_adam
                and train_loss <= args.switch_to_adam_loss
            ):
                optimizers = build_optimizers(model, adam_lr=trial.lr)
                switched_to_adam = True
                optimizer_name = "muon_to_adam"

            if step == 1 or step % args.eval_every == 0 or step == args.max_steps:
                eval_loss, eval_acc, eval_masks = evaluate(
                    model,
                    eval_loader,
                    device,
                    args.eval_batches,
                )
                elapsed = time.perf_counter() - trial_start
                best_acc = max(best_acc, eval_acc)
                best_loss = min(best_loss, eval_loss)
                eval_writer.writerow(
                    {
                        **asdict(trial),
                        "optimizer": optimizer_name,
                        "parameter_count": parameter_count,
                        "step": step,
                        "elapsed_seconds": elapsed,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "eval_masks": eval_masks,
                    }
                )
                print(
                    f"trial={trial.trial_id} step={step} "
                    f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if eval_acc >= args.target_acc and eval_loss <= args.target_loss:
                    status = "target_reached"
                    reached_step = step
                    reached_seconds = elapsed
                    break
                if (
                    args.prune_after_step > 0
                    and step >= args.prune_after_step
                    and eval_acc < args.prune_min_eval_acc
                ):
                    status = "pruned"
                    break
        if status == "running":
            status = "max_steps"
    except RuntimeError as error:
        if is_oom(error):
            status = "oom"
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            raise
    finally:
        elapsed = time.perf_counter() - trial_start
        summary_writer.writerow(
            {
                **asdict(trial),
                "optimizer": optimizer_name,
                "parameter_count": parameter_count,
                "status": status,
                "best_eval_loss": best_loss,
                "best_eval_acc": best_acc,
                "reached_step": reached_step,
                "reached_seconds": reached_seconds,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"trial={trial.trial_id} status={status} best_loss={best_loss:.4f} "
            f"best_acc={best_acc:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )


def main() -> None:
    args = get_args()
    seed_all(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PDBClusteredDataset(mode="single", split="train", seed=args.seed)
    dataset.set_epoch(0)

    results_dir = ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"hpo_memorize_summary_{stamp}.csv"
    eval_path = results_dir / f"hpo_memorize_eval_{stamp}.csv"

    summary_fields = [
        *asdict(sample_trial(random.Random(args.seed), 0, args)).keys(),
        "optimizer",
        "parameter_count",
        "status",
        "best_eval_loss",
        "best_eval_acc",
        "reached_step",
        "reached_seconds",
        "elapsed_seconds",
    ]
    eval_fields = [
        *asdict(sample_trial(random.Random(args.seed), 0, args)).keys(),
        "optimizer",
        "parameter_count",
        "step",
        "elapsed_seconds",
        "train_loss",
        "train_acc",
        "eval_loss",
        "eval_acc",
        "eval_masks",
    ]

    print(f"device={device}")
    print(f"dataset_size={len(dataset)}")
    print(f"train_batches={args.train_batches}")
    print(f"eval_batches={args.eval_batches}")
    print(f"summary_csv={summary_path}")
    print(f"eval_csv={eval_path}")

    with summary_path.open("w", newline="") as summary_handle:
        with eval_path.open("w", newline="") as eval_handle:
            summary_writer = csv.DictWriter(summary_handle, fieldnames=summary_fields)
            eval_writer = csv.DictWriter(eval_handle, fieldnames=eval_fields)
            summary_writer.writeheader()
            eval_writer.writeheader()
            for trial_id in range(args.trials):
                trial = sample_trial(rng, trial_id, args)
                print(f"trial={trial_id} config={asdict(trial)}", flush=True)
                run_trial(
                    trial,
                    args,
                    dataset,
                    device,
                    summary_writer,
                    eval_writer,
                )
                summary_handle.flush()
                eval_handle.flush()


if __name__ == "__main__":
    main()
