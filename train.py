import entrypoint_setup

import argparse
import copy

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import MODEL_CONFIGS
from data.sampler import PDBClusteredDataset, TokenizeCollator
from model.plm import PLM
from optimizer import build_optimizers
from utils import (
    apply_model_overrides,
    build_schedulers,
    dataloader_kwargs,
    move_to_device,
    seed_all,
    should_compile,
    step_optimizers,
    step_schedulers,
    update_muon_momentum,
    zero_grad,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        choices=[
            "small",
            "normal",
            "large",
            "wide",
            "deep"
        ],
        default="small"
    )
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--round_hidden_to", type=int, default=64)
    parser.add_argument("--head_size", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--expansion_ratio", type=float, default=None)
    parser.add_argument("--unet", action="store_true")
    parser.add_argument("--value_embeddings", action="store_true")
    parser.add_argument("--soft_logit_cap", type=float, default=None)
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--compile", dest="compile", action="store_true", default=None)
    parser.add_argument("--no_compile", dest="compile", action="store_false")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=True)
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--optimizer", type=str, choices=["adam", "muon"], default="adam")
    parser.add_argument("--muon_lr", type=float, default=1e-3)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_momentum_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_embed", type=float, default=None)
    parser.add_argument("--lr_head", type=float, default=None)
    parser.add_argument("--lr_scalar", type=float, default=None)
    parser.add_argument("--fused_adam", dest="fused_adam", action="store_true", default=None)
    parser.add_argument("--no_fused_adam", dest="fused_adam", action="store_false")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="none")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--switch_to_adam_loss", type=float, default=None)
    return parser.parse_args()


args = get_args()
seed_all()
model_config = apply_model_overrides(copy.deepcopy(MODEL_CONFIGS[args.config]), args)
args.plot_path = f"{args.config}.png"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compile_model = should_compile(args)
train_dataset = PDBClusteredDataset(mode="single", split="train", seed=42)
valid_dataset = PDBClusteredDataset(mode="single", split="valid", seed=42)
test_dataset = PDBClusteredDataset(mode="single", split="test", seed=42)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=TokenizeCollator(
        max_length=args.max_length,
        mask_rate=args.mask_rate,
    ),
    **dataloader_kwargs(
        args,
        device,
        drop_last=compile_model or model_config.attention_backend == "flex",
    ),
)
# we leave out works for eval sets since they are tiny
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=TokenizeCollator(
        max_length=args.max_length,
        mask_rate=args.mask_rate,
    ),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=TokenizeCollator(
        max_length=args.max_length,
        mask_rate=args.mask_rate,
    ),
)
train_steps = args.steps_per_epoch or len(train_loader)
valid_steps = len(valid_loader)
test_steps = len(test_loader)


model = PLM(config=model_config).to(device)
if args.bfloat16:
    model = model.bfloat16()
print(model)
print(f"attention_backend={model_config.attention_backend}")
print(f"compile_model={compile_model}")
print(f"bfloat16={args.bfloat16}")
if compile_model:
    model = torch.compile(model)
using_muon = args.optimizer == "muon"
optimizers = build_optimizers(
    model,
    adam_lr=args.lr,
    use_muon=using_muon,
    muon_lr=args.muon_lr,
    muon_momentum=args.muon_momentum,
    muon_ns_steps=args.muon_ns_steps,
    lr_embed=args.lr_embed,
    lr_head=args.lr_head,
    lr_scalar=args.lr_scalar,
    fused_adam=args.fused_adam,
    adam_betas=(args.adam_beta1, args.adam_beta2),
)
schedulers = build_schedulers(
    optimizers,
    args,
    total_steps=args.num_epochs * train_steps,
)

train_losses, valid_losses = [], []

for epoch in tqdm(range(args.num_epochs)):
    train_dataset.set_epoch(0) # repeatable selection to look at grokking
    model.train()
    train_loss_tally = 0.0
    for step, batch in tqdm(zip(range(train_steps), train_loader), total=train_steps):
        batch = move_to_device(batch, device)
        # train step
        zero_grad(optimizers)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        if using_muon:
            update_muon_momentum(
                optimizers,
                step=epoch * train_steps + step,
                warmup_steps=args.muon_momentum_warmup_steps,
            )
        step_optimizers(optimizers)
        step_schedulers(schedulers)

        train_loss_tally += loss.item()

    train_loss_average = train_loss_tally / train_steps
    train_losses.append(train_loss_average)
    if (
        using_muon
        and args.switch_to_adam_loss is not None
        and train_loss_average <= args.switch_to_adam_loss
    ):
        optimizers = build_optimizers(
            model,
            adam_lr=args.lr,
            lr_embed=args.lr_embed,
            lr_head=args.lr_head,
            lr_scalar=args.lr_scalar,
            fused_adam=args.fused_adam,
            adam_betas=(args.adam_beta1, args.adam_beta2),
        )
        schedulers = build_schedulers(
            optimizers,
            args,
            total_steps=args.num_epochs * train_steps,
        )
        using_muon = False
    
    valid_dataset.set_epoch(0)
    model.eval()
    valid_loss_tally = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, total=valid_steps):
            batch = move_to_device(batch, device)
            output = model(**batch)
            valid_loss_tally += output.loss.item()

    valid_loss_average = valid_loss_tally / valid_steps
    valid_losses.append(valid_loss_average)

    # plot every epoch so can look at over time
    assert len(train_losses) == len(valid_losses), "Expecting the same size"
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, valid_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=300)
    plt.close()
