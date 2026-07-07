import argparse

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.config import PLMConfig
from model.plm import PLM
from data.sampler import PDBClusteredDataset, TokenizeCollator
from configs import MODEL_CONFIGS


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
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    #parser.add_argument("--plot_path", type=str, default="loss.png")
    return parser.parse_args()


args = get_args()
model_config = MODEL_CONFIGS[args.config]
args.plot_path = f"{args.config}.png"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = PDBClusteredDataset(mode="single", split="train", seed=42)
valid_dataset = PDBClusteredDataset(mode="single", split="valid", seed=42)
test_dataset = PDBClusteredDataset(mode="single", split="test", seed=42)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=TokenizeCollator(max_length=args.max_length, device=device)
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=TokenizeCollator(max_length=args.max_length, device=device)
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=TokenizeCollator(max_length=args.max_length, device=device)
)
valid_steps = len(valid_loader)
test_steps = len(test_loader)


model = PLM(config=model_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_losses, valid_losses = [], []

for epoch in tqdm(range(args.num_epochs)):
    train_dataset.set_epoch(0) # repeatable selection to enable grokking
    model.train()
    train_loss_tally = 0.0
    for step, batch in tqdm(zip(range(args.steps_per_epoch), train_loader), total=args.steps_per_epoch):

        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()

        train_loss_tally += loss.item()

    train_loss_average = train_loss_tally / args.steps_per_epoch
    train_losses.append(train_loss_average)
    
    valid_dataset.set_epoch(0)
    model.eval()
    valid_loss_tally = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, total=valid_steps):
            output = model(**batch)
            valid_loss_tally += output.loss.item()

    valid_loss_average = valid_loss_tally / valid_steps
    valid_losses.append(valid_loss_average)


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
