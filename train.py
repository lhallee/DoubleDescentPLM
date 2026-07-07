import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


from model.config import PLMConfig
from model.plm import PLM
from data.sampler import PDBClusteredDataset, TokenizeCollator


NUM_EPOCHS = 10
STEPS_PER_EPOCH = 100
BATCH_SIZE = 2
MAX_LENGTH = 128
HIDDEN_SIZE = 512
NUM_HIDDEN_LAYERS = 2
LR = 1e-4
PLOT_PATH = "loss.png"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = PDBClusteredDataset(mode="single", split="train", seed=42)
valid_dataset = PDBClusteredDataset(mode="single", split="valid", seed=42)
test_dataset = PDBClusteredDataset(mode="single", split="test", seed=42)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=TokenizeCollator(max_length=MAX_LENGTH, device=device)
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=TokenizeCollator(max_length=MAX_LENGTH, device=device)
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=TokenizeCollator(max_length=MAX_LENGTH, device=device)
)
valid_steps = len(valid_loader)
test_steps = len(test_loader)


model = PLM(config = PLMConfig(
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_losses, valid_losses = [], []

for epoch in tqdm(range(NUM_EPOCHS)):
    train_dataset.set_epoch(0)
    model.train()
    train_loss_tally = 0.0
    for step, batch in tqdm(zip(range(STEPS_PER_EPOCH), train_loader), total=STEPS_PER_EPOCH):

        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()

        train_loss_tally += loss.item()

    train_loss_average = train_loss_tally / STEPS_PER_EPOCH
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
plt.show()
#plt.savefig(PLOT_PATH, dpi=300)
#plt.close()
