import torch
from torch.utils.data import dataloader
import torch.nn.functional as F
from config import GPTConfig
from dataset import GPTDataset
from model import GPT

config = GPTConfig()

batch_size = 16

train_dataset = GPTDataset(data_path="train.bin",block_size=config.block_size)

train_loader = dataloader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    pin_memory = True,
    num_workers = 2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = GPT(config)
model.to(device)

learning_rate = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

epochs = 100

model.train()

for step in range(epochs):
    x,y = next(iter(train_loader))
    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    B,T,C = logits.shape
    logits_flat = logits.view(B*T,C)
    y_flat = y.view(B*T)

    loss = F.cross_entropy(logits_flat,y_flat)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(f"Step {step} | Loss: {loss.item():.4f}")

