import torch
from torch.utils.data import dataloader
from config import GPTConfig
from dataset import GPTDataset

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

x,y = next(iter(train_loader))